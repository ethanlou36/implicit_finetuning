#!/usr/bin/env python3
"""
Minimal DPO + KTO training on real preference data (Anthropic/hh-rlhf).
"""

import argparse
import json
import math
import os
import random
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DPO or KTO on real preference data.")
    parser.add_argument("--mode", choices=["dpo", "kto"], required=True, help="Training objective.")
    parser.add_argument(
        "--dataset_name",
        default="Anthropic/hh-rlhf",
        help="HF dataset name with `chosen` and `rejected` columns.",
    )
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--eval_split", default="test")
    parser.add_argument("--model_name_or_path", default="distilgpt2")
    parser.add_argument(
        "--ref_model_name_or_path",
        default=None,
        help="Reference model for DPO/KTO. Defaults to --model_name_or_path.",
    )
    parser.add_argument("--output_dir", default="outputs/preference-model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_samples", type=int, default=2000)
    parser.add_argument("--eval_samples", type=int, default=500)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO/KTO temperature.")
    parser.add_argument(
        "--kto_target_kl",
        type=float,
        default=0.0,
        help="Simple KTO anchor term used for desirable/undesirable asymmetry.",
    )
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device. Use `cuda` on Google Colab GPU runtime.",
    )
    parser.add_argument(
        "--precision",
        choices=["auto", "fp32", "fp16", "bf16"],
        default="auto",
        help="Training precision. Mixed precision is only used on CUDA.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def shared_prefix(a: str, b: str) -> str:
    max_len = min(len(a), len(b))
    idx = 0
    while idx < max_len and a[idx] == b[idx]:
        idx += 1
    return a[:idx]


def split_pair(chosen: str, rejected: str) -> Optional[Tuple[str, str, str]]:
    markers = ["\n\nAssistant:", "\nAssistant:", "Assistant:"]
    for marker in markers:
        idx_c = chosen.rfind(marker)
        idx_r = rejected.rfind(marker)
        if idx_c != -1 and idx_r != -1 and idx_c == idx_r:
            prompt = chosen[: idx_c + len(marker)]
            chosen_resp = chosen[idx_c + len(marker) :].strip()
            rejected_resp = rejected[idx_r + len(marker) :].strip()
            if prompt.strip() and chosen_resp and rejected_resp:
                return prompt, chosen_resp, rejected_resp

    prefix = shared_prefix(chosen, rejected)
    chosen_resp = chosen[len(prefix) :].strip()
    rejected_resp = rejected[len(prefix) :].strip()
    prompt = prefix.strip()
    if prompt and chosen_resp and rejected_resp:
        return prompt, chosen_resp, rejected_resp
    return None


def build_pair_examples(dataset_name: str, split: str, limit: int) -> List[Dict[str, str]]:
    raw = load_dataset(dataset_name, split=split)
    pairs: List[Dict[str, str]] = []

    for ex in raw:
        chosen = ex.get("chosen")
        rejected = ex.get("rejected")
        if not chosen or not rejected:
            continue
        parsed = split_pair(chosen, rejected)
        if parsed is None:
            continue
        prompt, chosen_resp, rejected_resp = parsed
        pairs.append(
            {
                "prompt": prompt,
                "chosen": chosen_resp,
                "rejected": rejected_resp,
            }
        )
        if len(pairs) >= limit:
            break

    return pairs


def build_kto_examples_from_pairs(pairs: List[Dict[str, str]]) -> List[Dict[str, object]]:
    examples: List[Dict[str, object]] = []
    for pair in pairs:
        examples.append(
            {
                "prompt": pair["prompt"],
                "completion": pair["chosen"],
                "label": 1,
            }
        )
        examples.append(
            {
                "prompt": pair["prompt"],
                "completion": pair["rejected"],
                "label": 0,
            }
        )
    return examples


def encode_prompt_response(
    tokenizer: AutoTokenizer, prompt: str, response: str, max_length: int
) -> Optional[Dict[str, List[int]]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    response_ids = tokenizer(response, add_special_tokens=False).input_ids
    if tokenizer.eos_token_id is not None:
        response_ids = response_ids + [tokenizer.eos_token_id]

    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids

    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]

    if not any(label != -100 for label in labels):
        return None

    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def pad_batch(features: List[Dict[str, List[int]]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(len(feat["input_ids"]) for feat in features)

    input_ids, labels, attention_mask = [], [], []
    for feat in features:
        pad_len = max_len - len(feat["input_ids"])
        input_ids.append(feat["input_ids"] + [pad_token_id] * pad_len)
        labels.append(feat["labels"] + [-100] * pad_len)
        attention_mask.append(feat["attention_mask"] + [0] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def collate_pairs(
    batch: List[Dict[str, str]], tokenizer: AutoTokenizer, max_length: int
) -> Optional[Dict[str, torch.Tensor]]:
    chosen_features: List[Dict[str, List[int]]] = []
    rejected_features: List[Dict[str, List[int]]] = []
    for ex in batch:
        chosen_enc = encode_prompt_response(tokenizer, ex["prompt"], ex["chosen"], max_length=max_length)
        rejected_enc = encode_prompt_response(
            tokenizer, ex["prompt"], ex["rejected"], max_length=max_length
        )
        if chosen_enc is None or rejected_enc is None:
            continue
        chosen_features.append(chosen_enc)
        rejected_features.append(rejected_enc)

    if not chosen_features:
        return None

    chosen_batch = pad_batch(chosen_features, tokenizer.pad_token_id)
    rejected_batch = pad_batch(rejected_features, tokenizer.pad_token_id)

    out = {}
    for key, tensor in chosen_batch.items():
        out[f"chosen_{key}"] = tensor
    for key, tensor in rejected_batch.items():
        out[f"rejected_{key}"] = tensor
    return out


def collate_kto(
    batch: List[Dict[str, object]], tokenizer: AutoTokenizer, max_length: int
) -> Optional[Dict[str, torch.Tensor]]:
    features: List[Dict[str, List[int]]] = []
    labels: List[int] = []

    for ex in batch:
        enc = encode_prompt_response(tokenizer, str(ex["prompt"]), str(ex["completion"]), max_length=max_length)
        if enc is None:
            continue
        features.append(enc)
        labels.append(int(ex["label"]))

    if not features:
        return None

    tensor_batch = pad_batch(features, tokenizer.pad_token_id)
    tensor_batch["kto_label"] = torch.tensor(labels, dtype=torch.float)
    return tensor_batch


def sequence_log_probs(
    model: AutoModelForCausalLM, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    valid_mask = (shift_labels != -100).float()
    safe_labels = shift_labels.masked_fill(shift_labels == -100, 0)

    token_log_probs = F.log_softmax(shift_logits, dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    token_log_probs = token_log_probs * valid_mask
    return token_log_probs.sum(dim=-1)


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available.")
        return torch.device("cuda")
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested --device mps but MPS is not available.")
        return torch.device("mps")
    if device_arg == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_precision(precision_arg: str, device: torch.device) -> Tuple[str, Optional[torch.dtype]]:
    if device.type != "cuda":
        return "fp32", None
    if precision_arg == "fp32":
        return "fp32", None
    if precision_arg == "fp16":
        return "fp16", torch.float16
    if precision_arg == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Requested --precision bf16 but GPU does not support bf16.")
        return "bf16", torch.bfloat16

    if torch.cuda.is_bf16_supported():
        return "bf16", torch.bfloat16
    return "fp16", torch.float16


def dpo_forward(
    policy_model: AutoModelForCausalLM,
    ref_model: AutoModelForCausalLM,
    batch: Dict[str, torch.Tensor],
    beta: float,
) -> Dict[str, torch.Tensor]:
    policy_chosen = sequence_log_probs(
        policy_model, batch["chosen_input_ids"], batch["chosen_attention_mask"], batch["chosen_labels"]
    )
    policy_rejected = sequence_log_probs(
        policy_model,
        batch["rejected_input_ids"],
        batch["rejected_attention_mask"],
        batch["rejected_labels"],
    )

    with torch.no_grad():
        ref_chosen = sequence_log_probs(
            ref_model, batch["chosen_input_ids"], batch["chosen_attention_mask"], batch["chosen_labels"]
        )
        ref_rejected = sequence_log_probs(
            ref_model,
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["rejected_labels"],
        )

    policy_logratio = policy_chosen - policy_rejected
    ref_logratio = ref_chosen - ref_rejected
    advantage = beta * (policy_logratio - ref_logratio)

    loss = -F.logsigmoid(advantage).mean()
    pair_acc = (policy_chosen > policy_rejected).float().mean()
    dpo_acc = (advantage > 0).float().mean()
    margin = (policy_chosen - policy_rejected).mean()

    return {
        "loss": loss,
        "pair_acc": pair_acc,
        "dpo_acc": dpo_acc,
        "margin": margin,
        "reward": advantage.mean(),
    }


def kto_forward(
    policy_model: AutoModelForCausalLM,
    ref_model: AutoModelForCausalLM,
    batch: Dict[str, torch.Tensor],
    beta: float,
    target_kl: float,
) -> Dict[str, torch.Tensor]:
    policy_logp = sequence_log_probs(policy_model, batch["input_ids"], batch["attention_mask"], batch["labels"])
    with torch.no_grad():
        ref_logp = sequence_log_probs(ref_model, batch["input_ids"], batch["attention_mask"], batch["labels"])

    rewards = beta * (policy_logp - ref_logp)
    labels = batch["kto_label"]

    desirable_loss = -F.logsigmoid(rewards - target_kl)
    undesirable_loss = -F.logsigmoid(target_kl - rewards)
    loss = torch.where(labels > 0.5, desirable_loss, undesirable_loss).mean()

    pred = (rewards > 0).float()
    label_acc = (pred == labels).float().mean()
    pos_reward = rewards[labels > 0.5]
    neg_reward = rewards[labels <= 0.5]

    desired_reward = pos_reward.mean() if len(pos_reward) > 0 else torch.tensor(0.0, device=rewards.device)
    undesired_reward = neg_reward.mean() if len(neg_reward) > 0 else torch.tensor(0.0, device=rewards.device)

    return {
        "loss": loss,
        "label_acc": label_acc,
        "mean_reward": rewards.mean(),
        "desired_reward": desired_reward,
        "undesired_reward": undesired_reward,
    }


@torch.no_grad()
def evaluate_dpo(
    policy_model: AutoModelForCausalLM,
    ref_model: AutoModelForCausalLM,
    dataloader: DataLoader,
    device: torch.device,
    beta: float,
) -> Dict[str, float]:
    policy_model.eval()

    metrics = {
        "loss": 0.0,
        "pair_acc": 0.0,
        "dpo_acc": 0.0,
        "margin": 0.0,
        "reward": 0.0,
    }
    steps = 0

    for batch in dataloader:
        if batch is None:
            continue
        batch = to_device(batch, device)
        out = dpo_forward(policy_model, ref_model, batch, beta=beta)
        for key in metrics:
            metrics[key] += out[key].item()
        steps += 1

    if steps == 0:
        return {k: float("nan") for k in metrics}
    return {k: v / steps for k, v in metrics.items()}


@torch.no_grad()
def evaluate_kto(
    policy_model: AutoModelForCausalLM,
    ref_model: AutoModelForCausalLM,
    kto_dataloader: DataLoader,
    pair_dataloader: DataLoader,
    device: torch.device,
    beta: float,
    target_kl: float,
) -> Dict[str, float]:
    policy_model.eval()

    metrics = {
        "loss": 0.0,
        "label_acc": 0.0,
        "mean_reward": 0.0,
        "desired_reward": 0.0,
        "undesired_reward": 0.0,
    }
    steps = 0

    for batch in kto_dataloader:
        if batch is None:
            continue
        batch = to_device(batch, device)
        out = kto_forward(policy_model, ref_model, batch, beta=beta, target_kl=target_kl)
        for key in metrics:
            metrics[key] += out[key].item()
        steps += 1

    result = {k: (v / steps if steps else float("nan")) for k, v in metrics.items()}

    pair_eval = evaluate_dpo(policy_model, ref_model, pair_dataloader, device=device, beta=beta)
    result["pair_acc"] = pair_eval["pair_acc"]
    result["pair_margin"] = pair_eval["margin"]
    return result


def format_metric_value(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def print_metrics(title: str, metrics: Dict[str, float]) -> None:
    print(title)
    print("-" * len(title))
    key_width = max(len(k) for k in metrics)
    for key in sorted(metrics):
        print(f"{key.rjust(key_width)} : {format_metric_value(metrics[key])}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    precision, amp_dtype = resolve_precision(args.precision, device)
    print(f"Using device: {device}")
    print(f"Using precision: {precision}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    ref_name = args.ref_model_name_or_path or args.model_name_or_path
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=amp_dtype
    ).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(ref_name, torch_dtype=amp_dtype).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    print("Loading real preference dataset...")
    train_pairs = build_pair_examples(args.dataset_name, args.train_split, args.train_samples)
    eval_pairs = build_pair_examples(args.dataset_name, args.eval_split, args.eval_samples)
    print(f"Loaded {len(train_pairs)} train pairs and {len(eval_pairs)} eval pairs")

    if len(train_pairs) == 0 or len(eval_pairs) == 0:
        raise RuntimeError("No usable pairs were parsed from dataset. Check dataset columns and format.")

    pair_collator = partial(collate_pairs, tokenizer=tokenizer, max_length=args.max_length)
    train_pair_loader = DataLoader(
        train_pairs,
        batch_size=args.batch_size,
        shuffle=not args.eval_only,
        collate_fn=pair_collator,
        pin_memory=(device.type == "cuda"),
    )
    eval_pair_loader = DataLoader(
        eval_pairs,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pair_collator,
        pin_memory=(device.type == "cuda"),
    )

    if args.mode == "kto":
        train_kto = build_kto_examples_from_pairs(train_pairs)
        eval_kto = build_kto_examples_from_pairs(eval_pairs)
        kto_collator = partial(collate_kto, tokenizer=tokenizer, max_length=args.max_length)
        train_loader = DataLoader(
            train_kto,
            batch_size=args.batch_size,
            shuffle=not args.eval_only,
            collate_fn=kto_collator,
            pin_memory=(device.type == "cuda"),
        )
        eval_kto_loader = DataLoader(
            eval_kto,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=kto_collator,
            pin_memory=(device.type == "cuda"),
        )
        print(f"Built KTO examples: {len(train_kto)} train / {len(eval_kto)} eval")
    else:
        train_loader = train_pair_loader
        eval_kto_loader = None

    if not args.eval_only:
        optimizer = torch.optim.AdamW(
            policy_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        total_steps = math.ceil(len(train_loader) / args.grad_accum_steps) * args.epochs
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        print(f"Training {args.mode.upper()} for {args.epochs} epoch(s), total optimizer steps: {total_steps}")
        policy_model.train()
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and precision == "fp16"))
        global_step = 0
        optimizer.zero_grad(set_to_none=True)

        for epoch in range(args.epochs):
            progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
            running = {"loss": 0.0}
            running_steps = 0

            for step, batch in enumerate(progress):
                if batch is None:
                    continue
                batch = to_device(batch, device)

                use_amp = device.type == "cuda" and precision in {"fp16", "bf16"}
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    if args.mode == "dpo":
                        out = dpo_forward(policy_model, ref_model, batch, beta=args.beta)
                        loss = out["loss"]
                        running["pair_acc"] = running.get("pair_acc", 0.0) + out["pair_acc"].item()
                        running["dpo_acc"] = running.get("dpo_acc", 0.0) + out["dpo_acc"].item()
                    else:
                        out = kto_forward(
                            policy_model,
                            ref_model,
                            batch,
                            beta=args.beta,
                            target_kl=args.kto_target_kl,
                        )
                        loss = out["loss"]
                        running["label_acc"] = running.get("label_acc", 0.0) + out["label_acc"].item()

                running["loss"] += loss.item()
                running_steps += 1

                loss_for_backprop = loss / args.grad_accum_steps
                if scaler.is_enabled():
                    scaler.scale(loss_for_backprop).backward()
                else:
                    loss_for_backprop.backward()
                should_step = (
                    (step + 1) % args.grad_accum_steps == 0
                    or (step + 1) == len(train_loader)
                )
                if should_step:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.max_grad_norm)
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                if running_steps > 0 and (global_step % args.log_every == 0):
                    logs = {"loss": running["loss"] / running_steps}
                    if args.mode == "dpo":
                        logs["pair_acc"] = running.get("pair_acc", 0.0) / running_steps
                        logs["dpo_acc"] = running.get("dpo_acc", 0.0) / running_steps
                    else:
                        logs["label_acc"] = running.get("label_acc", 0.0) / running_steps
                    progress.set_postfix({k: f"{v:.4f}" for k, v in logs.items()})

            if args.mode == "dpo":
                eval_metrics = evaluate_dpo(
                    policy_model, ref_model, eval_pair_loader, device=device, beta=args.beta
                )
            else:
                eval_metrics = evaluate_kto(
                    policy_model,
                    ref_model,
                    eval_kto_loader,
                    eval_pair_loader,
                    device=device,
                    beta=args.beta,
                    target_kl=args.kto_target_kl,
                )
            print(f"Eval after epoch {epoch + 1}: {json.dumps(eval_metrics, indent=2)}")

    if args.mode == "dpo":
        final_metrics = evaluate_dpo(policy_model, ref_model, eval_pair_loader, device=device, beta=args.beta)
    else:
        final_metrics = evaluate_kto(
            policy_model,
            ref_model,
            eval_kto_loader,
            eval_pair_loader,
            device=device,
            beta=args.beta,
            target_kl=args.kto_target_kl,
        )

    print_metrics(f"Final metrics ({args.eval_split} split)", final_metrics)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    if args.save_model and not args.eval_only:
        policy_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Saved model + tokenizer to {args.output_dir}")
    else:
        print(f"Saved metrics to {args.output_dir}/metrics.json")


if __name__ == "__main__":
    main()
