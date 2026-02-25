#!/usr/bin/env python3
import argparse
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class RunResult:
    model: str
    seed: int
    stage: str
    pair_acc: float
    margin: float
    metrics_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 3-seed preference benchmarks for base and post-training pair accuracy."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt2", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"],
        help="Model names to benchmark. Include one base and one instruction-tuned model.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    parser.add_argument("--mode", choices=["dpo", "kto"], default="dpo")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train_samples", type=int, default=8000)
    parser.add_argument("--eval_samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--base_output_dir", default="outputs/sweeps")
    return parser.parse_args()


def run_cmd(cmd: List[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def load_metrics(metrics_path: Path) -> Dict[str, float]:
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize(values: List[float]) -> str:
    if not values:
        return "nan +/- nan"
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std = math.sqrt(variance)
    return f"{mean:.4f} +/- {std:.4f}"


def main() -> None:
    args = parse_args()
    results: List[RunResult] = []
    base_dir = Path(args.base_output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        safe_model = model.replace("/", "__")
        for seed in args.seeds:
            eval_dir = base_dir / safe_model / f"seed{seed}" / "eval_only"
            train_dir = base_dir / safe_model / f"seed{seed}" / "trained"
            eval_dir.mkdir(parents=True, exist_ok=True)
            train_dir.mkdir(parents=True, exist_ok=True)

            eval_cmd = [
                "python3",
                "train_preference.py",
                "--mode",
                args.mode,
                "--eval_only",
                "--model_name_or_path",
                model,
                "--ref_model_name_or_path",
                model,
                "--device",
                args.device,
                "--seed",
                str(seed),
                "--train_samples",
                "1",
                "--eval_samples",
                str(args.eval_samples),
                "--max_length",
                str(args.max_length),
                "--output_dir",
                str(eval_dir),
            ]
            run_cmd(eval_cmd)
            eval_metrics_path = eval_dir / "metrics.json"
            eval_metrics = load_metrics(eval_metrics_path)
            results.append(
                RunResult(
                    model=model,
                    seed=seed,
                    stage="baseline",
                    pair_acc=float(eval_metrics.get("pair_acc", float("nan"))),
                    margin=float(eval_metrics.get("margin", eval_metrics.get("pair_margin", float("nan")))),
                    metrics_path=str(eval_metrics_path),
                )
            )

            train_cmd = [
                "python3",
                "train_preference.py",
                "--mode",
                args.mode,
                "--model_name_or_path",
                model,
                "--ref_model_name_or_path",
                model,
                "--device",
                args.device,
                "--seed",
                str(seed),
                "--train_samples",
                str(args.train_samples),
                "--eval_samples",
                str(args.eval_samples),
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--grad_accum_steps",
                str(args.grad_accum_steps),
                "--max_length",
                str(args.max_length),
                "--output_dir",
                str(train_dir),
            ]
            run_cmd(train_cmd)
            train_metrics_path = train_dir / "metrics.json"
            train_metrics = load_metrics(train_metrics_path)
            results.append(
                RunResult(
                    model=model,
                    seed=seed,
                    stage="trained",
                    pair_acc=float(train_metrics.get("pair_acc", float("nan"))),
                    margin=float(train_metrics.get("margin", train_metrics.get("pair_margin", float("nan")))),
                    metrics_path=str(train_metrics_path),
                )
            )

    print("\nPer-run results:")
    for r in results:
        print(
            f"model={r.model} seed={r.seed} stage={r.stage} "
            f"pair_acc={r.pair_acc:.4f} margin={r.margin:.4f} metrics={r.metrics_path}"
        )

    print("\nSummary:")
    for model in args.models:
        baseline = [r.pair_acc for r in results if r.model == model and r.stage == "baseline"]
        trained = [r.pair_acc for r in results if r.model == model and r.stage == "trained"]
        deltas = [t - b for b, t in zip(baseline, trained)]
        print(f"model={model}")
        print(f"  baseline pair_acc: {summarize(baseline)}")
        print(f"  trained  pair_acc: {summarize(trained)}")
        print(f"  delta    pair_acc: {summarize(deltas)}")


if __name__ == "__main__":
    main()
