#!/usr/bin/env python3
"""Batch MedSAM experiment runner."""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

DEFAULT_DATASETS: List[str] = ["kvasir", "glas", "busi", "isic", "brats"]
MODEL_VARIANTS: Dict[str, str] = {
    "sam": "vit_b",
    "sam_adapter": "vit_adapter",
    "sam_lora": "vit_lora",
    "sam_prompt": "vit_prompt",
    "sam_adapter_lora": "vit_adapter_lora",
    "sam_adapter_prompt": "vit_adapter_prompt",
    "sam_lora_prompt": "vit_lora_prompt",
    "sam_adapter_lora_prompt": "vit_adapter_lora_prompt",
}
DEFAULT_TRAIN_RATIOS: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9]
METRIC_PATTERN = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+Train Loss:\s+([0-9.]+)\s+\|\s+Val Loss:\s+([0-9.]+)\s+"
    r"\|\s+Val mIoU:\s+([0-9.]+)\s+\|\s+Val Dice:\s+([0-9.]+)\s+\|\s+Val Acc:\s+([0-9.]+)"
)
SPLIT_PATTERN = re.compile(r"\[MedSAM\] train samples=(\d+) val samples=(\d+)")
DEFAULT_TRAIN_SCRIPT = Path(__file__).resolve().parent / "train_one_gpu_sam_adapter.py"


@dataclass
class RunResult:
    dataset: str
    model_name: str
    model_type: str
    train_ratio: float
    epochs: int
    status: str
    return_code: int
    duration_sec: float
    log_path: Path
    checkpoint_dir: Path
    task_name: str
    run_tag: str
    train_samples: Optional[int]
    val_samples: Optional[int]
    best_epoch: Optional[int]
    best_val_loss: Optional[float]
    best_miou: Optional[float]
    best_dice: Optional[float]
    best_acc: Optional[float]
    last_epoch: Optional[int]
    last_val_loss: Optional[float]
    last_miou: Optional[float]
    last_dice: Optional[float]
    last_acc: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MedSAM sweeps across models/datasets")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--models", nargs="+", default=list(MODEL_VARIANTS.keys()))
    parser.add_argument(
        "--train-ratios",
        nargs="+",
        type=float,
        default=DEFAULT_TRAIN_RATIOS,
        dest="train_ratios",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2, dest="batch_size")
    parser.add_argument("--num-workers", type=int, default=4, dest="num_workers")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/medsam_sweeps"),
        help="Directory to store logs/results",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("work_dir"),
        help="Directory that training scripts use for checkpoints",
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=DEFAULT_TRAIN_SCRIPT,
        help="Path to train_one_gpu_sam_adapter.py",
    )
    parser.add_argument("--log-filename", type=str, default="train.log")
    parser.add_argument("--task-prefix", type=str, default="auto_medsam")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--stream-stdout", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def slugify(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]+", "-", value)


def ratio_tag(ratio: float) -> str:
    return f"{int(round(ratio * 100)):03d}"


def validate_selections(options: Iterable[str], valid: Iterable[str], label: str) -> List[str]:
    valid_lower = {v.lower(): v.lower() for v in valid}
    ordered: List[str] = []
    for item in options:
        key = item.lower()
        if key not in valid_lower:
            raise ValueError(f"Unknown {label}: {item}")
        if key not in ordered:
            ordered.append(key)
    return ordered


def ensure_ratios(ratios: Iterable[float]) -> List[float]:
    cleaned: List[float] = []
    for ratio in ratios:
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"Train ratio must be within (0, 1), got {ratio}")
        cleaned.append(ratio)
    return cleaned


def run_trainer(cmd: List[str], log_path: Path, stream_stdout: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            if stream_stdout:
                sys.stdout.write(line)
        return process.wait()


def parse_log(log_path: Path) -> Dict[str, Optional[float]]:
    history = []
    train_samples = None
    val_samples = None
    if not log_path.exists():
        return {"history": history, "train_samples": None, "val_samples": None, "best": None, "last": None}
    with log_path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            metric_match = METRIC_PATTERN.search(line)
            if metric_match:
                epoch, train_loss, val_loss, miou, dice, acc = metric_match.groups()
                history.append(
                    {
                        "epoch": int(epoch),
                        "train_loss": float(train_loss),
                        "val_loss": float(val_loss),
                        "miou": float(miou),
                        "dice": float(dice),
                        "acc": float(acc),
                    }
                )
            split_match = SPLIT_PATTERN.search(line)
            if split_match:
                train_samples = int(split_match.group(1))
                val_samples = int(split_match.group(2))
    best = max(history, key=lambda item: item["dice"]) if history else None
    last = history[-1] if history else None
    return {
        "history": history,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "best": best,
        "last": last,
    }


def summarize_run(
    dataset: str,
    model_name: str,
    model_type: str,
    ratio: float,
    epochs: int,
    status: str,
    return_code: int,
    duration: float,
    log_path: Path,
    checkpoint_dir: Path,
    task_name: str,
    run_tag: str,
    log_summary: Dict[str, Optional[float]],
) -> RunResult:
    best = log_summary.get("best") or {}
    last = log_summary.get("last") or {}
    return RunResult(
        dataset=dataset,
        model_name=model_name,
        model_type=model_type,
        train_ratio=ratio,
        epochs=epochs,
        status=status,
        return_code=return_code,
        duration_sec=duration,
        log_path=log_path,
        checkpoint_dir=checkpoint_dir,
        task_name=task_name,
        run_tag=run_tag,
        train_samples=log_summary.get("train_samples"),
        val_samples=log_summary.get("val_samples"),
        best_epoch=best.get("epoch"),
        best_val_loss=best.get("val_loss"),
        best_miou=best.get("miou"),
        best_dice=best.get("dice"),
        best_acc=best.get("acc"),
        last_epoch=last.get("epoch"),
        last_val_loss=last.get("val_loss"),
        last_miou=last.get("miou"),
        last_dice=last.get("dice"),
        last_acc=last.get("acc"),
    )


def write_tables(results: List[RunResult], output_root: Path) -> None:
    if not results:
        return
    csv_path = output_root / "results.csv"
    json_path = output_root / "results.json"
    fieldnames = [
        "dataset",
        "model_name",
        "model_type",
        "train_ratio",
        "epochs",
        "status",
        "return_code",
        "duration_sec",
        "log_path",
        "checkpoint_dir",
        "task_name",
        "run_tag",
        "train_samples",
        "val_samples",
        "best_epoch",
        "best_val_loss",
        "best_miou",
        "best_dice",
        "best_acc",
        "last_epoch",
        "last_val_loss",
        "last_miou",
        "last_dice",
        "last_acc",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({
                "dataset": row.dataset,
                "model_name": row.model_name,
                "model_type": row.model_type,
                "train_ratio": row.train_ratio,
                "epochs": row.epochs,
                "status": row.status,
                "return_code": row.return_code,
                "duration_sec": f"{row.duration_sec:.1f}",
                "log_path": str(row.log_path),
                "checkpoint_dir": str(row.checkpoint_dir),
                "task_name": row.task_name,
                "run_tag": row.run_tag,
                "train_samples": row.train_samples,
                "val_samples": row.val_samples,
                "best_epoch": row.best_epoch,
                "best_val_loss": row.best_val_loss,
                "best_miou": row.best_miou,
                "best_dice": row.best_dice,
                "best_acc": row.best_acc,
                "last_epoch": row.last_epoch,
                "last_val_loss": row.last_val_loss,
                "last_miou": row.last_miou,
                "last_dice": row.last_dice,
                "last_acc": row.last_acc,
            })
    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump([row.__dict__ for row in results], json_file, indent=2, default=str)


def main() -> None:
    args = parse_args()
    datasets = validate_selections(args.datasets, DEFAULT_DATASETS, "dataset")
    models = validate_selections(args.models, MODEL_VARIANTS.keys(), "model")
    ratios = ensure_ratios(args.train_ratios)
    train_script = args.train_script.resolve()
    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")
    python_exec = sys.executable
    output_root = args.output_root.resolve()
    work_dir = args.work_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    plan = [(d, m, r) for d in datasets for m in models for r in ratios]
    if args.dry_run:
        for dataset, model, ratio in plan:
            model_type = MODEL_VARIANTS[model]
            print(
                f"[DRY-RUN] {python_exec} {train_script} -dataset_name {dataset} "
                f"-model_type {model_type} --train_ratio {ratio:.2f}"
            )
        return
    results: List[RunResult] = []
    summary_log = output_root / "experiment_log.txt"
    launched = 0
    for dataset, model, ratio in plan:
        if args.max_runs is not None and launched >= args.max_runs:
            break
        model_type = MODEL_VARIANTS[model]
        ratio_label = ratio_tag(ratio)
        run_dir = output_root / dataset / model / f"ratio_{ratio_label}"
        log_path = run_dir / args.log_filename
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_tag = slugify(f"{timestamp}_{dataset}_{model}_r{ratio_label}")
        task_name = slugify(f"{args.task_prefix}_{dataset}_{model}")
        checkpoint_dir = work_dir / task_name / f"{model_type}-{run_tag}"
        cmd = [
            python_exec,
            str(train_script),
            "-dataset_name",
            dataset,
            "-model_type",
            model_type,
            "-num_epochs",
            str(args.epochs),
            "-batch_size",
            str(args.batch_size),
            "-num_workers",
            str(args.num_workers),
            "--device",
            args.device,
            "--train_ratio",
            f"{ratio:.4f}",
            "--run_tag",
            run_tag,
            "-task_name",
            task_name,
            "-work_dir",
            str(work_dir),
        ]
        if args.skip_existing and log_path.exists():
            log_summary = parse_log(log_path)
            results.append(
                summarize_run(
                    dataset,
                    model,
                    model_type,
                    ratio,
                    args.epochs,
                    status="skipped",
                    return_code=0,
                    duration=0.0,
                    log_path=log_path,
                    checkpoint_dir=checkpoint_dir,
                    task_name=task_name,
                    run_tag=run_tag,
                    log_summary=log_summary,
                )
            )
            continue
        start = time.time()
        return_code = run_trainer(cmd, log_path, args.stream_stdout)
        duration = time.time() - start
        status = "success" if return_code == 0 else "failed"
        log_summary = parse_log(log_path)
        results.append(
            summarize_run(
                dataset,
                model,
                model_type,
                ratio,
                args.epochs,
                status,
                return_code,
                duration,
                log_path,
                checkpoint_dir,
                task_name,
                run_tag,
                log_summary,
            )
        )
        launched += 1
        with summary_log.open("a", encoding="utf-8") as tracker:
            tracker.write(
                f"{datetime.now().isoformat()} | {dataset} | {model} | ratio={ratio:.2f} | "
                f"status={status} | duration={duration/60:.1f} min | log={log_path}\n"
            )
    write_tables(results, output_root)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        sys.exit(1)
