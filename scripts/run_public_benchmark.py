from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_selection(path: Path) -> dict[str, dict[str, list[str] | str]]:
    groups: dict[str, dict[str, list[str] | str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            group = row["camels_group"]
            groups.setdefault(group, {"source": "", "targets": []})
            if row["role"] == "source":
                groups[group]["source"] = row["gauge_id"]
            elif row["role"] == "target":
                groups[group]["targets"].append(row["gauge_id"])
            else:
                raise ValueError(f"Unknown role in {path}: {row['role']}")
    for group, spec in groups.items():
        if not spec["source"] or not spec["targets"]:
            raise ValueError(f"Selection group {group} must contain one source and at least one target")
    return groups


def run_group(args, group: str, source: str, targets: list[str]) -> None:
    data_dir = args.data_root / group
    out_dir = args.out_dir / group
    cmd = [
        sys.executable,
        str(REPO_ROOT / "src" / "hydrodaf_benchmark.py"),
        "run-group",
        "--data-dir",
        str(data_dir),
        "--out-dir",
        str(out_dir),
        "--source",
        source,
        "--targets",
        *targets,
        "--models",
        *args.models,
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--target-fraction",
        str(args.target_fraction),
        "--input-len",
        str(args.input_len),
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--epochs",
        str(args.epochs),
        "--pretrain-epochs",
        str(args.pretrain_epochs),
        "--finetune-epochs",
        str(args.finetune_epochs),
        "--lr",
        str(args.lr),
        "--lambda-max",
        str(args.lambda_max),
        "--nse-gamma",
        str(args.nse_gamma),
        "--source-weight",
        str(args.source_weight),
        "--target-weight",
        str(args.target_weight),
        "--domain-weight",
        str(args.domain_weight),
        "--feature-mode",
        args.feature_mode,
        "--device",
        args.device,
    ]
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def combine_outputs(out_dir: Path, groups: list[str]) -> None:
    frames = []
    for group in groups:
        path = out_dir / group / "metrics_raw.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing group metrics: {path}")
        df = pd.read_csv(path, dtype={"source": str, "target": str})
        df["camels_group"] = group
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(out_dir / "public_camels_metrics_raw.csv", index=False)

    station_summary = (
        raw.groupby(["camels_group", "target", "model", "target_fraction"], as_index=False)
        .agg(
            NSE_mean=("NSE", "mean"),
            NSE_sd=("NSE", "std"),
            KGE_mean=("KGE", "mean"),
            KGE_sd=("KGE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_sd=("RMSE", "std"),
            PBIAS_mean=("PBIAS", "mean"),
            PBIAS_sd=("PBIAS", "std"),
            R2_mean=("R2", "mean"),
            R2_sd=("R2", "std"),
        )
        .sort_values(["camels_group", "target", "NSE_mean"], ascending=[True, True, False])
    )
    station_summary.to_csv(out_dir / "public_camels_metrics_summary.csv", index=False)

    average_summary = (
        raw.groupby(["model"], as_index=False)
        .agg(
            NSE_mean=("NSE", "mean"),
            NSE_sd=("NSE", "std"),
            KGE_mean=("KGE", "mean"),
            KGE_sd=("KGE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_sd=("RMSE", "std"),
            PBIAS_mean=("PBIAS", "mean"),
            PBIAS_sd=("PBIAS", "std"),
        )
        .sort_values("NSE_mean", ascending=False)
    )
    average_summary.to_csv(out_dir / "public_camels_average_summary.csv", index=False)
    print(f"[combine] wrote combined outputs to {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the public CAMELS-US HydroDAF benchmark")
    p.add_argument("--selection", type=Path, default=REPO_ROOT / "configs" / "public_benchmark_main.csv")
    p.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "camels_us" / "selected")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "public_camels_main")
    p.add_argument("--models", nargs="+", default=["local_lstm", "source_only", "finetune_lstm", "regional_lstm", "daf_shared_v", "daf"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--target-fraction", type=float, default=0.3)
    p.add_argument("--input-len", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--pretrain-epochs", type=int, default=5)
    p.add_argument("--finetune-epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-max", type=float, default=0.0001)
    p.add_argument("--nse-gamma", type=float, default=0.1)
    p.add_argument("--source-weight", type=float, default=1.0)
    p.add_argument("--target-weight", type=float, default=4.0)
    p.add_argument("--domain-weight", type=float, default=1.0)
    p.add_argument("--feature-mode", choices=["mean_rain", "all_rain", "rain_stats"], default="mean_rain")
    p.add_argument("--device", default="cpu")
    p.add_argument("--combine-only", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    groups = read_selection(args.selection)
    if not args.combine_only:
        for group, spec in groups.items():
            run_group(args, group, str(spec["source"]), list(spec["targets"]))
    combine_outputs(args.out_dir, list(groups))


if __name__ == "__main__":
    main()
