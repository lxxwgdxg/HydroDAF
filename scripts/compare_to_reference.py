from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description="Compare reproduced public benchmark averages with reference values")
    p.add_argument("--current", type=Path, default=REPO_ROOT / "outputs" / "public_camels_main" / "public_camels_average_summary.csv")
    p.add_argument("--reference", type=Path, default=REPO_ROOT / "reference_results" / "Table_public_CAMELS_main_recommended_average_formatted.csv")
    p.add_argument("--metric", default="NSE")
    p.add_argument("--tolerance", type=float, default=0.03)
    args = p.parse_args()

    current = pd.read_csv(args.current)
    reference = pd.read_csv(args.reference)
    metric_col = f"{args.metric}_mean"
    if metric_col not in current.columns:
        raise ValueError(f"Current file must contain {metric_col}")

    reference_mean = reference[args.metric].str.split("+/-", regex=False).str[0].astype(float)
    reference = reference.assign(reference_mean=reference_mean)
    merged = current[["model", metric_col]].merge(reference[["model", "reference_mean"]], on="model")
    merged["absolute_difference"] = (merged[metric_col] - merged["reference_mean"]).abs()
    print(merged.to_string(index=False))
    max_diff = merged["absolute_difference"].max()
    if max_diff > args.tolerance:
        raise SystemExit(f"Maximum difference {max_diff:.4f} exceeds tolerance {args.tolerance:.4f}")
    print(f"All {args.metric} means are within tolerance {args.tolerance:.4f}")


if __name__ == "__main__":
    main()
