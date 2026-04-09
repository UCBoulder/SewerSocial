import argparse
from pathlib import Path

import pandas as pd

# Alfonso G. Bastias, Ph.D.
# Date: 2026-04-04


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = ROOT / "All-Results" / "NewNNApproach" / "expanded" / "new_nn_per_drug_metrics.csv"
DEFAULT_OUTPUT = ROOT / "All-Results" / "NewNNApproach" / "expanded" / "top_classes_recall_ge_0_5.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a subgroup of drug classes from per-drug NN metrics.")
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-recall", type=float, default=0.5)
    parser.add_argument("--min-support", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.metrics_csv)

    required_cols = {"Drug", "recall", "support"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    filtered = df[(df["recall"] >= args.min_recall) & (df["support"] >= args.min_support)].copy()
    filtered = filtered.sort_values(["recall", "support", "f1-score"], ascending=[False, False, False])

    if args.top_k is not None:
        filtered = filtered.head(args.top_k).copy()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(args.output_csv, index=False)

    print(f"selected classes: {len(filtered)}")
    print(f"output: {args.output_csv}")
    if not filtered.empty:
        print(filtered[["Drug", "recall", "f1-score", "support"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
