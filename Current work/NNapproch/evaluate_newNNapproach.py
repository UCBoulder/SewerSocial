import json
from pathlib import Path

import pandas as pd

# Alfonso G. Bastias, Ph.D.
# date: 2026-04-04


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "All-Results"
DATA_DIR = ROOT / "All-Input-Data"
NN_RESULTS_ROOT = RESULTS_ROOT / "NewNNApproach"


def print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def print_metric_block(rows: list[tuple[str, object]]) -> None:
    label_width = max(len(label) for label, _ in rows)
    for label, value in rows:
        if isinstance(value, float):
            rendered = f"{value:.4f}"
        else:
            rendered = str(value)
        print(f"{label:<{label_width}} : {rendered}")


def print_table(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(df.to_string(index=False))


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def majority_baseline_accuracy() -> float:
    df = pd.read_csv(DATA_DIR / "integrated_data.csv", usecols=["Drug"])
    return float(df["Drug"].value_counts(normalize=True).iloc[0])


def load_run_config(mode_dir: Path) -> dict:
    return json.loads((mode_dir / "run_config.json").read_text())


def summarize_mode(mode: str) -> dict:
    mode_dir = NN_RESULTS_ROOT / mode
    cv_summary = load_csv(mode_dir / "new_nn_cv_summary.csv").iloc[0].to_dict()
    per_drug = load_csv(mode_dir / "new_nn_per_drug_metrics.csv")
    config = load_run_config(mode_dir)

    summary = {
        "mode": mode,
        "config": config,
        "accuracy_mean": float(cv_summary["accuracy_mean"]),
        "macro_recall_mean": float(cv_summary["macro_recall_mean"]),
        "macro_f1_mean": float(cv_summary["macro_f1_mean"]),
        "weighted_f1_mean": float(cv_summary["weighted_f1_mean"]),
        "median_recall": float(per_drug["recall"].median()),
        "median_f1": float(per_drug["f1-score"].median()),
        "classes": int(len(per_drug)),
        "classes_zero_recall": int((per_drug["recall"] == 0).sum()),
        "classes_recall_ge_0_5": int((per_drug["recall"] >= 0.5).sum()),
        "classes_recall_ge_0_8": int((per_drug["recall"] >= 0.8).sum()),
        "bottom_10": per_drug.sort_values("recall").head(10)[["Drug", "recall", "f1-score", "support"]].to_dict("records"),
        "top_10": per_drug.sort_values("recall", ascending=False).head(10)[["Drug", "recall", "f1-score", "support"]].to_dict("records"),
    }

    summary_path = mode_dir / "new_nn_quality_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def maybe_compare_realmlp() -> dict | None:
    path = RESULTS_ROOT / "RealMLP" / "realmlp_cv_results.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return {
        "accuracy_mean": float(df["accuracy"].mean()),
        "macro_recall_mean": float(df["macro_recall"].mean()),
        "macro_f1_mean": float(df["macro_f1"].mean()),
    }


def print_summary(summary: dict, majority_acc: float, realmlp: dict | None) -> None:
    print_section(f"Mode: {summary['mode']}")
    print("Config")
    print("------")
    print(json.dumps(summary["config"], indent=2))

    print("\nOverall Metrics")
    print("---------------")
    print_metric_block(
        [
            ("Accuracy", summary["accuracy_mean"]),
            ("Macro Recall", summary["macro_recall_mean"]),
            ("Macro F1", summary["macro_f1_mean"]),
            ("Weighted F1", summary["weighted_f1_mean"]),
            ("Median Recall", summary["median_recall"]),
            ("Median F1", summary["median_f1"]),
        ]
    )

    print("\nClass Coverage")
    print("--------------")
    print_metric_block(
        [
            ("Classes", summary["classes"]),
            ("Zero-recall classes", summary["classes_zero_recall"]),
            ("Classes with recall >= 0.5", summary["classes_recall_ge_0_5"]),
            ("Classes with recall >= 0.8", summary["classes_recall_ge_0_8"]),
            ("Majority-class baseline accuracy", majority_acc),
        ]
    )

    if realmlp:
        print("\nRealMLP Reference")
        print("-----------------")
        print_metric_block(
            [
                ("Accuracy", realmlp["accuracy_mean"]),
                ("Macro Recall", realmlp["macro_recall_mean"]),
                ("Macro F1", realmlp["macro_f1_mean"]),
            ]
        )

    print_table("Worst 10 classes by recall", pd.DataFrame(summary["bottom_10"]))
    print_table("Best 10 classes by recall", pd.DataFrame(summary["top_10"]))


def summarize_per_drug_subset(per_drug: pd.DataFrame, selected_drugs: list[str]) -> dict:
    subset = per_drug[per_drug["Drug"].isin(selected_drugs)].copy()
    missing = sorted(set(selected_drugs) - set(subset["Drug"]))
    return {
        "selected_classes": len(selected_drugs),
        "found_classes": int(len(subset)),
        "missing_classes": missing,
        "mean_recall": float(subset["recall"].mean()) if not subset.empty else float("nan"),
        "mean_f1": float(subset["f1-score"].mean()) if not subset.empty else float("nan"),
        "median_recall": float(subset["recall"].median()) if not subset.empty else float("nan"),
        "median_f1": float(subset["f1-score"].median()) if not subset.empty else float("nan"),
        "zero_recall_classes": int((subset["recall"] == 0).sum()) if not subset.empty else 0,
        "recall_ge_0_5_classes": int((subset["recall"] >= 0.5).sum()) if not subset.empty else 0,
        "recall_ge_0_8_classes": int((subset["recall"] >= 0.8).sum()) if not subset.empty else 0,
    }


def compare_full_vs_subgroup(subgroup_mode: str) -> dict | None:
    full_dir = NN_RESULTS_ROOT / "expanded"
    subgroup_dir = NN_RESULTS_ROOT / subgroup_mode
    if not full_dir.exists() or not subgroup_dir.exists():
        return None

    subgroup_config = load_run_config(subgroup_dir)
    class_filter_csv = subgroup_config.get("class_filter_csv")
    if not class_filter_csv:
        return None

    class_filter_path = Path(class_filter_csv)
    if not class_filter_path.is_absolute():
        class_filter_path = ROOT / class_filter_path

    selected_df = load_csv(class_filter_path)
    if "Drug" not in selected_df.columns:
        raise ValueError(f"Class filter file does not contain 'Drug': {class_filter_path}")
    selected_drugs = selected_df["Drug"].dropna().astype(str).tolist()

    full_per_drug = load_csv(full_dir / "new_nn_per_drug_metrics.csv")
    subgroup_per_drug = load_csv(subgroup_dir / "new_nn_per_drug_metrics.csv")
    full_cv = load_csv(full_dir / "new_nn_cv_summary.csv").iloc[0].to_dict()
    subgroup_cv = load_csv(subgroup_dir / "new_nn_cv_summary.csv").iloc[0].to_dict()

    full_subset = summarize_per_drug_subset(full_per_drug, selected_drugs)
    subgroup_subset = summarize_per_drug_subset(subgroup_per_drug, selected_drugs)

    comparison = {
        "subgroup_mode": subgroup_mode,
        "class_filter_csv": str(class_filter_path),
        "selected_classes": len(selected_drugs),
        "full_model_cv": {
            "accuracy_mean": float(full_cv["accuracy_mean"]),
            "macro_recall_mean": float(full_cv["macro_recall_mean"]),
            "macro_f1_mean": float(full_cv["macro_f1_mean"]),
            "weighted_f1_mean": float(full_cv["weighted_f1_mean"]),
        },
        "subgroup_model_cv": {
            "accuracy_mean": float(subgroup_cv["accuracy_mean"]),
            "macro_recall_mean": float(subgroup_cv["macro_recall_mean"]),
            "macro_f1_mean": float(subgroup_cv["macro_f1_mean"]),
            "weighted_f1_mean": float(subgroup_cv["weighted_f1_mean"]),
        },
        "full_model_on_selected_classes": full_subset,
        "subgroup_model_on_selected_classes": subgroup_subset,
        "delta_selected_classes": {
            "mean_recall": subgroup_subset["mean_recall"] - full_subset["mean_recall"],
            "mean_f1": subgroup_subset["mean_f1"] - full_subset["mean_f1"],
            "median_recall": subgroup_subset["median_recall"] - full_subset["median_recall"],
            "median_f1": subgroup_subset["median_f1"] - full_subset["median_f1"],
            "zero_recall_classes": subgroup_subset["zero_recall_classes"] - full_subset["zero_recall_classes"],
        },
    }

    comparison_path = subgroup_dir / "full_vs_subgroup_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2))
    return comparison


def print_full_vs_subgroup(comparison: dict) -> None:
    print_section(f"Full vs Subgroup Comparison: {comparison['subgroup_mode']}")
    print_metric_block(
        [
            ("Class filter", comparison["class_filter_csv"]),
            ("Selected classes", comparison["selected_classes"]),
        ]
    )

    full_cv = comparison["full_model_cv"]
    subgroup_cv = comparison["subgroup_model_cv"]
    comparison_table = pd.DataFrame(
        [
            {
                "Model": "Full expanded",
                "Accuracy": full_cv["accuracy_mean"],
                "Macro Recall": full_cv["macro_recall_mean"],
                "Macro F1": full_cv["macro_f1_mean"],
                "Weighted F1": full_cv["weighted_f1_mean"],
            },
            {
                "Model": "Subgroup",
                "Accuracy": subgroup_cv["accuracy_mean"],
                "Macro Recall": subgroup_cv["macro_recall_mean"],
                "Macro F1": subgroup_cv["macro_f1_mean"],
                "Weighted F1": subgroup_cv["weighted_f1_mean"],
            },
        ]
    )
    print_table("Cross-validation summary", comparison_table.round(4))

    full_subset = comparison["full_model_on_selected_classes"]
    subgroup_subset = comparison["subgroup_model_on_selected_classes"]
    delta = comparison["delta_selected_classes"]

    subset_table = pd.DataFrame(
        [
            {
                "View": "Full model on selected classes",
                "Mean Recall": full_subset["mean_recall"],
                "Mean F1": full_subset["mean_f1"],
                "Median Recall": full_subset["median_recall"],
                "Median F1": full_subset["median_f1"],
                "Zero Recall Classes": full_subset["zero_recall_classes"],
            },
            {
                "View": "Subgroup model on selected classes",
                "Mean Recall": subgroup_subset["mean_recall"],
                "Mean F1": subgroup_subset["mean_f1"],
                "Median Recall": subgroup_subset["median_recall"],
                "Median F1": subgroup_subset["median_f1"],
                "Zero Recall Classes": subgroup_subset["zero_recall_classes"],
            },
            {
                "View": "Delta (subgroup - full)",
                "Mean Recall": delta["mean_recall"],
                "Mean F1": delta["mean_f1"],
                "Median Recall": delta["median_recall"],
                "Median F1": delta["median_f1"],
                "Zero Recall Classes": delta["zero_recall_classes"],
            },
        ]
    )
    print_table("Selected-class subset comparison", subset_table.round(4))

    if full_subset["missing_classes"]:
        print(f"Missing classes in full model subset: {full_subset['missing_classes']}")
    if subgroup_subset["missing_classes"]:
        print(f"Missing classes in subgroup model subset: {subgroup_subset['missing_classes']}")


def main() -> None:
    majority_acc = majority_baseline_accuracy()
    realmlp = maybe_compare_realmlp()

    for mode in ["expanded", "demographics"]:
        mode_dir = NN_RESULTS_ROOT / mode
        if mode_dir.exists():
            summary = summarize_mode(mode)
            print_summary(summary, majority_acc, realmlp)
            print()

    for subgroup_dir in sorted(NN_RESULTS_ROOT.glob("expanded_subgroup*")):
        if subgroup_dir.is_dir():
            comparison = compare_full_vs_subgroup(subgroup_dir.name)
            if comparison is not None:
                print_full_vs_subgroup(comparison)
                print()


if __name__ == "__main__":
    main()
