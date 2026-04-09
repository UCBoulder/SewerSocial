import json
from pathlib import Path

import pandas as pd

# Alfonso G. Bastias, Ph.D.
# date: 2026-04-04

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "All-Results"
NN_ROOT = RESULTS_ROOT / "NewNNApproach"
CLASSICAL_ROOT = RESULTS_ROOT / "NewClassicalModels"
REALMLP_ROOT = RESULTS_ROOT / "RealMLP"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def summarize_per_drug_subset(per_drug: pd.DataFrame, selected_drugs: list[str]) -> dict:
    subset = per_drug[per_drug["Drug"].isin(selected_drugs)].copy()
    return {
        "classes_found": int(len(subset)),
        "mean_recall_selected": float(subset["recall"].mean()) if not subset.empty else float("nan"),
        "mean_f1_selected": float(subset["f1-score"].mean()) if not subset.empty else float("nan"),
        "median_recall_selected": float(subset["recall"].median()) if not subset.empty else float("nan"),
        "median_f1_selected": float(subset["f1-score"].median()) if not subset.empty else float("nan"),
        "zero_recall_selected": int((subset["recall"] == 0).sum()) if not subset.empty else 0,
        "recall_ge_0_8_selected": int((subset["recall"] >= 0.8).sum()) if not subset.empty else 0,
    }


def collect_realmlp() -> dict | None:
    path = REALMLP_ROOT / "realmlp_cv_results.csv"
    if not path.exists():
        return None
    df = load_csv(path)
    return {
        "model_group": "RealMLP",
        "run_name": "realmlp",
        "feature_set": "demographics",
        "scope": "full_217",
        "accuracy_mean": float(df["accuracy"].mean()),
        "macro_recall_mean": float(df["macro_recall"].mean()),
        "macro_f1_mean": float(df["macro_f1"].mean()),
        "weighted_f1_mean": float(df["micro_f1"].mean()) if "micro_f1" in df.columns else float(df["accuracy"].mean()),
    }


def collect_nn_models(selected_drugs: list[str]) -> list[dict]:
    rows = []
    candidates = [
        ("expanded", "NN", "full_217"),
        ("expanded_subgroup_recall_ge_0_5", "NN", "subgroup_59"),
    ]
    for folder, model_group, scope in candidates:
        mode_dir = NN_ROOT / folder
        if not mode_dir.exists():
            continue
        summary = load_csv(mode_dir / "new_nn_cv_summary.csv").iloc[0]
        per_drug = load_csv(mode_dir / "new_nn_per_drug_metrics.csv")
        config = load_json(mode_dir / "run_config.json")
        row = {
            "model_group": model_group,
            "run_name": folder,
            "feature_set": config.get("feature_set"),
            "scope": scope,
            "accuracy_mean": float(summary["accuracy_mean"]),
            "macro_recall_mean": float(summary["macro_recall_mean"]),
            "macro_f1_mean": float(summary["macro_f1_mean"]),
            "weighted_f1_mean": float(summary["weighted_f1_mean"]),
        }
        row.update(summarize_per_drug_subset(per_drug, selected_drugs))
        rows.append(row)
    return rows


def infer_model_group(run_name: str) -> str:
    lowered = run_name.lower()
    if "randomforest" in lowered:
        return "RandomForest"
    if "decisiontree" in lowered:
        return "DecisionTree"
    if "svm" in lowered:
        return "LinearSVM"
    if "knn" in lowered:
        return "KNN"
    return run_name


def collect_classical_models(selected_drugs: list[str]) -> list[dict]:
    rows = []
    for run_dir in sorted(CLASSICAL_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        run_name = run_dir.name
        if run_name.startswith("smoke_"):
            continue
        summary_files = list(run_dir.glob("*_cv_summary.csv"))
        per_drug_files = list(run_dir.glob("*_per_drug_metrics.csv"))
        if not summary_files or not per_drug_files:
            continue

        summary = load_csv(summary_files[0]).iloc[0]
        per_drug = load_csv(per_drug_files[0])
        config = load_json(run_dir / "run_config.json")
        row = {
            "model_group": infer_model_group(run_name),
            "run_name": run_name,
            "feature_set": config.get("feature_set"),
            "scope": "subgroup_59" if config.get("class_filter_csv") else "full_217",
            "accuracy_mean": float(summary["accuracy_mean"]),
            "macro_recall_mean": float(summary["macro_recall_mean"]),
            "macro_f1_mean": float(summary["macro_f1_mean"]),
            "weighted_f1_mean": float(summary["weighted_f1_mean"]),
        }
        row.update(summarize_per_drug_subset(per_drug, selected_drugs))
        rows.append(row)
    return rows


def main() -> None:
    subgroup_filter = load_csv(NN_ROOT / "expanded" / "top_classes_recall_ge_0_5.csv")
    selected_drugs = subgroup_filter["Drug"].dropna().astype(str).tolist()

    rows = []
    realmlp = collect_realmlp()
    if realmlp is not None:
        rows.append(realmlp)
    rows.extend(collect_nn_models(selected_drugs))
    rows.extend(collect_classical_models(selected_drugs))

    comparison = pd.DataFrame(rows)
    comparison = comparison.sort_values(
        ["scope", "macro_f1_mean", "macro_recall_mean", "accuracy_mean"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    output_path = RESULTS_ROOT / "model_comparison_all.csv"
    comparison.to_csv(output_path, index=False)

    display_cols = [
        "model_group",
        "run_name",
        "scope",
        "feature_set",
        "accuracy_mean",
        "macro_recall_mean",
        "macro_f1_mean",
        "weighted_f1_mean",
        "mean_recall_selected",
        "mean_f1_selected",
        "zero_recall_selected",
    ]

    print("\nAll model comparison")
    print("--------------------")
    print(comparison[display_cols].round(4).to_string(index=False))
    print(f"\nSaved comparison table to: {output_path}")


if __name__ == "__main__":
    main()
