import argparse
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from new_model_common import RESULTS_DIR, build_preprocessor, fit_final_model, prepare_training_data, run_cv, save_outputs

# Alfonso G. Bastias, Ph.D.
# date: 2026-04-04

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random Forest baseline for PharmShed subgroup or full-label training.")
    parser.add_argument("--feature-set", choices=["demographics", "expanded"], default="expanded")
    parser.add_argument("--class-filter-csv", type=Path, default=Path("All-Results/NewNNApproach/expanded/top_classes_recall_ge_0_5.csv"))
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default="new_randomforest_subgroup")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame, X, y, groups, label_encoder, categorical_cols, numeric_cols = prepare_training_data(
        feature_set=args.feature_set,
        class_filter_csv=args.class_filter_csv,
        sample_rows=args.sample_rows,
        seed=args.seed,
    )

    def model_factory():
        return Pipeline(
            [
                ("preprocessor", build_preprocessor(categorical_cols, numeric_cols, encoding="ordinal")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=args.n_estimators,
                        max_depth=args.max_depth,
                        min_samples_leaf=args.min_samples_leaf,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=args.seed,
                    ),
                ),
            ]
        )

    print(f"training rows: {len(frame):,}")
    print(f"target classes: {len(label_encoder.classes_)}")
    cv_results, per_class_reports = run_cv(X, y, groups, model_factory, args.n_splits, args.seed)
    final_model = fit_final_model(X, y, model_factory)

    output_dir = RESULTS_DIR / args.run_name
    save_outputs(
        output_dir=output_dir,
        model_name="RandomForest",
        run_name=args.run_name,
        cv_results=cv_results,
        per_class_reports=per_class_reports,
        label_encoder=label_encoder,
        final_model=final_model,
        config=vars(args),
    )


if __name__ == "__main__":
    main()
