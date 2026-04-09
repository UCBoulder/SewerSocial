import argparse
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from new_model_common import RESULTS_DIR, build_preprocessor, fit_final_model, prepare_training_data, run_cv, save_outputs

# Alfonso G. Bastias, Ph.D.
# date: 2026-04-04


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KNN baseline for PharmShed subgroup training.")
    parser.add_argument("--feature-set", choices=["demographics", "expanded"], default="expanded")
    parser.add_argument("--class-filter-csv", type=Path, default=Path("All-Results/NewNNApproach/expanded/top_classes_recall_ge_0_5.csv"))
    parser.add_argument("--sample-rows", type=int, default=40000)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default="new_knn_subgroup")
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--weights", choices=["uniform", "distance"], default="distance")
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
                ("preprocessor", build_preprocessor(categorical_cols, numeric_cols, encoding="onehot")),
                (
                    "model",
                    KNeighborsClassifier(
                        n_neighbors=args.n_neighbors,
                        weights=args.weights,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    print(f"training rows: {len(frame):,}")
    print(f"target classes: {len(label_encoder.classes_)}")
    print("note: KNN defaults to a sampled subgroup-sized run because full-scale training is impractical.")
    cv_results, per_class_reports = run_cv(X, y, groups, model_factory, args.n_splits, args.seed)
    final_model = fit_final_model(X, y, model_factory)

    output_dir = RESULTS_DIR / args.run_name
    save_outputs(
        output_dir=output_dir,
        model_name="KNN",
        run_name=args.run_name,
        cv_results=cv_results,
        per_class_reports=per_class_reports,
        label_encoder=label_encoder,
        final_model=final_model,
        config=vars(args),
    )


if __name__ == "__main__":
    main()
