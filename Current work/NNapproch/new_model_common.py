import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler

# Alfonso G. Bastias, Ph.D.
# date: 2026-04-04

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "All-Input-Data"
RESULTS_DIR = ROOT / "All-Results" / "NewClassicalModels"
TARGET_COL = "Drug"
GROUP_COL = "Person_ID"
DROP_COLUMNS = ["Unnamed: 0"]

DEMOGRAPHIC_CATEGORICAL = ["Sex", "Insurance_coverage", "Race_ethnicity"]
DEMOGRAPHIC_NUMERIC = ["Age", "Family_income"]
EXPANDED_CATEGORICAL = DEMOGRAPHIC_CATEGORICAL + ["Form", "Form.Units", "Administration_route"]
EXPANDED_NUMERIC = DEMOGRAPHIC_NUMERIC + ["Quantity", "Strength", "Day_Supply", "Daily_Frequency", "Daily_Dosage"]


def cleaned_read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    drop_cols = [col for col in DROP_COLUMNS if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def get_feature_columns(feature_set: str) -> Tuple[List[str], List[str]]:
    if feature_set == "expanded":
        return EXPANDED_CATEGORICAL, EXPANDED_NUMERIC
    return DEMOGRAPHIC_CATEGORICAL, DEMOGRAPHIC_NUMERIC


def load_training_frame(feature_set: str) -> pd.DataFrame:
    integrated = cleaned_read_csv(DATA_DIR / "integrated_data.csv")
    metadata = cleaned_read_csv(DATA_DIR / "metadata.csv")[["Observation_ID", "Person_ID"]]
    frame = integrated.merge(metadata, on="Observation_ID", how="left")

    if feature_set == "expanded":
        prescription = cleaned_read_csv(DATA_DIR / "prescription.csv").drop(columns=[TARGET_COL], errors="ignore")
        frame = frame.merge(prescription, on="Observation_ID", how="left")

    return frame


def load_class_filter(class_filter_csv: Path | None) -> List[str] | None:
    if class_filter_csv is None:
        return None
    if not class_filter_csv.is_absolute():
        class_filter_csv = ROOT / class_filter_csv
    class_df = pd.read_csv(class_filter_csv)
    if "Drug" not in class_df.columns:
        raise ValueError("Class filter CSV must contain a 'Drug' column.")
    classes = class_df["Drug"].dropna().astype(str).tolist()
    if not classes:
        raise ValueError("Class filter CSV did not contain any classes.")
    return classes


def sanitize_frame(frame: pd.DataFrame, categorical_cols: List[str], numeric_cols: List[str]) -> pd.DataFrame:
    expected_cols = categorical_cols + numeric_cols + [TARGET_COL]
    if GROUP_COL in frame.columns:
        expected_cols.append(GROUP_COL)

    missing_cols = [col for col in categorical_cols + numeric_cols + [TARGET_COL] if col not in frame.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    data = frame[expected_cols].copy()
    for col in categorical_cols:
        data[col] = data[col].fillna("__MISSING__").astype(str)
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    return data


def prepare_training_data(feature_set: str, class_filter_csv: Path | None, sample_rows: int | None, seed: int):
    categorical_cols, numeric_cols = get_feature_columns(feature_set)
    frame = load_training_frame(feature_set)
    frame = sanitize_frame(frame, categorical_cols, numeric_cols)

    selected_classes = load_class_filter(class_filter_csv)
    if selected_classes is not None:
        frame = frame[frame[TARGET_COL].isin(selected_classes)].reset_index(drop=True)

    if sample_rows is not None and sample_rows < len(frame):
        frame = frame.sample(n=sample_rows, random_state=seed).reset_index(drop=True)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(frame[TARGET_COL])
    X = frame[categorical_cols + numeric_cols].copy()
    groups = frame[GROUP_COL].astype(str).values if GROUP_COL in frame.columns else np.arange(len(frame))
    return frame, X, y, groups, label_encoder, categorical_cols, numeric_cols


def build_preprocessor(categorical_cols: List[str], numeric_cols: List[str], encoding: str) -> ColumnTransformer:
    if encoding == "onehot":
        categorical_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        numeric_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    elif encoding == "ordinal":
        categorical_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]
        )
        numeric_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")

    return ColumnTransformer(
        [
            ("cat", categorical_transformer, categorical_cols),
            ("num", numeric_transformer, numeric_cols),
        ]
    )


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def run_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    model_factory,
    n_splits: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_metrics = []
    per_class_reports = []

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups), start=1):
        print(f"\n{'=' * 72}")
        print(f"fold {fold_idx}/{n_splits}")
        print(f"{'=' * 72}")
        print(f"train rows: {len(train_idx):,} | val rows: {len(val_idx):,}")

        model = model_factory()
        model.fit(X.iloc[train_idx], y[train_idx])
        preds = model.predict(X.iloc[val_idx])

        scores = metric_summary(y[val_idx], preds)
        scores["fold"] = fold_idx
        fold_metrics.append(scores)
        print(json.dumps(scores, indent=2))

        report = classification_report(y[val_idx], preds, zero_division=0, output_dict=True)
        report_df = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"})
        report_df["fold"] = fold_idx
        per_class_reports.append(report_df)

    return pd.DataFrame(fold_metrics), pd.concat(per_class_reports, ignore_index=True)


def fit_final_model(X: pd.DataFrame, y: np.ndarray, model_factory):
    model = model_factory()
    model.fit(X, y)
    return model


def save_outputs(
    output_dir: Path,
    model_name: str,
    run_name: str,
    cv_results: pd.DataFrame,
    per_class_reports: pd.DataFrame,
    label_encoder: LabelEncoder,
    final_model,
    config: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cv_results.to_csv(output_dir / f"{run_name}_cv_results.csv", index=False)

    cv_summary = pd.DataFrame(
        [
            {
                "model": model_name,
                "accuracy_mean": cv_results["accuracy"].mean(),
                "accuracy_std": cv_results["accuracy"].std(ddof=1),
                "macro_recall_mean": cv_results["macro_recall"].mean(),
                "macro_recall_std": cv_results["macro_recall"].std(ddof=1),
                "macro_f1_mean": cv_results["macro_f1"].mean(),
                "macro_f1_std": cv_results["macro_f1"].std(ddof=1),
                "weighted_f1_mean": cv_results["weighted_f1"].mean(),
                "weighted_f1_std": cv_results["weighted_f1"].std(ddof=1),
            }
        ]
    )
    cv_summary.to_csv(output_dir / f"{run_name}_cv_summary.csv", index=False)

    per_class = per_class_reports.copy()
    mask = ~per_class["label"].isin(["accuracy", "macro avg", "weighted avg"])
    per_class = per_class[mask].copy()
    per_class["label"] = per_class["label"].astype(int)
    per_class["Drug"] = label_encoder.inverse_transform(per_class["label"])
    per_drug = (
        per_class.groupby("Drug", as_index=False)[["precision", "recall", "f1-score", "support"]]
        .mean()
        .sort_values("recall", ascending=False)
    )
    per_drug.to_csv(output_dir / f"{run_name}_per_drug_metrics.csv", index=False)

    with open(output_dir / f"{run_name}_model.pkl", "wb") as handle:
        pickle.dump(final_model, handle)

    serializable_config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    (output_dir / "run_config.json").write_text(json.dumps(serializable_config, indent=2))
