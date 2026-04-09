import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, Dataset


# Custom Neural Network — PharmShed Super Dataset
# Based on architecture by Alfonso G. Bastias, Ph.D. (2026-04-04)
# Adapted to use super dataset (demographics + prescription features)

# How to run:
#
# Run with expanded features (demographics + prescription):
#   python custom_nn_super_dataset.py --feature-set expanded
#
# Run with demographics only (baseline comparison):
#   python custom_nn_super_dataset.py --feature-set demographics
#
# Run with class filtering (only drugs with recall >= 0.5 in CV):
#   python custom_nn_super_dataset.py --feature-set expanded --class-filter-csv ../Custom_NN/Results/expanded/new_nn_per_drug_metrics.csv --run-name subgroup
#
# Results are saved to:
#   Current work/Custom_NN/Results/<feature_set>/

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "superdataset_construction"
METADATA_DIR = ROOT / "grouped_dataset"
RESULTS_DIR = ROOT / "Custom_NN" / "Results"

DEMOGRAPHIC_CATEGORICAL = ["Sex", "Insurance_coverage", "Race_ethnicity"]
DEMOGRAPHIC_NUMERIC = ["Age", "Family_income"]
EXPANDED_CATEGORICAL = DEMOGRAPHIC_CATEGORICAL + ["Form"]
EXPANDED_NUMERIC = DEMOGRAPHIC_NUMERIC + ["Quantity", "Strength", "Day_Supply"]

DROP_COLUMNS = ["Unnamed: 0"]
TARGET_COL = "Drug"
GROUP_COL = "Person_ID"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cleaned_read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    drop_cols = [col for col in DROP_COLUMNS if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def load_training_frame(feature_set: str) -> pd.DataFrame:
    frame = cleaned_read_csv(DATA_DIR / "super_integrated_data.csv")
    metadata = cleaned_read_csv(METADATA_DIR / "metadata.csv")[["Observation_ID", "Person_ID"]]
    frame = frame.merge(metadata, on="Observation_ID", how="left")
    return frame


def load_validation_frame(feature_set: str) -> pd.DataFrame | None:
    return cleaned_read_csv(DATA_DIR / "super_data_2022.csv")


def get_feature_columns(feature_set: str) -> Tuple[List[str], List[str]]:
    if feature_set == "expanded":
        return EXPANDED_CATEGORICAL, EXPANDED_NUMERIC
    return DEMOGRAPHIC_CATEGORICAL, DEMOGRAPHIC_NUMERIC


def load_class_filter(class_filter_csv: Path | None) -> List[str] | None:
    if class_filter_csv is None:
        return None
    class_df = pd.read_csv(class_filter_csv)
    if "Drug" not in class_df.columns:
        raise ValueError("Class filter CSV must contain a 'Drug' column.")
    classes = class_df["Drug"].dropna().astype(str).tolist()
    if not classes:
        raise ValueError("Class filter CSV did not contain any classes.")
    return classes


def build_output_dir(base_dir: Path, feature_set: str, run_name: str | None) -> Path:
    if run_name:
        return base_dir / f"{feature_set}_{run_name}"
    return base_dir / feature_set


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


@dataclass
class TabularPreprocessor:
    categorical_cols: List[str]
    numeric_cols: List[str]
    categorical_maps: Dict[str, Dict[str, int]]
    numeric_medians: Dict[str, float]
    numeric_means: Dict[str, float]
    numeric_stds: Dict[str, float]

    @classmethod
    def fit(cls, frame: pd.DataFrame, categorical_cols: List[str], numeric_cols: List[str]) -> "TabularPreprocessor":
        categorical_maps: Dict[str, Dict[str, int]] = {}
        numeric_medians: Dict[str, float] = {}
        numeric_means: Dict[str, float] = {}
        numeric_stds: Dict[str, float] = {}

        for col in categorical_cols:
            values = frame[col].fillna("__MISSING__").astype(str)
            unique_values = sorted(values.unique().tolist())
            value_to_idx = {"__UNK__": 0}
            for idx, value in enumerate(unique_values, start=1):
                value_to_idx[value] = idx
            categorical_maps[col] = value_to_idx

        for col in numeric_cols:
            series = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            median = float(series.median()) if not series.dropna().empty else 0.0
            filled = series.fillna(median)
            mean = float(filled.mean())
            std = float(filled.std(ddof=0))
            numeric_medians[col] = median
            numeric_means[col] = mean
            numeric_stds[col] = std if std > 1e-6 else 1.0

        return cls(
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            categorical_maps=categorical_maps,
            numeric_medians=numeric_medians,
            numeric_means=numeric_means,
            numeric_stds=numeric_stds,
        )

    def transform(self, frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        cat_arrays: List[np.ndarray] = []
        for col in self.categorical_cols:
            mapping = self.categorical_maps[col]
            encoded = frame[col].fillna("__MISSING__").astype(str).map(lambda x: mapping.get(x, 0)).to_numpy(dtype=np.int64)
            cat_arrays.append(encoded)

        num_arrays: List[np.ndarray] = []
        for col in self.numeric_cols:
            series = (
                pd.to_numeric(frame[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(self.numeric_medians[col])
            )
            normalized = (series - self.numeric_means[col]) / self.numeric_stds[col]
            num_arrays.append(normalized.to_numpy(dtype=np.float32))

        cat_matrix = np.column_stack(cat_arrays).astype(np.int64) if cat_arrays else np.zeros((len(frame), 0), dtype=np.int64)
        num_matrix = np.column_stack(num_arrays).astype(np.float32) if num_arrays else np.zeros((len(frame), 0), dtype=np.float32)
        return cat_matrix, num_matrix

    def cardinalities(self) -> List[int]:
        return [len(self.categorical_maps[col]) for col in self.categorical_cols]


class MixedTabularDataset(Dataset):
    def __init__(self, cat_data: np.ndarray, num_data: np.ndarray, labels: np.ndarray):
        self.cat_data = torch.as_tensor(cat_data, dtype=torch.long)
        self.num_data = torch.as_tensor(num_data, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.cat_data[idx], self.num_data[idx], self.labels[idx]


class TabularClassifier(nn.Module):
    def __init__(
        self,
        cardinalities: List[int],
        num_numeric: int,
        num_classes: int,
        hidden_dims: List[int],
        dropout: float,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList()
        embedding_dims: List[int] = []

        for cardinality in cardinalities:
            dim = min(64, max(4, int(math.ceil(math.sqrt(cardinality)))))
            self.embeddings.append(nn.Embedding(cardinality, dim))
            embedding_dims.append(dim)

        input_dim = sum(embedding_dims) + num_numeric
        layers: List[nn.Module] = []

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, cat_x: torch.Tensor, num_x: torch.Tensor) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        if len(self.embeddings) > 0:
            embedded = [emb(cat_x[:, idx]) for idx, emb in enumerate(self.embeddings)]
            parts.append(torch.cat(embedded, dim=1))
        if num_x.shape[1] > 0:
            parts.append(num_x)
        x = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        return self.network(x)


def build_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes)
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for cat_x, num_x, _ in loader:
            logits = model(cat_x.to(device), num_x.to(device))
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def train_one_fold(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    categorical_cols: List[str],
    numeric_cols: List[str],
    label_encoder: LabelEncoder,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    device: torch.device,
) -> Tuple[nn.Module, TabularPreprocessor, LabelEncoder, Dict[str, float], pd.DataFrame]:
    y_train = label_encoder.transform(train_frame[TARGET_COL])
    y_val = label_encoder.transform(val_frame[TARGET_COL])

    preprocessor = TabularPreprocessor.fit(train_frame, categorical_cols, numeric_cols)
    X_train_cat, X_train_num = preprocessor.transform(train_frame)
    X_val_cat, X_val_num = preprocessor.transform(val_frame)

    train_ds = MixedTabularDataset(X_train_cat, X_train_num, y_train)
    val_ds = MixedTabularDataset(X_val_cat, X_val_num, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)

    model = TabularClassifier(
        cardinalities=preprocessor.cardinalities(),
        num_numeric=len(numeric_cols),
        num_classes=len(label_encoder.classes_),
        hidden_dims=[512, 256, 128],
        dropout=0.25,
    ).to(device)

    class_weights = build_class_weights(y_train, len(label_encoder.classes_)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_state = None
    best_score = -np.inf
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for cat_x, num_x, labels in train_loader:
            cat_x = cat_x.to(device)
            num_x = num_x.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(cat_x, num_x)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)

        val_pred = predict(model, val_loader, device)
        scores = metric_summary(y_val, val_pred)
        score = 0.7 * scores["macro_f1"] + 0.3 * scores["macro_recall"]

        print(
            f"    epoch {epoch:02d} | "
            f"train_loss={running_loss / len(train_ds):.4f} | "
            f"val_macro_f1={scores['macro_f1']:.4f} | "
            f"val_macro_recall={scores['macro_recall']:.4f}"
        )

        if score > best_score:
            best_score = score
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"    early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training failed to produce a model state.")

    model.load_state_dict(best_state)
    val_pred = predict(model, val_loader, device)
    scores = metric_summary(y_val, val_pred)

    report = classification_report(
        y_val,
        val_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0,
        output_dict=True,
    )
    per_class = (
        pd.DataFrame(report)
        .transpose()
        .reset_index()
        .rename(columns={"index": "Drug"})
    )
    per_class = per_class[~per_class["Drug"].isin(["accuracy", "macro avg", "weighted avg"])]
    return model, preprocessor, label_encoder, scores, per_class


def train_final_model(
    frame: pd.DataFrame,
    categorical_cols: List[str],
    numeric_cols: List[str],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> Tuple[nn.Module, TabularPreprocessor, LabelEncoder]:
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(frame[TARGET_COL])
    preprocessor = TabularPreprocessor.fit(frame, categorical_cols, numeric_cols)
    cat_data, num_data = preprocessor.transform(frame)
    dataset = MixedTabularDataset(cat_data, num_data, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = TabularClassifier(
        cardinalities=preprocessor.cardinalities(),
        num_numeric=len(numeric_cols),
        num_classes=len(label_encoder.classes_),
        hidden_dims=[512, 256, 128],
        dropout=0.20,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=build_class_weights(labels, len(label_encoder.classes_)).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for cat_x, num_x, batch_labels in loader:
            cat_x = cat_x.to(device)
            num_x = num_x.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(cat_x, num_x)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_labels.size(0)

        print(f"final epoch {epoch:02d} | train_loss={running_loss / len(dataset):.4f}")

    return model, preprocessor, label_encoder


def evaluate_external(
    model: nn.Module,
    preprocessor: TabularPreprocessor,
    label_encoder: LabelEncoder,
    frame: pd.DataFrame,
    device: torch.device,
    batch_size: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    known_mask = frame[TARGET_COL].isin(label_encoder.classes_)
    filtered = frame.loc[known_mask].reset_index(drop=True)
    y_true = label_encoder.transform(filtered[TARGET_COL])
    cat_data, num_data = preprocessor.transform(filtered)
    dataset = MixedTabularDataset(cat_data, num_data, y_true)
    loader = DataLoader(dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    y_pred = predict(model, loader, device)
    scores = metric_summary(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0,
        output_dict=True,
    )
    per_class = (
        pd.DataFrame(report)
        .transpose()
        .reset_index()
        .rename(columns={"index": "Drug"})
    )
    per_class = per_class[~per_class["Drug"].isin(["accuracy", "macro avg", "weighted avg"])]
    return scores, per_class


def save_bundle(
    output_dir: Path,
    model: nn.Module,
    preprocessor: TabularPreprocessor,
    label_encoder: LabelEncoder,
    categorical_cols: List[str],
    numeric_cols: List[str],
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "categorical_maps": preprocessor.categorical_maps,
        "numeric_medians": preprocessor.numeric_medians,
        "numeric_means": preprocessor.numeric_means,
        "numeric_stds": preprocessor.numeric_stds,
        "classes": label_encoder.classes_.tolist(),
    }
    torch.save(payload, output_dir / "new_nn_model.pt")


def run_pipeline(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device()
    categorical_cols, numeric_cols = get_feature_columns(args.feature_set)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = build_output_dir(RESULTS_DIR, args.feature_set, args.run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"device: {device}")
    print(f"feature_set: {args.feature_set}")

    frame = load_training_frame(args.feature_set)
    frame = sanitize_frame(frame, categorical_cols, numeric_cols)
    selected_classes = load_class_filter(args.class_filter_csv)
    if selected_classes is not None:
        frame = frame[frame[TARGET_COL].isin(selected_classes)].reset_index(drop=True)
        print(f"class-filtered rows: {len(frame):,}")
        print(f"class-filtered target classes: {frame[TARGET_COL].nunique()}")
    print(f"training rows: {len(frame):,}")
    print(f"target classes: {frame[TARGET_COL].nunique()}")

    if args.sample_rows:
        frame = frame.sample(n=min(args.sample_rows, len(frame)), random_state=args.seed).reset_index(drop=True)
        print(f"sampled training rows: {len(frame):,}")

    sgkf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    label_encoder_all = LabelEncoder()
    y_all = label_encoder_all.fit_transform(frame[TARGET_COL])
    groups = frame[GROUP_COL].astype(str).values if GROUP_COL in frame.columns else np.arange(len(frame))

    fold_metrics: List[Dict[str, float]] = []
    per_class_reports: List[pd.DataFrame] = []

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(frame, y_all, groups), start=1):
        print(f"\n{'=' * 72}")
        print(f"fold {fold_idx}/{args.n_splits}")
        print(f"{'=' * 72}")

        train_frame = frame.iloc[train_idx].reset_index(drop=True)
        val_frame = frame.iloc[val_idx].reset_index(drop=True)
        print(f"train rows: {len(train_frame):,} | val rows: {len(val_frame):,}")

        _, _, _, scores, per_class = train_one_fold(
            train_frame=train_frame,
            val_frame=val_frame,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            label_encoder=label_encoder_all,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            device=device,
        )

        scores["fold"] = fold_idx
        fold_metrics.append(scores)
        per_class["fold"] = fold_idx
        per_class_reports.append(per_class)
        print(json.dumps(scores, indent=2))

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(output_dir / "new_nn_cv_results.csv", index=False)

    cv_summary = pd.DataFrame(
        [
            {
                "model": "NewNNApproach",
                "feature_set": args.feature_set,
                "accuracy_mean": fold_metrics_df["accuracy"].mean(),
                "accuracy_std": fold_metrics_df["accuracy"].std(ddof=1),
                "macro_recall_mean": fold_metrics_df["macro_recall"].mean(),
                "macro_recall_std": fold_metrics_df["macro_recall"].std(ddof=1),
                "macro_f1_mean": fold_metrics_df["macro_f1"].mean(),
                "macro_f1_std": fold_metrics_df["macro_f1"].std(ddof=1),
                "weighted_f1_mean": fold_metrics_df["weighted_f1"].mean(),
                "weighted_f1_std": fold_metrics_df["weighted_f1"].std(ddof=1),
            }
        ]
    )
    cv_summary.to_csv(output_dir / "new_nn_cv_summary.csv", index=False)

    per_class_df = pd.concat(per_class_reports, ignore_index=True)
    per_class_mean = (
        per_class_df.groupby("Drug", as_index=False)[["precision", "recall", "f1-score", "support"]]
        .mean()
        .sort_values("recall", ascending=False)
    )
    per_class_mean.to_csv(output_dir / "new_nn_per_drug_metrics.csv", index=False)

    print("\ntraining final model on full 2014-2021 dataset")
    final_model, final_preprocessor, final_label_encoder = train_final_model(
        frame=frame,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        epochs=max(args.epochs, 10),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
    )
    save_bundle(
        output_dir=output_dir,
        model=final_model,
        preprocessor=final_preprocessor,
        label_encoder=final_label_encoder,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
    )

    validation_frame = load_validation_frame(args.feature_set)
    if validation_frame is not None:
        validation_frame = sanitize_frame(validation_frame, categorical_cols, numeric_cols)
        if selected_classes is not None:
            validation_frame = validation_frame[validation_frame[TARGET_COL].isin(selected_classes)].reset_index(drop=True)
        validation_scores, validation_per_class = evaluate_external(
            model=final_model,
            preprocessor=final_preprocessor,
            label_encoder=final_label_encoder,
            frame=validation_frame,
            device=device,
            batch_size=args.batch_size,
        )
        pd.DataFrame([validation_scores]).to_csv(output_dir / "new_nn_2022_summary.csv", index=False)
        validation_per_class.to_csv(output_dir / "new_nn_2022_per_drug_metrics.csv", index=False)
        print("\n2022 validation")
        print(json.dumps(validation_scores, indent=2))
    else:
        print("\nno external validation run for expanded mode because 2022 prescription features are not present.")

    run_config = {
        "feature_set": args.feature_set,
        "run_name": args.run_name,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "n_splits": args.n_splits,
        "sample_rows": args.sample_rows,
        "device": str(device),
        "class_filter_csv": str(args.class_filter_csv) if args.class_filter_csv else None,
        "selected_classes_count": len(selected_classes) if selected_classes is not None else None,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neural-network replacement for the RealMLP PharmShed workflow.")
    parser.add_argument("--feature-set", choices=["demographics", "expanded"], default="expanded")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--class-filter-csv", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
