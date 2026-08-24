from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "readmitted_within_30_days"
PATIENT_COLUMN = "patient_nbr"
RAW_TARGET_COLUMN = "readmitted"
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

CATEGORICAL_FEATURES = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "medical_specialty",
    "diag_1",
    "diag_2",
    "diag_3",
    "max_glu_serum",
    "A1Cresult",
    "change",
    "diabetesMed",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class DatasetSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def create_target(df: pd.DataFrame) -> pd.Series:
    """Return 1 only for readmission within 30 days, otherwise 0."""
    if RAW_TARGET_COLUMN not in df.columns:
        raise KeyError(f"Missing required target column: {RAW_TARGET_COLUMN}")
    return df[RAW_TARGET_COLUMN].astype("string").eq("<30").astype(int)


def prepare_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select model features and normalize types consistently for train/inference."""
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"Dataset is missing required feature columns: {missing}")

    X = df[FEATURE_COLUMNS].copy()
    X = X.replace("?", np.nan)

    for column in NUMERIC_FEATURES:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    for column in CATEGORICAL_FEATURES:
        values = X[column]
        X[column] = values.where(values.isna(), values.astype(str))

    return X


def build_classifier(random_state: int = RANDOM_STATE) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=20),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2_000,
        solver="liblinear",
        random_state=random_state,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def _valid_stratify(labels: pd.Series) -> pd.Series | None:
    counts = labels.value_counts(dropna=False)
    return labels if len(counts) > 1 and counts.min() >= 2 else None


def split_by_patient(
    df: pd.DataFrame,
    *,
    patient_column: str = PATIENT_COLUMN,
    target_column: str = TARGET_COLUMN,
    train_size: float = 0.70,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> DatasetSplit:
    """Create disjoint patient-level train/validation/test encounter splits.

    Patients are stratified by whether they have at least one positive encounter.
    This prevents encounters from the same patient appearing in multiple splits.
    """
    if not np.isclose(train_size + validation_size + test_size, 1.0):
        raise ValueError("train_size + validation_size + test_size must equal 1.0")
    if patient_column not in df.columns or target_column not in df.columns:
        raise KeyError(f"Required columns: {patient_column}, {target_column}")

    patient_summary = (
        df[[patient_column, target_column]]
        .groupby(patient_column, as_index=False)[target_column]
        .max()
    )

    train_patients, holdout_patients = train_test_split(
        patient_summary,
        test_size=validation_size + test_size,
        random_state=random_state,
        stratify=_valid_stratify(patient_summary[target_column]),
    )

    holdout_validation_fraction = validation_size / (validation_size + test_size)
    validation_patients, test_patients = train_test_split(
        holdout_patients,
        train_size=holdout_validation_fraction,
        random_state=random_state,
        stratify=_valid_stratify(holdout_patients[target_column]),
    )

    train_ids = set(train_patients[patient_column])
    validation_ids = set(validation_patients[patient_column])
    test_ids = set(test_patients[patient_column])

    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise RuntimeError("Patient leakage detected while creating data splits")

    return DatasetSplit(
        train=df[df[patient_column].isin(train_ids)].copy(),
        validation=df[df[patient_column].isin(validation_ids)].copy(),
        test=df[df[patient_column].isin(test_ids)].copy(),
    )


def classification_metrics(
    y_true: Iterable[int],
    scores: Iterable[float],
    threshold: float,
) -> dict[str, object]:
    y_true_array = np.asarray(list(y_true), dtype=int)
    score_array = np.asarray(list(scores), dtype=float)
    predictions = (score_array >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true_array, predictions, labels=[0, 1]
    ).ravel()

    metrics: dict[str, object] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true_array, predictions)),
        "precision": float(
            precision_score(y_true_array, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_true_array, predictions, zero_division=0)),
        "f1": float(f1_score(y_true_array, predictions, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    if len(np.unique(y_true_array)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true_array, score_array))
        metrics["pr_auc"] = float(average_precision_score(y_true_array, score_array))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics


def threshold_sweep(
    y_true: Iterable[int],
    scores: Iterable[float],
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    rows = []
    for threshold in thresholds:
        metrics = classification_metrics(y_true, scores, float(threshold))
        rows.append(
            {
                "threshold": metrics["threshold"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
            }
        )
    return pd.DataFrame(rows)


def choose_validation_threshold(sweep: pd.DataFrame) -> float:
    """Choose the validation threshold that maximizes F1, then recall."""
    required = {"threshold", "f1", "recall"}
    if not required.issubset(sweep.columns):
        raise KeyError(f"Threshold sweep must contain: {sorted(required)}")

    best = sweep.sort_values(
        ["f1", "recall", "threshold"],
        ascending=[False, False, True],
    ).iloc[0]
    return float(best["threshold"])


def risk_thresholds_from_training_scores(scores: Iterable[float]) -> dict[str, float]:
    values = np.asarray(list(scores), dtype=float)
    if values.size == 0:
        raise ValueError("Training scores cannot be empty")
    low_to_medium, medium_to_higher = np.quantile(values, [0.50, 0.80])
    return {
        "low_to_medium_threshold": float(low_to_medium),
        "medium_to_higher_threshold": float(medium_to_higher),
    }


def risk_category(score: float, thresholds: dict[str, float]) -> str:
    low_to_medium = thresholds["low_to_medium_threshold"]
    medium_to_higher = thresholds["medium_to_higher_threshold"]
    if score < low_to_medium:
        return "Low"
    if score < medium_to_higher:
        return "Medium"
    return "Higher"
