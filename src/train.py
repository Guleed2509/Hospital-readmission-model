from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance

from src.modeling import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    build_classifier,
    choose_validation_threshold,
    classification_metrics,
    create_target,
    prepare_feature_matrix,
    risk_category,
    risk_thresholds_from_training_scores,
    split_by_patient,
    threshold_sweep,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "final_readmission_classifier.joblib"
RESULTS_PATH = OUTPUT_DIR / "evaluation_results.json"
THRESHOLD_PATH = OUTPUT_DIR / "threshold_analysis.csv"
TEST_PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"
COEFFICIENT_PATH = OUTPUT_DIR / "coefficient_magnitudes.csv"
PERMUTATION_PATH = OUTPUT_DIR / "permutation_importance.csv"


def find_data_file() -> Path:
    environment_path = os.getenv("READMISSION_DATA_PATH")
    candidates = [
        Path(environment_path).expanduser() if environment_path else None,
        PROJECT_ROOT / "data" / "diabetic_data.csv",
        PROJECT_ROOT / "Data" / "diabetic_data.csv",
        PROJECT_ROOT / "data/readmissions.csv",
        PROJECT_ROOT / "diabetic_data.csv",
    ]

    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate.resolve()

    matches = [
        path
        for path in PROJECT_ROOT.rglob("diabetic_data.csv")
        if ".venv" not in path.parts and "venv" not in path.parts
    ]
    if matches:
        return matches[0].resolve()

    raise FileNotFoundError(
        "Could not find diabetic_data.csv. Place it in data/diabetic_data.csv "
        "or set READMISSION_DATA_PATH."
    )


def _input_schema(X_train: pd.DataFrame) -> dict[str, object]:
    schema: dict[str, object] = {"numeric": {}, "categorical": {}}

    for column in NUMERIC_FEATURES:
        values = pd.to_numeric(X_train[column], errors="coerce")
        schema["numeric"][column] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(values.median()),
        }

    for column in CATEGORICAL_FEATURES:
        non_missing = X_train[column].dropna().astype(str)
        mode = str(non_missing.mode().iloc[0]) if not non_missing.empty else "Unknown"
        top_values = non_missing.value_counts().head(50).index.tolist()
        if mode not in top_values:
            top_values.insert(0, mode)
        schema["categorical"][column] = {
            "default": mode,
            "values": [str(value) for value in top_values],
        }

    return schema


def _coefficient_table(model) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    transformed_names = list(preprocessor.get_feature_names_out())
    coefficients = classifier.coef_[0]

    if len(transformed_names) != len(coefficients):
        raise RuntimeError("Feature names and coefficients do not align")

    original_features = sorted(FEATURE_COLUMNS, key=len, reverse=True)

    def original_feature(name: str) -> str:
        if name in FEATURE_COLUMNS:
            return name
        for feature in original_features:
            if name.startswith(f"{feature}_"):
                return feature
        return name

    detail = pd.DataFrame(
        {
            "transformed_feature": transformed_names,
            "coefficient": coefficients,
        }
    )
    detail["absolute_coefficient"] = detail["coefficient"].abs()
    detail["original_feature"] = detail["transformed_feature"].map(original_feature)

    aggregated = (
        detail.groupby("original_feature", as_index=False)["absolute_coefficient"]
        .sum()
        .rename(columns={"absolute_coefficient": "aggregated_coefficient_magnitude"})
        .sort_values("aggregated_coefficient_magnitude", ascending=False)
    )
    return aggregated


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data_path = find_data_file()

    raw = pd.read_csv(data_path, na_values=["?"])
    raw[TARGET_COLUMN] = create_target(raw)

    split = split_by_patient(raw, target_column=TARGET_COLUMN)

    X_train = prepare_feature_matrix(split.train)
    X_validation = prepare_feature_matrix(split.validation)
    X_test = prepare_feature_matrix(split.test)
    y_train = split.train[TARGET_COLUMN].astype(int)
    y_validation = split.validation[TARGET_COLUMN].astype(int)
    y_test = split.test[TARGET_COLUMN].astype(int)

    model = build_classifier()
    model.fit(X_train, y_train)

    train_scores = model.predict_proba(X_train)[:, 1]
    validation_scores = model.predict_proba(X_validation)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]

    sweep = threshold_sweep(y_validation, validation_scores)
    selected_threshold = choose_validation_threshold(sweep)
    sweep.to_csv(THRESHOLD_PATH, index=False)

    risk_thresholds = risk_thresholds_from_training_scores(train_scores)

    model_test_metrics = classification_metrics(
        y_test, test_scores, selected_threshold
    )
    model_test_metrics_default = classification_metrics(y_test, test_scores, 0.50)

    dummy = DummyClassifier(strategy="prior", random_state=RANDOM_STATE)
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    dummy_scores = dummy.predict_proba(np.zeros((len(y_test), 1)))[:, 1]
    dummy_metrics = classification_metrics(y_test, dummy_scores, 0.50)

    permutation = permutation_importance(
        model,
        X_validation,
        y_validation,
        scoring="average_precision",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    permutation_table = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance_mean": permutation.importances_mean,
            "importance_std": permutation.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    permutation_table.to_csv(PERMUTATION_PATH, index=False)

    coefficient_table = _coefficient_table(model)
    coefficient_table.to_csv(COEFFICIENT_PATH, index=False)

    prediction_export = pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "risk_score": test_scores,
            "predicted": (test_scores >= selected_threshold).astype(int),
            "risk_category": [
                risk_category(float(score), risk_thresholds) for score in test_scores
            ],
        }
    )
    if "encounter_id" in split.test.columns:
        prediction_export.insert(0, "encounter_id", split.test["encounter_id"].to_numpy())
    prediction_export.to_csv(TEST_PREDICTIONS_PATH, index=False)

    results = {
        "model": {
            "name": "LogisticRegression",
            "class_weight": "balanced",
            "solver": "liblinear",
            "random_state": RANDOM_STATE,
            "score_semantics": (
                "Relative model risk score. It is not presented as a clinically "
                "validated individual probability."
            ),
        },
        "dataset": {
            "filename": data_path.name,
            "rows": int(len(raw)),
            "patients": int(raw["patient_nbr"].nunique()),
            "positive_rate": float(raw[TARGET_COLUMN].mean()),
        },
        "split": {
            "strategy": "patient-level 70/15/15 split stratified by patient ever-positive label",
            "train_encounters": int(len(split.train)),
            "validation_encounters": int(len(split.validation)),
            "test_encounters": int(len(split.test)),
            "train_patients": int(split.train["patient_nbr"].nunique()),
            "validation_patients": int(split.validation["patient_nbr"].nunique()),
            "test_patients": int(split.test["patient_nbr"].nunique()),
            "train_positive_rate": float(y_train.mean()),
            "validation_positive_rate": float(y_validation.mean()),
            "test_positive_rate": float(y_test.mean()),
        },
        "threshold_selection": {
            "method": "maximize F1 on validation set; tie-break on recall",
            "selected_threshold": float(selected_threshold),
            "default_threshold": 0.50,
        },
        "risk_categories": {
            **risk_thresholds,
            "method": "50th and 80th percentiles of training-set risk scores",
        },
        "metrics": {
            "test_selected_threshold": model_test_metrics,
            "test_default_threshold": model_test_metrics_default,
            "dummy_prior_baseline": dummy_metrics,
        },
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
        "input_schema": _input_schema(X_train),
        "artifacts": {
            "model": MODEL_PATH.name,
            "threshold_analysis": THRESHOLD_PATH.name,
            "test_predictions": TEST_PREDICTIONS_PATH.name,
            "coefficient_magnitudes": COEFFICIENT_PATH.name,
            "permutation_importance": PERMUTATION_PATH.name,
        },
    }

    joblib.dump(model, MODEL_PATH)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Evaluation saved to: {RESULTS_PATH}")
    print(json.dumps(results["metrics"]["test_selected_threshold"], indent=2))


if __name__ == "__main__":
    main()
