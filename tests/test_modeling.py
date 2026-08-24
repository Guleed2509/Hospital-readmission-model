from __future__ import annotations

import numpy as np
import pandas as pd

from src.modeling import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_classifier,
    choose_validation_threshold,
    create_target,
    prepare_feature_matrix,
    risk_category,
    split_by_patient,
    threshold_sweep,
)


def synthetic_dataset(n_patients: int = 60, encounters_per_patient: int = 2) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    encounter_id = 1

    for patient in range(n_patients):
        patient_positive = patient % 4 == 0
        for encounter in range(encounters_per_patient):
            row: dict[str, object] = {
                "encounter_id": encounter_id,
                "patient_nbr": patient,
                "readmitted": "<30" if patient_positive and encounter == 0 else "NO",
            }
            for feature in NUMERIC_FEATURES:
                row[feature] = (patient + encounter) % 10 + 1
            for feature in CATEGORICAL_FEATURES:
                row[feature] = f"category_{(patient + encounter) % 3}"
            rows.append(row)
            encounter_id += 1

    return pd.DataFrame(rows)


def test_create_target_maps_only_under_30_to_positive() -> None:
    frame = pd.DataFrame({"readmitted": ["<30", ">30", "NO"]})
    assert create_target(frame).tolist() == [1, 0, 0]


def test_patient_split_has_no_patient_overlap() -> None:
    frame = synthetic_dataset()
    frame[TARGET_COLUMN] = create_target(frame)

    split = split_by_patient(frame)
    train_patients = set(split.train["patient_nbr"])
    validation_patients = set(split.validation["patient_nbr"])
    test_patients = set(split.test["patient_nbr"])

    assert not (train_patients & validation_patients)
    assert not (train_patients & test_patients)
    assert not (validation_patients & test_patients)
    assert len(split.train) + len(split.validation) + len(split.test) == len(frame)


def test_pipeline_predict_proba_returns_bounded_scores() -> None:
    frame = synthetic_dataset()
    y = create_target(frame)
    X = prepare_feature_matrix(frame)

    model = build_classifier()
    model.fit(X, y)
    probabilities = model.predict_proba(X)[:, 1]

    assert probabilities.shape == (len(frame),)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert set(X.columns) == set(FEATURE_COLUMNS)


def test_threshold_selection_returns_candidate_threshold() -> None:
    y_true = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.6, 0.9]
    sweep = threshold_sweep(y_true, scores, thresholds=[0.3, 0.5, 0.7])
    selected = choose_validation_threshold(sweep)
    assert selected in {0.3, 0.5, 0.7}


def test_risk_categories_use_shared_thresholds() -> None:
    thresholds = {
        "low_to_medium_threshold": 0.4,
        "medium_to_higher_threshold": 0.7,
    }
    assert risk_category(0.2, thresholds) == "Low"
    assert risk_category(0.5, thresholds) == "Medium"
    assert risk_category(0.8, thresholds) == "Higher"
