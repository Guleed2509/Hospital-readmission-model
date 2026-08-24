"""
Classification pipeline
Project: Hospital 30-Day Readmission Risk Prediction

This version replaces Linear Regression with Logistic Regression for the
binary target readmitted_30_days.

Pipeline:
1. Load dataset
2. Create binary target
3. Remove problematic columns
4. Preprocess numeric and categorical features
5. Split train/test data
6. Train Logistic Regression classifier
7. Evaluate classification performance
8. Save model and dashboard-ready outputs
9. Export coefficient-based feature importance
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_PATH = Path("Dataset/readmissions.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
DECISION_THRESHOLD = 0.50


# --------------------------------------------------
# Helper
# --------------------------------------------------

def assign_risk_category(
    score: float,
    medium_threshold: float,
    high_threshold: float,
) -> str:
    """
    Relative risk categories based on the TRAINING prediction distribution.

    These categories are intended for presentation only and are NOT
    clinically validated thresholds.
    """
    if score < medium_threshold:
        return "Low Risk"
    if score < high_threshold:
        return "Medium Risk"
    return "Higher Risk"


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found. Expected file: {DATA_PATH}. "
        "Rename your dataset to readmissions.csv or update DATA_PATH."
    )

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully")
print(f"Shape: {df.shape}")


# --------------------------------------------------
# 2. Create binary target: readmitted_30_days
# --------------------------------------------------

if "readmitted" not in df.columns:
    raise ValueError("Column 'readmitted' not found in dataset.")

df["readmitted_30_days"] = (df["readmitted"] == "<30").astype(int)

print("\nTarget distribution:")
print(
    df["readmitted_30_days"]
    .value_counts(normalize=True)
    .sort_index()
    .rename("proportion")
)


# --------------------------------------------------
# 3. Remove problematic columns
# --------------------------------------------------

columns_to_drop = [
    "readmitted",          # original target replaced by readmitted_30_days
    "encounter_id",        # administrative identifier
    "patient_nbr",         # administrative identifier
    "weight",              # too many missing values
    "payer_code",          # many missing values / not clinically central
    "medical_specialty",   # many missing values
]

df_model = df.drop(columns=columns_to_drop, errors="ignore")

X = df_model.drop(columns=["readmitted_30_days"])
y = df_model["readmitted_30_days"]

# Convert dataset-specific missing marker to NaN
X = X.replace("?", np.nan)


# --------------------------------------------------
# 4. Preprocess data
# --------------------------------------------------

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print(f"\nNumerical features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)


# --------------------------------------------------
# 5. Train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nTrain target distribution:")
print(y_train.value_counts(normalize=True).sort_index())

print("\nTest target distribution:")
print(y_test.value_counts(normalize=True).sort_index())


# --------------------------------------------------
# 6. Train classifier
# --------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

model.fit(X_train, y_train)

print("\nLogistic Regression classifier trained successfully")


# --------------------------------------------------
# 7. Predict
# --------------------------------------------------

# Probability-like score for the positive class (<30 day readmission)
y_probability = model.predict_proba(X_test)[:, 1]

# Binary prediction using a transparent baseline threshold
y_pred = (y_probability >= DECISION_THRESHOLD).astype(int)

# Risk-category thresholds are learned ONLY from training predictions,
# so the test set does not determine its own category boundaries.
train_probability = model.predict_proba(X_train)[:, 1]
medium_risk_threshold = float(np.quantile(train_probability, 0.50))
high_risk_threshold = float(np.quantile(train_probability, 0.80))


# --------------------------------------------------
# 8. Evaluate classification performance
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_probability)
pr_auc = average_precision_score(y_test, y_probability)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0,
)

evaluation_results = {
    "model": "Logistic Regression Classifier",
    "target": "readmitted_30_days",
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "class_weight": "balanced",
    "decision_threshold": DECISION_THRESHOLD,
    "target_positive_rate": round(float(y.mean()), 4),
    "metrics": {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
    },
    "confusion_matrix": {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    },
    "risk_categories": {
        "method": "Training prediction quantiles",
        "low_to_medium_threshold": round(medium_risk_threshold, 6),
        "medium_to_higher_threshold": round(high_risk_threshold, 6),
        "note": "Relative presentation categories only; not clinically validated.",
    },
    "interpretation": {
        "roc_auc": "Measures how well the model ranks readmitted patients above non-readmitted patients across thresholds.",
        "pr_auc": "Precision-recall area under the curve; especially useful for an imbalanced positive class.",
        "precision": "Among encounters predicted as readmitted within 30 days, the fraction that were actually readmitted.",
        "recall": "Among encounters actually readmitted within 30 days, the fraction identified by the model.",
        "f1": "Harmonic mean of precision and recall.",
        "predicted_probability": (
            "Model score from Logistic Regression. Because class_weight='balanced' is used, "
            "it should not be presented as a clinically calibrated probability."
        ),
    },
    "classification_report": report,
}

print("\nEvaluation Results:")
print(json.dumps(evaluation_results, indent=4))


# --------------------------------------------------
# 9. Save model
# --------------------------------------------------

model_path = OUTPUT_DIR / "final_readmission_classifier.joblib"
joblib.dump(model, model_path)
print(f"\nModel saved to: {model_path}")


# --------------------------------------------------
# 10. Save evaluation results
# --------------------------------------------------

results_path = OUTPUT_DIR / "evaluation_results.json"

with open(results_path, "w", encoding="utf-8") as f:
    json.dump(evaluation_results, f, indent=4)

print(f"Evaluation results saved to: {results_path}")


# --------------------------------------------------
# 11. Save test predictions
# --------------------------------------------------

predictions_df = pd.DataFrame(
    {
        "actual": y_test.values,
        "predicted_probability": y_probability,
        "predicted_class": y_pred,
    }
)

predictions_df["risk_category"] = predictions_df["predicted_probability"].apply(
    lambda score: assign_risk_category(
        score,
        medium_risk_threshold,
        high_risk_threshold,
    )
)

# Compatibility aliases for an existing dashboard that may still
# reference the old regression output column names.
predictions_df["predicted_risk_raw"] = predictions_df["predicted_probability"]
predictions_df["predicted_risk_clipped"] = predictions_df["predicted_probability"]

predictions_path = OUTPUT_DIR / "test_predictions.csv"
predictions_df.to_csv(predictions_path, index=False)

print(f"Predictions saved to: {predictions_path}")


# --------------------------------------------------
# 12. Save model input columns
# --------------------------------------------------

input_columns_path = OUTPUT_DIR / "model_input_columns.json"

with open(input_columns_path, "w", encoding="utf-8") as f:
    json.dump(list(X.columns), f, indent=4)

print(f"Model input columns saved to: {input_columns_path}")


# --------------------------------------------------
# 13. Save test-set model input for dashboard inference
# --------------------------------------------------

X_test_output = X_test.copy()
X_test_output.insert(0, "row_id", X_test.index)

x_test_path = OUTPUT_DIR / "X_test_model_input.csv"
X_test_output.to_csv(x_test_path, index=False)

print(f"Test model input saved to: {x_test_path}")


# --------------------------------------------------
# 14. Save patient-level predictions for dashboard
# --------------------------------------------------

patient_predictions = df.loc[X_test.index].copy()

patient_predictions["row_id"] = X_test.index
patient_predictions["actual_readmitted_30_days"] = y_test.values
patient_predictions["predicted_probability"] = y_probability
patient_predictions["predicted_class"] = y_pred

patient_predictions["risk_category"] = patient_predictions[
    "predicted_probability"
].apply(
    lambda score: assign_risk_category(
        score,
        medium_risk_threshold,
        high_risk_threshold,
    )
)

# Compatibility aliases for the current dashboard.
patient_predictions["predicted_risk_raw"] = patient_predictions[
    "predicted_probability"
]
patient_predictions["predicted_risk_clipped"] = patient_predictions[
    "predicted_probability"
]

columns_to_export = [
    "row_id",
    "encounter_id",
    "patient_nbr",
    "age",
    "gender",
    "race",
    "readmitted",
    "readmitted_30_days",
    "actual_readmitted_30_days",
    "predicted_probability",
    "predicted_class",
    "predicted_risk_raw",
    "predicted_risk_clipped",
    "risk_category",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "diag_1",
    "diag_2",
    "diag_3",
]

columns_to_export = [
    col for col in columns_to_export if col in patient_predictions.columns
]

patient_predictions_path = OUTPUT_DIR / "patient_level_predictions.csv"
patient_predictions[columns_to_export].to_csv(
    patient_predictions_path,
    index=False,
)

print(f"Patient-level predictions saved to: {patient_predictions_path}")


# --------------------------------------------------
# 15. Save feature importance at original feature level
# --------------------------------------------------

try:
    preprocessor_fitted = model.named_steps["preprocessor"]
    classifier_fitted = model.named_steps["classifier"]

    feature_names = preprocessor_fitted.get_feature_names_out()
    coefficients = classifier_fitted.coef_[0]

    importance_df = pd.DataFrame(
        {
            "encoded_feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
        }
    )

    def recover_original_feature(encoded_name: str) -> str:
        """
        Converts encoded feature names such as:
        cat__age_[70-80) -> age
        cat__diag_1_250.83 -> diag_1
        num__time_in_hospital -> time_in_hospital
        """
        if encoded_name.startswith("num__"):
            return encoded_name.replace("num__", "")

        if encoded_name.startswith("cat__"):
            cleaned = encoded_name.replace("cat__", "")
            for original_col in categorical_features:
                if cleaned.startswith(original_col + "_"):
                    return original_col
            return cleaned

        return encoded_name

    importance_df["original_feature"] = importance_df[
        "encoded_feature"
    ].apply(recover_original_feature)

    # Save encoded-level coefficients too.
    encoded_importance_path = OUTPUT_DIR / "encoded_feature_importance.csv"
    importance_df.sort_values(
        "absolute_coefficient",
        ascending=False,
    ).to_csv(encoded_importance_path, index=False)

    original_importance = (
        importance_df
        .groupby("original_feature", as_index=False)["absolute_coefficient"]
        .sum()
        .rename(
            columns={
                "absolute_coefficient": "total_absolute_coefficient"
            }
        )
        .sort_values("total_absolute_coefficient", ascending=False)
    )

    importance_csv_path = OUTPUT_DIR / "feature_importance_original_level.csv"
    original_importance.to_csv(importance_csv_path, index=False)

    print(f"Feature importance CSV saved to: {importance_csv_path}")

    import matplotlib.pyplot as plt

    top_20 = (
        original_importance
        .head(20)
        .sort_values("total_absolute_coefficient")
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(
        top_20["original_feature"],
        top_20["total_absolute_coefficient"],
    )
    ax.set_title(
        "Top 20 Feature Importances - Logistic Regression"
    )
    ax.set_xlabel("Sum of absolute coefficient values")
    ax.set_ylabel("Original feature")
    fig.tight_layout()

    importance_png_path = (
        OUTPUT_DIR / "feature_importance_original_top20.png"
    )
    fig.savefig(importance_png_path, dpi=200)
    plt.close(fig)

    print(f"Feature importance image saved to: {importance_png_path}")

except Exception as error:
    print("\nFeature importance export failed.")
    print(f"Reason: {error}")


print("\nClassification pipeline completed successfully.")
