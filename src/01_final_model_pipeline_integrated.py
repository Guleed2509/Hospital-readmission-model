"""
01_final_model_pipeline.py

Final model pipeline
Project: Hospital 30-Day Readmission Risk Prediction

This script contains only the final modelling approach:
1. Load dataset
2. Create target variable
3. Remove problematic columns
4. Preprocess data
5. Train final regression baseline
6. Evaluate model
7. Save model
8. Save dashboard-ready outputs
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path("Dataset/readmissions.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2


def assign_risk_category(score: float) -> str:
    """
    These thresholds are illustrative and based on the observed score distribution.
    They are not clinically validated.
    """
    if score < 0.06:
        return "Low Risk"
    if score < 0.15:
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
# 2. Create final target: readmitted_30_days
# --------------------------------------------------

if "readmitted" not in df.columns:
    raise ValueError("Column 'readmitted' not found in dataset.")

df["readmitted_30_days"] = (df["readmitted"] == "<30").astype(int)

print("\nTarget distribution:")
print(df["readmitted_30_days"].value_counts(normalize=True).rename("proportion"))


# --------------------------------------------------
# 3. Remove problematic columns
# --------------------------------------------------

columns_to_drop = [
    "readmitted",          # original target replaced by readmitted_30_days
    "encounter_id",        # administrative identifier
    "patient_nbr",         # administrative identifier
    "weight",              # too many missing values
    "payer_code",          # many missing values / not clinically central
    "medical_specialty"    # many missing values
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

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)


# --------------------------------------------------
# 5. Train final regression baseline
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)

print("\nModel trained successfully")


# --------------------------------------------------
# 6. Evaluate model
# --------------------------------------------------

y_pred_raw = model.predict(X_test)

# Linear regression can predict below 0 or above 1.
# Clipping is only used for stakeholder-friendly risk-score interpretation.
y_pred_clipped = np.clip(y_pred_raw, 0, 1)

mae = mean_absolute_error(y_test, y_pred_clipped)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_clipped))
r2 = r2_score(y_test, y_pred_clipped)

evaluation_results = {
    "model": "Linear Regression Baseline",
    "target": "readmitted_30_days",
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "MAE": round(float(mae), 4),
    "RMSE": round(float(rmse), 4),
    "R2": round(float(r2), 4),
    "target_positive_rate": round(float(y.mean()), 4),
    "mean_predicted_risk": round(float(np.mean(y_pred_clipped)), 4),
    "interpretation": {
        "MAE": "Average prediction error on a 0 to 1 risk-score scale.",
        "RMSE": "Penalizes larger prediction errors more strongly.",
        "R2": "Explains how much variation in the target is captured by the model.",
        "risk_score": "The output is a relative risk score, not a calibrated probability."
    }
}

print("\nEvaluation Results:")
print(json.dumps(evaluation_results, indent=4))


# --------------------------------------------------
# 7. Save model
# --------------------------------------------------

model_path = OUTPUT_DIR / "final_readmission_regression_model.joblib"
joblib.dump(model, model_path)
print(f"\nModel saved to: {model_path}")


# --------------------------------------------------
# 8. Save evaluation results
# --------------------------------------------------

results_path = OUTPUT_DIR / "evaluation_results.json"

with open(results_path, "w", encoding="utf-8") as f:
    json.dump(evaluation_results, f, indent=4)

print(f"Evaluation results saved to: {results_path}")


# --------------------------------------------------
# 9. Save simple test predictions
# --------------------------------------------------

predictions_df = pd.DataFrame({
    "actual": y_test.values,
    "predicted_risk_raw": y_pred_raw,
    "predicted_risk_clipped": y_pred_clipped
})

predictions_df["risk_category"] = predictions_df["predicted_risk_clipped"].apply(assign_risk_category)

predictions_path = OUTPUT_DIR / "test_predictions.csv"
predictions_df.to_csv(predictions_path, index=False)

print(f"Predictions saved to: {predictions_path}")


# --------------------------------------------------
# 10. Save model input columns
# --------------------------------------------------

input_columns_path = OUTPUT_DIR / "model_input_columns.json"

with open(input_columns_path, "w", encoding="utf-8") as f:
    json.dump(list(X.columns), f, indent=4)

print(f"Model input columns saved to: {input_columns_path}")


# --------------------------------------------------
# 11. Save test-set model input for dashboard inference
# --------------------------------------------------

X_test_output = X_test.copy()
X_test_output.insert(0, "row_id", X_test.index)

x_test_path = OUTPUT_DIR / "X_test_model_input.csv"
X_test_output.to_csv(x_test_path, index=False)

print(f"Test model input saved to: {x_test_path}")


# --------------------------------------------------
# 12. Save patient-level predictions for dashboard
# --------------------------------------------------

patient_predictions = df.loc[X_test.index].copy()

patient_predictions["row_id"] = X_test.index
patient_predictions["actual_readmitted_30_days"] = y_test.values
patient_predictions["predicted_risk_raw"] = y_pred_raw
patient_predictions["predicted_risk_clipped"] = y_pred_clipped
patient_predictions["risk_category"] = patient_predictions["predicted_risk_clipped"].apply(assign_risk_category)

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
    "diag_3"
]

columns_to_export = [col for col in columns_to_export if col in patient_predictions.columns]

patient_predictions_path = OUTPUT_DIR / "patient_level_predictions.csv"
patient_predictions[columns_to_export].to_csv(patient_predictions_path, index=False)

print(f"Patient-level predictions saved to: {patient_predictions_path}")


# --------------------------------------------------
# 13. Save feature importance at original feature level
# --------------------------------------------------

try:
    preprocessor_fitted = model.named_steps["preprocessor"]
    regressor_fitted = model.named_steps["regressor"]

    feature_names = preprocessor_fitted.get_feature_names_out()
    coefficients = regressor_fitted.coef_

    importance_df = pd.DataFrame({
        "encoded_feature": feature_names,
        "coefficient": coefficients,
        "absolute_coefficient": np.abs(coefficients)
    })

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

    importance_df["original_feature"] = importance_df["encoded_feature"].apply(recover_original_feature)

    original_importance = (
        importance_df
        .groupby("original_feature", as_index=False)["absolute_coefficient"]
        .sum()
        .rename(columns={"absolute_coefficient": "total_absolute_coefficient"})
        .sort_values("total_absolute_coefficient", ascending=False)
    )

    importance_csv_path = OUTPUT_DIR / "feature_importance_original_level.csv"
    original_importance.to_csv(importance_csv_path, index=False)

    print(f"Feature importance CSV saved to: {importance_csv_path}")

    import matplotlib.pyplot as plt

    top_20 = original_importance.head(20).sort_values("total_absolute_coefficient")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(top_20["original_feature"], top_20["total_absolute_coefficient"])
    ax.set_title("Top 20 Feature Importances - Original Feature Level")
    ax.set_xlabel("Total absolute coefficient value")
    ax.set_ylabel("Original feature")

    fig.tight_layout()

    importance_png_path = OUTPUT_DIR / "feature_importance_original_top20.png"
    fig.savefig(importance_png_path, dpi=200)
    plt.close(fig)

    print(f"Feature importance image saved to: {importance_png_path}")

except Exception as error:
    print("\nFeature importance export failed.")
    print(f"Reason: {error}")


print("\nFinal pipeline completed successfully.")
