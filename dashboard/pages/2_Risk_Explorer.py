from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.shared import require_artifacts, risk_category
from src.modeling import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

st.set_page_config(page_title="Risk Explorer", page_icon="🧪", layout="wide")
st.title("Risk Explorer")
st.caption(
    "Synthetic what-if demonstration. Do not enter real identifiable patient information."
)

model, results = require_artifacts()
if model is not None and results is not None:
    schema = results["input_schema"]

    record: dict[str, object] = {}
    for feature in NUMERIC_FEATURES:
        record[feature] = schema["numeric"][feature]["median"]
    for feature in CATEGORICAL_FEATURES:
        record[feature] = schema["categorical"][feature]["default"]

    st.subheader("Example encounter inputs")
    c1, c2 = st.columns(2)

    with c1:
        age_options = schema["categorical"]["age"]["values"]
        record["age"] = st.selectbox("Age band", age_options, index=0)

        gender_options = schema["categorical"]["gender"]["values"]
        record["gender"] = st.selectbox("Gender", gender_options, index=0)

        record["time_in_hospital"] = st.slider(
            "Time in hospital (days)",
            int(schema["numeric"]["time_in_hospital"]["min"]),
            int(schema["numeric"]["time_in_hospital"]["max"]),
            int(round(schema["numeric"]["time_in_hospital"]["median"])),
        )

        record["num_lab_procedures"] = st.slider(
            "Lab procedures",
            int(schema["numeric"]["num_lab_procedures"]["min"]),
            int(schema["numeric"]["num_lab_procedures"]["max"]),
            int(round(schema["numeric"]["num_lab_procedures"]["median"])),
        )

        record["num_medications"] = st.slider(
            "Number of medications",
            int(schema["numeric"]["num_medications"]["min"]),
            int(schema["numeric"]["num_medications"]["max"]),
            int(round(schema["numeric"]["num_medications"]["median"])),
        )

    with c2:
        record["number_emergency"] = st.slider(
            "Emergency visits in prior year",
            int(schema["numeric"]["number_emergency"]["min"]),
            int(schema["numeric"]["number_emergency"]["max"]),
            int(round(schema["numeric"]["number_emergency"]["median"])),
        )

        record["number_inpatient"] = st.slider(
            "Inpatient visits in prior year",
            int(schema["numeric"]["number_inpatient"]["min"]),
            int(schema["numeric"]["number_inpatient"]["max"]),
            int(round(schema["numeric"]["number_inpatient"]["median"])),
        )

        record["number_diagnoses"] = st.slider(
            "Number of diagnoses",
            int(schema["numeric"]["number_diagnoses"]["min"]),
            int(schema["numeric"]["number_diagnoses"]["max"]),
            int(round(schema["numeric"]["number_diagnoses"]["median"])),
        )

        a1c_options = schema["categorical"]["A1Cresult"]["values"]
        record["A1Cresult"] = st.selectbox("A1C result", a1c_options, index=0)

        diabetes_med_options = schema["categorical"]["diabetesMed"]["values"]
        record["diabetesMed"] = st.selectbox(
            "Diabetes medication recorded", diabetes_med_options, index=0
        )

    input_frame = pd.DataFrame([{feature: record[feature] for feature in FEATURE_COLUMNS}])
    risk_score = float(model.predict_proba(input_frame)[0, 1])
    operating_threshold = float(results["threshold_selection"]["selected_threshold"])
    predicted_class = int(risk_score >= operating_threshold)
    band = risk_category(risk_score, results)

    st.subheader("Model output")
    r1, r2, r3 = st.columns(3)
    r1.metric("Relative risk score", f"{risk_score:.3f}")
    r2.metric("Relative risk band", band)
    r3.metric("Classifier output", "Flag" if predicted_class else "No flag")

    thresholds = results["risk_categories"]
    st.caption(
        "Risk bands are defined by training-score percentiles: "
        f"Low < {thresholds['low_to_medium_threshold']:.3f}; "
        f"Medium < {thresholds['medium_to_higher_threshold']:.3f}; "
        "Higher otherwise. These are relative ranking bands, not clinical cut-offs."
    )

    st.info(
        "Unedited features use training-set median/mode defaults. This page is a "
        "model-behavior demo, not a clinical decision-support interface."
    )
