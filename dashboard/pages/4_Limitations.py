from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Limitations", page_icon="⚠️", layout="wide")
st.title("Limitations and Responsible Use")

st.markdown(
    """
### Research/portfolio prototype only
This project has **not** been clinically validated, prospectively evaluated, or approved for patient-care decisions. It should not be used to diagnose, treat, discharge, or prioritize real patients.

### Historical and limited dataset
The UCI data covers diabetic inpatient encounters from **1999–2008** across 130 US hospitals and integrated delivery networks. Clinical practice, coding, populations and care pathways have changed since then.

### Sensitive attributes
The dataset contains demographic attributes such as age, gender and race. Any real deployment would require subgroup performance analysis, fairness review, governance and clinical oversight.

### Relative score, not a validated probability
The Logistic Regression model uses balanced class weights to address class imbalance. The raw score is therefore presented as a **relative risk score** rather than a clinically calibrated individual probability.

### Moderate predictive discrimination
Readmission is influenced by many social, clinical and operational factors that are not fully captured by this dataset. Model performance should be interpreted as a baseline demonstration, not as evidence of clinical utility.

### No causal interpretation
Coefficients and permutation importance describe model behavior and association. They do not establish that changing a feature would cause readmission risk to change.

### Split design
The project uses patient-level train/validation/test splits so repeated encounters from one patient do not leak across evaluation splits. Threshold selection is performed on validation data; final metrics are reported on the held-out test set.
    """
)
