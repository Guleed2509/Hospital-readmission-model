from __future__ import annotations

import streamlit as st

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.shared import load_results, pct, score

st.set_page_config(
    page_title="Hospital Readmission Model",
    page_icon="🏥",
    layout="wide",
)

st.title("Hospital Readmission Risk Model")
st.caption(
    "Portfolio demonstration using the UCI Diabetes 130-US Hospitals dataset. "
    "Not clinically validated and not intended for patient-care decisions."
)

results = load_results()

if results is None:
    st.warning(
        "No trained artifacts found yet. From the repository root, run "
        "`python -m src.train`, then reload this app."
    )
else:
    metrics = results["metrics"]["test_selected_threshold"]
    baseline = results["metrics"]["dummy_prior_baseline"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC", score(metrics.get("roc_auc")))
    col2.metric("PR-AUC", score(metrics.get("pr_auc")))
    col3.metric("Recall", pct(metrics.get("recall")))
    col4.metric("Precision", pct(metrics.get("precision")))

    st.subheader("What changed in this version")
    st.markdown(
        """
- **Patient-level splitting:** encounters from one patient cannot occur in multiple splits.
- **Three-way evaluation:** 70% train, 15% validation, 15% final test at patient level.
- **Consistent classification pipeline:** Logistic Regression, `predict_proba`, shared model artifacts and thresholds.
- **Imbalance-aware evaluation:** ROC-AUC, PR-AUC, precision, recall, F1 and a dummy prior baseline.
- **Relative risk bands:** Low/Medium/Higher thresholds come only from training-set scores.
        """
    )

    st.subheader("Current evaluation context")
    a, b, c = st.columns(3)
    a.metric("Test positive rate", pct(results["split"]["test_positive_rate"]))
    b.metric("Selected threshold", score(results["threshold_selection"]["selected_threshold"]))
    c.metric("Dummy PR-AUC", score(baseline.get("pr_auc")))

    st.info(
        "The displayed model score is used for relative ranking. Because the "
        "classifier uses balanced class weights and has not undergone clinical "
        "probability calibration, it is not presented as a validated individual "
        "readmission probability."
    )

st.markdown("Use the pages in the sidebar for detailed results, explainability and a safe what-if demo.")
