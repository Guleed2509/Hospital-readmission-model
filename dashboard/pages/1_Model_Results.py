from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.shared import THRESHOLD_PATH, load_csv, pct, require_artifacts, score

st.set_page_config(page_title="Model Results", page_icon="📊", layout="wide")
st.title("Model Results")

model, results = require_artifacts()
if model is not None and results is not None:
    selected = results["metrics"]["test_selected_threshold"]
    default = results["metrics"]["test_default_threshold"]
    baseline = results["metrics"]["dummy_prior_baseline"]

    st.subheader("Final test-set metrics")
    cols = st.columns(6)
    cols[0].metric("ROC-AUC", score(selected.get("roc_auc")))
    cols[1].metric("PR-AUC", score(selected.get("pr_auc")))
    cols[2].metric("Recall", pct(selected.get("recall")))
    cols[3].metric("Precision", pct(selected.get("precision")))
    cols[4].metric("F1", pct(selected.get("f1")))
    cols[5].metric("Accuracy", pct(selected.get("accuracy")))

    st.caption(
        "The operating threshold was selected on the validation set only. The test "
        "set is used for final reporting."
    )

    st.subheader("Confusion matrix")
    cm = selected["confusion_matrix"]
    matrix = pd.DataFrame(
        [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"],
    )
    st.dataframe(matrix, use_container_width=True)

    st.subheader("Threshold analysis")
    threshold_data = load_csv(THRESHOLD_PATH)
    if threshold_data is not None:
        chart = threshold_data.set_index("threshold")[["precision", "recall", "f1"]]
        st.line_chart(chart)
        st.dataframe(threshold_data, use_container_width=True, hide_index=True)

    st.subheader("Comparison")
    comparison = pd.DataFrame(
        [
            {
                "Evaluation": "Logistic Regression — selected validation threshold",
                "Threshold": selected["threshold"],
                "ROC-AUC": selected.get("roc_auc"),
                "PR-AUC": selected.get("pr_auc"),
                "Precision": selected["precision"],
                "Recall": selected["recall"],
                "F1": selected["f1"],
            },
            {
                "Evaluation": "Logistic Regression — default threshold",
                "Threshold": default["threshold"],
                "ROC-AUC": default.get("roc_auc"),
                "PR-AUC": default.get("pr_auc"),
                "Precision": default["precision"],
                "Recall": default["recall"],
                "F1": default["f1"],
            },
            {
                "Evaluation": "Dummy prior baseline",
                "Threshold": baseline["threshold"],
                "ROC-AUC": baseline.get("roc_auc"),
                "PR-AUC": baseline.get("pr_auc"),
                "Precision": baseline["precision"],
                "Recall": baseline["recall"],
                "F1": baseline["f1"],
            },
        ]
    )
    st.dataframe(comparison, use_container_width=True, hide_index=True)
