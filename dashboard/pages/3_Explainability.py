from __future__ import annotations

import streamlit as st

from dashboard.shared import (
    COEFFICIENT_PATH,
    PERMUTATION_PATH,
    load_csv,
    require_artifacts,
)

st.set_page_config(page_title="Explainability", page_icon="🔎", layout="wide")
st.title("Explainability")

model, results = require_artifacts()
if model is not None and results is not None:
    st.subheader("Permutation importance")
    permutation = load_csv(PERMUTATION_PATH)
    if permutation is not None:
        top = permutation.head(15).set_index("feature")["importance_mean"]
        st.bar_chart(top)
        st.dataframe(permutation, use_container_width=True, hide_index=True)
        st.caption(
            "Permutation importance is measured on the validation set using "
            "average precision. It is preferred here for comparing original features."
        )

    st.subheader("Aggregated coefficient magnitude")
    coefficients = load_csv(COEFFICIENT_PATH)
    if coefficients is not None:
        top_coefficients = coefficients.head(15).set_index("original_feature")[
            "aggregated_coefficient_magnitude"
        ]
        st.bar_chart(top_coefficients)
        st.dataframe(coefficients, use_container_width=True, hide_index=True)
        st.warning(
            "Aggregated absolute coefficients are not a causal importance measure. "
            "Categorical features can contribute multiple one-hot coefficients, so "
            "their summed magnitude can be larger simply because they have more levels."
        )
