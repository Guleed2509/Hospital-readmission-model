# """
# 02_dashboard.py

# Streamlit dashboard
# Project: Hospital 30-Day Readmission Risk Prediction

# This dashboard is integrated with:
# - the original dataset
# - saved model evaluation results
# - saved test predictions
# - saved trained model pipeline
# - patient-level test-set predictions
# - feature importance output
# """

# import json
# from pathlib import Path

# import joblib
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import streamlit as st


# # --------------------------------------------------
# # Page settings
# # --------------------------------------------------

# st.set_page_config(
#     page_title="Hospital Readmission Dashboard",
#     layout="wide"
# )

# st.title("Hospital 30-Day Readmission Risk Dashboard")

# st.markdown("""
# This dashboard presents the final results of the hospital readmission prediction project.
# The goal is to estimate a **relative risk score** for diabetes patients being readmitted
# within 30 days after discharge.

# The score should be interpreted as a **decision-support signal**, not as a calibrated probability.
# """)


# # --------------------------------------------------
# # Paths
# # --------------------------------------------------

# DATA_PATH = Path("readmissions.csv")
# OUTPUT_DIR = Path("outputs")

# EVAL_PATH = OUTPUT_DIR / "evaluation_results.json"
# PRED_PATH = OUTPUT_DIR / "test_predictions.csv"
# MODEL_PATH = OUTPUT_DIR / "final_readmission_regression_model.joblib"
# X_TEST_INPUT_PATH = OUTPUT_DIR / "X_test_model_input.csv"
# PATIENT_PRED_PATH = OUTPUT_DIR / "patient_level_predictions.csv"
# FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance_original_top20.png"
# FEATURE_IMPORTANCE_CSV_PATH = OUTPUT_DIR / "feature_importance_original_level.csv"


# # --------------------------------------------------
# # Cached loaders
# # --------------------------------------------------

# @st.cache_data
# def load_data(path: Path) -> pd.DataFrame:
#     data = pd.read_csv(path)
#     data["readmitted_30_days"] = (data["readmitted"] == "<30").astype(int)
#     return data


# @st.cache_data
# def load_csv(path: Path) -> pd.DataFrame:
#     return pd.read_csv(path)


# @st.cache_resource
# def load_model(path: Path):
#     return joblib.load(path)


# @st.cache_data
# def load_json(path: Path) -> dict:
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)


# # --------------------------------------------------
# # Helper functions
# # --------------------------------------------------

# def assign_risk_category(score: float):
#     """
#     These thresholds are illustrative and not clinically validated.
#     They are based on the observed distribution of model scores.
#     """
#     if score < 0.06:
#         return "Low Risk", "Standard discharge process."
#     if score < 0.15:
#         return "Medium Risk", "Consider follow-up contact."
#     return "Higher Risk", "Additional review or support."


# def show_missing_file_warning(path: Path):
#     st.error(f"Required file not found: `{path}`")
#     st.info("Run `python 01_final_model_pipeline.py` first and check your file paths.")


# # --------------------------------------------------
# # Initial dataset loading
# # --------------------------------------------------

# if not DATA_PATH.exists():
#     st.error("Dataset file not found. Expected file: `readmissions.csv`")
#     st.stop()

# df = load_data(DATA_PATH)


# # --------------------------------------------------
# # Sidebar
# # --------------------------------------------------

# page = st.sidebar.radio(
#     "Navigation",
#     [
#         "Project Overview",
#         "Dataset Insights",
#         "Model Results",
#         "Prediction Analysis",
#         "Explainability",
#         "Stakeholder Advice",
#         "Limitations",
#         "Patient Risk Prediction"
#     ]
# )

# st.sidebar.markdown("---")
# st.sidebar.caption("Final Challenge - Hospital Readmission")


# # --------------------------------------------------
# # Page 1: Project Overview
# # --------------------------------------------------

# if page == "Project Overview":
#     st.header("Project Overview")

#     st.subheader("Stakeholder")
#     st.write("""
#     The stakeholder is a healthcare professional or hospital decision-maker who wants to identify
#     patients who may need additional support after discharge.
#     """)

#     st.subheader("Research Question")
#     st.write("""
#     How can historical patient data be used to estimate the risk of hospital readmission
#     within 30 days for patients with diabetes?
#     """)

#     st.subheader("Why Readmission Matters")
#     st.write("""
#     Hospital readmissions can indicate complications, insufficient follow-up care,
#     or patient vulnerability. Predicting readmission risk may help hospitals prioritize
#     follow-up care and use limited resources more effectively.
#     """)

#     st.info("""
#     The model output should be interpreted as a relative risk score, not as an exact probability.
#     """)

#     st.subheader("Dashboard Purpose")
#     st.write("""
#     This dashboard makes the project results understandable for stakeholders by showing:
#     dataset insights, model performance, prediction distributions, explainability,
#     limitations, and patient-level risk-score examples.
#     """)


# # --------------------------------------------------
# # Page 2: Dataset Insights
# # --------------------------------------------------

# elif page == "Dataset Insights":
#     st.header("Dataset Insights")

#     col1, col2, col3 = st.columns(3)

#     col1.metric("Rows", f"{df.shape[0]:,}")
#     col2.metric("Columns", df.shape[1])
#     col3.metric(
#         "30-day readmission rate",
#         f"{df['readmitted_30_days'].mean() * 100:.2f}%"
#     )

#     st.subheader("Original Target Distribution")

#     target_counts = df["readmitted"].value_counts()

#     fig, ax = plt.subplots()
#     target_counts.plot(kind="bar", ax=ax)
#     ax.set_title("Original Readmission Labels")
#     ax.set_xlabel("Readmission Category")
#     ax.set_ylabel("Number of Encounters")
#     st.pyplot(fig)

#     st.write("""
#     The original target contains three categories: `NO`, `<30`, and `>30`.
#     For this project, the target was transformed into `readmitted_30_days`,
#     where `<30` is labelled as 1 and all other cases are labelled as 0.
#     """)

#     st.warning("""
#     This transformation fits the 30-day readmission goal, but it also causes information loss.
#     Patients readmitted after more than 30 days are treated the same as patients who were never readmitted.
#     """)

#     st.subheader("Binary Target Distribution")

#     binary_counts = df["readmitted_30_days"].value_counts(normalize=True).sort_index() * 100

#     fig, ax = plt.subplots()
#     binary_counts.plot(kind="bar", ax=ax)
#     ax.set_title("30-Day Readmission Target")
#     ax.set_xlabel("Readmitted within 30 days")
#     ax.set_ylabel("Percentage")
#     ax.set_xticklabels(["No", "Yes"], rotation=0)
#     st.pyplot(fig)

#     st.warning("""
#     The target is imbalanced: only around 11% of patients were readmitted within 30 days.
#     This makes the prediction task more difficult.
#     """)

#     st.subheader("Missing Values")

#     missing_values = df.replace("?", pd.NA).isna().mean().sort_values(ascending=False) * 100
#     missing_values = missing_values[missing_values > 0].head(10)

#     if missing_values.empty:
#         st.success("No missing values found after checking for '?' values.")
#     else:
#         fig, ax = plt.subplots()
#         missing_values.plot(kind="bar", ax=ax)
#         ax.set_title("Top Missing Values")
#         ax.set_ylabel("Missing Percentage")
#         ax.set_xlabel("Feature")
#         st.pyplot(fig)

#     st.subheader("Key Data Insight")
#     st.write("""
#     The dataset is large, but readmission is difficult to predict because important context
#     such as social support, disease severity, medication adherence and discharge quality is missing.
#     """)


# # --------------------------------------------------
# # Page 3: Model Results
# # --------------------------------------------------

# elif page == "Model Results":
#     st.header("Model Results")

#     if not EVAL_PATH.exists():
#         show_missing_file_warning(EVAL_PATH)
#         st.stop()

#     results = load_json(EVAL_PATH)

#     col1, col2, col3 = st.columns(3)

#     col1.metric("MAE", results["MAE"])
#     col2.metric("RMSE", results["RMSE"])
#     col3.metric("R²", results["R2"])

#     st.subheader("Metric Interpretation")

#     st.write(f"""
#     **MAE = {results['MAE']}**  
#     MAE shows the average prediction error on a 0 to 1 risk-score scale.
#     This means the model prediction is on average about {results['MAE']} away from the true target value.
#     """)

#     st.write(f"""
#     **RMSE = {results['RMSE']}**  
#     RMSE penalizes larger errors more strongly. A higher RMSE than MAE indicates
#     that some predictions had larger errors.
#     """)

#     st.write(f"""
#     **R² = {results['R2']}**  
#     R² shows how much variation in the target is explained by the model.
#     The low R² score indicates that the model has limited predictive power.
#     """)

#     st.warning("""
#     The low performance should not only be interpreted as a model problem.
#     It also reflects limitations in the dataset, the target design, and the quality of the labels.
#     """)

#     st.subheader("Main Technical Conclusion")
#     st.write("""
#     The model learned the general readmission rate, but it struggled to explain individual variation.
#     This suggests that data quality and target design are stronger limitations than model complexity.
#     """)


# # --------------------------------------------------
# # Page 4: Prediction Analysis
# # --------------------------------------------------

# elif page == "Prediction Analysis":
#     st.header("Prediction Analysis")

#     if not PRED_PATH.exists():
#         show_missing_file_warning(PRED_PATH)
#         st.stop()

#     predictions = load_csv(PRED_PATH)

#     st.subheader("Prediction Distribution")

#     fig, ax = plt.subplots()
#     predictions["predicted_risk_clipped"].hist(bins=30, ax=ax)
#     ax.set_title("Distribution of Predicted Risk Scores")
#     ax.set_xlabel("Predicted Risk Score")
#     ax.set_ylabel("Number of Patients")
#     st.pyplot(fig)

#     st.write("""
#     Most predictions are close to the average readmission rate. This suggests that the model
#     is conservative and struggles to clearly separate low-risk and high-risk patients.
#     """)

#     st.subheader("Prediction Summary")
#     st.dataframe(predictions["predicted_risk_clipped"].describe())

#     st.subheader("Highest Risk Patients")

#     top_risk = predictions.sort_values(
#         "predicted_risk_clipped",
#         ascending=False
#     ).head(20)

#     st.dataframe(top_risk)

#     st.warning("""
#     These scores are relative risk scores, not calibrated probabilities.
#     A score of 0.20 should be interpreted as higher than average risk, not automatically as a 20% probability.
#     """)

#     st.write("""
#     Some high-risk predictions may be false positives. This may happen because patients readmitted
#     after more than 30 days are treated as 0, even though they may still look clinically high-risk.
#     """)


# # --------------------------------------------------
# # Page 5: Explainability
# # --------------------------------------------------

# elif page == "Explainability":
#     st.header("Explainability")

#     st.write("""
#     This page explains which original feature groups contributed most strongly to the linear model.
#     """)

#     if FEATURE_IMPORTANCE_PATH.exists():
#         st.image(str(FEATURE_IMPORTANCE_PATH), caption="Top 20 Feature Importances")
#     else:
#         st.warning("Feature importance image not found.")

#     if FEATURE_IMPORTANCE_CSV_PATH.exists():
#         importance_df = load_csv(FEATURE_IMPORTANCE_CSV_PATH)
#         st.subheader("Feature Importance Table")
#         st.dataframe(importance_df.head(20))
#     else:
#         st.info("Feature importance CSV not found.")

#     st.warning("""
#     This feature importance is based on total absolute coefficient values after one-hot encoding.
#     It should not be interpreted causally. Features with many encoded categories, such as diagnosis
#     variables, may receive larger total coefficient values because they are represented by many columns.
#     """)

#     st.write("""
#     The result suggests that diagnosis-related features are important for the model, but it does not prove
#     that these features directly cause readmission. The explanation should be used to understand model
#     behaviour, not to make clinical conclusions.
#     """)


# # --------------------------------------------------
# # Page 6: Stakeholder Advice
# # --------------------------------------------------

# elif page == "Stakeholder Advice":
#     st.header("Stakeholder Advice")

#     st.write("""
#     The model should be used as a decision-support tool, not as an automated decision-making system.
#     """)

#     st.subheader("Possible Risk Categories")

#     advice_df = pd.DataFrame({
#         "Risk Level": ["Low Risk", "Medium Risk", "Higher Risk"],
#         "Risk Score": ["0.00 - 0.06", "0.06 - 0.15", "> 0.15"],
#         "Possible Action": [
#             "Standard discharge process",
#             "Consider follow-up contact",
#             "Additional review or support"
#         ]
#     })

#     st.table(advice_df)

#     st.info("""
#     These thresholds are examples only. In a real hospital setting, the thresholds should be chosen
#     together with healthcare professionals and validated on real-world outcomes.
#     """)

#     st.warning("""
#     The score should support clinical judgement, not replace it.
#     """)

#     st.subheader("Practical Use")
#     st.write("""
#     The most realistic use of this model is patient ranking. A hospital could use the score to identify
#     which patients may need additional review, while still leaving the final decision to healthcare professionals.
#     """)


# # --------------------------------------------------
# # Page 7: Limitations
# # --------------------------------------------------

# elif page == "Limitations":
#     st.header("Limitations")

#     st.subheader("1. Coarse Target Labels")
#     st.write("""
#     The dataset does not contain exact readmission days. It only contains `NO`, `<30`, and `>30`.
#     This means that a patient readmitted after 2 days and a patient readmitted after 29 days receive
#     the same label, even though their clinical situations may be different.
#     """)

#     st.subheader("2. Information Loss")
#     st.write("""
#     Patients readmitted after more than 30 days are treated as 0 in the final target.
#     This fits the 30-day readmission goal, but it removes information about later readmissions.
#     """)

#     st.subheader("3. Missing Context")
#     st.write("""
#     The dataset does not include important factors such as social support, discharge quality,
#     disease severity, medication adherence, or socioeconomic circumstances.
#     """)

#     st.subheader("4. Limited Predictive Power")
#     st.write("""
#     The low R² score shows that the model explains only a small part of the variation in readmission outcomes.
#     This suggests that the main limitation is not only the model, but also the available data and target design.
#     """)

#     st.subheader("5. Risk Score Is Not a Probability")
#     st.write("""
#     The Linear Regression model produces a continuous score. This score can be useful for ranking patients,
#     but it should not be interpreted as a calibrated probability of readmission.
#     """)

#     st.subheader("6. Not Clinically Validated")
#     st.write("""
#     The dashboard and thresholds were created for educational and exploratory purposes.
#     They have not been clinically validated and should not be used for real patient care.
#     """)


# # --------------------------------------------------
# # Page 8: Patient Risk Prediction
# # --------------------------------------------------

# elif page == "Patient Risk Prediction":
#     st.header("Patient Risk Prediction")

#     st.write("""
#     This page uses the trained model pipeline to generate a readmission risk score
#     for real patients from the test dataset.
#     """)

#     st.warning("""
#     The model output is a relative risk score, not a calibrated probability.
#     It should support clinical judgement, not replace it.
#     """)

#     required_files = [MODEL_PATH, X_TEST_INPUT_PATH, PATIENT_PRED_PATH]
#     missing_files = [str(path) for path in required_files if not path.exists()]

#     if missing_files:
#         st.error("The following required files are missing:")
#         st.write(missing_files)
#         st.stop()

#     model = load_model(MODEL_PATH)
#     patient_predictions = load_csv(PATIENT_PRED_PATH)
#     X_test_input = load_csv(X_TEST_INPUT_PATH)

#     X_test_input = X_test_input.set_index("row_id")

#     st.subheader("Select a Patient from the Test Set")

#     selected_row_id = st.selectbox(
#         "Select patient row ID",
#         patient_predictions["row_id"].astype(int).tolist()
#     )

#     selected_patient_info = patient_predictions[
#         patient_predictions["row_id"] == selected_row_id
#     ].iloc[0]

#     selected_model_input = X_test_input.loc[[selected_row_id]]

#     raw_prediction = float(model.predict(selected_model_input)[0])
#     risk_score = float(np.clip(raw_prediction, 0, 1))

#     risk_category, advice = assign_risk_category(risk_score)

#     col1, col2, col3 = st.columns(3)

#     col1.metric("Model Risk Score", f"{risk_score:.3f}")
#     col2.metric("Risk Category", risk_category)
#     col3.metric(
#         "Actual 30-Day Readmission",
#         int(selected_patient_info["actual_readmitted_30_days"])
#     )

#     st.caption(f"Raw model output before clipping: {raw_prediction:.4f}")

#     st.subheader("Original Patient Information")

#     display_columns = [
#         "age",
#         "gender",
#         "race",
#         "readmitted",
#         "time_in_hospital",
#         "num_lab_procedures",
#         "num_procedures",
#         "num_medications",
#         "number_outpatient",
#         "number_emergency",
#         "number_inpatient",
#         "number_diagnoses",
#         "diag_1",
#         "diag_2",
#         "diag_3"
#     ]

#     display_columns = [col for col in display_columns if col in selected_patient_info.index]

#     st.dataframe(
#         selected_patient_info[display_columns].to_frame(name="Value")
#     )

#     st.subheader("Recommended Action")
#     st.write(advice)

#     st.info("""
#     This prediction is calculated using the saved trained pipeline from the final model script.
#     The patient row contains the full model input structure, so the prediction is integrated
#     with the actual trained model and dataset.
#     """)

#     st.divider()

#     st.subheader("What-if Simulation")

#     st.write("""
#     This section starts from the selected real patient and allows a few values to be adjusted.
#     The remaining features stay unchanged, so the input remains compatible with the trained model.
#     """)

#     what_if_input = selected_model_input.copy()

#     editable_features = [
#         "time_in_hospital",
#         "num_lab_procedures",
#         "num_procedures",
#         "num_medications",
#         "number_outpatient",
#         "number_emergency",
#         "number_inpatient",
#         "number_diagnoses"
#     ]

#     existing_editable_features = [
#         feature for feature in editable_features if feature in what_if_input.columns
#     ]

#     slider_ranges = {
#         "time_in_hospital": (1, 14),
#         "num_lab_procedures": (1, 132),
#         "num_procedures": (0, 6),
#         "num_medications": (1, 81),
#         "number_outpatient": (0, 40),
#         "number_emergency": (0, 76),
#         "number_inpatient": (0, 21),
#         "number_diagnoses": (1, 16)
#     }

#     for feature in existing_editable_features:
#         min_value, max_value = slider_ranges[feature]
#         current_value = int(what_if_input[feature].iloc[0])
#         current_value = max(min_value, min(max_value, current_value))

#         new_value = st.slider(feature, min_value, max_value, current_value)
#         what_if_input[feature] = new_value

#     if st.button("Calculate What-if Risk Score"):
#         what_if_raw = float(model.predict(what_if_input)[0])
#         what_if_score = float(np.clip(what_if_raw, 0, 1))
#         what_if_category, what_if_advice = assign_risk_category(what_if_score)

#         col1, col2 = st.columns(2)

#         col1.metric("Original Risk Score", f"{risk_score:.3f}")
#         col2.metric("What-if Risk Score", f"{what_if_score:.3f}")

#         st.metric("What-if Risk Category", what_if_category)

#         st.subheader("What-if Recommended Action")
#         st.write(what_if_advice)

#         st.caption("""
#         The what-if result is based on changing selected variables while keeping the rest of the
#         patient profile unchanged. It should be interpreted as a demonstration of model behaviour,
#         not as clinical advice.
#         """)


"""
02_dashboard.py

Streamlit dashboard
Project: Hospital 30-Day Readmission Risk Prediction
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Page settings
# --------------------------------------------------

st.set_page_config(
    page_title="Hospital Readmission Dashboard",
    page_icon="🏥",
    layout="wide"
)


# --------------------------------------------------
# Custom UI Styling
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #F6F9FC;
    color: #102A43;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #EEF6FF 100%);
    border-right: 1px solid #D9EAF7;
}

section[data-testid="stSidebar"] * {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
    color: #0B2E4A;
    font-weight: 700;
}

h1 {
    font-size: 2.2rem;
    margin-bottom: 0.3rem;
}

h2 {
    margin-top: 1.6rem;
}

p, li, div {
    color: #334E68;
}

.dashboard-card {
    background: #FFFFFF;
    padding: 1.4rem 1.6rem;
    border-radius: 18px;
    border: 1px solid #E3EDF7;
    box-shadow: 0 8px 24px rgba(16, 42, 67, 0.06);
    margin-bottom: 1.2rem;
    animation: fadeIn 0.35s ease-in-out;
}

.hero-card {
    background: linear-gradient(135deg, #EAF6FF 0%, #FFFFFF 100%);
    padding: 1.8rem;
    border-radius: 22px;
    border: 1px solid #D6EAF8;
    box-shadow: 0 10px 28px rgba(16, 42, 67, 0.07);
    margin-bottom: 1.5rem;
}

.badge {
    display: inline-block;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    background: #E3F2FD;
    color: #0B5CAD;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
}

.warning-badge {
    display: inline-block;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    background: #FFF4E5;
    color: #9A5B00;
    font-size: 0.8rem;
    font-weight: 600;
}

.success-badge {
    display: inline-block;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    background: #E8F7EF;
    color: #1B7F4C;
    font-size: 0.8rem;
    font-weight: 600;
}

.metric-card {
    background: #FFFFFF;
    padding: 1.2rem;
    border-radius: 18px;
    border: 1px solid #E3EDF7;
    box-shadow: 0 6px 20px rgba(16, 42, 67, 0.05);
    text-align: center;
    transition: all 0.2s ease-in-out;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 28px rgba(16, 42, 67, 0.09);
}

.metric-value {
    font-size: 1.7rem;
    font-weight: 700;
    color: #0B5CAD;
}

.metric-label {
    font-size: 0.85rem;
    color: #627D98;
    margin-top: 0.2rem;
}

.stMetric {
    background: #FFFFFF;
    padding: 1rem;
    border-radius: 16px;
    border: 1px solid #E3EDF7;
    box-shadow: 0 6px 18px rgba(16, 42, 67, 0.05);
}

div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid #E3EDF7;
}

button[kind="primary"], .stButton > button {
    border-radius: 12px;
    border: none;
    background: #0B5CAD;
    color: white;
    font-weight: 600;
    padding: 0.6rem 1.1rem;
    transition: all 0.2s ease-in-out;
}

.stButton > button:hover {
    background: #084B8A;
    transform: translateY(-1px);
}

.stAlert {
    border-radius: 14px;
}

hr {
    border: none;
    height: 1px;
    background: #D9EAF7;
    margin: 1.5rem 0;
}

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(8px);}
    to {opacity: 1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Helper UI components
# --------------------------------------------------

def page_header(icon: str, title: str, subtitle: str):
    st.markdown(f"""
    <div class="hero-card">
        <div class="badge">{icon} Hospital Readmission Project</div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def card(title: str, body: str, icon: str = "ℹ️"):
    st.markdown(f"""
    <div class="dashboard-card">
        <h3>{icon} {title}</h3>
        <p>{body}</p>
    </div>
    """, unsafe_allow_html=True)


def custom_metric(label: str, value: str, icon: str = "📊"):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{icon} {value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def styled_plot(fig):
    fig.patch.set_facecolor("#FFFFFF")
    ax = fig.axes[0]
    ax.set_facecolor("#FFFFFF")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D9EAF7")
    ax.spines["bottom"].set_color("#D9EAF7")
    ax.title.set_color("#0B2E4A")
    ax.xaxis.label.set_color("#334E68")
    ax.yaxis.label.set_color("#334E68")
    return fig


# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_PATH = Path("Dataset/readmissions.csv")
OUTPUT_DIR = Path("outputs")

EVAL_PATH = OUTPUT_DIR / "evaluation_results.json"
PRED_PATH = OUTPUT_DIR / "test_predictions.csv"
MODEL_PATH = OUTPUT_DIR / "final_readmission_regression_model.joblib"
X_TEST_INPUT_PATH = OUTPUT_DIR / "X_test_model_input.csv"
PATIENT_PRED_PATH = OUTPUT_DIR / "patient_level_predictions.csv"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance_original_top20.png"
FEATURE_IMPORTANCE_CSV_PATH = OUTPUT_DIR / "feature_importance_original_level.csv"


# --------------------------------------------------
# Cached loaders
# --------------------------------------------------

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["readmitted_30_days"] = (data["readmitted"] == "<30").astype(int)
    return data


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)


@st.cache_data
def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def assign_risk_category(score: float):
    if score < 0.06:
        return "Low Risk", "Standard discharge process."
    if score < 0.15:
        return "Medium Risk", "Consider follow-up contact."
    return "Higher Risk", "Additional review or support."


def show_missing_file_warning(path: Path):
    st.error(f"Required file not found: `{path}`")
    st.info("Run `python 01_final_model_pipeline.py` first and check your file paths.")


# --------------------------------------------------
# Initial dataset loading
# --------------------------------------------------

if not DATA_PATH.exists():
    st.error("Dataset file not found. Expected file: `readmissions.csv`")
    st.stop()

df = load_data(DATA_PATH)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.markdown("## 🏥 Readmission AI")
st.sidebar.caption("Decision-support prototype")

page = st.sidebar.radio(
    "Navigation",
    [
        "Project Overview",
        "Dataset Insights",
        "Model Results",
        "Prediction Analysis",
        "Explainability",
        "Stakeholder Advice",
        "Limitations",
        "Patient Risk Prediction"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Final Challenge - Faisal Guleed")


# --------------------------------------------------
# Page 1: Project Overview
# --------------------------------------------------

if page == "Project Overview":
    page_header(
        "🏥",
        "Hospital 30-Day Readmission Risk Dashboard",
        "A stakeholder-friendly dashboard for understanding readmission risk, model behaviour and project limitations."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        custom_metric("Rows in dataset", f"{df.shape[0]:,}", "📄")
    with col2:
        custom_metric("Features", str(df.shape[1]), "🧬")
    with col3:
        custom_metric("30-day readmission rate", f"{df['readmitted_30_days'].mean() * 100:.2f}%", "📈")

    card(
        "Stakeholder",
        "The stakeholder is a healthcare professional or hospital decision-maker who wants to identify patients who may need additional support after discharge.",
        "👥"
    )

    card(
        "Research Question",
        "How can historical patient data be used to estimate the risk of hospital readmission within 30 days for patients with diabetes?",
        "🎯"
    )

    card(
        "Important Interpretation",
        "The model output is a relative risk score, not a calibrated probability. It should support clinical judgement, not replace it.",
        "⚠️"
    )


# --------------------------------------------------
# Page 2: Dataset Insights
# --------------------------------------------------

elif page == "Dataset Insights":
    page_header(
        "📊",
        "Dataset Insights",
        "Overview of target design, class imbalance, missing values and key data limitations."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        custom_metric("Rows", f"{df.shape[0]:,}", "📄")
    with col2:
        custom_metric("Columns", str(df.shape[1]), "📚")
    with col3:
        custom_metric("30-day readmission", f"{df['readmitted_30_days'].mean() * 100:.2f}%", "🏥")

    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("Original Target Distribution")
    target_counts = df["readmitted"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 4))
    target_counts.plot(kind="bar", ax=ax)
    ax.set_title("Original Readmission Labels")
    ax.set_xlabel("Readmission Category")
    ax.set_ylabel("Number of Encounters")
    st.pyplot(styled_plot(fig))
    st.markdown("</div>", unsafe_allow_html=True)

    st.warning("""
    The original target contains `NO`, `<30`, and `>30`.  
    For this project, `<30` is labelled as 1 and all other cases as 0.
    This fits the 30-day readmission goal, but it also causes information loss.
    """)

    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("Binary Target Distribution")
    binary_counts = df["readmitted_30_days"].value_counts(normalize=True).sort_index() * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    binary_counts.plot(kind="bar", ax=ax)
    ax.set_title("30-Day Readmission Target")
    ax.set_xlabel("Readmitted within 30 days")
    ax.set_ylabel("Percentage")
    ax.set_xticklabels(["No", "Yes"], rotation=0)
    st.pyplot(styled_plot(fig))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("Missing Values")
    missing_values = df.replace("?", pd.NA).isna().mean().sort_values(ascending=False) * 100
    missing_values = missing_values[missing_values > 0].head(10)

    if missing_values.empty:
        st.success("No missing values found after checking for '?' values.")
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        missing_values.plot(kind="bar", ax=ax)
        ax.set_title("Top Missing Values")
        ax.set_ylabel("Missing Percentage")
        ax.set_xlabel("Feature")
        st.pyplot(styled_plot(fig))
    st.markdown("</div>", unsafe_allow_html=True)

    card(
        "Key Data Insight",
        "The dataset is large, but readmission is difficult to predict because important context such as social support, disease severity, medication adherence and discharge quality is missing.",
        "💡"
    )


# --------------------------------------------------
# Page 3: Model Results
# --------------------------------------------------

elif page == "Model Results":
    page_header(
        "📈",
        "Model Results",
        "Final baseline regression results and interpretation of predictive performance."
    )

    if not EVAL_PATH.exists():
        show_missing_file_warning(EVAL_PATH)
        st.stop()

    results = load_json(EVAL_PATH)

    col1, col2, col3 = st.columns(3)
    with col1:
        custom_metric("MAE", str(results["MAE"]), "📉")
    with col2:
        custom_metric("RMSE", str(results["RMSE"]), "📊")
    with col3:
        custom_metric("R²", str(results["R2"]), "🧪")

    card(
        "MAE Interpretation",
        f"MAE = {results['MAE']}. This means the model prediction is on average about {results['MAE']} away from the true target value on a 0 to 1 risk-score scale.",
        "📉"
    )

    card(
        "RMSE Interpretation",
        f"RMSE = {results['RMSE']}. Because RMSE is higher than MAE, some predictions contain larger errors.",
        "📊"
    )

    card(
        "R² Interpretation",
        f"R² = {results['R2']}. This low score indicates that the model explains only a small part of the variation in readmission outcomes.",
        "🧪"
    )

    st.warning("""
    The low performance should not only be interpreted as a model problem.
    It also reflects limitations in the dataset, target design and label quality.
    """)

    card(
        "Main Technical Conclusion",
        "The model learned the general readmission rate, but struggled to explain individual variation. This suggests that data quality and target design are stronger limitations than model complexity.",
        "✅"
    )


# --------------------------------------------------
# Page 4: Prediction Analysis
# --------------------------------------------------

elif page == "Prediction Analysis":
    page_header(
        "🔎",
        "Prediction Analysis",
        "Distribution and interpretation of model risk scores on the test set."
    )

    if not PRED_PATH.exists():
        show_missing_file_warning(PRED_PATH)
        st.stop()

    predictions = load_csv(PRED_PATH)

    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("Prediction Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    predictions["predicted_risk_clipped"].hist(bins=30, ax=ax)
    ax.set_title("Distribution of Predicted Risk Scores")
    ax.set_xlabel("Predicted Risk Score")
    ax.set_ylabel("Number of Patients")
    st.pyplot(styled_plot(fig))
    st.markdown("</div>", unsafe_allow_html=True)

    card(
        "Prediction Interpretation",
        "Most predictions are close to the average readmission rate. This suggests that the model is conservative and struggles to clearly separate low-risk and high-risk patients.",
        "🧠"
    )

    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("Prediction Summary")
    st.dataframe(predictions["predicted_risk_clipped"].describe())
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("Highest Risk Patients")
    top_risk = predictions.sort_values("predicted_risk_clipped", ascending=False).head(20)
    st.dataframe(top_risk)
    st.markdown("</div>", unsafe_allow_html=True)

    st.warning("""
    These scores are relative risk scores, not calibrated probabilities.
    A score of 0.20 means higher than average risk, not automatically a 20% probability.
    """)


# --------------------------------------------------
# Page 5: Explainability
# --------------------------------------------------

elif page == "Explainability":
    page_header(
        "🧩",
        "Explainability",
        "Understanding which feature groups influenced the linear model."
    )

    card(
        "Important Warning",
        "Feature importance is based on total absolute coefficient values after one-hot encoding. It explains model behaviour, not causal clinical importance.",
        "⚠️"
    )

    if FEATURE_IMPORTANCE_PATH.exists():
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.image(str(FEATURE_IMPORTANCE_PATH), caption="Top 20 Feature Importances")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Feature importance image not found.")

    if FEATURE_IMPORTANCE_CSV_PATH.exists():
        importance_df = load_csv(FEATURE_IMPORTANCE_CSV_PATH)
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("Feature Importance Table")
        st.dataframe(importance_df.head(20))
        st.markdown("</div>", unsafe_allow_html=True)

    st.warning("""
    Diagnosis variables contain many categories and are expanded into many one-hot encoded columns.
    Therefore, they may receive higher total importance scores simply because they consist of many encoded variables.
    """)


# --------------------------------------------------
# Page 6: Stakeholder Advice
# --------------------------------------------------

elif page == "Stakeholder Advice":
    page_header(
        "🩺",
        "Stakeholder Advice",
        "How the model output could support healthcare professionals."
    )

    card(
        "Decision Support Only",
        "The model should be used as a decision-support tool, not as an automated decision-making system.",
        "⚠️"
    )

    advice_df = pd.DataFrame({
        "Risk Level": ["Low Risk", "Medium Risk", "Higher Risk"],
        "Risk Score": ["0.00 - 0.06", "0.06 - 0.15", "> 0.15"],
        "Possible Action": [
            "Standard discharge process",
            "Consider follow-up contact",
            "Additional review or support"
        ]
    })

    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("Possible Risk Categories")
    st.table(advice_df)
    st.markdown("</div>", unsafe_allow_html=True)

    st.info("""
    These thresholds are examples only. In a real hospital setting, thresholds should be chosen
    together with healthcare professionals and validated on real-world outcomes.
    """)

    card(
        "Practical Use",
        "The most realistic use of this model is patient ranking. A hospital could use the score to identify which patients may need additional review.",
        "✅"
    )


# --------------------------------------------------
# Page 7: Limitations
# --------------------------------------------------

elif page == "Limitations":
    page_header(
        "⚠️",
        "Limitations",
        "Key reasons why this model should not be used as a clinical system."
    )

    limitations = [
        ("Coarse Target Labels", "The dataset does not contain exact readmission days. It only contains NO, <30 and >30."),
        ("Information Loss", "Patients readmitted after more than 30 days are treated as 0 in the final target."),
        ("Missing Context", "The dataset lacks social support, discharge quality, disease severity and medication adherence."),
        ("Limited Predictive Power", "The low R² score shows that the model explains only a small part of the variation."),
        ("Risk Score Is Not a Probability", "The Linear Regression output is useful for ranking, but not as a calibrated probability."),
        ("Not Clinically Validated", "The dashboard and thresholds were created for educational and exploratory purposes.")
    ]

    for title, body in limitations:
        card(title, body, "⚠️")


# --------------------------------------------------
# Page 8: Patient Risk Prediction
# --------------------------------------------------

elif page == "Patient Risk Prediction":
    page_header(
        "🧑‍⚕️",
        "Patient Risk Prediction",
        "Demonstration of the saved model pipeline on historical test-set patients."
    )

    st.warning("""
    This is historical test data. The actual outcome is shown only for evaluation.
    The model output is a relative risk score, not a calibrated probability.
    """)

    required_files = [MODEL_PATH, X_TEST_INPUT_PATH, PATIENT_PRED_PATH]
    missing_files = [str(path) for path in required_files if not path.exists()]

    if missing_files:
        st.error("The following required files are missing:")
        st.write(missing_files)
        st.stop()

    model = load_model(MODEL_PATH)
    patient_predictions = load_csv(PATIENT_PRED_PATH)
    X_test_input = load_csv(X_TEST_INPUT_PATH)
    X_test_input = X_test_input.set_index("row_id")

    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("Select a Patient from the Test Set")

    selected_row_id = st.selectbox(
        "Select patient row ID",
        patient_predictions["row_id"].astype(int).tolist()
    )

    selected_patient_info = patient_predictions[
        patient_predictions["row_id"] == selected_row_id
    ].iloc[0]

    selected_model_input = X_test_input.loc[[selected_row_id]]

    raw_prediction = float(model.predict(selected_model_input)[0])
    risk_score = float(np.clip(raw_prediction, 0, 1))
    risk_category, advice = assign_risk_category(risk_score)

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Risk Score", f"{risk_score:.3f}")
    col2.metric("Risk Category", risk_category)
    col3.metric("Actual 30-Day Readmission", int(selected_patient_info["actual_readmitted_30_days"]))

    st.caption(f"Raw model output before clipping: {raw_prediction:.4f}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("Original Patient Information")

    display_columns = [
        "age", "gender", "race", "readmitted",
        "time_in_hospital", "num_lab_procedures",
        "num_procedures", "num_medications",
        "number_outpatient", "number_emergency",
        "number_inpatient", "number_diagnoses",
        "diag_1", "diag_2", "diag_3"
    ]

    display_columns = [col for col in display_columns if col in selected_patient_info.index]

    st.dataframe(selected_patient_info[display_columns].to_frame(name="Value"))
    st.markdown("</div>", unsafe_allow_html=True)

    card("Recommended Action", advice, "🩺")

    st.divider()

    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("What-if Simulation")

    st.write("""
    This section starts from the selected real patient and allows a few values to be adjusted.
    The remaining features stay unchanged.
    """)

    what_if_input = selected_model_input.copy()

    editable_features = [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses"
    ]

    slider_ranges = {
        "time_in_hospital": (1, 14),
        "num_lab_procedures": (1, 132),
        "num_procedures": (0, 6),
        "num_medications": (1, 81),
        "number_outpatient": (0, 40),
        "number_emergency": (0, 76),
        "number_inpatient": (0, 21),
        "number_diagnoses": (1, 16)
    }

    for feature in editable_features:
        if feature in what_if_input.columns:
            min_value, max_value = slider_ranges[feature]
            current_value = int(what_if_input[feature].iloc[0])
            current_value = max(min_value, min(max_value, current_value))
            new_value = st.slider(feature, min_value, max_value, current_value)
            what_if_input[feature] = new_value

    if st.button("Calculate What-if Risk Score"):
        what_if_raw = float(model.predict(what_if_input)[0])
        what_if_score = float(np.clip(what_if_raw, 0, 1))
        what_if_category, what_if_advice = assign_risk_category(what_if_score)

        col1, col2 = st.columns(2)
        col1.metric("Original Risk Score", f"{risk_score:.3f}")
        col2.metric("What-if Risk Score", f"{what_if_score:.3f}")

        st.metric("What-if Risk Category", what_if_category)
        st.write(what_if_advice)

        st.caption("""
        The what-if result demonstrates model behaviour only.
        It should not be interpreted as clinical advice or causal evidence.
        """)

    st.markdown("</div>", unsafe_allow_html=True)
