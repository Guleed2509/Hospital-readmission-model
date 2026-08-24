# Hospital Readmission Risk Prediction

An end-to-end machine learning project for estimating the relative risk of **30-day hospital readmission among patients with diabetes**.

The project includes data preprocessing, model training, evaluation, explainability, risk categorisation and an interactive **Streamlit dashboard** for presenting results to stakeholders.

> **Important:** This project is intended for educational and analytical purposes. The model is not clinically validated and should not be used for individual medical decision-making.

---

## Project Overview

Hospital readmissions are an important challenge in healthcare. Identifying patients with an elevated risk of being readmitted may help healthcare organisations better understand patterns in historical patient data and explore opportunities for targeted follow-up.

This project investigates whether historical hospital data can be used to estimate the **relative risk of readmission within 30 days** for patients with diabetes.

The final system consists of two main components:

1. A machine learning pipeline that preprocesses the data, trains and evaluates the model, and generates prediction and explainability outputs.
2. A Streamlit dashboard that presents model results and risk information in a stakeholder-friendly format.

The model output should be interpreted as a **relative risk score**, not as a clinically calibrated probability.

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Streamlit
* Joblib

---

## Machine Learning Workflow

The project follows the following workflow:

```text
Historical hospital data
        ↓
Data preprocessing
        ↓
Feature preparation
        ↓
Train / test split
        ↓
Logistic Regression
        ↓
Model evaluation
        ↓
Feature importance analysis
        ↓
Patient-level risk scores
        ↓
Streamlit dashboard
```

---

## Dataset

The dataset contains historical hospital encounters involving patients with diabetes.

The target variable is:

```text
readmitted_30_days
```

It represents whether a patient was readmitted to the hospital within 30 days.

The positive class is relatively uncommon:

```text
30-day readmission rate: 11.16%
```

This creates a **class-imbalanced machine learning problem**.

Because of this imbalance, accuracy alone is not sufficient for evaluating the model. Metrics such as **recall, precision, F1-score, ROC-AUC and PR-AUC** are also used.

> Add the original dataset source here if the dataset is publicly available.

---

## Final Model

The final model is a:

**Logistic Regression Classifier**

Configuration:

| Setting            |               Value |
| ------------------ | ------------------: |
| Model              | Logistic Regression |
| Test size          |                 20% |
| Random state       |                  42 |
| Class weighting    |            Balanced |
| Decision threshold |                0.50 |

The model uses:

```python
class_weight="balanced"
```

to give additional importance to the minority readmission class during training.

Logistic Regression also provides an interpretable linear model, which is useful in a healthcare-oriented project where understanding model behaviour is important.

---

## Model Performance

The final model achieved the following results on the test set:

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **64.56%** |
| Precision | **16.76%** |
| Recall    | **54.87%** |
| F1-score  | **25.67%** |
| ROC-AUC   | **0.6426** |
| PR-AUC    | **0.2030** |

### Interpretation

**ROC-AUC — 0.6426**

Measures how well the model ranks readmitted patients above non-readmitted patients across different classification thresholds.

**PR-AUC — 0.2030**

Precision-Recall Area Under the Curve is particularly relevant because only approximately 11% of encounters belong to the positive class.

**Recall — 54.87%**

Of the patients in the test set who were actually readmitted within 30 days, the model identified approximately **55%**.

**Precision — 16.76%**

Of the encounters classified as 30-day readmission cases, approximately **17%** were actually readmitted.

Because the dataset is imbalanced, the model involves a trade-off between identifying more readmission cases and generating more false-positive predictions.

---

## Confusion Matrix

At a decision threshold of `0.50`, the final model produced:

|                           | Predicted No Readmission | Predicted Readmission |
| ------------------------- | -----------------------: | --------------------: |
| **Actual No Readmission** |                   11,894 |                 6,189 |
| **Actual Readmission**    |                    1,025 |                 1,246 |

This corresponds to:

* **True Negatives:** 11,894
* **False Positives:** 6,189
* **False Negatives:** 1,025
* **True Positives:** 1,246

The model identifies a meaningful proportion of readmitted patients, but this comes with a relatively large number of false-positive classifications.

This trade-off is especially important when considering potential healthcare applications.

---

## Why Accuracy Is Not Enough

Approximately **88.84%** of observations are not readmitted within 30 days.

Because of this class imbalance, a model could achieve high accuracy simply by predicting the majority class most of the time.

For this reason, this project focuses on several complementary evaluation metrics:

* **Recall** — how many actual readmission cases are identified.
* **Precision** — how many predicted readmissions are correct.
* **F1-score** — balance between precision and recall.
* **ROC-AUC** — overall ranking performance across thresholds.
* **PR-AUC** — performance focused on the minority positive class.

---

## Relative Risk Scores

The Logistic Regression model generates prediction scores that are used to compare the relative readmission risk between encounters.

Because the model is trained with:

```python
class_weight="balanced"
```

these scores should **not be interpreted directly as clinically calibrated probabilities**.

For example, a score of:

```text
0.70
```

does not necessarily mean that a patient has a clinically validated 70% probability of being readmitted.

Instead, higher scores indicate **higher model-estimated relative risk**.

---

## Risk Categories

For presentation in the dashboard, model scores are divided into relative risk categories using quantiles derived from the training predictions.

Current thresholds:

| Risk transition |    Score |
| --------------- | -------: |
| Low → Medium    | 0.448295 |
| Medium → Higher | 0.579669 |

These categories are intended to make model output easier to communicate to stakeholders.

They are:

**relative presentation categories only and are not clinically validated thresholds.**

---

## Explainability

The project includes feature-importance outputs to help investigate which variables influence the model predictions.

Generated outputs include:

```text
outputs/encoded_feature_importance.csv
outputs/feature_importance.csv
outputs/feature_importance_original_features.csv
outputs/feature_importance_original_level.csv
outputs/feature_importance_top15.png
outputs/feature_importance_top20.png
outputs/feature_importance_original_top20.png
```

These outputs make it possible to inspect model behaviour at both encoded-feature and original-feature level.

Explainability is especially important in healthcare-related machine learning because model predictions should not be treated as black-box decisions.

---

## Dashboard

An interactive dashboard is provided using **Streamlit**.

The dashboard is designed to communicate the model results in a more accessible format for stakeholders.

It can be launched with:

```bash
streamlit run dashboard/app.py
```

After starting Streamlit, the application is normally available at:

```text
http://localhost:8501
```

### Recommended screenshot

For the GitHub portfolio version of this repository, add a screenshot of the dashboard to:

```text
assets/dashboard.png
```

and add the following to this section:

```markdown
![Hospital Readmission Dashboard](assets/dashboard.png)
```

---

## Project Structure

```text
hospital-readmission-risk/
│
├── dashboard/
│   └── app.py
│
├── Dataset/
│   └── readmissions.csv
│
├── Docs/
│   ├── Domain analysis document.docx
│   └── final_notebook_conclusions.html
│
├── outputs/
│   ├── evaluation_results.json
│   ├── patient_level_predictions.csv
│   ├── test_predictions.csv
│   ├── feature_importance.csv
│   ├── feature_importance_top15.png
│   ├── feature_importance_top20.png
│   ├── final_readmission_regression_model.joblib
│   └── ...
│
├── src/
│   └── train.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

### Main Files

#### `src/train.py`

Runs the machine learning pipeline.

The script handles model training, evaluation and generation of model output files.

#### `dashboard/app.py`

Streamlit application used to explore and present the model results.

#### `outputs/evaluation_results.json`

Contains the final model configuration, evaluation metrics, confusion matrix and risk-category thresholds.

#### `outputs/`

Contains generated predictions, model artifacts, evaluation files and feature-importance outputs.

#### `Docs/`

Contains supporting domain analysis and project documentation.

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd hospital-readmission-risk
```

### 2. Create a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

---

## How to Run

### Train and evaluate the model

From the project root:

```bash
python src/train.py
```

The pipeline generates model artifacts, predictions, evaluation metrics and explainability outputs inside:

```text
outputs/
```

### Start the dashboard

```bash
streamlit run dashboard/app.py
```

Streamlit will display a local address, normally:

```text
http://localhost:8501
```

Open this address in a browser to use the dashboard.

---

## Generated Outputs

The machine learning pipeline produces several artifacts, including:

```text
evaluation_results.json
patient_level_predictions.csv
test_predictions.csv
feature_importance.csv
feature_importance_original_features.csv
feature_importance_original_level.csv
model_input_columns.json
final_readmission_regression_model.joblib
```

These files support model evaluation, explainability and the Streamlit dashboard.

---

## Limitations

Several limitations should be considered when interpreting the results.

### Class imbalance

Only approximately **11.16%** of encounters represent a 30-day readmission.

This makes predicting the minority class substantially more difficult and results in a trade-off between recall and precision.

### Moderate predictive performance

The final ROC-AUC of **0.6426** indicates that the model contains predictive signal, but its ability to distinguish between readmitted and non-readmitted patients is limited.

The model should therefore not be interpreted as a production-ready clinical prediction system.

### False positives

The model produces a substantial number of false-positive classifications.

At the current threshold:

```text
False positives: 6,189
True positives:  1,246
```

Any real-world implementation would therefore require careful evaluation of the consequences of unnecessary interventions.

### Probability calibration

Because Logistic Regression uses balanced class weights, its prediction scores are used as **relative risk scores** rather than validated clinical probabilities.

Probability calibration would need to be evaluated separately before interpreting scores as absolute probabilities.

### Risk thresholds

The Low, Medium and Higher risk categories used in the dashboard are derived from model score distributions.

They have **not been clinically validated**.

### Generalisability

Performance on this dataset does not guarantee equivalent performance on patients from different hospitals, populations, geographic regions or time periods.

External validation would be required before considering real-world use.

---

## Ethical Considerations

Machine learning in healthcare requires additional care because incorrect predictions may affect real people.

Important considerations include:

* Potential demographic or historical bias in the dataset.
* Differences in healthcare access and treatment patterns.
* False negatives that may fail to identify high-risk patients.
* False positives that may lead to unnecessary follow-up.
* The need for transparency and explainability.
* The importance of human oversight.
* Validation on external patient populations.

This model should therefore be viewed as an **analytical and educational decision-support prototype**, not as an autonomous medical decision system.

---

## Possible Future Improvements

Potential next steps include:

* Compare additional classification algorithms.
* Perform more extensive hyperparameter optimisation.
* Evaluate different classification thresholds.
* Investigate probability calibration.
* Perform subgroup fairness analysis.
* Expand explainability analysis.
* Test performance on external datasets.
* Investigate temporal validation.
* Improve feature engineering.
* Deploy the dashboard as an online demonstration.
* Add automated tests and CI using GitHub Actions.

---

## Author

**Faisal Guleed**

Data & AI Student
Fontys ICT — AI & Machine Learning
