# Hospital Readmission Risk Model

An end-to-end machine-learning portfolio project that predicts **30-day hospital readmission risk for diabetic inpatient encounters** using the UCI Diabetes 130-US Hospitals dataset.

> **Important:** this is a research/portfolio prototype. It has not been clinically validated and must not be used for patient-care decisions.

## What this project demonstrates

- Leakage-aware **patient-level** train/validation/test splitting
- Scikit-learn `Pipeline` + `ColumnTransformer`
- Numeric imputation and `StandardScaler`
- Categorical imputation and one-hot encoding with unknown-category handling
- Class-imbalance handling with balanced Logistic Regression
- Validation-only operating-threshold selection
- Evaluation with ROC-AUC, PR-AUC, precision, recall, F1 and confusion matrix
- A `DummyClassifier` prior baseline
- Training-derived Low / Medium / Higher relative-risk bands
- Permutation importance plus clearly labelled coefficient-magnitude analysis
- A Streamlit dashboard that loads the exact artifacts produced by training
- Unit tests and GitHub Actions CI

## Dataset

This project uses **Diabetes 130-US Hospitals for Years 1999–2008** from the UCI Machine Learning Repository. The dataset contains 101,766 hospital encounters and includes repeated encounters for some patients, which is why this project splits by `patient_nbr` instead of randomly splitting individual encounters.

UCI source: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

Citation:

> Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). *Diabetes 130-US Hospitals for Years 1999–2008* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J

The dataset is licensed under **CC BY 4.0**. See [`DATASET_ATTRIBUTION.md`](DATASET_ATTRIBUTION.md).

## Target

The original `readmitted` field contains three outcomes:

- `<30` — readmitted within 30 days
- `>30` — readmitted after 30 days
- `NO` — no recorded readmission

The binary target used here is:

```text
1 = <30
0 = >30 or NO
```

## Leakage-aware split

A normal encounter-level random split can place two encounters belonging to the same patient in different partitions. This project avoids that by splitting **unique patients first**, then assigning all encounters for each patient to exactly one partition:

```text
70% train / 15% validation / 15% test
```

Patient groups are stratified by whether a patient has at least one positive encounter. The validation set is used to select the classifier operating threshold. The final test set is reserved for final reporting.

## Model

The baseline model is a balanced Logistic Regression classifier. Numeric and categorical preprocessing are fitted only inside the scikit-learn pipeline.

Key design choices:

- `SimpleImputer(strategy="median")` + `StandardScaler()` for numeric features
- `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")` for categorical features
- `class_weight="balanced"` for the imbalanced target
- `predict_proba()` for ranking scores

Because balanced class weights change the relationship between model scores and the observed event rate, the dashboard labels the output as a **relative risk score**, not as a clinically calibrated individual probability.

## Thresholds

Two different threshold concepts are intentionally separated:

1. **Classifier operating threshold** — selected on the validation set by maximizing F1, with recall used as a tie-breaker.
2. **Risk-band thresholds** — the 50th and 80th percentiles of **training-set** risk scores. These define Low / Medium / Higher relative-risk groups and are not clinical cut-offs.

The full validation threshold sweep is exported to `outputs/threshold_analysis.csv`.

## Results

Run the training pipeline to generate current results:

```bash
python -m src.train
```

The refactor changes the evaluation split, so older encounter-level metrics should **not** be reused as if they were directly comparable. Current final metrics are generated in:

```text
outputs/evaluation_results.json
```

The Streamlit dashboard reads those metrics directly, preventing README/dashboard/model drift.

## Project structure

```text
Hospital-readmission-model/
├── .github/
│   └── workflows/
│       └── ci.yml
├── dashboard/
│   ├── app.py
│   ├── shared.py
│   └── pages/
│       ├── 1_Model_Results.py
│       ├── 2_Risk_Explorer.py
│       ├── 3_Explainability.py
│       └── 4_Limitations.py
├── data/
│   └── diabetic_data.csv          # local dataset; use your existing copy
├── outputs/                       # generated; ignored by git
├── src/
│   ├── __init__.py
│   ├── modeling.py
│   └── train.py
├── tests/
│   └── test_modeling.py
├── .gitignore
├── DATASET_ATTRIBUTION.md
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Guleed2509/Hospital-readmission-model.git
cd Hospital-readmission-model
```

### 2. Create a virtual environment

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For development/tests:

```bash
pip install -r requirements-dev.txt
```

### 4. Add the dataset

Place `diabetic_data.csv` in:

```text
data/diabetic_data.csv
```

The training script also recognizes `Data/diabetic_data.csv`, a root-level file, or a custom location provided through `READMISSION_DATA_PATH`.

### 5. Train and evaluate

```bash
python -m src.train
```

Generated artifacts include:

```text
outputs/
├── final_readmission_classifier.joblib
├── evaluation_results.json
├── threshold_analysis.csv
├── test_predictions.csv
├── coefficient_magnitudes.csv
└── permutation_importance.csv
```

### 6. Run the dashboard

```bash
streamlit run dashboard/app.py
```

## Testing and CI

Run locally:

```bash
ruff check src dashboard tests
pytest -q
python -m compileall -q src dashboard
```

The GitHub Actions workflow runs the same checks on pushes and pull requests.

## Interpretation and limitations

- The model is a **baseline**, not a clinical product.
- Data is historical (1999–2008) and limited to diabetic inpatient encounters in the source dataset.
- Demographic attributes require careful subgroup/fairness evaluation before any real-world use.
- Model associations and coefficients are **not causal effects**.
- Risk bands are relative ranking groups, not medical thresholds.
- The model score is not presented as a clinically calibrated probability.
- Real deployment would require external validation, calibration, fairness analysis, clinical governance, monitoring, privacy/security controls and prospective evaluation.

## Privacy and responsible demo use

The Streamlit Risk Explorer is intentionally framed as a **synthetic what-if demo**. Do not enter identifiable patient information.

## Future improvements

- Add subgroup performance/fairness reporting
- Add probability calibration experiments on a dedicated calibration/validation strategy
- Compare Logistic Regression against tree-based baselines using the same patient-level split
- Add temporal or hospital-level external validation if appropriate data becomes available
- Add model-card documentation and deployment monitoring

## Reproducibility

The model and split use a fixed random seed (`42`). Generated model outputs are intentionally excluded from git so results are produced from the code and local dataset rather than silently drifting out of sync.
