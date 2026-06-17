# Hospital Readmission Risk Prediction

## Project overview

This project investigates whether historical hospital data can be used to estimate the relative risk of 30-day hospital readmission among patients with diabetes.

The final output is intended as a decision-support tool for healthcare stakeholders. The generated risk score should be interpreted as a relative risk score rather than a calibrated probability.

## Project structure

* `01_final_model_pipeline_integrated.py`
  Trains the final model, evaluates performance, and generates output files.

* `02_dashboard_integrated.py`
  Streamlit dashboard used to present the project results to stakeholders.

* `outputs/`
  Contains generated files such as evaluation results, predictions, and feature importance outputs.

* `readmissions.csv`
  Original dataset used for modelling.

* `Domain analysis document.docx`
  Domain analysis and literature research.

* `Main challenge final presentation.pptx`
  Final presentation.

* `final_notebook_conclusions.html`
  Export of the notebook containing conclusions and additional analyses.

## How to run the project

1. Install the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib streamlit joblib
```

2. Run the modelling pipeline:

```bash
python 01_final_model_pipeline_integrated.py
```

3. Launch the dashboard:

```bash
streamlit run 02_dashboard_integrated.py
```

## Final model

The final selected model is Linear Regression. Although alternative models were explored, they did not provide substantial improvements while reducing interpretability.

## Important notes

* The dataset is imbalanced, with approximately 11% of patients readmitted within 30 days.
* The model output represents a relative risk score and should not be interpreted as a clinical probability.
* The demonstrated risk thresholds are illustrative and not clinically validated.

## Author

Faisal Guleed
Fontys ICT – AI & Machine Learning Main Semester
