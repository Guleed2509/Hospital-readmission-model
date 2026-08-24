# Refactor summary

This patch addresses the issues found in the repository review:

- Replaced regression-era dashboard assumptions with one consistent Logistic Regression classification pipeline.
- Standardized the model artifact name to `final_readmission_classifier.joblib`.
- Replaced dashboard `.predict()` risk scoring with `.predict_proba()`.
- Removed hard-coded risk thresholds; dashboard reads thresholds generated during training.
- Replaced MAE/RMSE/R² reporting with classification metrics.
- Added patient-level train/validation/test splitting to prevent repeated-patient leakage across splits.
- Added validation-set threshold analysis and selection.
- Added `DummyClassifier` prior baseline.
- Added numeric feature scaling.
- Added permutation importance and renamed coefficient output to aggregated coefficient magnitude.
- Added UCI dataset citation and CC BY 4.0 attribution.
- Replaced the monolithic dashboard with a small multipage Streamlit app.
- Broke modeling logic into reusable/testable functions.
- Replaced environment-dump requirements with direct dependencies plus a dev requirements file.
- Expanded `.gitignore` for local secrets, caches and editor files.
- Added pytest coverage and GitHub Actions CI.
- Corrected Windows virtual-environment activation instructions.
- Removed hard-coded legacy metrics from documentation because the new patient-level split requires re-evaluation.
