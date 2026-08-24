from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "final_readmission_classifier.joblib"
RESULTS_PATH = OUTPUT_DIR / "evaluation_results.json"
THRESHOLD_PATH = OUTPUT_DIR / "threshold_analysis.csv"
PERMUTATION_PATH = OUTPUT_DIR / "permutation_importance.csv"
COEFFICIENT_PATH = OUTPUT_DIR / "coefficient_magnitudes.csv"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_results() -> dict | None:
    if not RESULTS_PATH.exists():
        return None
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def require_artifacts() -> tuple[object, dict] | tuple[None, None]:
    model = load_model()
    results = load_results()
    if model is None or results is None:
        st.error(
            "Model artifacts are missing. Run `python -m src.train` from the "
            "repository root first."
        )
        return None, None
    return model, results


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100 * float(value):.1f}%"


def score(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def risk_category(value: float, results: dict) -> str:
    thresholds = results["risk_categories"]
    if value < thresholds["low_to_medium_threshold"]:
        return "Low"
    if value < thresholds["medium_to_higher_threshold"]:
        return "Medium"
    return "Higher"
