"""
predict.py — Load trained XGBoost model and run inference + SHAP explanations.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

MODEL_PATH = Path("models/xgb_port_congestion.json")
EXPLAINER_PATH = Path("models/explainer.pkl")
FEATURE_COLS_PATH = Path("models/feature_cols.json")


def load_artifacts() -> tuple[xgb.XGBRegressor, shap.TreeExplainer, list[str]]:
    """Load model, SHAP explainer, and feature column list from disk."""
    if not all(p.exists() for p in [MODEL_PATH, EXPLAINER_PATH, FEATURE_COLS_PATH]):
        raise FileNotFoundError(
            "Model files not found. Run `python scripts/train.py` first, "
            "then commit the generated models/ and data/ directories to GitHub."
        )

    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)

    with open(EXPLAINER_PATH, "rb") as f:
        explainer = pickle.load(f)

    with open(FEATURE_COLS_PATH) as f:
        feature_cols = json.load(f)

    return model, explainer, feature_cols


def predict(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
) -> np.ndarray:
    """Run inference and clip predictions to non-negative values."""
    preds = model.predict(X)
    return np.clip(preds, 0, None)


def explain(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
) -> np.ndarray:
    """Return SHAP values for the given feature matrix."""
    return explainer.shap_values(X)
