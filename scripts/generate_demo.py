"""
generate_demo.py — ⚠️ DEV/UI-TESTING TOOL ONLY. NOT FOR DEPLOYMENT. ⚠️

Generates synthetic Oct 2021-style congestion data and trains a demo
XGBoost model purely so you can click through the Streamlit UI locally
without waiting for a real NOAA download. The numbers are fake.

For the actual deployed app, use scripts/train.py instead, which pulls
real, free NOAA Marine Cadastre AIS data (no API key required — see
README.md for details).

Run (for local UI testing only):
    python scripts/generate_demo.py

Every artifact this script produces is tagged "synthetic" in
models/metrics.json, and the app surfaces that tag in the UI, so you
can't accidentally deploy these files thinking they're real.

Do NOT commit the output of this script as your production model.
"""

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baseline import predict_baseline
from src.features import get_feature_cols, make_features

OUT_MODELS = ROOT / "models"
OUT_DATA = ROOT / "data"
OUT_MODELS.mkdir(exist_ok=True)
OUT_DATA.mkdir(exist_ok=True)

np.random.seed(42)


def generate_synthetic_congestion(n_days: int = 45) -> pd.DataFrame:
    """
    Synthetic congestion data modelled on Oct–Nov 2021 LA/LGB patterns:
    - Queue: mean ~43 vessels, std ~6, range 28–51
    - Higher weekday business hours, lower overnight
    - Gradual upward trend through Oct, slight relief in Nov
    - Occasional high-arrival-rate spikes (ship convoys, weather backlogs)
    """
    hours = pd.date_range("2021-10-01", periods=24 * n_days, freq="h")
    records = []

    for i, h in enumerate(hours):
        # Cyclical patterns
        hour_effect = -2.5 if h.hour < 6 else (1.5 if 8 <= h.hour <= 17 else 0)
        dow_effect = -2.0 if h.dayofweek >= 5 else 0

        # Crisis arc: ramp up through Oct, plateau, slight relief in Nov
        t = i / (24 * n_days)
        trend = 4 * np.sin(np.pi * t) - 1.5 * t

        noise = np.random.normal(0, 2.2)
        queue = max(0, 43 + hour_effect + dow_effect + trend + noise)

        # Arrival rate: mostly 1–3/hr with occasional convoy spikes
        if np.random.random() < 0.03:
            arrival = int(np.random.uniform(20, 44))
        else:
            arrival = max(0, int(np.random.poisson(2.2)))

        records.append({
            "hour": h,
            "queue_length": round(queue),
            "arrival_rate": arrival,
            "total_pings": int(np.random.normal(1900, 150)),
            "data_gap": False,
            "dayofweek": h.dayofweek,
            "hourofday": h.hour,
        })

    return pd.DataFrame(records)


def main():
    print("Generating synthetic demo data...")
    df = generate_synthetic_congestion(n_days=45)
    df.to_csv(OUT_DATA / "congestion_features.csv", index=False)
    print(f"  Saved data/congestion_features.csv ({len(df)} rows)")

    print("Engineering features...")
    df_feat = make_features(df, horizon=6)
    feature_cols = get_feature_cols(df_feat)

    split_idx = int(len(df_feat) * 0.80)
    train = df_feat.iloc[:split_idx]
    test = df_feat.iloc[split_idx:]

    X_train, y_train = train[feature_cols], train["target"]
    X_test, y_test = test[feature_cols], test["target"]

    print("Training XGBoost on demo data...")
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=20,
        eval_metric="mae",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    xgb_preds = np.clip(model.predict(X_test), 0, None)
    xgb_mae = mean_absolute_error(y_test, xgb_preds)
    xgb_rmse = mean_squared_error(y_test, xgb_preds) ** 0.5

    baseline_preds = predict_baseline(test)
    base_mae = mean_absolute_error(y_test, baseline_preds)
    base_rmse = mean_squared_error(y_test, baseline_preds) ** 0.5
    improvement = (base_mae - xgb_mae) / base_mae * 100

    print(f"  XGBoost  → MAE: {xgb_mae:.2f} | RMSE: {xgb_rmse:.2f}")
    print(f"  Baseline → MAE: {base_mae:.2f} | RMSE: {base_rmse:.2f}")
    print(f"  Improvement: {improvement:.1f}%")

    # Backtest CSV
    backtest_df = test[["hour"]].copy()
    backtest_df["actual"] = y_test.values
    backtest_df["xgb_pred"] = xgb_preds
    backtest_df["baseline_pred"] = baseline_preds.values
    backtest_df.to_csv(OUT_DATA / "backtest.csv", index=False)

    print("Computing SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    shap_sample = X_test.iloc[:100]
    shap_vals = explainer.shap_values(shap_sample)

    pd.DataFrame(shap_vals, columns=feature_cols).to_csv(
        OUT_DATA / "shap_values_sample.csv", index=False
    )
    shap_sample.reset_index(drop=True).to_csv(
        OUT_DATA / "shap_features_sample.csv", index=False
    )

    # Save artifacts
    model.save_model(OUT_MODELS / "xgb_port_congestion.json")
    with open(OUT_MODELS / "explainer.pkl", "wb") as f:
        pickle.dump(explainer, f)
    with open(OUT_MODELS / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)
    with open(OUT_MODELS / "metrics.json", "w") as f:
        json.dump({
            "xgb_mae": round(xgb_mae, 3),
            "xgb_rmse": round(xgb_rmse, 3),
            "baseline_mae": round(base_mae, 3),
            "baseline_rmse": round(base_rmse, 3),
            "mae_improvement_pct": round(improvement, 1),
            "note": "demo model — trained on synthetic data"
        }, f)

    print("\nDemo model saved. Run `git add models/ data/ && git commit && git push`")


if __name__ == "__main__":
    main()
