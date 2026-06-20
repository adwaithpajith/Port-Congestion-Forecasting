"""
train.py — Full training pipeline for port congestion forecasting.

Run this script locally or in Google Colab after cloning the repo:
    python scripts/train.py

What it does:
  1. Downloads NOAA AIS data for the specified date range
  2. Builds congestion features (queue length, arrival rate)
  3. Trains XGBoost forecaster (6h-ahead queue length)
  4. Computes SHAP explainer
  5. Saves model + explainer + feature list to models/
  6. Saves processed data to data/

After running, commit the generated files:
    git add models/ data/
    git commit -m "add trained model and processed data"
    git push
"""

import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Allow imports from project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baseline import predict_baseline
from src.features import get_feature_cols, make_features
from src.pipeline import (
    build_congestion_features,
    compute_wait_times,
    download_date_range,
    extract_events,
)

# ── Config ───────────────────────────────────────────────────────────────────
START_DATE = datetime(2021, 10, 1)
N_DAYS = 45          # Oct 1 – Nov 14 2021 — captures full crisis arc
FORECAST_HORIZON = 6  # hours ahead

OUT_MODELS = ROOT / "models"
OUT_DATA = ROOT / "data"
OUT_MODELS.mkdir(exist_ok=True)
OUT_DATA.mkdir(exist_ok=True)


def main():
    print("=" * 60)
    print("Port Congestion Forecasting — Training Pipeline")
    print("=" * 60)

    # ── 1. Download AIS data ─────────────────────────────────────────────────
    print(f"\n[1/5] Downloading {N_DAYS} days of NOAA AIS data...")
    ais_df = download_date_range(START_DATE, N_DAYS, verbose=True)

    # ── 2. Build congestion features ─────────────────────────────────────────
    print("\n[2/5] Building congestion features...")
    congestion = build_congestion_features(ais_df)

    # Drop flagged data gaps before training
    clean = congestion[~congestion["data_gap"]].reset_index(drop=True)
    print(f"  Clean hours: {len(clean)} / {len(congestion)} total")

    clean.to_csv(OUT_DATA / "congestion_features.csv", index=False)
    print(f"  Saved → data/congestion_features.csv")

    # Wait times (ground truth for the queue model)
    event_log = extract_events(ais_df)
    wait_df = compute_wait_times(event_log)
    wait_df.to_csv(OUT_DATA / "wait_times.csv", index=False)
    print(f"  Saved → data/wait_times.csv  ({len(wait_df)} vessel calls)")

    # ── 3. Feature engineering ───────────────────────────────────────────────
    print("\n[3/5] Engineering features...")
    df_feat = make_features(clean, horizon=FORECAST_HORIZON)
    feature_cols = get_feature_cols(df_feat)
    print(f"  Features: {len(feature_cols)}")

    # Temporal 80/20 split — never shuffle time-series data
    split_idx = int(len(df_feat) * 0.80)
    train = df_feat.iloc[:split_idx]
    test = df_feat.iloc[split_idx:]

    X_train, y_train = train[feature_cols], train["target"]
    X_test, y_test = test[feature_cols], test["target"]

    # ── 4. Train XGBoost ─────────────────────────────────────────────────────
    print("\n[4/5] Training XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=30,
        eval_metric="mae",
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    xgb_preds = np.clip(model.predict(X_test), 0, None)
    xgb_mae = mean_absolute_error(y_test, xgb_preds)
    xgb_rmse = mean_squared_error(y_test, xgb_preds) ** 0.5

    # Baseline comparison
    baseline_preds = predict_baseline(test)
    base_mae = mean_absolute_error(y_test, baseline_preds)
    base_rmse = mean_squared_error(y_test, baseline_preds) ** 0.5

    improvement = (base_mae - xgb_mae) / base_mae * 100
    print(f"  XGBoost  → MAE: {xgb_mae:.2f} vessels | RMSE: {xgb_rmse:.2f}")
    print(f"  Baseline → MAE: {base_mae:.2f} vessels | RMSE: {base_rmse:.2f}")
    print(f"  MAE improvement over baseline: {improvement:.1f}%")

    # Save metrics alongside the model
    metrics = {
        "xgb_mae": round(xgb_mae, 3),
        "xgb_rmse": round(xgb_rmse, 3),
        "baseline_mae": round(base_mae, 3),
        "baseline_rmse": round(base_rmse, 3),
        "mae_improvement_pct": round(improvement, 1),
        "train_rows": len(train),
        "test_rows": len(test),
        "n_features": len(feature_cols),
        "forecast_horizon_h": FORECAST_HORIZON,
    }

    # Persist backtest predictions for the Streamlit dashboard
    backtest_df = test[["hour"]].copy()
    backtest_df["actual"] = y_test.values
    backtest_df["xgb_pred"] = xgb_preds
    backtest_df["baseline_pred"] = baseline_preds.values
    backtest_df.to_csv(OUT_DATA / "backtest.csv", index=False)
    print("  Saved → data/backtest.csv")

    # ── 5. SHAP + save artifacts ─────────────────────────────────────────────
    print("\n[5/5] Computing SHAP explainer and saving artifacts...")
    explainer = shap.TreeExplainer(model)

    model.save_model(OUT_MODELS / "xgb_port_congestion.json")

    with open(OUT_MODELS / "explainer.pkl", "wb") as f:
        pickle.dump(explainer, f)

    with open(OUT_MODELS / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    with open(OUT_MODELS / "metrics.json", "w") as f:
        json.dump(metrics, f)

    # Save a small SHAP values sample for the dashboard
    shap_sample = X_test.iloc[:100]
    shap_vals = explainer.shap_values(shap_sample)
    shap_df = pd.DataFrame(shap_vals, columns=feature_cols)
    shap_df.to_csv(OUT_DATA / "shap_values_sample.csv", index=False)
    shap_sample.reset_index(drop=True).to_csv(
        OUT_DATA / "shap_features_sample.csv", index=False
    )

    print("  Saved → models/xgb_port_congestion.json")
    print("  Saved → models/explainer.pkl")
    print("  Saved → models/feature_cols.json")
    print("  Saved → models/metrics.json")
    print("  Saved → data/shap_values_sample.csv")

    print("\n" + "=" * 60)
    print("Training complete. Now commit and push:")
    print("  git add models/ data/")
    print('  git commit -m "add trained model and data"')
    print("  git push")
    print("=" * 60)


if __name__ == "__main__":
    main()
