"""
pages/1_Dashboard.py — Main forecast dashboard.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import build_single_row
from src.predict import load_artifacts, predict

st.set_page_config(page_title="Dashboard · Port Congestion", page_icon="📊", layout="wide")

st.title("📊 Congestion Dashboard")
st.caption("Port of LA / Long Beach — 6-Hour Queue Forecast")


# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def get_artifacts():
    return load_artifacts()


@st.cache_data(show_spinner="Loading data...")
def load_data():
    congestion = pd.read_csv(ROOT / "data/congestion_features.csv", parse_dates=["hour"])
    backtest = pd.read_csv(ROOT / "data/backtest.csv", parse_dates=["hour"])
    with open(ROOT / "models/metrics.json") as f:
        metrics = json.load(f)
    return congestion, backtest, metrics


try:
    model, explainer, feature_cols = get_artifacts()
    congestion, backtest, metrics = load_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ── Top metrics ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model MAE", f"{metrics['xgb_mae']} vessels", f"vs {metrics['baseline_mae']} baseline")
c2.metric("Model RMSE", f"{metrics['xgb_rmse']} vessels")
c3.metric("MAE improvement", f"{metrics['mae_improvement_pct']}%", "over M/M/c baseline")
c4.metric("Training rows", metrics.get("train_rows", "—"))

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Historical Queue", "🔁 Backtest", "🔮 Live Forecast"])

# ── Tab 1: Historical queue ───────────────────────────────────────────────────
with tab1:
    st.subheader("Anchorage Queue Length Over Time")

    # Date range filter
    min_d = congestion["hour"].min().date()
    max_d = congestion["hour"].max().date()
    col_a, col_b = st.columns(2)
    start_d = col_a.date_input("From", value=min_d, min_value=min_d, max_value=max_d)
    end_d = col_b.date_input("To", value=max_d, min_value=min_d, max_value=max_d)

    mask = (congestion["hour"].dt.date >= start_d) & (congestion["hour"].dt.date <= end_d)
    filtered = congestion[mask]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered["hour"], y=filtered["queue_length"],
        mode="lines", name="Queue Length",
        line=dict(color="#0066CC", width=2),
        fill="tozeroy", fillcolor="rgba(0,102,204,0.1)",
    ))
    # Mark data gaps
    gaps = filtered[filtered["data_gap"]]
    if not gaps.empty:
        fig.add_trace(go.Scatter(
            x=gaps["hour"], y=gaps["queue_length"],
            mode="markers", name="Data Gap",
            marker=dict(color="red", size=4, symbol="x"),
        ))
    fig.update_layout(
        xaxis_title="Hour",
        yaxis_title="Vessels in Anchorage",
        height=380,
        template="plotly_dark",
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, width="stretch")

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean queue", f"{filtered['queue_length'].mean():.1f} vessels")
    col2.metric("Peak queue", f"{filtered['queue_length'].max():.0f} vessels")
    col3.metric("Mean arrival rate", f"{filtered['arrival_rate'].mean():.1f} vessels/hr")

# ── Tab 2: Backtest ───────────────────────────────────────────────────────────
with tab2:
    st.subheader("Backtest: XGBoost vs M/M/c Baseline vs Actual")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=backtest["hour"], y=backtest["actual"],
        name="Actual", line=dict(color="#FAFAFA", width=2),
    ))
    fig2.add_trace(go.Scatter(
        x=backtest["hour"], y=backtest["xgb_pred"],
        name="XGBoost (6h ahead)", line=dict(color="#00CC66", width=1.5, dash="dash"),
    ))
    fig2.add_trace(go.Scatter(
        x=backtest["hour"], y=backtest["baseline_pred"],
        name="M/M/c Baseline", line=dict(color="#FF4444", width=1.5, dash="dot"),
    ))
    fig2.update_layout(
        xaxis_title="Hour",
        yaxis_title="Vessels in Anchorage",
        height=380,
        template="plotly_dark",
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig2, width="stretch")

    # Error distribution
    st.subheader("Error Distribution")
    backtest["xgb_error"] = backtest["xgb_pred"] - backtest["actual"]
    backtest["baseline_error"] = backtest["baseline_pred"] - backtest["actual"]

    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(
        x=backtest["xgb_error"], name="XGBoost error",
        opacity=0.7, nbinsx=30, marker_color="#00CC66",
    ))
    fig3.add_trace(go.Histogram(
        x=backtest["baseline_error"], name="Baseline error",
        opacity=0.7, nbinsx=30, marker_color="#FF4444",
    ))
    fig3.update_layout(
        barmode="overlay",
        xaxis_title="Prediction Error (vessels)",
        yaxis_title="Count",
        height=300,
        template="plotly_dark",
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig3, width="stretch")

# ── Tab 3: Live forecast widget ───────────────────────────────────────────────
with tab3:
    st.subheader("6-Hour Ahead Forecast")
    st.caption("Input current port conditions to generate a forecast.")

    c1, c2 = st.columns(2)
    with c1:
        current_queue = st.slider(
            "Current anchorage queue (vessels)", 0, 80, 43,
            help="Number of vessels currently waiting in the anchorage zone."
        )
        arrival_rate = st.slider(
            "Arrival rate (vessels/hr)", 0, 50, 2,
            help="Number of new vessels arriving in the anchorage per hour."
        )
    with c2:
        now = datetime.now()
        hourofday = st.slider("Hour of day (0–23)", 0, 23, now.hour)
        dayofweek = st.slider(
            "Day of week (0=Mon … 6=Sun)", 0, 6, now.weekday(),
            format="%d",
        )

    if st.button("🔮 Generate Forecast", type="primary"):
        X_live = build_single_row(
            queue_now=current_queue,
            arrival_rate_now=arrival_rate,
            hourofday=hourofday,
            dayofweek=dayofweek,
        )[feature_cols]

        pred = predict(model, X_live)[0]
        delta = pred - current_queue
        delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
        trend = "📈 Rising" if delta > 1 else ("📉 Falling" if delta < -1 else "➡️ Stable")

        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Current queue", f"{current_queue} vessels")
        r2.metric(
            "Forecast (in 6h)", f"{pred:.0f} vessels",
            delta=delta_str,
            delta_color="inverse",
        )
        r3.metric("Trend", trend)

        # Severity indicator
        if pred >= 48:
            st.error("🚨 **High congestion** — predicted queue exceeds 48 vessels. Significant berthing delays expected.")
        elif pred >= 40:
            st.warning("⚠️ **Moderate congestion** — queue above average. Monitor closely.")
        else:
            st.success("✅ **Normal conditions** — queue within typical range.")

        st.caption("Navigate to **Explainability** in the sidebar to see which factors drove this prediction.")
