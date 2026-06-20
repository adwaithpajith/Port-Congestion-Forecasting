"""
app.py — Home page for Port Congestion Forecasting System.
"""

import json
import os
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Port Congestion Forecast",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path("models/xgb_port_congestion.json")
METRICS_PATH = Path("models/metrics.json")
DATA_PATH = Path("data/congestion_features.csv")

# ── Header ───────────────────────────────────────────────────────────────────
st.title("⚓ Port Congestion Forecasting System")
st.caption("Port of Los Angeles / Long Beach · AIS-Based Anchorage Queue Prediction")
st.divider()

# ── Status row ───────────────────────────────────────────────────────────────
model_ready = MODEL_PATH.exists()
data_ready = DATA_PATH.exists()

col1, col2, col3, col4 = st.columns(4)
col1.metric("📍 Case Study", "LA / Long Beach", "San Pedro Bay")
col2.metric("🎯 Forecast Horizon", "6 Hours Ahead", "XGBoost + SHAP")
col3.metric("🚢 Vessel Types", "Cargo & Tanker", "AIS Types 70–89")

if model_ready and METRICS_PATH.exists():
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    col4.metric(
        "📉 Model MAE",
        f"{metrics['xgb_mae']} vessels",
        f"{metrics['mae_improvement_pct']}% vs M/M/c baseline",
    )
    if "synthetic" in metrics.get("note", "").lower():
        st.error(
            "🚧 **This model was trained on synthetic data** (`scripts/generate_demo.py`) "
            "— for local UI testing only. Run `python scripts/train.py` to train on real "
            "NOAA AIS data before treating any numbers here as meaningful.",
            icon="🚧",
        )
else:
    col4.metric("⚠️ Model", "Not trained", "Run scripts/train.py")

st.divider()

# ── Setup warning ────────────────────────────────────────────────────────────
if not model_ready:
    st.warning(
        "**Model files not found.** This app needs a trained model before it can run.",
        icon="⚠️",
    )
    st.code(
        "python scripts/train.py",
        language="bash",
    )
    st.caption(
        "Pulls free, real NOAA Marine Cadastre AIS data — no API key required. "
        "Takes ~30–60 min depending on connection speed."
    )
    st.info(
        "After it finishes, commit the generated `models/` and `data/` directories to "
        "GitHub and Streamlit Cloud will pick them up on the next deploy.",
        icon="ℹ️",
    )
    with st.expander("Just testing the UI locally? (not for deployment)"):
        st.caption(
            "`python scripts/generate_demo.py` generates synthetic data and trains a "
            "throwaway model in seconds, purely so you can click through the dashboard "
            "without waiting on a real download. Outputs are tagged `synthetic` in "
            "`models/metrics.json` — don't commit them as your production model."
        )
    st.stop()

# ── Architecture overview ────────────────────────────────────────────────────
st.subheader("System Architecture")

c1, c2, c3, c4, c5 = st.columns(5)
layers = [
    ("📡", "Ingestion", "NOAA AIS\nbulk CSV"),
    ("⚙️", "Processing", "Geofence\nclassification\n+ event detection"),
    ("🤖", "Modeling", "XGBoost\n+ M/M/c\nbaseline"),
    ("🔍", "Explainability", "SHAP\nwaterfall\n+ summary"),
    ("📊", "Dashboard", "Streamlit\n+ Folium\n+ Plotly"),
]
for col, (icon, title, desc) in zip([c1, c2, c3, c4, c5], layers):
    with col:
        st.markdown(
            f"""
            <div style="background:#1A1F2E;border-radius:10px;padding:16px;text-align:center;height:140px;">
                <div style="font-size:28px">{icon}</div>
                <div style="font-weight:bold;margin:6px 0 4px">{title}</div>
                <div style="font-size:12px;color:#aaa;white-space:pre-line">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# ── About section ────────────────────────────────────────────────────────────
with st.expander("📖 About this project", expanded=False):
    st.markdown("""
    **Port Congestion Forecasting** predicts the anchorage queue length at the
    Port of Los Angeles / Long Beach 6 hours ahead using historical AIS
    (Automatic Identification System) vessel data.

    **Why LA/Long Beach?**
    The October–November 2021 congestion crisis created unprecedented queue
    lengths (100+ vessels waiting offshore), with average wait times rising
    from under a day to over two weeks. That dynamic range makes it an ideal
    training signal for a forecasting model.

    **Data source:**
    NOAA Marine Cadastre AIS dataset — free, publicly available daily CSV
    archives covering all US coastal waters.

    **Methodology:**
    1. Geofence the San Pedro Bay anchorage and berth zones
    2. Classify every AIS ping into `anchorage / berth / transit`
    3. Detect vessel arrival → berth events to compute realized wait times
    4. Build hourly congestion features (queue length, arrival rate, lags)
    5. Train XGBoost forecaster and compare against M/M/c queueing baseline
    6. SHAP explainability layer surfaces which features drive each prediction

    **Navigate using the sidebar →**
    """)

st.caption("Built with Streamlit · XGBoost · SHAP · Folium · NOAA AIS Data")
