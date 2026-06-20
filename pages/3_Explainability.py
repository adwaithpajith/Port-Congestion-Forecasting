"""
pages/3_Explainability.py — SHAP-based explainability for the congestion forecast.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict import load_artifacts

st.set_page_config(
    page_title="Explainability · Port Congestion", page_icon="🔍", layout="wide"
)

st.title("🔍 Model Explainability")
st.caption("SHAP (SHapley Additive exPlanations) — What drives each forecast?")


# ── Load artifacts + SHAP sample ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def get_artifacts():
    return load_artifacts()


@st.cache_data(show_spinner="Loading SHAP data...")
def load_shap_data():
    shap_vals = pd.read_csv(ROOT / "data/shap_values_sample.csv").values
    shap_feats = pd.read_csv(ROOT / "data/shap_features_sample.csv")
    return shap_vals, shap_feats


try:
    model, explainer, feature_cols = get_artifacts()
    shap_values, shap_features = load_shap_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

n_samples = len(shap_features)
st.caption(f"Showing explanations for {n_samples} test-set predictions.")

st.divider()
tab1, tab2, tab3 = st.tabs(["🌐 Global Importance", "🔎 Single Prediction", "📖 Feature Guide"])

# ── Tab 1: Global feature importance ─────────────────────────────────────────
with tab1:
    st.subheader("Global Feature Importance (SHAP Summary)")
    st.caption(
        "Each point is one prediction. Position on X = SHAP value (impact on forecast). "
        "Color = feature value (red = high, blue = low)."
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")
    shap.summary_plot(
        shap_values, shap_features,
        feature_names=feature_cols,
        show=False,
        plot_size=None,
        color_bar=True,
    )
    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close()

    st.divider()
    st.subheader("Mean |SHAP| — Average Impact Per Feature")

    mean_shap = np.abs(shap_values).mean(axis=0)
    importance_df = (
        pd.DataFrame({"feature": feature_cols, "mean_shap": mean_shap})
        .sort_values("mean_shap", ascending=True)
        .tail(15)
    )

    fig2 = go.Figure(go.Bar(
        x=importance_df["mean_shap"],
        y=importance_df["feature"],
        orientation="h",
        marker_color="#0066CC",
    ))
    fig2.update_layout(
        xaxis_title="Mean |SHAP value| (vessels)",
        height=420,
        template="plotly_dark",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig2, width="stretch")

# ── Tab 2: Single prediction waterfall ───────────────────────────────────────
with tab2:
    st.subheader("Waterfall Plot — Single Prediction Breakdown")
    st.caption(
        "Shows exactly which features pushed this prediction above or below the baseline."
    )

    idx = st.slider("Select test sample index", 0, n_samples - 1, 0)

    queue_val = shap_features.iloc[idx].get("queue_lag_1h", "—")
    st.caption(f"Sample {idx} — current queue (lag 1h): {queue_val:.0f} vessels")

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    fig3.patch.set_facecolor("#0E1117")
    ax3.set_facecolor("#0E1117")
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[idx],
            base_values=explainer.expected_value,
            data=shap_features.iloc[idx].values,
            feature_names=feature_cols,
        ),
        show=False,
        max_display=12,
    )
    plt.tight_layout()
    st.pyplot(fig3, width="stretch")
    plt.close()

    # Feature values table for this sample
    with st.expander("Feature values for this prediction"):
        sample_df = pd.DataFrame({
            "Feature": feature_cols,
            "Value": shap_features.iloc[idx].values.round(3),
            "SHAP impact": shap_values[idx].round(3),
        }).sort_values("SHAP impact", key=abs, ascending=False)
        st.dataframe(sample_df, width="stretch", hide_index=True)

# ── Tab 3: Feature guide ──────────────────────────────────────────────────────
with tab3:
    st.subheader("Feature Dictionary")
    features_guide = {
        "queue_lag_Nh": "Anchorage queue length N hours ago. "
                         "Most informative feature — queue is strongly autocorrelated.",
        "queue_roll_mean_Nh": "Rolling mean of queue length over last N hours. "
                               "Captures trend without noise.",
        "queue_roll_std_Nh": "Rolling std of queue length. High std = volatile conditions.",
        "arrival_lag_Nh": "Number of new vessels entering the anchorage N hours ago. "
                           "Leading indicator of future queue build-up.",
        "hour_sin / hour_cos": "Cyclically encoded hour-of-day (preserves 23→0 adjacency). "
                                "Captures daily berthing patterns.",
        "dow_sin / dow_cos": "Cyclically encoded day-of-week. "
                              "Weekends see slightly lower activity.",
    }
    for feat, desc in features_guide.items():
        st.markdown(f"**`{feat}`** — {desc}")

    st.divider()
    st.markdown("""
    **How to read SHAP values:**
    - A SHAP value of **+3.5** means that feature pushed the prediction **3.5 vessels higher**
      than the model's average prediction.
    - A SHAP value of **-2.0** means it pulled the prediction **2 vessels lower**.
    - Values sum to: `prediction = base_value + Σ SHAP values`
    """)
