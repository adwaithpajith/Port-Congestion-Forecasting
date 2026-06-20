"""
features.py — Feature engineering for the port congestion forecasting model.
"""

import numpy as np
import pandas as pd

FORECAST_HORIZON = 6  # hours ahead


def make_features(df: pd.DataFrame, horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
    """
    Build lag, rolling, and cyclical features from hourly congestion data.
    Target: queue_length N hours in the future.
    """
    d = df.copy().sort_values("hour").reset_index(drop=True)

    # Target
    d["target"] = d["queue_length"].shift(-horizon)

    # Lag features — past queue state
    for lag in [1, 2, 3, 6, 12, 24]:
        d[f"queue_lag_{lag}h"] = d["queue_length"].shift(lag)

    # Rolling statistics on queue
    for window in [3, 6, 12]:
        rolled = d["queue_length"].shift(1).rolling(window)
        d[f"queue_roll_mean_{window}h"] = rolled.mean()
        d[f"queue_roll_std_{window}h"] = rolled.std()

    # Arrival rate lags
    for lag in [1, 3, 6]:
        d[f"arrival_lag_{lag}h"] = d["arrival_rate"].shift(lag)

    # Cyclical time encoding (avoids ordinality issues in tree models)
    d["hour_sin"] = np.sin(2 * np.pi * d["hourofday"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hourofday"] / 24)
    d["dow_sin"] = np.sin(2 * np.pi * d["dayofweek"] / 7)
    d["dow_cos"] = np.cos(2 * np.pi * d["dayofweek"] / 7)

    return d.dropna().reset_index(drop=True)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the list of feature columns (excludes metadata and target)."""
    exclude = {
        "hour", "total_pings", "data_gap", "target",
        "queue_length", "arrival_rate", "dayofweek", "hourofday",
    }
    return [c for c in df.columns if c not in exclude]


def build_single_row(
    queue_now: float,
    arrival_rate_now: float,
    hourofday: int,
    dayofweek: int,
    queue_history: list[float] | None = None,
) -> pd.DataFrame:
    """
    Build a single-row feature dataframe for live inference from the dashboard.
    `queue_history` should be a list of the last 24 hourly queue lengths, most
    recent last. If not provided, all lags are filled with queue_now.
    """
    if queue_history is None or len(queue_history) < 24:
        queue_history = [queue_now] * 24

    h = queue_history  # alias

    row = {
        # Lags
        "queue_lag_1h": h[-1],
        "queue_lag_2h": h[-2],
        "queue_lag_3h": h[-3],
        "queue_lag_6h": h[-6],
        "queue_lag_12h": h[-12],
        "queue_lag_24h": h[-24],
        # Rolling means
        "queue_roll_mean_3h": np.mean(h[-3:]),
        "queue_roll_std_3h": np.std(h[-3:]),
        "queue_roll_mean_6h": np.mean(h[-6:]),
        "queue_roll_std_6h": np.std(h[-6:]),
        "queue_roll_mean_12h": np.mean(h[-12:]),
        "queue_roll_std_12h": np.std(h[-12:]),
        # Arrival rate lags (same value used for all lags in manual mode)
        "arrival_lag_1h": arrival_rate_now,
        "arrival_lag_3h": arrival_rate_now,
        "arrival_lag_6h": arrival_rate_now,
        # Cyclical time
        "hour_sin": np.sin(2 * np.pi * hourofday / 24),
        "hour_cos": np.cos(2 * np.pi * hourofday / 24),
        "dow_sin": np.sin(2 * np.pi * dayofweek / 7),
        "dow_cos": np.cos(2 * np.pi * dayofweek / 7),
    }
    return pd.DataFrame([row])
