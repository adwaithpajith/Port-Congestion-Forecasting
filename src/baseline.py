"""
baseline.py — M/M/c queueing-theory baseline for port congestion.

Models the port as a multi-server queue:
  - Arrivals: Poisson process with rate λ (vessels/hour)
  - Service:  Exponential with rate μ per berth
  - Servers:  c berths
"""

import math
import numpy as np
import pandas as pd

# Port of LA/Long Beach: ~27 active container berths during Oct 2021
C_BERTHS = 27

# Mean service rate: 1 vessel per 48 hours per berth (~2 day turnaround)
MU = 1 / 48


def mmc_lq(lam: float, mu: float = MU, c: int = C_BERTHS) -> float | None:
    """
    Expected queue length (vessels *waiting*, not being served) for M/M/c.
    Returns None if the system is unstable (ρ >= 1).
    """
    if mu <= 0 or c <= 0 or lam <= 0:
        return None

    rho = lam / (c * mu)
    if rho >= 1:
        return None  # unstable — queue grows without bound

    sum_term = sum((c * rho) ** n / math.factorial(n) for n in range(c))
    erlang_term = (c * rho) ** c / (math.factorial(c) * (1 - rho))
    p0 = 1.0 / (sum_term + erlang_term)

    lq = (rho * (c * rho) ** c * p0) / (math.factorial(c) * (1 - rho) ** 2)
    return lq


def predict_baseline(df: pd.DataFrame, fallback_col: str = "queue_lag_1h") -> pd.Series:
    """
    Apply M/M/c baseline to each row of a feature dataframe.
    Falls back to the last known queue length when the system is unstable.
    """
    preds = []
    for _, row in df.iterrows():
        lam = max(row.get("arrival_lag_1h", 0.001), 0.001)
        lq = mmc_lq(lam)
        if lq is None or np.isnan(lq):
            lq = row.get(fallback_col, 43.0)
        preds.append(lq)
    return pd.Series(preds, index=df.index, name="baseline_pred")
