# core.py — logic-only helpers (no Streamlit/UI)
# Purpose: Pure functions for features, simple backtests, screener scoring, and a reliability score.
# Notes: Expect daily OHLCV input; Callers handle data fetching/cleaning.

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, brier_score_loss

# ---------- Data/Feature Helpers ----------
def _ensure_flat(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with flat (non-MultiIndex) columns."""
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d

def make_features(df: pd.DataFrame):
    """Add SMA5/10/20 and Return; return (feature_df, X=[Close,SMA5,SMA10,SMA20])."""
    d = _ensure_flat(df)
    d["SMA5"]  = d["Close"].rolling(5).mean()
    d["SMA10"] = d["Close"].rolling(10).mean()
    d["SMA20"] = d["Close"].rolling(20).mean()
    d["Return"] = d["Close"].pct_change()
    d.dropna(inplace=True)
    feature_cols = ["Close", "SMA5", "SMA10", "SMA20"]
    return d, d[feature_cols].astype("float32").values

# ---------- Backtests ----------
def backtest_buy_hold(df: pd.DataFrame, start_idx: int) -> float:
    """Buy & hold total return from start_idx to end; 0.0 if insufficient data."""
    d = _ensure_flat(df)
    d["Return"] = d["Close"].pct_change()
    d.dropna(inplace=True)
    if len(d) <= start_idx:
        return 0.0
    c0 = d["Close"].iloc[start_idx]
    c1 = d["Close"].iloc[-1]
    return float(c1 / c0 - 1.0)

def backtest_sma_only(df: pd.DataFrame, sma_window: int, start_idx: int, cost_bps: int = 0) -> float:
    """Long when Close>SMA(sma_window); apply yesterday’s signal to today’s return; simple cost model."""
    d = _ensure_flat(df)
    d["SMA_GATE"] = d["Close"].rolling(sma_window).mean()
    d["Return"]   = d["Close"].pct_change()
    d.dropna(inplace=True)
    if len(d) <= start_idx:
        return 0.0

    sig = (d["Close"] > d["SMA_GATE"]).astype(int)
    strat = [0.0]
    for t in range(1, len(d)):
        r_t = d["Return"].iloc[t]
        strat.append(sig.iloc[t - 1] * (0 if pd.isna(r_t) else r_t))
    eq = (1.0 + pd.Series(strat, index=d.index)).iloc[start_idx:].cumprod()

    if cost_bps > 0 and len(eq) > 0:
        pos = sig.iloc[start_idx:].to_numpy()
        transitions = np.abs(np.diff(pos)).sum() + (1 if pos[0] == 1 else 0)
        eq *= (1 - cost_bps / 10000.0) ** transitions

    return float(eq.iloc[-1] - 1.0)

# ---------- Screener Weights ----------
def risk_weights(risk: str) -> Tuple[float, float, float, float]:
    """Return weights (ret1m, ret3m, low_vol, sma_bonus) for Low/Medium/High risk."""
    r = (risk or "Medium").lower()
    if r == "low":
        return (0.30, 0.30, 0.35, 0.05)
    if r == "high":
        return (0.50, 0.45, 0.00, 0.05)
    return (0.40, 0.40, 0.15, 0.05)

def score_row(row: pd.Series, ranks: Dict[str, pd.Series], w: Tuple[float, float, float, float]) -> float:
    """Percentile-rank blend (ret1m, ret3m, low-vol) + small SMA-above bonus."""
    r1, r3, rlv, w_sma = ranks["ret1m"][row.name], ranks["ret3m"][row.name], ranks["lowvol"][row.name], w[3]
    base = w[0]*r1 + w[1]*r3 + w[2]*rlv
    bonus = w_sma * (1.0 if row.get("above_sma", False) else 0.0)
    return float(base + bonus)

# ---------- Reliability ----------
def reliability_score(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Combine AUC, balanced accuracy (0.5 threshold), Brier, and class balance into [0,1] score + band.
    Returns: {"score","band","auc","ba","brier","balance"} where band∈{Low,Medium,High}.
    """
    auc = float(roc_auc_score(y_true, y_prob))
    ba  = float(balanced_accuracy_score(y_true, (y_prob >= 0.5).astype("int32")))
    brier = float(brier_score_loss(y_true, y_prob))
    pos_rate = float(np.mean(y_true))
    balance = float(1 - min(abs(pos_rate - 0.5) / 0.5, 1.0))  # 1 balanced → 0 extreme

    auc_n = max(0.0, min((auc - 0.5) / 0.5, 1.0))
    brier_n = 1.0 - max(0.0, min(brier / 0.25, 1.0))

    score = 0.35*auc_n + 0.35*ba + 0.15*brier_n + 0.15*balance
    band = "High" if score >= 0.65 else "Medium" if score >= 0.45 else "Low"
    return {"score": float(score), "band": band, "auc": auc, "ba": ba, "brier": brier, "balance": balance}
