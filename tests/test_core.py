# tests/test_core.py
# Purpose: Unit tests for core.py helper functions (features, backtests, scoring, reliability).

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    # Make project root importable so `from core import ...` succeeds when running pytest from repo root.
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from core import (
    make_features, backtest_buy_hold, backtest_sma_only,
    reliability_score, risk_weights, score_row, _ensure_flat
)

# Helper: synthetic “steady uptrend” price series for predictable expectations in unit tests.
def _uptrend_df(n: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    close = pd.Series(np.linspace(100, 120, n), index=idx)
    return pd.DataFrame({
        "Open": close, "High": close, "Low": close, "Close": close, "Volume": 1_000_000
    })

# Test 1 — Unit test: feature engineering returns expected columns/shape and drops NaNs.
def test_make_features_has_expected_cols():
    df = _uptrend_df(50)
    feat_df, X = make_features(df)
    assert {"SMA5", "SMA10", "SMA20", "Return"} <= set(feat_df.columns)
    assert X.shape[1] == 4
    assert len(feat_df) > 0

# Test 2 — Unit test: buy-and-hold return should be positive on a synthetic uptrend.
def test_buy_hold_positive_on_uptrend():
    df = _uptrend_df(200)
    r = backtest_buy_hold(df, start_idx=20)
    assert r > 0

# Test 3 — Unit test: SMA-only strategy should not be negative on a clean uptrend (no costs).
def test_sma_only_non_negative_on_uptrend():
    df = _uptrend_df(200)
    r = backtest_sma_only(df, sma_window=20, start_idx=20, cost_bps=0)
    assert r >= 0

# Test 4 — Unit test: reliability score stays in [0,1], AUC > 0.5 on a sensible toy example, band is valid.
def test_reliability_score_reasonable():
    y_true = np.array([0,0,1,1,1,0,1,0,1,1])
    y_prob = np.array([0.1,0.2,0.9,0.7,0.8,0.4,0.65,0.3,0.75,0.6])
    stats = reliability_score(y_true, y_prob)
    assert 0.0 <= stats["score"] <= 1.0
    assert stats["auc"] > 0.5
    assert stats["band"] in {"Low", "Medium", "High"}

# Test 5 — Unit test: sector scoring prefers higher momentum and lower volatility, with small SMA bonus.
def test_scoring_prefers_better_momentum_and_lower_vol():
    df = pd.DataFrame([
        dict(ret1m=0.10, ret3m=0.20, vol30=0.20, above_sma=True),   # stronger candidate
        dict(ret1m=-0.05, ret3m=0.02, vol30=0.30, above_sma=False), # weaker candidate
    ])
    df["ret1m_rank"] = df["ret1m"].rank(pct=True)
    df["ret3m_rank"] = df["ret3m"].rank(pct=True)
    df["lowvol_rank"] = (-df["vol30"]).rank(pct=True)
    ranks = {"ret1m": df["ret1m_rank"], "ret3m": df["ret3m_rank"], "lowvol": df["lowvol_rank"]}

    w = risk_weights("Medium")
    s0 = score_row(df.iloc[0], ranks, w)
    s1 = score_row(df.iloc[1], ranks, w)
    assert s0 > s1
