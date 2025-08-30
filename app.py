# app.py
import streamlit as st
st.set_page_config(page_title="Explainable Financial Advisor Bot", layout="wide")

from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import json, requests, math
from typing import List, Tuple, Dict

from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    roc_auc_score,
    brier_score_loss,
)

from lstm_model import train_or_load_model, SEQ_LEN

from core import (
    _ensure_flat, make_features, backtest_buy_hold, backtest_sma_only,
    reliability_score, risk_weights, score_row
)

# =========================
# Cached data/model helpers
# =========================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_prices(ticker: str, start: str) -> pd.DataFrame:
    """Download prices (cached ~5min)."""
    return yf.download(ticker, start=start, auto_adjust=True, progress=False)

@st.cache_data(ttl=600, show_spinner=False)
def make_features(df: pd.DataFrame):
    """Features for Close + SMAs, returns (feature_df, features_array)."""
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d["SMA5"]  = d["Close"].rolling(5).mean()
    d["SMA10"] = d["Close"].rolling(10).mean()
    d["SMA20"] = d["Close"].rolling(20).mean()
    d["Return"] = d["Close"].pct_change()
    d.dropna(inplace=True)
    feature_cols = ["Close", "SMA5", "SMA10", "SMA20"]
    return d, d[feature_cols].astype("float32").values

@st.cache_resource(show_spinner=False)
def get_lstm_resource(ticker: str, start: str):
    """
    Cache the heavy LSTM work per (ticker, start).
    Returns (model, scaler, X_test, y_test, df_prices).
    """
    df = fetch_prices(ticker, start=start)
    model_path = f"models/{ticker.upper()}_lstm.keras"
    scaler_path = f"models/{ticker.upper()}_scaler.pkl"
    model, scaler, X_test, y_test, _ = train_or_load_model(
        df, model_path=model_path, scaler_path=scaler_path
    )
    return model, scaler, X_test, y_test, df

# =========================
# Screener (S&P500 universe, snapshots, scoring)
# =========================
@st.cache_data(ttl=3600, show_spinner=False)
def get_sp500_universe() -> pd.DataFrame:
    """
    Returns a DataFrame with columns: Symbol, Name, Sector.
    Tries Wikipedia; falls back to a tiny static sample if offline.
    """
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0][["Symbol", "Security", "GICS Sector"]].copy()
        df.columns = ["Symbol", "Name", "Sector"]
        # yfinance uses "-" instead of "." for tickers like BRK.B
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
        return df
    except Exception:
        data = [
            ("AAPL", "Apple", "Information Technology"),
            ("MSFT", "Microsoft", "Information Technology"),
            ("NVDA", "NVIDIA", "Information Technology"),
            ("BAC", "Bank of America", "Financials"),
            ("JPM", "JPMorgan Chase", "Financials"),
            ("BRK-B", "Berkshire Hathaway", "Financials"),
            ("JNJ", "Johnson & Johnson", "Health Care"),
            ("PFE", "Pfizer", "Health Care"),
            ("LLY", "Eli Lilly", "Health Care"),
        ]
        return pd.DataFrame(data, columns=["Symbol", "Name", "Sector"])

@st.cache_data(ttl=900, show_spinner=False)
def download_history(ticker: str, start: str) -> pd.DataFrame:
    """Auto-adjusted daily OHLCV for one ticker."""
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def compute_snapshot(df: pd.DataFrame) -> Dict[str, float]:
    """Compute price/returns/vol/SMA/avgvol snapshot for the latest date."""
    if df.empty or len(df) < 70:  # need ~3m for metrics
        return {}
    close = df["Close"]
    price = float(close.iloc[-1])
    ret1m = float(close.pct_change(21).iloc[-1])    # ~1 month
    ret3m = float(close.pct_change(63).iloc[-1])    # ~3 months
    daily = close.pct_change()
    vol30 = float(daily.rolling(30).std().iloc[-1])
    sma20 = float(close.rolling(20).mean().iloc[-1])
    above_sma = bool(price > sma20)
    avgvol30 = float(df["Volume"].rolling(30).mean().iloc[-1]) if "Volume" in df else math.nan
    return dict(price=price, ret1m=ret1m, ret3m=ret3m, vol30=vol30, above_sma=above_sma, avgvol30=avgvol30)

def risk_weights(risk: str) -> Tuple[float, float, float, float]:
    """
    Return weights for (ret1m, ret3m, low_vol, sma_bonus).
    """
    r = (risk or "Medium").lower()
    if r == "low":
        return (0.30, 0.30, 0.35, 0.05)
    if r == "high":
        return (0.50, 0.45, 0.00, 0.05)
    return (0.40, 0.40, 0.15, 0.05)

def score_row(row: pd.Series, ranks: Dict[str, pd.Series], w: Tuple[float, float, float, float]) -> float:
    r1, r3, rlv, w_sma = ranks["ret1m"][row.name], ranks["ret3m"][row.name], ranks["lowvol"][row.name], w[3]
    base = w[0]*r1 + w[1]*r3 + w[2]*rlv
    bonus = w_sma * (1.0 if row["above_sma"] else 0.0)
    return float(base + bonus)

def select_ranked_candidates(
    universe: pd.DataFrame, sectors: List[str], risk: str, start_iso: str, top_n: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (all_scored_df, top_df). The top_df:
      1) prefers positive return names ((ret1m>0) or (ret3m>0))
      2) if not enough, fills with least-loss (by score then max(ret1m,ret3m))
    """
    if not len(sectors):
        return pd.DataFrame(), pd.DataFrame()

    sectors = list(sectors)[:3]  # safety
    uni = universe[universe["Sector"].isin(sectors)].copy()
    if uni.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    for sym, name, sec in uni[["Symbol", "Name", "Sector"]].itertuples(index=False):
        hist = download_history(sym, start_iso)
        snap = compute_snapshot(hist)
        if not snap:
            continue
        rows.append({"Symbol": sym, "Name": name, "Sector": sec, **snap})
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    w = risk_weights(risk)
    df["ret1m_rank"] = df["ret1m"].rank(pct=True)       # high better
    df["ret3m_rank"] = df["ret3m"].rank(pct=True)       # high better
    df["lowvol_rank"] = (-df["vol30"]).rank(pct=True)   # low vol ⇒ higher rank
    ranks = {"ret1m": df["ret1m_rank"], "ret3m": df["ret3m_rank"], "lowvol": df["lowvol_rank"]}

    df["score"] = df.apply(lambda r: score_row(r, ranks, w), axis=1)
    df["best_ret"] = df[["ret1m", "ret3m"]].max(axis=1)

    pos = df[(df["ret1m"] > 0) | (df["ret3m"] > 0)].sort_values(["score", "best_ret"], ascending=[False, False])
    neg = df[(df["ret1m"] <= 0) & (df["ret3m"] <= 0)].sort_values(["score", "best_ret"], ascending=[False, False])

    top = pd.concat([pos.head(top_n), neg.head(max(0, top_n - len(pos)))], axis=0)
    display = top[["Symbol", "Name", "Sector", "price", "ret1m", "ret3m", "vol30", "above_sma", "avgvol30", "score"]].copy()
    return df, display


# =========================
# Backtests
# =========================
def _ensure_flat(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d

def backtest_buy_hold(df: pd.DataFrame, start_idx: int) -> float:
    d = _ensure_flat(df)
    d["Return"] = d["Close"].pct_change()
    d.dropna(inplace=True)
    if len(d) <= start_idx:
        return 0.0
    c0 = d["Close"].iloc[start_idx]
    c1 = d["Close"].iloc[-1]
    return float(c1 / c0 - 1.0)

def backtest_sma_only(df: pd.DataFrame, sma_window: int, start_idx: int, cost_bps: int = 0) -> float:
    d = _ensure_flat(df)
    d["SMA_GATE"] = d["Close"].rolling(sma_window).mean()
    d["Return"]   = d["Close"].pct_change()
    d.dropna(inplace=True)
    if len(d) <= start_idx:
        return 0.0

    sig = (d["Close"] > d["SMA_GATE"]).astype(int)   # long when price > SMA(window)
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

def backtest_lstm_sma_fast(
    df: pd.DataFrame,
    lstm_model,
    scaler,
    prob_thr: float = 0.60,
    sma_window: int = 20,
    cost_bps: int = 5,
    batch_size: int = 512,
):
    """
    Long when: (prob_up >= prob_thr) AND (Close > SMA(sma_window)).
    Walk-forward; no look-ahead. Returns total return and # signal days.
    """
    d, feats = make_features(df)        # has SMA5/10/20 + Return
    d["SMA_GATE"] = d["Close"].rolling(sma_window).mean()
    d.dropna(inplace=True)
    if len(d) < max(SEQ_LEN, sma_window) + 2:
        return {"total_return": 0.0, "signal_days": 0}

    feature_cols = ["Close", "SMA5", "SMA10", "SMA20"]
    feats = d[feature_cols].astype("float32").values
    scaled = scaler.transform(feats)

    start = max(SEQ_LEN, sma_window)
    n = len(d)
    idx = np.arange(start, n)

    X_all = np.stack([scaled[i-SEQ_LEN:i, :] for i in idx], axis=0)
    p_all = lstm_model.predict(X_all, verbose=0, batch_size=batch_size).ravel()

    price_above = (d["Close"].values[idx-1] > d["SMA_GATE"].values[idx-1])
    long_ok = (p_all >= prob_thr) & price_above

    r = np.nan_to_num(d["Return"].values[idx])
    strat = long_ok * r
    eq = np.cumprod(1.0 + strat)

    if cost_bps > 0 and eq.size > 0:
        pos = long_ok.astype(int)
        transitions = np.abs(np.diff(pos)).sum() + (1 if pos[0] == 1 else 0)
        eq *= (1 - cost_bps / 10000.0) ** transitions

    total_return = float(eq[-1] - 1.0) if eq.size else 0.0
    return {"total_return": total_return, "signal_days": int(long_ok.sum())}


# =========================
# Reliability weighting
# =========================
def reliability_score(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Combine AUC, balanced acc, Brier, class balance → [0,1] score."""
    auc = float(roc_auc_score(y_true, y_prob))
    ba  = float(balanced_accuracy_score(y_true, (y_prob >= 0.5).astype("int32")))
    brier = float(brier_score_loss(y_true, y_prob))
    pos_rate = float(np.mean(y_true))
    balance = float(1 - min(abs(pos_rate - 0.5) / 0.5, 1.0))  # 1=balanced

    auc_n = max(0.0, min((auc - 0.5) / 0.5, 1.0))  # 0 @0.5 AUC → 1 @1.0 AUC
    brier_n = 1.0 - max(0.0, min(brier / 0.25, 1.0))  # lower brier → higher score

    score = 0.35 * auc_n + 0.35 * ba + 0.15 * brier_n + 0.15 * balance
    if score >= 0.65:
        band = "High"
    elif score >= 0.45:
        band = "Medium"
    else:
        band = "Low"
    return {"score": float(score), "band": band, "auc": auc, "ba": ba, "brier": brier, "balance": balance}


# =========================
# Preferences UI helpers (limit to 3 sectors, per-sector tables)
# =========================
SECTOR_OPTIONS = [
    "Information Technology", "Financials", "Health Care", "Energy",
    "Consumer Staples", "Consumer Discretionary", "Industrials",
    "Utilities", "Materials", "Real Estate", "Communication Services"
]
SECTOR_PICKER_KEY = "sectors_multiselect"

def _enforce_max_sectors():
    sel = st.session_state.get(SECTOR_PICKER_KEY, [])
    if len(sel) > 3:
        st.session_state[SECTOR_PICKER_KEY] = sel[:3]
        st.warning("You can pick at most **3** sectors. I kept the first three.", icon="⚠️")

def _nice_table(df: pd.DataFrame) -> pd.DataFrame:
    show = df.copy()
    show.rename(columns={
        "price": "Price",
        "ret1m": "Ret 1m",
        "ret3m": "Ret 3m",
        "vol30": "Volatility (30d)",
        "above_sma": "Above 20-day SMA",
        "avgvol30": "Avg Vol (30d)",
        "score": "Score",
    }, inplace=True)
    show["Price"] = show["Price"].map(lambda x: f"${x:,.2f}")
    show["Ret 1m"] = show["Ret 1m"].map(lambda x: f"{x:+.2%}")
    show["Ret 3m"] = show["Ret 3m"].map(lambda x: f"{x:+.2%}")
    show["Volatility (30d)"] = show["Volatility (30d)"].map(lambda x: f"{x:.3f}")
    show["Avg Vol (30d)"] = show["Avg Vol (30d)"].map(lambda x: f"{x:,.0f}")
    show["Score"] = show["Score"].map(lambda x: f"{x:.3f}")
    return show.reset_index(drop=True)

def build_explanations(all_df: pd.DataFrame, top_df: pd.DataFrame) -> str:
    """One compact markdown block explaining selection + a tiny per-ticker table."""
    if all_df.empty or top_df.empty:
        return ""
    pos_mask = (all_df["ret1m"] > 0) | (all_df["ret3m"] > 0)
    pct_pos = 100.0 * pos_mask.mean()
    pct_above = 100.0 * all_df["above_sma"].mean()

    # Small markdown table
    lines = [
        "| Ticker | 1m | 3m | σ30d | SMA20 |",
        "|:------|---:|---:|----:|:----:|",
    ]
    for _, r in top_df.iterrows():
        sma = "✅" if r["above_sma"] else "❌"
        lines.append(f"| **{r['Symbol']}** | {r['ret1m']:+.1%} | {r['ret3m']:+.1%} | {r['vol30']:.3f} | {sma} |")
    table_md = "\n".join(lines)

    overview = (
        "Why these? We prioritised names with **positive 1–3m momentum** and price **above the 20-day SMA**. "
        "If none were positive, we chose the **least losses** by score.\n\n"
        f"**Sector breadth:** {pct_pos:.0f}% of screened names are positive on 1–3m; "
        f"{pct_above:.0f}% sit above their 20-day SMA.\n\n"
        + table_md
    )
    return overview

def suggest_across_sectors(universe: pd.DataFrame, sectors: list[str], risk: str, start_iso: str, top_n: int):
    """Run screener per sector so we can render one table per sector."""
    results = {}
    for sec in sectors[:3]:
        all_df, top_df = select_ranked_candidates(
            universe=universe, sectors=[sec], risk=risk, start_iso=start_iso, top_n=top_n
        )
        results[sec] = (all_df, top_df)
    return results


# =========================
# App chrome
# =========================
st.title("📊 Explainable AI Financial Advisor Bot")
st.caption("Educational tool: This bot explains signals to help you learn. It doesn’t know your full situation and isn’t a recommendation to buy or sell. Past performance ≠ future results")

st.sidebar.title("User Menu")
page = st.sidebar.selectbox("Go to", ["Home", "LSTM Forecast", "Preferences", "Contact Support"])


# =========================
# HOME
# =========================
if page == "Home":
    st.header("Welcome")
    st.markdown("""
This bot is built for **novice investors** who want to see the *why* behind signals—not just the signal.

### ℹ️ What should I pick? (short guide)
- **New here?** Keep defaults and use **Balanced** decision style.
- **Fewer whipsaws (choppy in/out)?** Increase **SMA gate** (e.g., 30–50 days).
- **Faster signals?** Use **History 2–3 years** and **SMA 10–20**, but watch **Max Drawdown** in backtests.
- **Trading cost (bps):** If unsure, use **5 bps**. Illiquid names? Try **10 bps** and confirm CAGR stays positive.

### 📚 Mini-glossary
- **Whipsaw:** Frequent flips in/out when a signal hugs a threshold or MA—often causes repeated small losses.
- **Max Drawdown:** Biggest peak-to-trough loss over the period—how painful the worst slump was.
- **CAGR:** Compound Annual Growth Rate—the steady annual return that gets from start value to end value.

Head to **LSTM Forecast** to run the model, and **Preferences** to get sector-based suggestions.
""")


# =========================
# LSTM Forecast
# =========================
elif page == "LSTM Forecast":
    st.header("📈 LSTM Trend Forecast")

    c_top = st.columns(3)
    stock = c_top[0].text_input("Enter Stock Ticker", value="AAPL", key="lstm_input")
    lookback_years = c_top[1].slider(
        "History (years)", 1, 10, 5, 1,
        help="More years = slower but more stable. Fewer = faster, more recent regime."
    )
    sma_gate = c_top[2].slider(
        "SMA gate (days)", 10, 50, 20, 5,
        help="Only buy when price is above this moving average. Higher = fewer whipsaws."
    )

    c_mid = st.columns(2)
    trade_cost_bps = c_mid[0].slider(
        "Trading cost (bps)", 0, 20, 5, 1,
        help="Slippage/fees per entry/exit. 1 bp = 0.01%."
    )
    decision_style = c_mid[1].select_slider(
        "Decision style",
        options=["Conservative", "Balanced", "Aggressive"],
        value="Balanced",
        help="Conservative raises the trading threshold; Aggressive lowers it."
    )

    st.caption("⏱️ Tip: Years ↑ = slower but more stable. SMA gate ↑ = fewer whipsaws. Threshold ↓ (Aggressive) = more trades.")

    # placeholders
    prob_ph = st.empty()
    decision_hdr_ph = st.empty()
    decision_ph = st.empty()
    compare_hdr_ph = st.empty()
    compare_box = st.container()
    reliability_hdr_ph = st.empty()
    reliability_box = st.container()
    transparent_ph = st.empty()

    if st.button("Predict"):
        if not stock:
            st.warning("Please enter a valid stock ticker.")
        else:
            try:
                start_iso = (pd.Timestamp.today() - pd.DateOffset(years=lookback_years)).date().isoformat()
                with st.status("", expanded=False) as status:
                    status.update(label="Step 1/6: Fetching historical prices")
                    lstm_model, scaler, X_test, y_test, df = get_lstm_resource(stock, start_iso)

                    df = _ensure_flat(df)
                    if df.empty or len(df) < SEQ_LEN + 2:
                        status.update(label="Not enough data", state="error")
                        st.warning("Not enough historical data (need at least ~62 trading days).")
                        st.stop()

                    status.update(label="Step 2/6: Preparing features")
                    feat_df, features = make_features(df)
                    scaled = scaler.transform(features)
                    feature_cols = ["Close", "SMA5", "SMA10", "SMA20"]

                    # size diagnostics
                    samples_total = max(0, len(features) - SEQ_LEN - 1)
                    dA, dB, dC = st.columns(3)
                    dA.metric("Days loaded", len(feat_df))
                    dB.metric("Sequences", samples_total)
                    dC.metric("Holdout size (~20%)", int(samples_total * 0.2))

                    status.update(label="Step 3/6: Evaluating model & tuning threshold")
                    y_prob = lstm_model.predict(X_test, verbose=0)
                    y_true = y_test.astype("int32")

                    thr_grid = np.linspace(0.30, 0.70, 41)  # tune quickly
                    ba_scores = [balanced_accuracy_score(y_true, (y_prob >= t).astype("int32")) for t in thr_grid]
                    best_thr = float(thr_grid[int(np.argmax(ba_scores))])
                    y_pred_best = (y_prob >= best_thr).astype("int32")
                    acc = float((y_pred_best == y_true).mean())

                    # reliability & headline probability (reliability-weighted)
                    stats = reliability_score(y_true, y_prob)

                    status.update(label="Step 4/6: Computing live probability")
                    base_prob_up = None
                    if len(scaled) >= SEQ_LEN:
                        last_window = scaled[-SEQ_LEN:, :]
                        X_last = last_window.reshape(1, SEQ_LEN, scaled.shape[1])
                        base_prob_up = float(lstm_model.predict(X_last, verbose=0)[0][0])

                    # shrink probability toward 0.5 by reliability
                    if base_prob_up is None:
                        p_cal = 0.5
                    else:
                        R = stats["score"]  # 0..1
                        p_cal = 0.5 + (base_prob_up - 0.5) * R

                    # decision threshold (style-adjusted)
                    if decision_style == "Conservative":
                        trade_thr = min(0.80, best_thr + 0.07)
                    elif decision_style == "Aggressive":
                        trade_thr = max(0.50, best_thr - 0.05)
                    else:
                        trade_thr = min(0.75, best_thr + 0.05)

                    # headline
                    prob_ph.metric("Probability of Up (next day)", f"{p_cal:.1%}")

                    # Decision & risk filter
                    status.update(label="Step 5/6: Backtesting & preparing outputs")
                    decision_hdr_ph.markdown("### 🧭 Decision with risk filter")

                    gate_series = feat_df["Close"].rolling(sma_gate).mean()
                    price_above_gate = float(feat_df["Close"].iloc[-1] > gate_series.iloc[-1])

                    if p_cal >= trade_thr and price_above_gate:
                        decision_ph.success(
                            f"✅ **Buy signal** (adjusted prob ≥ trading threshold **and** Price > SMA{sma_gate})."
                        )
                    elif p_cal <= (1 - trade_thr):
                        decision_ph.error("⛔ **Avoid buying** (adjusted prob ≤ short-side threshold).")
                    else:
                        decision_ph.warning("⚖️ Borderline / medium confidence → consider waiting or very small size.")

                    st.caption(
                        f"Edge vs trading threshold: **{(p_cal - trade_thr):+.1%}** "
                        f"(thr={trade_thr:.2f}, style={decision_style}, reliability={stats['band']})."
                    )

                    # Baseline comparisons (fast LSTM backtest)
                    compare_hdr_ph.markdown("### 📈 How does this compare?")
                    start_idx = max(SEQ_LEN, sma_gate)
                    bh = backtest_buy_hold(df, start_idx)
                    sma_ret = backtest_sma_only(df, sma_window=sma_gate, start_idx=start_idx, cost_bps=trade_cost_bps)
                    combo = backtest_lstm_sma_fast(
                        df, lstm_model, scaler,
                        prob_thr=trade_thr, sma_window=sma_gate, cost_bps=trade_cost_bps,
                    )

                    with compare_box:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Buy & Hold", f"{bh*100:+.2f}%")
                        c2.metric(f"SMA{sma_gate}-only", f"{sma_ret*100:+.2f}%")
                        c3.metric("LSTM ∩ SMA", f"{combo['total_return']*100:+.2f}%")
                        st.caption(
                            f"Aligned from first valid window (≈{start_idx} days in). "
                            f"Costs {trade_cost_bps} bps/turn. Signal days (LSTM∩SMA): {combo['signal_days']}."
                        )

                    # Reliability panel
                    reliability_hdr_ph.markdown("### 🧪 How reliable is this model?")
                    conf_mx = confusion_matrix(y_true, y_pred_best)
                    conf_df = pd.DataFrame(
                        conf_mx,
                        index=["Actual Down", "Actual Up"],
                        columns=["Predicted Down", "Predicted Up"],
                    )
                    with reliability_box:
                        colA, colB, colC, colD, colE = st.columns(5)
                        colA.metric("Accuracy", f"{acc*100:.2f}%")
                        colB.metric("Balanced Acc.", f"{stats['ba']*100:.2f}%")
                        colC.metric("AUC", f"{stats['auc']:.2f}")
                        colD.metric("Brier score", f"{stats['brier']:.3f}")
                        colE.metric("Reliability score", f"{stats['score']:.2f}")
                        with st.expander("Confusion Matrix"):
                            st.dataframe(conf_df, use_container_width=True)

                    # Transparent mode
                    with transparent_ph.expander("🔎 How this forecast was generated (transparent mode)"):
                        st.markdown(
                            f"""
**Pipeline (this run):**
1. **Data:** Yahoo Finance daily prices (last **{lookback_years}y**).
2. **Features:** `Close`, `SMA5`, `SMA10`, `SMA20`; scaled with `MinMaxScaler`.
3. **Sequences:** rolling windows of **{SEQ_LEN}** days → predict **next-day** up/down.
4. **Split:** chronological **80/20** train/holdout (no shuffle).
5. **Thresholds:** tuned **{best_thr:.2f}** (classification), style-adjusted trading threshold **{trade_thr:.2f}**.
6. **Headline probability:** shrunk toward 50% by reliability (AUC, BA, Brier, class balance).
"""
                        )
                        try:
                            preview = feat_df[["Close"]].copy()
                            preview[f"SMA{sma_gate}"] = preview["Close"].rolling(sma_gate).mean()
                            preview = preview.iloc[-SEQ_LEN:].reset_index()
                            st.line_chart(preview.set_index(preview.columns[0])[["Close", f"SMA{sma_gate}"]])
                        except Exception:
                            st.caption("Window preview unavailable.")

                    status.update(label="Forecast complete", state="complete")

            except Exception as e:
                st.error(f"Error during LSTM prediction: {e}")


# =========================
# PREFERENCES (dynamic screener, max 3 sectors, per-sector tables)
# =========================
elif page == "Preferences":
    st.header("⚙️ Your Investment Preferences")

    st.info(
        "Default risk = **Medium (Balanced)**. Your risk setting nudges which stocks are suggested "
        "(momentum vs. volatility tolerance). After choosing sectors and risk, click **Save Preferences** "
        "before pressing **Suggest 3–5 trending stocks**.",
        icon="ℹ️"
    )

    # Risk
    risk = st.radio("Preferred risk level", ["Low", "Medium", "High"], index=1)

    # Sectors (hard-limit to 3 via on_change)
    sectors_selected = st.multiselect(
        "Preferred sectors (pick up to 3)",
        SECTOR_OPTIONS,
        default=["Information Technology", "Financials"],
        key=SECTOR_PICKER_KEY,
        on_change=_enforce_max_sectors,
        help="We’ll filter the S&P 500 to these sectors. Limited to 3."
    )

    if st.button("💾 Save Preferences"):
        st.session_state["preferences"] = {
            "risk": risk,
            "sectors": sectors_selected[:3],
        }
        st.success("✅ Preferences saved!")

    if "preferences" in st.session_state:
        prefs = st.session_state["preferences"]
    else:
        prefs = {"risk": risk, "sectors": sectors_selected[:3]}

    st.divider()
    st.subheader("🔎 Suggested tickers (based on sectors & risk)")

    c1, c2 = st.columns(2)
    num_to_show = c1.slider("Number to show **per sector**", 3, 5, 4,
                            help="Applies to each selected sector (max 3).")
    years_hist  = c2.slider("History (years)", 1, 5, 3,
                            help="More years = more stable metrics; fewer = more responsive.")

    if st.button("Suggest 3–5 trending stocks"):
        with st.status("Screening…", expanded=True) as status:
            status.write("Loading S&P 500 universe…")
            uni = get_sp500_universe()

            sectors = prefs.get("sectors", [])[:3]
            if not sectors:
                status.update(label="Pick at least one sector", state="error")
                st.warning("Pick at least one sector above.")
                st.stop()

            start_iso = (pd.Timestamp.today() - pd.DateOffset(years=int(years_hist))).date().isoformat()
            status.write(f"Downloading price histories since {start_iso}…")

            results = suggest_across_sectors(
                universe=uni,
                sectors=sectors,
                risk=prefs.get("risk", "Medium"),
                start_iso=start_iso,
                top_n=num_to_show
            )

            any_rows = False
            for sec in sectors:
                all_df, top_df = results.get(sec, (pd.DataFrame(), pd.DataFrame()))
                st.markdown(f"#### {sec}")
                if top_df.empty:
                    st.caption("No eligible names found for this sector in the selected timeframe.")
                    continue

                any_rows = True
                st.dataframe(_nice_table(top_df), use_container_width=True, hide_index=True)

                expl = build_explanations(all_df, top_df)
                st.markdown(expl)

            if any_rows:
                st.caption(
                    "Heuristic: prefers **positive 1–3m returns**, price **above 20-day SMA**, "
                    "and volatility consistent with your **risk**. Universe: S&P 500 only."
                )
                status.update(label="Done", state="complete")
            else:
                status.update(label="No eligible names found", state="error")


# =========================
# CONTACT SUPPORT
# =========================
elif page == "Contact Support":
    st.header("📬 Contact Support")

    with st.form("support_form"):
        kind = st.selectbox(
            "Type of question",
            ["Bug", "Feature request", "Question", "Data issue", "Other"],
            index=2,
            help="This helps us route your message."
        )
        email = st.text_input("Your Email")
        message = st.text_area("Your Message", height=150)

        submitted = st.form_submit_button("📨 Submit")

        if submitted:
            if not email or "@" not in email:
                st.error("⚠️ Please enter a valid email address.")
            elif not message.strip():
                st.error("⚠️ Message cannot be empty.")
            else:
                from datetime import datetime, timezone
                # Create a static, human-readable timestamp (no formulas)
                ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")

                try:
                    # Your NoCodeAPI endpoint + target sheet tab
                    endpoint = "https://v1.nocodeapi.com/xiangen/google_sheets/SUiLiedWCJVRgivB"
                    tab_name = "Sheet1"

                    # Construct URL with tab name
                    url = f"{endpoint}?tabId={tab_name}"  

                    headers = {"Content-Type": "application/json"}
                    # Order will match Google Sheets header row: Timestamp | Type | Email | Message
                    payload = [[ts, kind, email, message]]

                    response = requests.post(url, headers=headers, data=json.dumps(payload))
                    res_json = {}
                    try:
                        res_json = response.json()
                    except Exception:
                        pass

                    if response.status_code == 200:
                        st.success("✅ Your message was sent successfully!")
                        st.markdown("### 📄 Submitted Details")
                        st.markdown(f"- **When**: `{ts}`")
                        st.markdown(f"- **Type**: `{kind}`")
                        st.markdown(f"- **Email**: `{email}`")
                        st.markdown(f"- **Message**: {message}")
                    else:
                        st.error(f"❌ Failed to send message. Reason: {res_json.get('message', 'Unknown error')}")
                except Exception as e:
                    st.error(f"❌ Error sending message: {e}")