import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

# =========================================================
# Alex AI Infrastructure Institutional Engine
# Final Mobile Dashboard Version
# No news / no analyst API
# Dynamic Theme Score + Institutional Ranking
# =========================================================

st.set_page_config(
    page_title="Alex AI Infrastructure Engine",
    layout="wide"
)

# =========================================================
# Universe
# =========================================================

UNIVERSE = [
    "NVDA", "AVGO", "MU", "TSM", "ASML",
    "VRT", "ANET", "ARM", "AMD", "MRVL",
    "AMAT", "LRCX", "KLAC", "COHR", "LITE",
    "PLTR", "CIEN", "SNDK", "DELL", "MSFT",
    "ORCL", "SMCI", "AAOI", "GLW", "INTC"
]

BENCHMARKS = ["QQQ", "SOXX", "SPY"]

NYSE_TICKERS = {
    "TSM", "VRT", "DELL", "ORCL", "CIEN", "GLW"
}

# =========================================================
# AI Value Chain Classification
# Manual classification, dynamic scoring later
# =========================================================

AI_VALUE_CHAIN = {
    "NVDA": {
        "category": "AI Compute",
        "ai_relevance": 100,
        "capex_sensitivity": 100,
        "strategic_importance": 100,
        "cyclicality_risk": 30,
    },
    "AVGO": {
        "category": "ASIC / Networking",
        "ai_relevance": 96,
        "capex_sensitivity": 95,
        "strategic_importance": 98,
        "cyclicality_risk": 35,
    },
    "MU": {
        "category": "HBM / Memory",
        "ai_relevance": 94,
        "capex_sensitivity": 92,
        "strategic_importance": 92,
        "cyclicality_risk": 65,
    },
    "TSM": {
        "category": "Foundry",
        "ai_relevance": 92,
        "capex_sensitivity": 88,
        "strategic_importance": 96,
        "cyclicality_risk": 40,
    },
    "ASML": {
        "category": "Equipment",
        "ai_relevance": 88,
        "capex_sensitivity": 82,
        "strategic_importance": 96,
        "cyclicality_risk": 45,
    },
    "VRT": {
        "category": "Power / Cooling",
        "ai_relevance": 90,
        "capex_sensitivity": 92,
        "strategic_importance": 88,
        "cyclicality_risk": 45,
    },
    "ANET": {
        "category": "Networking",
        "ai_relevance": 88,
        "capex_sensitivity": 90,
        "strategic_importance": 86,
        "cyclicality_risk": 42,
    },
    "ARM": {
        "category": "AI Compute / IP",
        "ai_relevance": 86,
        "capex_sensitivity": 78,
        "strategic_importance": 88,
        "cyclicality_risk": 45,
    },
    "AMD": {
        "category": "AI Compute",
        "ai_relevance": 86,
        "capex_sensitivity": 86,
        "strategic_importance": 82,
        "cyclicality_risk": 55,
    },
    "MRVL": {
        "category": "Optical / Custom Silicon",
        "ai_relevance": 86,
        "capex_sensitivity": 88,
        "strategic_importance": 84,
        "cyclicality_risk": 55,
    },
    "AMAT": {
        "category": "Equipment",
        "ai_relevance": 84,
        "capex_sensitivity": 82,
        "strategic_importance": 84,
        "cyclicality_risk": 55,
    },
    "LRCX": {
        "category": "Equipment",
        "ai_relevance": 84,
        "capex_sensitivity": 82,
        "strategic_importance": 84,
        "cyclicality_risk": 55,
    },
    "KLAC": {
        "category": "Equipment / Inspection",
        "ai_relevance": 82,
        "capex_sensitivity": 78,
        "strategic_importance": 86,
        "cyclicality_risk": 50,
    },
    "COHR": {
        "category": "Optical",
        "ai_relevance": 84,
        "capex_sensitivity": 88,
        "strategic_importance": 82,
        "cyclicality_risk": 60,
    },
    "LITE": {
        "category": "Optical",
        "ai_relevance": 82,
        "capex_sensitivity": 86,
        "strategic_importance": 80,
        "cyclicality_risk": 62,
    },
    "PLTR": {
        "category": "AI Software",
        "ai_relevance": 82,
        "capex_sensitivity": 65,
        "strategic_importance": 78,
        "cyclicality_risk": 45,
    },
    "CIEN": {
        "category": "Optical Networking",
        "ai_relevance": 80,
        "capex_sensitivity": 82,
        "strategic_importance": 78,
        "cyclicality_risk": 55,
    },
    "SNDK": {
        "category": "Storage",
        "ai_relevance": 80,
        "capex_sensitivity": 78,
        "strategic_importance": 76,
        "cyclicality_risk": 70,
    },
    "DELL": {
        "category": "AI Servers",
        "ai_relevance": 80,
        "capex_sensitivity": 85,
        "strategic_importance": 76,
        "cyclicality_risk": 55,
    },
    "MSFT": {
        "category": "Cloud / AI Platform",
        "ai_relevance": 78,
        "capex_sensitivity": 68,
        "strategic_importance": 84,
        "cyclicality_risk": 25,
    },
    "ORCL": {
        "category": "Cloud / AI Infrastructure",
        "ai_relevance": 78,
        "capex_sensitivity": 72,
        "strategic_importance": 78,
        "cyclicality_risk": 35,
    },
    "SMCI": {
        "category": "AI Servers",
        "ai_relevance": 78,
        "capex_sensitivity": 88,
        "strategic_importance": 72,
        "cyclicality_risk": 75,
    },
    "AAOI": {
        "category": "Optical High Beta",
        "ai_relevance": 76,
        "capex_sensitivity": 85,
        "strategic_importance": 68,
        "cyclicality_risk": 85,
    },
    "GLW": {
        "category": "Fiber / Materials",
        "ai_relevance": 72,
        "capex_sensitivity": 70,
        "strategic_importance": 72,
        "cyclicality_risk": 45,
    },
    "INTC": {
        "category": "Turnaround Semiconductor",
        "ai_relevance": 68,
        "capex_sensitivity": 75,
        "strategic_importance": 78,
        "cyclicality_risk": 80,
    },
}

# =========================================================
# Core Utility Functions
# =========================================================

def clamp(x, lo=0, hi=100):
    if x is None or pd.isna(x):
        return 50.0
    return max(lo, min(hi, float(x)))


def pct_return(series, days):
    if len(series) <= days:
        return np.nan
    base = series.iloc[-days]
    if pd.isna(base) or base == 0:
        return np.nan
    return series.iloc[-1] / base - 1


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def tradingview_url(ticker):
    exchange = "NYSE" if ticker in NYSE_TICKERS else "NASDAQ"
    return f"https://www.tradingview.com/chart/?symbol={exchange}:{ticker}"


def signal_color(signal):
    if signal == "Strong Long":
        return "🟢"
    if signal == "Buy":
        return "🟩"
    if signal == "Watch":
        return "🟨"
    if signal == "Extended":
        return "🟧"
    if signal == "Risk Off":
        return "🔴"
    return "⚪"


# =========================================================
# Data Download
# 5-minute cache for mobile-friendly refresh
# =========================================================

@st.cache_data(ttl=300, show_spinner=False)
def download_market_data():
    tickers = UNIVERSE + BENCHMARKS
    data = yf.download(
        tickers,
        period="18mo",
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    return data


def get_df(data, ticker):
    if isinstance(data.columns, pd.MultiIndex):
        df = data[ticker].copy()
    else:
        df = data.copy()

    df.columns = [str(c).lower() for c in df.columns]
    return df.dropna()


# =========================================================
# Dynamic Theme Score
# Key upgrade:
# Theme is no longer purely static.
# It adjusts based on price leadership, relative strength,
# AI value-chain importance, and cyclicality penalty.
# =========================================================

def calculate_dynamic_theme_score(
    ticker,
    ret63,
    ret126,
    rel63,
    high_prox,
    market_regime
):
    profile = AI_VALUE_CHAIN[ticker]

    ai_relevance = profile["ai_relevance"]
    capex_sensitivity = profile["capex_sensitivity"]
    strategic_importance = profile["strategic_importance"]
    cyclicality_risk = profile["cyclicality_risk"]

    structural_score = (
        ai_relevance * 0.40
        + capex_sensitivity * 0.25
        + strategic_importance * 0.25
        + (100 - cyclicality_risk) * 0.10
    )

    leadership_adjustment = 0

    if ret63 > 0:
        leadership_adjustment += 4
    if ret126 > 0:
        leadership_adjustment += 4
    if rel63 > 0:
        leadership_adjustment += 6
    if high_prox > 0.92:
        leadership_adjustment += 4
    if high_prox > 0.97:
        leadership_adjustment += 4

    if market_regime >= 80:
        leadership_adjustment += 3
    elif market_regime < 50:
        leadership_adjustment -= 5

    cyclical_penalty = 0
    if cyclicality_risk >= 75 and ret63 < 0:
        cyclical_penalty -= 8
    elif cyclicality_risk >= 65 and ret63 < 0:
        cyclical_penalty -= 5

    dynamic_theme = structural_score + leadership_adjustment + cyclical_penalty

    return clamp(dynamic_theme)
# =========================================================
# Ticker-Level Institutional Scoring Engine
# =========================================================

def score_ticker(ticker, data, bench20, bench63, market_regime):
    df = get_df(data, ticker)

    if len(df) < 260:
        return {
            "ticker": ticker,
            "category": AI_VALUE_CHAIN[ticker]["category"],
            "signal": "Data Error",
            "price": np.nan,
            "institutional_score": 0,
            "base_score": 0,
            "short": 0,
            "swing": 0,
            "position": 0,
            "theme": 0,
            "quality": 0,
            "momentum": 0,
            "technical": 0,
            "flow": 0,
            "leadership": 0,
            "risk": 0,
            "ret20_%": np.nan,
            "ret63_%": np.nan,
            "rel20_%": np.nan,
            "rel63_%": np.nan,
            "rel_vol20": np.nan,
            "rsi": np.nan,
            "high_prox_%": np.nan,
            "chart": tradingview_url(ticker),
        }

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    price = close.iloc[-1]

    sma10 = close.rolling(10).mean().iloc[-1]
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]

    prev_sma20 = close.rolling(20).mean().iloc[-2]
    prev_sma50 = close.rolling(50).mean().iloc[-2]

    vol20 = volume.rolling(20).mean().iloc[-1]
    vol50 = volume.rolling(50).mean().iloc[-1]

    ret1 = pct_return(close, 2)
    ret5 = pct_return(close, 6)
    ret10 = pct_return(close, 11)
    ret20 = pct_return(close, 21)
    ret63 = pct_return(close, 64)
    ret126 = pct_return(close, 127)

    rel20 = ret20 - bench20
    rel63 = ret63 - bench63

    rel_vol20 = volume.iloc[-1] / vol20 if vol20 and not pd.isna(vol20) else 1.0
    rel_vol50 = volume.iloc[-1] / vol50 if vol50 and not pd.isna(vol50) else 1.0

    high252 = high.rolling(252).max().iloc[-1]
    high_prox = price / high252 if high252 and not pd.isna(high252) else np.nan

    atr14 = (high - low).rolling(14).mean().iloc[-1]
    atr_pct = atr14 / price if price and not pd.isna(price) else np.nan

    rsi14 = rsi(close).iloc[-1]

    # -----------------------------------------------------
    # Dynamic Theme Score
    # -----------------------------------------------------
    theme_score = calculate_dynamic_theme_score(
        ticker=ticker,
        ret63=ret63,
        ret126=ret126,
        rel63=rel63,
        high_prox=high_prox,
        market_regime=market_regime,
    )

    # -----------------------------------------------------
    # Quality / Growth Proxy
    # This is not accounting-grade fundamentals.
    # It is a market-implied quality/growth proxy:
    # long-term trend, 50/200 structure, and medium-term price strength.
    # -----------------------------------------------------
    quality_score = clamp(
        35
        + (20 if ret126 > 0.25 else 10 if ret126 > 0.05 else 0)
        + (15 if price > sma200 else 0)
        + (15 if sma50 > sma200 else 0)
        + (15 if price > sma50 else 0)
    )

    # -----------------------------------------------------
    # Momentum Score
    # Measures whether money is currently flowing into the name.
    # Includes absolute momentum and benchmark-relative momentum.
    # -----------------------------------------------------
    momentum_score = clamp(
        (14 if ret20 > 0 else 0)
        + (14 if ret63 > 0 else 0)
        + (8 if ret126 > 0 else 0)
        + (20 if rel20 > 0 else 0)
        + (20 if rel63 > 0 else 0)
        + (10 if price > sma20 else 0)
        + (10 if price > sma50 else 0)
        + (
            14
            if high_prox > 0.97
            else 9
            if high_prox > 0.92
            else 4
            if high_prox > 0.87
            else 0
        )
    )

    # -----------------------------------------------------
    # Technical Setup Score
    # Captures actionable chart setups:
    # breakout, healthy pullback, trend stack, moving-average reclaim.
    # -----------------------------------------------------
    breakout = price > high.iloc[-2] and rel_vol20 > 1.2
    pullback = price > sma50 and price < sma20 and 42 <= rsi14 <= 58
    trend_stack = price > sma20 > sma50 > sma200
    reclaim20 = price > sma20 and close.iloc[-2] < prev_sma20
    reclaim50 = price > sma50 and close.iloc[-2] < prev_sma50

    technical_score = clamp(
        (28 if breakout else 0)
        + (22 if pullback else 0)
        + (25 if trend_stack else 0)
        + (10 if reclaim20 else 0)
        + (10 if reclaim50 else 0)
        + (5 if rsi14 >= 50 else 0)
    )

    # -----------------------------------------------------
    # Flow Score
    # Relative volume + positive price action.
    # Proxy for institutional accumulation.
    # -----------------------------------------------------
    flow_score = clamp(
        (
            35
            if rel_vol20 > 2.0 and ret1 > 0
            else 25
            if rel_vol20 > 1.5 and ret1 > 0
            else 15
            if rel_vol20 > 1.2 and ret1 > 0
            else 5
        )
        + (
            25
            if rel_vol50 > 1.2 and ret5 > 0
            else 15
            if rel_vol50 > 1.0 and ret5 > 0
            else 5
        )
        + (20 if ret5 > 0 else 0)
        + (20 if price > sma10 else 0)
    )

    # -----------------------------------------------------
    # Earnings Reaction Proxy
    # Without using a paid earnings API, this captures the market reaction
    # often seen after earnings or major fundamental catalysts.
    # -----------------------------------------------------
    earnings_reaction_score = (
        100
        if ret5 > 0.08 and rel_vol20 > 1.3
        else 85
        if ret5 > 0.04 and rel_vol20 > 1.1
        else 65
        if ret5 > 0
        else 45
        if ret5 > -0.04
        else 25
    )

    # -----------------------------------------------------
    # Risk Score
    # Higher = safer setup.
    # Penalizes high volatility, extended RSI, and trend breakdowns.
    # -----------------------------------------------------
    risk_score = (
        85
        if atr_pct < 0.03
        else 75
        if atr_pct < 0.045
        else 60
        if atr_pct < 0.06
        else 45
        if atr_pct < 0.08
        else 30
    )

    if rsi14 > 80:
        risk_score -= 20
    if price < sma50:
        risk_score -= 25
    if price < sma200:
        risk_score -= 20

    risk_score = clamp(risk_score)

    # -----------------------------------------------------
    # Time-Horizon Scores
    # -----------------------------------------------------

    short_score = clamp(
        momentum_score * 0.28
        + technical_score * 0.28
        + flow_score * 0.24
        + market_regime * 0.10
        + risk_score * 0.10
    )

    swing_score = clamp(
        momentum_score * 0.30
        + technical_score * 0.22
        + flow_score * 0.13
        + theme_score * 0.12
        + quality_score * 0.10
        + market_regime * 0.07
        + risk_score * 0.06
    )

    position_score = clamp(
        theme_score * 0.30
        + quality_score * 0.24
        + momentum_score * 0.20
        + market_regime * 0.10
        + risk_score * 0.10
        + technical_score * 0.06
    )

    # -----------------------------------------------------
    # Base Score
    # News/analyst removed.
    # Leadership will be added later cross-sectionally in Part 3.
    # -----------------------------------------------------
    base_score = clamp(
        theme_score * 0.10
        + quality_score * 0.15
        + momentum_score * 0.20
        + technical_score * 0.15
        + flow_score * 0.15
        + earnings_reaction_score * 0.10
        + market_regime * 0.05
        + risk_score * 0.10
    )

    # -----------------------------------------------------
    # Signal Logic
    # -----------------------------------------------------
    if price < sma50 or risk_score < 40:
        signal = "Risk Off"
    elif base_score >= 85 and rsi14 < 74:
        signal = "Strong Long"
    elif base_score >= 75 and rsi14 < 74:
        signal = "Buy"
    elif base_score >= 75 and rsi14 >= 74:
        signal = "Extended"
    elif base_score >= 65:
        signal = "Watch"
    else:
        signal = "Avoid"

    return {
        "ticker": ticker,
        "category": AI_VALUE_CHAIN[ticker]["category"],
        "signal": signal,
        "price": round(float(price), 2),
        "base_score": round(base_score, 1),
        "short": round(short_score, 1),
        "swing": round(swing_score, 1),
        "position": round(position_score, 1),
        "theme": round(theme_score, 1),
        "quality": round(quality_score, 1),
        "momentum": round(momentum_score, 1),
        "technical": round(technical_score, 1),
        "flow": round(flow_score, 1),
        "earnings_reaction": round(earnings_reaction_score, 1),
        "leadership": 0,
        "risk": round(risk_score, 1),
        "ret20_%": round(ret20 * 100, 2),
        "ret63_%": round(ret63 * 100, 2),
        "rel20_%": round(rel20 * 100, 2),
        "rel63_%": round(rel63 * 100, 2),
        "rel_vol20": round(float(rel_vol20), 2),
        "rsi": round(float(rsi14), 1),
        "high_prox_%": round(high_prox * 100, 1),
        "chart": tradingview_url(ticker),
    }
# =========================================================
# Ranking Engine + Dashboard
# =========================================================

@st.cache_data(ttl=300, show_spinner=False)
def build_rankings():
    data = download_market_data()

    qqq = get_df(data, "QQQ")["close"]
    soxx = get_df(data, "SOXX")["close"]

    qqq20 = pct_return(qqq, 21)
    soxx20 = pct_return(soxx, 21)
    qqq63 = pct_return(qqq, 64)
    soxx63 = pct_return(soxx, 64)

    bench20 = np.nanmean([qqq20, soxx20])
    bench63 = np.nanmean([qqq63, soxx63])

    market_regime = (
        100 if qqq20 > 0 and soxx20 > 0 and soxx20 > qqq20 else
        82 if qqq20 > 0 and soxx20 > 0 else
        62 if qqq20 > 0 or soxx20 > 0 else
        35
    )

    rows = []
    for ticker in UNIVERSE:
        try:
            row = score_ticker(ticker, data, bench20, bench63, market_regime)
            if row:
                rows.append(row)
            time.sleep(0.01)
        except Exception as exc:
            rows.append({
                "ticker": ticker,
                "category": AI_VALUE_CHAIN[ticker]["category"],
                "signal": "Data Error",
                "price": np.nan,
                "base_score": 0,
                "short": 0,
                "swing": 0,
                "position": 0,
                "theme": 0,
                "quality": 0,
                "momentum": 0,
                "technical": 0,
                "flow": 0,
                "earnings_reaction": 0,
                "leadership": 0,
                "risk": 0,
                "ret20_%": np.nan,
                "ret63_%": np.nan,
                "rel20_%": np.nan,
                "rel63_%": np.nan,
                "rel_vol20": np.nan,
                "rsi": np.nan,
                "high_prox_%": np.nan,
                "chart": tradingview_url(ticker),
                "error": str(exc),
            })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["rs_rank_20"] = df["rel20_%"].rank(pct=True) * 100
    df["rs_rank_63"] = df["rel63_%"].rank(pct=True) * 100
    df["leadership"] = (
        df["rs_rank_20"] * 0.40
        + df["rs_rank_63"] * 0.45
        + df["flow"] * 0.15
    ).round(1).fillna(0)

    category_leadership = []
    for _, row in df.iterrows():
        category = row["category"]
        same_group = df[df["category"] == category]
        if len(same_group) <= 1:
            category_leadership.append(70)
        else:
            rank_pct = same_group["base_score"].rank(pct=True).loc[row.name] * 100
            category_leadership.append(rank_pct)

    df["category_leadership"] = pd.Series(category_leadership).round(1).fillna(50)

    df["institutional_score"] = (
        df["base_score"] * 0.62
        + df["leadership"] * 0.25
        + df["category_leadership"] * 0.08
        + df["flow"] * 0.05
    ).round(1)

    df = df.sort_values("institutional_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    return df


# =========================================================
# Streamlit UI
# =========================================================

st.title("Alex AI Infrastructure Institutional Engine")
st.caption(
    "25-stock AI infrastructure dashboard · Dynamic Theme Score · "
    "Leadership Rank · Flow · Short / Swing / Position · 5-min refresh"
)

with st.spinner("Building institutional rankings..."):
    rankings = build_rankings()

if rankings.empty:
    st.error("No data available. Please refresh later.")
    st.stop()

top = rankings.iloc[0]
best_short = rankings.sort_values("short", ascending=False).iloc[0]
best_swing = rankings.sort_values("swing", ascending=False).iloc[0]
best_position = rankings.sort_values("position", ascending=False).iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Best Overall", top["ticker"], top["signal"])
c2.metric("Institutional Score", top["institutional_score"])
c3.metric("Best Short", best_short["ticker"], best_short["short"])
c4.metric("Best Swing", best_swing["ticker"], best_swing["swing"])

c5, c6, c7, c8 = st.columns(4)
c5.metric("Best Position", best_position["ticker"], best_position["position"])
c6.metric("Market Leader", rankings.sort_values("leadership", ascending=False).iloc[0]["ticker"])
c7.metric("Best Flow", rankings.sort_values("flow", ascending=False).iloc[0]["ticker"])
c8.metric("Generated", datetime.now().strftime("%H:%M"))

st.subheader("Institutional Ranking Table")

view_cols = [
    "rank", "ticker", "category", "signal", "institutional_score",
    "short", "swing", "position",
    "theme", "quality", "momentum", "technical", "flow",
    "leadership", "category_leadership", "risk",
    "price", "rsi", "rel_vol20", "ret20_%", "ret63_%", "high_prox_%"
]

st.dataframe(
    rankings[view_cols],
    use_container_width=True,
    hide_index=True
)

st.subheader("Top 10 Setups")

for _, row in rankings.head(10).iterrows():
    with st.container(border=True):
        st.markdown(
            f"### #{int(row['rank'])} {row['ticker']} "
            f"{signal_color(row['signal'])} — {row['signal']}"
        )

        a, b, c, d = st.columns(4)
        a.metric("Institutional", row["institutional_score"])
        b.metric("Short", row["short"])
        c.metric("Swing", row["swing"])
        d.metric("Position", row["position"])

        e, f, g, h = st.columns(4)
        e.metric("Theme", row["theme"])
        f.metric("Momentum", row["momentum"])
        g.metric("Flow", row["flow"])
        h.metric("Leadership", row["leadership"])

        st.write(
            f"**Category:** {row['category']} · "
            f"**Technical:** {row['technical']} · "
            f"**Risk:** {row['risk']} · "
            f"**RS 20D:** {row['rel20_%']}% · "
            f"**RS 63D:** {row['rel63_%']}%"
        )

        st.link_button("Open TradingView Chart", row["chart"])


st.subheader("Category Leaders")

category_rows = []
for category in sorted(rankings["category"].unique()):
    group = rankings[rankings["category"] == category].sort_values(
        "institutional_score",
        ascending=False
    )
    leader = group.iloc[0]
    category_rows.append({
        "category": category,
        "leader": leader["ticker"],
        "signal": leader["signal"],
        "score": leader["institutional_score"],
        "momentum": leader["momentum"],
        "flow": leader["flow"],
        "risk": leader["risk"],
    })

category_df = pd.DataFrame(category_rows)
st.dataframe(category_df, use_container_width=True, hide_index=True)

csv = rankings.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download CSV",
    csv,
    "alex_ai_infrastructure_rankings.csv",
    "text/csv"
)
