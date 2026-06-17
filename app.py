
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Alex AI Infrastructure Engine V3",
    layout="wide"
)

# =========================================================
# Alex AI Infrastructure Institutional Engine V3 Final
# 31-stock mobile dashboard
# No news / no analyst API
# 5-minute refresh
# =========================================================

COMPANY_PROFILES = {
    "NVDA": {"category": "Compute", "ai_exposure": 100, "tier": "Core Leader"},
    "AMD":  {"category": "Compute", "ai_exposure": 87,  "tier": "Challenger"},
    "ARM":  {"category": "Compute", "ai_exposure": 86,  "tier": "Strategic IP"},

    "AVGO": {"category": "ASIC / Custom Silicon", "ai_exposure": 98, "tier": "Core Leader"},
    "MRVL": {"category": "ASIC / Custom Silicon", "ai_exposure": 90, "tier": "Challenger"},

    "MU":   {"category": "Memory / HBM", "ai_exposure": 92, "tier": "Core Leader"},
    "SNDK": {"category": "Storage", "ai_exposure": 76, "tier": "Cyclical Satellite"},

    "COHR": {"category": "Optical / Interconnect", "ai_exposure": 86, "tier": "Optical Leader"},
    "CRDO": {"category": "Optical / Interconnect", "ai_exposure": 85, "tier": "High-Growth Interconnect"},
    "ALAB": {"category": "Optical / Interconnect", "ai_exposure": 84, "tier": "High-Growth Interconnect"},
    "LITE": {"category": "Optical / Interconnect", "ai_exposure": 83, "tier": "Optical Challenger"},
    "AAOI": {"category": "Optical / Interconnect", "ai_exposure": 66, "tier": "Speculative High Beta"},

    "ANET": {"category": "Networking", "ai_exposure": 94, "tier": "Core Leader"},
    "CIEN": {"category": "Networking", "ai_exposure": 78, "tier": "Networking Satellite"},

    "VRT":  {"category": "Power / Cooling", "ai_exposure": 95, "tier": "Core Leader"},
    "ETN":  {"category": "Power / Cooling", "ai_exposure": 79, "tier": "Power Infrastructure"},

    "TSM":  {"category": "Foundry", "ai_exposure": 94, "tier": "Core Leader"},
    "INTC": {"category": "Foundry", "ai_exposure": 64, "tier": "Turnaround / Speculative"},

    "ASML": {"category": "Equipment", "ai_exposure": 89, "tier": "Equipment Leader"},
    "LRCX": {"category": "Equipment", "ai_exposure": 88, "tier": "Equipment Leader"},
    "KLAC": {"category": "Equipment", "ai_exposure": 82, "tier": "Quality Equipment"},
    "AMAT": {"category": "Equipment", "ai_exposure": 81, "tier": "Quality Equipment"},
    "TER":  {"category": "Equipment", "ai_exposure": 72, "tier": "Testing / Automation"},

    "SNPS": {"category": "EDA", "ai_exposure": 71, "tier": "EDA Quality"},
    "CDNS": {"category": "EDA", "ai_exposure": 71, "tier": "EDA Quality"},

    "MSFT": {"category": "Cloud / Platform", "ai_exposure": 74, "tier": "Mega-Cap Platform"},
    "ORCL": {"category": "Cloud / Platform", "ai_exposure": 73, "tier": "AI Cloud"},
    "PLTR": {"category": "AI Software", "ai_exposure": 72, "tier": "AI Software"},
    "SMCI": {"category": "Servers", "ai_exposure": 77, "tier": "High-Beta Server"},
    "DELL": {"category": "Servers", "ai_exposure": 78, "tier": "Server Infrastructure"},

    "GLW":  {"category": "Fiber / Materials", "ai_exposure": 68, "tier": "Infrastructure Satellite"},
    "NBIS": {"category": "AI Infrastructure", "ai_exposure": 62, "tier": "Speculative Infrastructure"},
}

UNIVERSE = list(COMPANY_PROFILES.keys())
BENCHMARKS = ["QQQ", "SOXX", "SPY"]
NYSE_TICKERS = {"TSM", "VRT", "ETN", "CIEN", "DELL", "ORCL", "GLW", "NBIS"}


def clamp(x, lo=0, hi=100):
    if x is None or pd.isna(x) or np.isinf(x):
        return 50.0
    return max(lo, min(hi, float(x)))


def pct_return(series, days):
    if len(series) <= days:
        return np.nan
    base = series.iloc[-days]
    last = series.iloc[-1]
    if pd.isna(base) or pd.isna(last) or base == 0:
        return np.nan
    return last / base - 1


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def max_drawdown(series, window=126):
    s = series.tail(window)
    if len(s) < 5:
        return np.nan
    rolling_max = s.cummax()
    drawdown = s / rolling_max - 1
    return drawdown.min()


def risk_adjusted_return(series, window=126):
    s = series.tail(window).pct_change().dropna()
    if len(s) < 20 or s.std() == 0:
        return np.nan
    return (s.mean() / s.std()) * np.sqrt(252)


def trend_quality_score(series, window=126):
    s = series.tail(window).dropna()
    if len(s) < 40:
        return 50.0

    x = np.arange(len(s))
    y = np.log(s.values)

    try:
        slope, intercept = np.polyfit(x, y, 1)
        fitted = slope * x + intercept
        residual = y - fitted
        denom = np.sum((y - y.mean()) ** 2)
        r2 = 0 if denom == 0 else 1 - (np.sum(residual ** 2) / denom)
        annualized_slope = slope * 252
        score = 50 + annualized_slope * 25 + r2 * 35
        return clamp(score)
    except Exception:
        return 50.0


def tradingview_url(ticker):
    exchange = "NYSE" if ticker in NYSE_TICKERS else "NASDAQ"
    return f"https://www.tradingview.com/chart/?symbol={exchange}:{ticker}"


def signal_icon(signal):
    return {
        "Strong Long": "🟢",
        "Buy": "🟩",
        "Watch": "🟨",
        "Extended": "🟧",
        "Risk Off": "🔴",
        "Avoid": "⚪",
        "Data Error": "⚫",
    }.get(signal, "⚪")


@st.cache_data(ttl=300, show_spinner=False)
def download_market_data():
    tickers = UNIVERSE + BENCHMARKS
    data = yf.download(
        tickers,
        period="24mo",
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    return data


def get_df(data, ticker):
    if isinstance(data.columns, pd.MultiIndex):
        if ticker not in data.columns.get_level_values(0):
            return pd.DataFrame()
        df = data[ticker].copy()
    else:
        df = data.copy()

    df.columns = [str(c).lower() for c in df.columns]
    return df.dropna()


def calculate_market_regime(data):
    qqq = get_df(data, "QQQ")["close"]
    soxx = get_df(data, "SOXX")["close"]
    spy = get_df(data, "SPY")["close"]

    qqq21 = pct_return(qqq, 21)
    soxx21 = pct_return(soxx, 21)
    spy21 = pct_return(spy, 21)
    qqq63 = pct_return(qqq, 64)
    soxx63 = pct_return(soxx, 64)

    regime = (
        100 if qqq21 > 0 and soxx21 > 0 and soxx21 > qqq21 else
        85 if qqq21 > 0 and soxx21 > 0 else
        65 if qqq21 > 0 or soxx21 > 0 else
        40 if spy21 > 0 else
        25
    )

    bench_returns = {
        "bench21": np.nanmean([qqq21, soxx21]),
        "bench63": np.nanmean([qqq63, soxx63]),
    }
    return regime, bench_returns


def score_ticker(ticker, data, market_regime, bench_returns):
    profile = COMPANY_PROFILES[ticker]
    df = get_df(data, ticker)

    if df.empty or len(df) < 260:
        return {
            "ticker": ticker,
            "category": profile["category"],
            "tier": profile["tier"],
            "signal": "Data Error",
            "ai_exposure": profile["ai_exposure"],
            "institutional_score": 0,
            "short": 0,
            "swing": 0,
            "position": 0,
            "price": np.nan,
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
    ret21 = pct_return(close, 22)
    ret63 = pct_return(close, 64)
    ret126 = pct_return(close, 127)
    ret252 = pct_return(close, 253)

    rel21 = ret21 - bench_returns["bench21"]
    rel63 = ret63 - bench_returns["bench63"]

    high252 = high.rolling(252).max().iloc[-1]
    high_prox = price / high252 if high252 and not pd.isna(high252) else np.nan

    rel_vol20 = volume.iloc[-1] / vol20 if vol20 and not pd.isna(vol20) else 1.0
    rel_vol50 = volume.iloc[-1] / vol50 if vol50 and not pd.isna(vol50) else 1.0

    rsi14 = rsi(close).iloc[-1]
    atr_pct = (high - low).rolling(14).mean().iloc[-1] / price

    ai_exposure_score = profile["ai_exposure"]

    rs_raw = clamp(
        (18 if ret21 > 0 else 0)
        + (20 if ret63 > 0 else 0)
        + (18 if ret126 > 0 else 0)
        + (14 if ret252 > 0 else 0)
        + (15 if rel21 > 0 else 0)
        + (15 if rel63 > 0 else 0)
    )

    flow_score = clamp(
        (35 if rel_vol20 > 2.0 and ret1 > 0 else
         27 if rel_vol20 > 1.5 and ret1 > 0 else
         18 if rel_vol20 > 1.2 and ret1 > 0 else 6)
        + (25 if rel_vol50 > 1.2 and ret5 > 0 else
           15 if rel_vol50 > 1.0 and ret5 > 0 else 5)
        + (20 if ret5 > 0 else 0)
        + (20 if price > sma10 else 0)
    )

    earnings_power_score = (
        100 if ret5 > 0.08 and rel_vol20 > 1.30 else
        88 if ret5 > 0.04 and rel_vol20 > 1.10 else
        70 if ret5 > 0 else
        45 if ret5 > -0.04 else
        25
    )

    trend_quality = trend_quality_score(close, 126)

    dd = max_drawdown(close, 126)
    drawdown_quality = (
        90 if dd > -0.08 else
        78 if dd > -0.13 else
        62 if dd > -0.20 else
        45 if dd > -0.30 else
        28
    )

    sharpe_like = risk_adjusted_return(close, 126)
    risk_adjusted = (
        95 if sharpe_like > 2.0 else
        82 if sharpe_like > 1.2 else
        68 if sharpe_like > 0.6 else
        50 if sharpe_like > 0 else
        35
    )

    breakout = price > high.iloc[-2] and rel_vol20 > 1.2
    pullback = price > sma50 and price < sma20 and 42 <= rsi14 <= 58
    trend_stack = price > sma20 > sma50 > sma200
    reclaim20 = price > sma20 and close.iloc[-2] < prev_sma20
    reclaim50 = price > sma50 and close.iloc[-2] < prev_sma50

    technical_setup = clamp(
        (28 if breakout else 0)
        + (22 if pullback else 0)
        + (25 if trend_stack else 0)
        + (10 if reclaim20 else 0)
        + (10 if reclaim50 else 0)
        + (5 if rsi14 >= 50 else 0)
    )

    risk_score = (
        88 if atr_pct < 0.03 else
        76 if atr_pct < 0.045 else
        62 if atr_pct < 0.06 else
        45 if atr_pct < 0.08 else
        30
    )
    if rsi14 > 80:
        risk_score -= 15
    if price < sma50:
        risk_score -= 22
    if price < sma200:
        risk_score -= 18
    risk_score = clamp(risk_score)

    short_score = clamp(
        rs_raw * 0.25
        + technical_setup * 0.28
        + flow_score * 0.27
        + earnings_power_score * 0.10
        + risk_score * 0.10
    )

    swing_score = clamp(
        rs_raw * 0.32
        + flow_score * 0.18
        + trend_quality * 0.15
        + technical_setup * 0.15
        + ai_exposure_score * 0.10
        + market_regime * 0.05
        + risk_score * 0.05
    )

    position_score = clamp(
        ai_exposure_score * 0.25
        + rs_raw * 0.25
        + trend_quality * 0.15
        + drawdown_quality * 0.10
        + risk_adjusted * 0.10
        + market_regime * 0.10
        + risk_score * 0.05
    )

    base_score = clamp(
        ai_exposure_score * 0.15
        + rs_raw * 0.20
        + flow_score * 0.15
        + earnings_power_score * 0.10
        + trend_quality * 0.10
        + drawdown_quality * 0.05
        + risk_adjusted * 0.05
        + market_regime * 0.05
        + technical_setup * 0.10
        + risk_score * 0.05
    )

    if price < sma50 or risk_score < 38:
        signal = "Risk Off"
    elif base_score >= 85 and rsi14 < 76:
        signal = "Strong Long"
    elif base_score >= 76 and rsi14 < 76:
        signal = "Buy"
    elif base_score >= 76 and rsi14 >= 76:
        signal = "Extended"
    elif base_score >= 65:
        signal = "Watch"
    else:
        signal = "Avoid"

    return {
        "ticker": ticker,
        "category": profile["category"],
        "tier": profile["tier"],
        "signal": signal,
        "price": round(float(price), 2),
        "ai_exposure": round(ai_exposure_score, 1),
        "rs_raw": round(rs_raw, 1),
        "flow": round(flow_score, 1),
        "earnings_power": round(earnings_power_score, 1),
        "trend_quality": round(trend_quality, 1),
        "drawdown_quality": round(drawdown_quality, 1),
        "risk_adjusted": round(risk_adjusted, 1),
        "technical": round(technical_setup, 1),
        "risk": round(risk_score, 1),
        "base_score": round(base_score, 1),
        "short": round(short_score, 1),
        "swing": round(swing_score, 1),
        "position": round(position_score, 1),
        "ret21_%": round(ret21 * 100, 2),
        "ret63_%": round(ret63 * 100, 2),
        "ret126_%": round(ret126 * 100, 2),
        "ret252_%": round(ret252 * 100, 2),
        "rel21_%": round(rel21 * 100, 2),
        "rel63_%": round(rel63 * 100, 2),
        "rel_vol20": round(float(rel_vol20), 2),
        "rsi": round(float(rsi14), 1),
        "max_dd_6m_%": round(float(dd * 100), 2) if not pd.isna(dd) else np.nan,
        "high_prox_%": round(float(high_prox * 100), 1) if not pd.isna(high_prox) else np.nan,
        "chart": tradingview_url(ticker),
    }


@st.cache_data(ttl=300, show_spinner=False)
def build_rankings():
    data = download_market_data()
    market_regime, bench_returns = calculate_market_regime(data)

    rows = []
    for ticker in UNIVERSE:
        try:
            rows.append(score_ticker(ticker, data, market_regime, bench_returns))
            time.sleep(0.01)
        except Exception as exc:
            profile = COMPANY_PROFILES[ticker]
            rows.append({
                "ticker": ticker,
                "category": profile["category"],
                "tier": profile["tier"],
                "signal": "Data Error",
                "institutional_score": 0,
                "chart": tradingview_url(ticker),
                "error": str(exc),
            })

    df = pd.DataFrame(rows)

    for col in ["ret21_%", "ret63_%", "ret126_%", "ret252_%", "rel21_%", "rel63_%"]:
        if col not in df.columns:
            df[col] = np.nan

    df["rs_pct_21"] = df["ret21_%"].rank(pct=True) * 100
    df["rs_pct_63"] = df["ret63_%"].rank(pct=True) * 100
    df["rs_pct_126"] = df["ret126_%"].rank(pct=True) * 100
    df["rs_pct_252"] = df["ret252_%"].rank(pct=True) * 100
    df["relative_rs_pct"] = (
        df["rel21_%"].rank(pct=True) * 0.45
        + df["rel63_%"].rank(pct=True) * 0.55
    ) * 100

    df["rs_percentile"] = (
        df["rs_pct_21"] * 0.25
        + df["rs_pct_63"] * 0.30
        + df["rs_pct_126"] * 0.25
        + df["rs_pct_252"] * 0.10
        + df["relative_rs_pct"] * 0.10
    ).round(1).fillna(0)

    df["category_rank"] = df.groupby("category")["base_score"].rank(ascending=False, method="first")
    df["category_count"] = df.groupby("category")["ticker"].transform("count")
    df["category_leadership"] = (
        (1 - (df["category_rank"] - 1) / df["category_count"].replace(0, np.nan)) * 100
    ).round(1).fillna(70)

    df["leader_status"] = np.where(
        df["category_rank"] == 1,
        "Leader",
        np.where(df["category_rank"] <= 2, "Follower", "Lagging")
    )

    df["institutional_score"] = (
        df["ai_exposure"] * 0.15
        + df["rs_percentile"] * 0.25
        + df["flow"] * 0.15
        + df["earnings_power"] * 0.10
        + df["trend_quality"] * 0.10
        + df["category_leadership"] * 0.10
        + df["drawdown_quality"] * 0.05
        + df["risk_adjusted"] * 0.05
        + df["risk"] * 0.05
    ).round(1)

    df = df.sort_values("institutional_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    return df, market_regime


st.title("Alex AI Infrastructure Institutional Engine V3")
st.caption(
    "31-stock AI infrastructure dashboard · AI Exposure + RS Percentile + Flow + Leadership · 5-minute refresh"
)

with st.spinner("Building V3 institutional rankings..."):
    rankings, market_regime = build_rankings()

if rankings.empty:
    st.error("No ranking data available. Refresh later.")
    st.stop()

top = rankings.iloc[0]
best_short = rankings.sort_values("short", ascending=False).iloc[0]
best_swing = rankings.sort_values("swing", ascending=False).iloc[0]
best_position = rankings.sort_values("position", ascending=False).iloc[0]
best_flow = rankings.sort_values("flow", ascending=False).iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Best Overall", top["ticker"], top["signal"])
c2.metric("Institutional Score", top["institutional_score"])
c3.metric("Best Short", best_short["ticker"], best_short["short"])
c4.metric("Best Swing", best_swing["ticker"], best_swing["swing"])

c5, c6, c7, c8 = st.columns(4)
c5.metric("Best Position", best_position["ticker"], best_position["position"])
c6.metric("Best Flow", best_flow["ticker"], best_flow["flow"])
c7.metric("Market Regime", round(market_regime, 1))
c8.metric("Generated", datetime.now().strftime("%H:%M"))

st.subheader("Key Value-Chain Leaders")

key_categories = [
    "Compute",
    "ASIC / Custom Silicon",
    "Memory / HBM",
    "Optical / Interconnect",
    "Networking",
    "Power / Cooling",
    "Foundry",
    "Equipment",
]

leader_cards = []
for cat in key_categories:
    group = rankings[rankings["category"] == cat]
    if not group.empty:
        leader = group.sort_values("institutional_score", ascending=False).iloc[0]
        leader_cards.append((cat, leader))

cols = st.columns(4)
for i, (cat, leader) in enumerate(leader_cards):
    with cols[i % 4]:
        st.metric(cat, leader["ticker"], leader["signal"])

st.subheader("Institutional Ranking Table")

view_cols = [
    "rank", "ticker", "category", "tier", "leader_status", "signal",
    "institutional_score", "ai_exposure", "rs_percentile",
    "flow", "earnings_power", "trend_quality",
    "category_leadership", "risk_adjusted", "risk",
    "short", "swing", "position",
    "price", "rsi", "rel_vol20", "ret21_%", "ret63_%", "ret126_%", "max_dd_6m_%"
]

safe_view_cols = [c for c in view_cols if c in rankings.columns]
st.dataframe(rankings[safe_view_cols], use_container_width=True, hide_index=True)

st.subheader("Top 10 Setups")

for _, row in rankings.head(10).iterrows():
    with st.container(border=True):
        st.markdown(
            f"### #{int(row['rank'])} {row['ticker']} {signal_icon(row['signal'])} — {row['signal']}"
        )

        a, b, c, d = st.columns(4)
        a.metric("Institutional", row["institutional_score"])
        b.metric("Short", row["short"])
        c.metric("Swing", row["swing"])
        d.metric("Position", row["position"])

        e, f, g, h = st.columns(4)
        e.metric("AI Exposure", row["ai_exposure"])
        f.metric("RS Percentile", row["rs_percentile"])
        g.metric("Flow", row["flow"])
        h.metric("Leadership", row["leader_status"])

        st.write(
            f"**Category:** {row['category']} · "
            f"**Trend Quality:** {row['trend_quality']} · "
            f"**Risk Adj:** {row['risk_adjusted']} · "
            f"**Drawdown Quality:** {row['drawdown_quality']} · "
            f"**Risk:** {row['risk']}"
        )

        st.link_button("Open TradingView Chart", row["chart"])

st.subheader("All Category Leaders")

category_summary = []
for cat in sorted(rankings["category"].dropna().unique()):
    group = rankings[rankings["category"] == cat].sort_values("institutional_score", ascending=False)
    leader = group.iloc[0]
    category_summary.append({
        "category": cat,
        "leader": leader["ticker"],
        "signal": leader["signal"],
        "score": leader["institutional_score"],
        "rs_percentile": leader["rs_percentile"],
        "flow": leader["flow"],
        "risk": leader["risk"],
    })

category_df = pd.DataFrame(category_summary)
st.dataframe(category_df, use_container_width=True, hide_index=True)

csv = rankings.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download CSV",
    csv,
    "alex_ai_infrastructure_v3_rankings.csv",
    "text/csv"
)
