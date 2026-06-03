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
