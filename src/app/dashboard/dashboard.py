from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────
# CONFIGURATION STREAMLIT & UI
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SignalConso · Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# THEME & CSS
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* fond général */
.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* sidebar */
section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

/* tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0px;
    background: #161b22;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #21262d;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8b949e;
    border-radius: 8px;
    padding: 8px 18px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.stTabs [aria-selected="true"] {
    background: #1f6feb !important;
    color: #ffffff !important;
}

/* metric cards */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace;
    font-size: 28px !important;
    color: #58a6ff;
}
[data-testid="stMetricLabel"] {
    font-size: 12px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* section headers */
.sec-header {
    font-size: 20px;
    font-weight: 700;
    color: #e6edf3;
    border-left: 3px solid #1f6feb;
    padding-left: 12px;
    margin: 24px 0 16px;
}

/* KPI badge */
.kpi-badge {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
}
.kpi-num  { font-family:'DM Mono',monospace; font-size:26px; font-weight:600; color:#3fb950; }
.kpi-lbl  { font-size:11px; color:#8b949e; text-transform:uppercase; letter-spacing:.08em; }

/* leaderboard table */
.lb-table {
    width:100%; border-collapse:collapse;
    font-family:'DM Mono',monospace; font-size:13px;
}
.lb-table th {
    background:#161b22; color:#8b949e;
    padding:10px 14px; text-align:left;
    font-size:11px; letter-spacing:.08em;
    border-bottom:1px solid #21262d;
}
.lb-table td {
    padding:10px 14px; border-bottom:1px solid #21262d;
    color:#e6edf3;
}
.lb-table tr:hover td { background:#161b22; }
.badge-gold   { background:#b8860b22; color:#e3b341; border:1px solid #b8860b; border-radius:6px; padding:2px 8px; }
.badge-silver { background:#30363d; color:#8b949e; border:1px solid #30363d; border-radius:6px; padding:2px 8px; }
.bar-fill { height:6px; background:#1f6feb; border-radius:3px; display:inline-block; }

/* pipeline log */
.pipeline-log {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 16px;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    line-height: 1.8;
    color: #c9d1d9;
    max-height: 400px;
    overflow-y: auto;
}

/* button override */
.stButton button {
    background: #1f6feb;
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    padding: 10px 24px;
    transition: background .2s;
}
.stButton button:hover { background: #388bfd; }

/* progress bar */
.stProgress > div > div { background: #1f6feb; }

/* dividers */
hr { border-color: #21262d; }

/* dataframe */
.stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""",
    unsafe_allow_html=True,
)
