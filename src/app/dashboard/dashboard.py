from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────
# CONFIG & API URLS
# ─────────────────────────────────────────────
API_BASE_URL = os.getenv("API_URL", "http://api:8000").rstrip("/")
PREDICTION_URL = f"{API_BASE_URL}/predictions/"
FLOWS_API_URL = f"{API_BASE_URL}/flows"

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "clean_complaints")
GCS_RESULTS_PREFIX = os.getenv("PREFECT_RESULTS_PREFIX", "prefect-results/")
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
DEFAULT_MODEL_VER = os.getenv("MODEL_VERSION", "logreg-v1")
MODEL_REFRESH_SECONDS = 90


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


# ─────────────────────────────────────────────
# API HELPER
# ─────────────────────────────────────────────
PREDICTION_URL = os.getenv("PREDICTION_URL", "http://localhost:8000/predictions")
MODEL_REFRESH_SECONDS = 90


def predict_api(text: str, model_blob: str | None = None) -> dict:
    payload = {
        "text": text,
        "model": model_blob,
        "model_blob": model_blob,
        "model_path": model_blob,
        "model_version": Path(model_blob).stem if model_blob else None,
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    r = requests.post(PREDICTION_URL, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tabs = st.tabs(
    [
        "📋 Vue d'ensemble",
        "⏱️ Flows temps réel",
        "🗺️ Cartographie",
        "🤖 Prédiction",
        "🧠 Modèles ML",
        "⚙️ Pipeline",
        "☁️ GCS",
    ]
)
tab_overview, tab_flows, tab_map, tab_predict, tab_ml, tab_pipeline, tab_gcs = tabs


# ══════════════════════════════════════════════
# TAB 4 — PRÉDICTION
# ══════════════════════════════════════════════
with tab_predict:
    st.markdown(
        '<div class="sec-header">🤖 Classification de signalement</div>', unsafe_allow_html=True
    )

    selected_model_blob = st.session_state.get("selected_model_blob", DEFAULT_MODEL_PATH)
    if not selected_model_blob:
        selected_model_blob = DEFAULT_MODEL_PATH

    st.markdown(
        f"""
        <div style="background:#161b22;border:1px solid #21262d;border-radius:10px;padding:14px 16px;margin-bottom:16px;">
          <div style="font-size:13px;color:#8b949e;line-height:1.8;">
            <b style="color:#e6edf3;">Modèle actif :</b> {Path(selected_model_blob).name}<br>
            <span style="color:#58a6ff;">Endpoint :</span> <code>{PREDICTION_URL}</code><br>
            <span style="color:#8b949e;">Sélectionnez un modèle dans la sidebar pour changer le comportement de la classification.</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([3, 1])
    with col_left:
        user_text = st.text_area(
            "Décrivez le signalement",
            height=150,
            placeholder="Ex : j'ai commandé un produit sur un site internet et je n'ai jamais reçu ma commande...",
        )
    with col_right:
        st.markdown("<br>", unsafe_allow_html=True)
        run_pred = st.button("🚀 Classifier", width="stretch")
        st.markdown("---")
        st.markdown(
            f"<div style='font-size:11px;color:#8b949e'>Modèle : `{Path(selected_model_blob).name}`</div>",
            unsafe_allow_html=True,
        )

    if run_pred:
        if not user_text.strip():
            st.warning("Veuillez saisir un texte.")
        else:
            with st.spinner("Analyse en cours…"):
                try:
                    result = predict_api(user_text, selected_model_blob)
                    cat = result.get("predicted_category") or result.get("category", "–")
                    conf = result.get("confidence", 0)
                    api_model = (
                        result.get("model_version")
                        or result.get("model")
                        or Path(selected_model_blob).stem
                    )

                    st.markdown("<br>", unsafe_allow_html=True)
                    r1, r2, r3 = st.columns(3)
                    r1.metric("Catégorie prédite", cat)
                    r2.metric("Confiance", f"{conf:.2%}")
                    r3.metric("Modèle", api_model)

                    st.success(f"Classification lancée avec **{Path(selected_model_blob).name}**.")

                    with st.expander("Réponse API complète"):
                        st.json(result)
                except Exception as e:
                    st.error(f"Erreur API : {e}")
                    st.info(f"Vérifiez que l'API est démarrée sur `{PREDICTION_URL}`")
