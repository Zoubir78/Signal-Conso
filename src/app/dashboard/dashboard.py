from __future__ import annotations

import ast
import json
import os
import sys
from collections import Counter
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
from google.cloud import storage

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
# HELPERS UTILS
# ─────────────────────────────────────────────
STOPWORDS_FR = {
    "a",
    "alors",
    "au",
    "aucuns",
    "aussi",
    "autre",
    "avant",
    "avec",
    "avoir",
    "bon",
    "car",
    "ce",
    "cela",
    "ces",
    "ceux",
    "chaque",
    "ci",
    "comme",
    "comment",
    "dans",
    "des",
    "du",
    "dedans",
    "dehors",
    "depuis",
    "devrait",
    "doit",
    "donc",
    "dos",
    "debut",
    "elle",
    "elles",
    "en",
    "encore",
    "essai",
    "est",
    "et",
    "eu",
    "fait",
    "faites",
    "fois",
    "font",
    "hors",
    "ici",
    "il",
    "ils",
    "je",
    "la",
    "le",
    "les",
    "leur",
    "ma",
    "maintenant",
    "mais",
    "mes",
    "mine",
    "moins",
    "mon",
    "mot",
    "meme",
    "ni",
    "nommes",
    "notre",
    "nous",
    "nouveaux",
    "ou",
    "par",
    "parce",
    "parole",
    "pas",
    "personnes",
    "peu",
    "peut",
    "piece",
    "plupart",
    "pour",
    "pourquoi",
    "quand",
    "que",
    "quel",
    "quelle",
    "quelles",
    "quels",
    "qui",
    "sa",
    "sans",
    "ses",
    "seulement",
    "si",
    "sien",
    "son",
    "sont",
    "sous",
    "soyez",
    "sujet",
    "sur",
    "ta",
    "tandis",
    "te",
    "tes",
    "ton",
    "tous",
    "tout",
    "trop",
    "tres",
    "tu",
    "un",
    "une",
    "vos",
    "votre",
    "vous",
    "vu",
    "ca",
    "etaient",
    "etat",
    "etions",
    "ete",
    "etre",
}


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    return bool(isinstance(v, str) and not v.strip())


def _parse_multivalue(value: Any) -> list[str]:
    if _is_missing(value):
        return []
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        raw = str(value).strip()
        if not raw:
            return []
        if raw[0] in "[({":
            try:
                parsed = ast.literal_eval(raw)
                items = list(parsed) if isinstance(parsed, (list, tuple, set)) else [parsed]
            except (ValueError, SyntaxError):
                items = [p.strip() for p in raw.strip("[](){} ").split(",") if p.strip()]
        else:
            items = [raw]
    return [str(i).strip() for i in items if str(i).strip()]


def _normalize_label(t: str) -> str:
    return " ".join(str(t).strip().split())


def _bool_series(s: pd.Series) -> pd.Series:
    TRUTHY = {"1", "true", "t", "yes", "y", "oui", "vrai", "on"}

    def _tb(v):
        if _is_missing(v):
            return False
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        return str(v).strip().lower() in TRUTHY

    return s.apply(_tb)


def _department_label(row: pd.Series) -> str:
    def _clean_dep_code(v: Any) -> str:
        if _is_missing(v):
            return ""
        txt = str(v).strip().replace(".0", "")
        txt = txt.zfill(2) if txt.isdigit() and len(txt) <= 2 else txt
        return _normalize_label(txt)

    code = _clean_dep_code(row.get("dep_code", ""))
    name = _normalize_label(row.get("dep_name", "")) if not _is_missing(row.get("dep_name")) else ""
    if code and name:
        return f"{code} – {name}"
    return code or name or "Inconnu"


def _frequency(df: pd.DataFrame, col: str, limit: int = 15) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=int)
    counter: Counter = Counter()
    display: dict = {}
    for raw in df[col].dropna():
        for v in _parse_multivalue(raw) or [str(raw)]:
            lbl = _normalize_label(v)
            if not lbl:
                continue
            key = lbl.casefold()
            counter[key] += 1
            display.setdefault(key, lbl)
    if not counter:
        return pd.Series(dtype=int)
    top = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:limit]
    return pd.Series({display[k]: c for k, c in top})


def _keyword_freq(df: pd.DataFrame, limit: int = 20) -> pd.Series:
    counter: Counter = Counter()
    display: dict = {}
    cols = [c for c in ["tags", "subcategories", "clean_text"] if c in df.columns]
    if not cols:
        return pd.Series(dtype=int)
    for _, row in df[cols].iterrows():
        tokens = []
        for c in ["tags", "subcategories"]:
            if c in row and not _is_missing(row[c]):
                tokens.extend(_parse_multivalue(row[c]))
        if not tokens and "clean_text" in row and not _is_missing(row["clean_text"]):
            tokens = [
                t
                for t in str(row["clean_text"]).lower().split()
                if len(t) > 3 and t not in STOPWORDS_FR
            ]
        for tok in tokens:
            lbl = _normalize_label(tok)
            if not lbl:
                continue
            key = lbl.casefold()
            counter[key] += 1
            display.setdefault(key, lbl)
    if not counter:
        return pd.Series(dtype=int)
    top = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:limit]
    return pd.Series({display[k]: c for k, c in top})


# ─────────────────────────────────────────────
# GCS HELPERS
# ─────────────────────────────────────────────
@st.cache_resource
def _gcs() -> storage.Client:
    return storage.Client()


@st.cache_data(ttl=120)
def list_blobs(prefix: str) -> list[str]:
    return [b.name for b in _gcs().bucket(GCS_BUCKET_NAME).list_blobs(prefix=prefix)]


@st.cache_data(ttl=120)
def load_latest_dataset() -> tuple[pd.DataFrame, str | None]:
    blobs = list(_gcs().bucket(GCS_BUCKET_NAME).list_blobs(prefix="processed/"))
    if not blobs:
        return pd.DataFrame(), None
    latest = max(blobs, key=lambda b: b.updated or datetime.min.replace(tzinfo=datetime.UTC))
    data = latest.download_as_bytes()
    return pd.read_csv(BytesIO(data)), latest.name


@st.cache_data(ttl=120)
def load_evaluation_report() -> dict | None:
    try:
        blob = _gcs().bucket(GCS_BUCKET_NAME).blob("models/evaluation_report.json")
        if blob.exists():
            return json.loads(blob.download_as_text())
    except Exception:
        pass
    return None


def download_model(blob_name: str, local: str = "/tmp/tmp_model.joblib") -> str:
    _gcs().bucket(GCS_BUCKET_NAME).blob(blob_name).download_to_filename(local)
    return local


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
# DATASET (chargé une fois)
# ─────────────────────────────────────────────
df, source_name = load_latest_dataset()

if not df.empty:
    if "creationdate" in df.columns:
        df["creationdate"] = pd.to_datetime(df["creationdate"], errors="coerce")
    df["department_label"] = (
        df.apply(_department_label, axis=1)
        if ("dep_name" in df.columns or "dep_code" in df.columns)
        else "Inconnu"
    )


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


# ══════════════════════════════════════════════
# TAB 6 — PIPELINE
# ══════════════════════════════════════════════
with tab_pipeline:
    st.markdown(
        '<div class="sec-header">⚙️ Pipeline complet SignalConso</div>', unsafe_allow_html=True
    )

    st.markdown(
        """
    <div style="background:#161b22;border:1px solid #21262d;border-radius:10px;padding:16px;margin-bottom:16px;">
      <div style="font-size:13px;color:#8b949e;line-height:2;">
        <b style="color:#e6edf3;">Flux :</b><br>
        <span style="color:#58a6ff;">①</span> Extract API SignalConso (10 000 enregistrements)<br>
        <span style="color:#58a6ff;">②</span> Upload GCS <code>raw/</code> → table externe BigQuery<br>
        <span style="color:#58a6ff;">③</span> dbt run → staging → intermediate → mart_signalconso<br>
        <span style="color:#58a6ff;">④</span> Lecture mart depuis BigQuery<br>
        <span style="color:#58a6ff;">⑤</span> Entraînement multi-modèles (LogReg · SGD · LinearSVC · NB · RF)<br>
        <span style="color:#58a6ff;">⑥</span> Leaderboard + sélection du meilleur modèle<br>
        <span style="color:#58a6ff;">⑦</span> Upload GCS <code>models/</code> + rapport JSON
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if st.button("🚀 Lancer le pipeline", width="content"):
        log_lines: list[str] = []
        log_box = st.empty()

        def _log(msg: str):
            log_lines.append(str(msg))
            log_html = "<br>".join(
                f'<span style="color:#3fb950">{line}</span>'
                if any(line.startswith(p) for p in ["✔", "🏁", "🏆"])
                else f'<span style="color:#f85149">{line}</span>'
                if any(line.startswith(p) for p in ["✖", "❌"])
                else f'<span style="color:#e3b341">{line}</span>'
                if any(line.startswith(p) for p in ["⚠", "📊 Leaderboard"])
                else f'<span style="color:#58a6ff">{line}</span>'
                if any(line.startswith(p) for p in ["🚀", "📥", "🔧", "🤖", "📤", "☁️"])
                else f'<span style="color:#c9d1d9">{line}</span>'
                for line in log_lines[-80:]
            )
            log_box.markdown(
                f'<div class="pipeline-log">{log_html}</div>',
                unsafe_allow_html=True,
            )

        try:
            from scripts.pipeline import run_pipeline

            result = run_pipeline(_log)
            st.success("✅ Pipeline terminé avec succès !")

            # Résumé des résultats
            st.divider()
            st.markdown('<div class="sec-header">📊 Résultats du run</div>', unsafe_allow_html=True)

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Enregistrements bruts", f"{result.get('raw_rows', 0):,}")
            r2.metric("Lignes du mart dbt", f"{result.get('mart_rows', 0):,}")
            r3.metric("Meilleur modèle", result.get("best_model", "–"))
            r4.metric("Accuracy", f"{result.get('accuracy', 0):.2%}")

            # Leaderboard du run
            if "leaderboard" in result and result["leaderboard"]:
                st.markdown(
                    '<div class="sec-header">🏆 Leaderboard de ce run</div>', unsafe_allow_html=True
                )
                lb_run = sorted(result["leaderboard"], key=lambda x: x["accuracy"], reverse=True)
                html = '<table class="lb-table"><thead><tr><th>Rang</th><th>Modèle</th><th>Accuracy</th><th>F1-macro</th><th>Train</th><th>Test</th></tr></thead><tbody>'
                for i, r in enumerate(lb_run, 1):
                    badge = (
                        '<span class="badge-gold">🥇</span>'
                        if i == 1
                        else f'<span class="badge-silver">#{i}</span>'
                    )
                    html += f'<tr><td>{badge}</td><td style="font-weight:600">{r["model"]}</td>'
                    html += f"<td>{r['accuracy']:.2%}</td><td>{r['f1_macro']:.2%}</td>"
                    html += f'<td style="color:#8b949e">{r.get("n_train", "–"):,}</td>'
                    html += f'<td style="color:#8b949e">{r.get("n_test", "–"):,}</td></tr>'
                html += "</tbody></table>"
                st.markdown(html, unsafe_allow_html=True)

            # Rafraîchit le cache pour voir les nouveaux modèles
            st.cache_data.clear()

        except Exception as e:
            st.error(f"Erreur pipeline : {e}")


# ══════════════════════════════════════════════
# TAB 7 — GCS
# ══════════════════════════════════════════════
with tab_gcs:
    st.markdown(
        f'<div class="sec-header">☁️ Explorateur GCS — {GCS_BUCKET_NAME}</div>',
        unsafe_allow_html=True,
    )

    g1, g2 = st.columns([1, 3])
    with g1:
        prefix = st.radio(
            "Dossier",
            ["raw/", "processed/", "models/", "predictions/"],
            label_visibility="collapsed",
        )
    with g2:
        blobs = list_blobs(prefix)
        if blobs:
            blob_df = pd.DataFrame({"Fichier": blobs})
            blob_df["Extension"] = blob_df["Fichier"].apply(lambda x: Path(x).suffix or "–")
            st.dataframe(blob_df, width="stretch", height=400)
            st.caption(f"{len(blobs)} fichier(s) dans `{prefix}`")
        else:
            st.info(f"Aucun fichier dans `{prefix}`.")
