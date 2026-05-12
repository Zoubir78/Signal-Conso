"""
Orchestration Prefect des KPIs Signal Conso depuis Google Cloud Storage.

Flows orchestrés :
  1. nombre_signalements       — Comptage total des signalements
  2. signalements_transmis     — Part des signalements transmis aux entreprises
  3. signalements_transmis_lus — Part des signalements transmis qui ont été lus
  4. signalements_lus_reponse   — Part des signalements lus ayant reçu une réponse

Usage :
  # Lancement ponctuel
  python signal_conso_flows.py

  # Déploiement avec schedule
  prefect deploy --all
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from io import BytesIO
from typing import Any

import pandas as pd
from google.cloud import storage
from prefect import get_run_logger, task

# -- Config --------------------------------------------------------------------
GCS_BUCKET_NAME: str = os.getenv("GCS_BUCKET_NAME", "clean_complaints")
GCS_PROCESSED_PREFIX: str = os.getenv("GCS_PROCESSED_PREFIX", "processed/")
GCS_RESULTS_PREFIX: str = os.getenv("GCS_RESULTS_PREFIX", "prefect-results/")
BOOL_TRUE_VALUES: frozenset[str] = frozenset({"1", "true", "t", "yes", "y", "oui", "vrai", "on"})


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return bool(isinstance(value, str) and not value.strip())


def _to_bool(value: Any) -> bool:
    if _is_missing(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in BOOL_TRUE_VALUES


def _bool_series(series: pd.Series) -> pd.Series:
    return series.apply(_to_bool)


def _now_iso() -> str:
    """Heure UTC courante en ISO 8601 (datetime.utcnow() deprecie en Python 3.12)."""
    return datetime.now(datetime.UTC).isoformat()


# ═══════════════════════════════════════════════════════════════════════════════
# TASKS GCS
# ═══════════════════════════════════════════════════════════════════════════════


@task(
    name="get-gcs-client",
    description="Initialise le client Google Cloud Storage.",
    retries=2,
    retry_delay_seconds=5,
    tags=["gcs", "infra"],
    persist_result=False,
)
def get_gcs_client_task() -> storage.Client:
    logger = get_run_logger()
    logger.info("Initialisation du client GCS.")
    return storage.Client()


@task(
    name="find-latest-blob",
    description="Trouve le blob le plus recent dans le prefix GCS.",
    retries=3,
    retry_delay_seconds=10,
    tags=["gcs", "extract"],
)
def find_latest_blob_task(client: storage.Client, bucket_name: str, prefix: str) -> str | None:
    logger = get_run_logger()
    logger.info(f"Recherche du blob le plus recent dans gs://{bucket_name}/{prefix}")

    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    if not blobs:
        logger.warning(f"Aucun blob trouve dans gs://{bucket_name}/{prefix}")
        return None

    fallback_dt = datetime.min.replace(tzinfo=datetime.UTC)
    latest = max(blobs, key=lambda b: b.updated or fallback_dt)
    logger.info(f"Blob le plus recent : {latest.name} (mis a jour le {latest.updated})")
    return latest.name


@task(
    name="download-dataset",
    description="Telecharge le dataset CSV depuis GCS et le charge en DataFrame.",
    retries=3,
    retry_delay_seconds=15,
    tags=["gcs", "extract"],
    persist_result=False,
)
def download_dataset_task(client: storage.Client, bucket_name: str, blob_name: str) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info(f"Telechargement de gs://{bucket_name}/{blob_name}")

    blob = client.bucket(bucket_name).blob(blob_name)
    data = blob.download_as_bytes()
    df = pd.read_csv(BytesIO(data))

    logger.info(f"Dataset charge : {len(df)} lignes x {len(df.columns)} colonnes")
    return df


@task(
    name="preprocess-dataframe",
    description="Normalise les types (dates, booleens) du DataFrame brut.",
    tags=["transform"],
    persist_result=False,
)
def preprocess_task(df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()

    if "creationdate" in df.columns:
        df["creationdate"] = pd.to_datetime(df["creationdate"], errors="coerce")
        logger.info("Colonne 'creationdate' convertie en datetime.")

    for bool_col in ["signalement_transmis", "signalement_lu", "signalement_reponse"]:
        if bool_col in df.columns:
            df[bool_col] = _bool_series(df[bool_col])

    if "department_label" not in df.columns and (
        "dep_code" in df.columns or "dep_name" in df.columns
    ):

        def _dept_label(row: pd.Series) -> str:
            parts: list[str] = []
            if "dep_code" in row.index and not _is_missing(row.get("dep_code")):
                code = str(row.get("dep_code")).strip().replace(".0", "")
                if code.isdigit() and len(code) <= 2:
                    code = code.zfill(2)
                parts.append(code)
            if "dep_name" in row.index and not _is_missing(row.get("dep_name")):
                parts.append(str(row.get("dep_name")).strip())
            if not parts:
                return "Inconnu"
            return " – ".join(parts)

        df["department_label"] = df.apply(_dept_label, axis=1)
        logger.info("Colonne 'department_label' créée pour harmoniser le filtrage département.")
    elif "department_label" in df.columns:
        df["department_label"] = df["department_label"].astype(str).str.strip()

    logger.info("Pre-traitement termine.")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# TASKS FILTRAGE
# ═══════════════════════════════════════════════════════════════════════════════


@task(
    name="apply-temporal-filter",
    description="Filtre le DataFrame sur une période temporelle.",
    tags=["filter"],
)
def apply_temporal_filter_task(
    df: pd.DataFrame,
    reference_date: date | None = None,
    period: str = "Depuis le début du mois",
) -> pd.DataFrame:
    logger = get_run_logger()

    if "creationdate" not in df.columns:
        logger.warning("Colonne 'creationdate' absente — pas de filtre temporel.")
        return df

    ref = reference_date or date.today()
    df = df[df["creationdate"].notna()].copy()

    if period == "Depuis le début du mois":
        start = ref.replace(day=1)
        end = ref
    elif period == "7 derniers jours":
        start = ref - timedelta(days=6)
        end = ref
    else:
        logger.info("Période = 'Toutes les données' — pas de filtre temporel.")
        return df

    filtered = df[(df["creationdate"].dt.date >= start) & (df["creationdate"].dt.date <= end)]

    logger.info(f"Filtre temporel [{start} → {end}] : {len(df)} → {len(filtered)} lignes.")
    return filtered


@task(
    name="apply-geo-filter",
    description="Filtre le DataFrame par région et/ou département.",
    tags=["filter"],
)
def apply_geo_filter_task(
    df: pd.DataFrame,
    region: str | None = None,
    department_label: str | None = None,
) -> pd.DataFrame:
    logger = get_run_logger()

    if region and "reg_name" in df.columns:
        before = len(df)
        df = df[df["reg_name"].astype(str) == region]
        logger.info(f"Filtre région '{region}' : {before} → {len(df)} lignes.")

    if department_label and "department_label" in df.columns:
        before = len(df)
        df = df[df["department_label"].astype(str) == department_label]
        logger.info(f"Filtre département '{department_label}' : {before} → {len(df)} lignes.")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# TASKS KPI
# ═══════════════════════════════════════════════════════════════════════════════


@task(
    name="kpi-nombre-signalements",
    description="Calcule le nombre total de signalements.",
    tags=["kpi"],
)
def kpi_nombre_signalements_task(df: pd.DataFrame) -> dict[str, Any]:
    logger = get_run_logger()
    total = len(df)
    logger.info(f"[KPI] Nombre de signalements = {total}")
    return {
        "kpi": "nombre_signalements",
        "label": "Nombre de signalements",
        "value": total,
        "unit": "signalements",
        "computed_at": _now_iso(),
    }


@task(
    name="kpi-signalements-transmis",
    description="Calcule la part des signalements transmis aux entreprises.",
    tags=["kpi"],
)
def kpi_signalements_transmis_task(df: pd.DataFrame) -> dict[str, Any]:
    logger = get_run_logger()

    total = len(df)
    if "signalement_transmis" not in df.columns:
        logger.warning("Colonne 'signalement_transmis' absente.")
        return {
            "kpi": "signalements_transmis",
            "label": "Part de signalements transmis",
            "value": None,
            "numerator": None,
            "denominator": total,
            "error": "Colonne manquante",
            "computed_at": _now_iso(),
        }

    transmitted = int(df["signalement_transmis"].sum())
    rate = transmitted / total if total else 0.0

    logger.info(f"[KPI] Signalements transmis = {transmitted}/{total} = {rate:.2%}")
    return {
        "kpi": "signalements_transmis",
        "label": "Part de signalements transmis",
        "value": round(rate, 6),
        "value_pct": f"{rate:.2%}",
        "numerator": transmitted,
        "denominator": total,
        "computed_at": _now_iso(),
    }


@task(
    name="kpi-signalements-transmis-lus",
    description="Calcule la part des signalements transmis qui ont été lus.",
    tags=["kpi"],
)
def kpi_signalements_transmis_lus_task(df: pd.DataFrame) -> dict[str, Any]:
    logger = get_run_logger()

    missing_cols = [c for c in ["signalement_transmis", "signalement_lu"] if c not in df.columns]
    if missing_cols:
        logger.warning(f"Colonnes manquantes : {missing_cols}")
        return {
            "kpi": "signalements_transmis_lus",
            "label": "Part des signalements transmis lus",
            "value": None,
            "error": f"Colonnes manquantes : {missing_cols}",
            "computed_at": _now_iso(),
        }

    transmitted = int(df["signalement_transmis"].sum())
    transmitted_df = df[df["signalement_transmis"]]
    read = int(transmitted_df["signalement_lu"].sum())
    rate = read / transmitted if transmitted else 0.0

    logger.info(f"[KPI] Signalements transmis lus = {read}/{transmitted} = {rate:.2%}")
    return {
        "kpi": "signalements_transmis_lus",
        "label": "Part des signalements transmis lus",
        "value": round(rate, 6),
        "value_pct": f"{rate:.2%}",
        "numerator": read,
        "denominator": transmitted,
        "computed_at": _now_iso(),
    }


@task(
    name="kpi-signalements-lus-reponse",
    description="Calcule la part des signalements lus ayant reçu une réponse.",
    tags=["kpi"],
)
def kpi_signalements_lus_reponse_task(df: pd.DataFrame) -> dict[str, Any]:
    logger = get_run_logger()

    missing_cols = [c for c in ["signalement_lu", "signalement_reponse"] if c not in df.columns]
    if missing_cols:
        logger.warning(f"Colonnes manquantes : {missing_cols}")
        return {
            "kpi": "signalements_lus_reponse",
            "label": "Part des signalements lus ayant une réponse",
            "value": None,
            "error": f"Colonnes manquantes : {missing_cols}",
            "computed_at": _now_iso(),
        }

    read = int(df["signalement_lu"].sum())
    read_df = df[df["signalement_lu"]]
    response = int(read_df["signalement_reponse"].sum())
    rate = response / read if read else 0.0

    logger.info(f"[KPI] Signalements lus avec réponse = {response}/{read} = {rate:.2%}")
    return {
        "kpi": "signalements_lus_reponse",
        "label": "Part des signalements lus ayant une réponse",
        "value": round(rate, 6),
        "value_pct": f"{rate:.2%}",
        "numerator": response,
        "denominator": read,
        "computed_at": _now_iso(),
    }
