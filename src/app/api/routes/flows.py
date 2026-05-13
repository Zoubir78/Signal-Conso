"""
Routes FastAPI qui déclenchent les flows Prefect via l'API Prefect Cloud.

Architecture :
  POST /flows/pipeline          → déclenche kpi_pipeline_flow complet
  POST /flows/nombre-signalements → déclenche flow_nombre_signalements
  POST /flows/transmis          → déclenche flow_transmis_global
  POST /flows/lus-reponse       → déclenche flow_signalements_lus_reponse
  GET  /flows/status/{run_id}   → statut d'un flow run
  GET  /flows/latest-kpis       → derniers KPIs publiés depuis GCS
"""

from __future__ import annotations

import os
from datetime import date
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()

# ── Config ────────────────────────────────────────────────────────────────────
GCS_BUCKET_NAME: str = os.getenv("GCS_BUCKET_NAME", "clean_complaints")
GCS_RESULTS_PREFIX: str = os.getenv("GCS_RESULTS_PREFIX", "prefect-results/")

# Nom du déploiement Prefect Cloud (défini dans prefect.yaml)
PREFECT_DEPLOYMENT_PIPELINE = os.getenv(
    "PREFECT_DEPLOYMENT_PIPELINE",
    "kpi-pipeline-flow/signal-conso-pipeline",
)
PREFECT_DEPLOYMENT_NOMBRE = os.getenv(
    "PREFECT_DEPLOYMENT_NOMBRE",
    "flow-nombre-signalements/kpi-nombre-signalements",
)
PREFECT_DEPLOYMENT_TRANSMIS = os.getenv(
    "PREFECT_DEPLOYMENT_TRANSMIS",
    "flow-transmis-global/kpi-signalements-transmis",
)
PREFECT_DEPLOYMENT_REPONSE = os.getenv(
    "PREFECT_DEPLOYMENT_REPONSE",
    "flow-signalements-lus-reponse/kpi-signalements-lus-reponse",
)


# ══════════════════════════════════════════════════════════════════════════════
# SCHÉMAS (Pydantic v2)
# ══════════════════════════════════════════════════════════════════════════════


class PipelineRequest(BaseModel):
    """Paramètres du pipeline KPI complet."""

    bucket_name: str = Field(default=GCS_BUCKET_NAME, description="Nom du bucket GCS")
    prefix: str = Field(default="processed/", description="Préfixe GCS des fichiers traités")
    reference_date: date | None = Field(
        default=None, description="Date de référence (défaut: aujourd'hui)"
    )
    period: str = Field(
        default="Depuis le début du mois",
        description="'Depuis le début du mois' | '7 derniers jours' | 'Toutes les données'",
    )
    region: str | None = Field(default=None, description="Filtre région (ex: 'Île-de-France')")
    department_label: str | None = Field(
        default=None, description="Filtre département (ex: '75 - Paris')"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "period": "7 derniers jours",
                "region": "Île-de-France",
            }
        }
    }


class TransmisRequest(BaseModel):
    """Paramètres du flow transmis (KPI 2 et/ou 3)."""

    kpi_type: str = Field(
        default="both",
        description="'transmis' | 'transmis_lus' | 'both'",
    )
    bucket_name: str = Field(default=GCS_BUCKET_NAME)
    prefix: str = Field(default="processed/")
    period: str = Field(default="Depuis le début du mois")
    region: str | None = None
    department_label: str | None = None


class FlowRunResponse(BaseModel):
    """Réponse après déclenchement d'un flow."""

    flow_run_id: str
    flow_run_name: str
    deployment_name: str
    state: str
    message: str


class KpiSummaryResponse(BaseModel):
    """Résumé des KPIs calculés."""

    status: str
    source: str | None = None
    computed_at: str | None = None
    kpis: list[dict[str, Any]] = []
