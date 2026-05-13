"""
Routes FastAPI qui déclenchent les flows Prefect via l'API Prefect Cloud.

Architecture :
  POST /flows/pipeline            → déclenche kpi_pipeline_flow complet
  POST /flows/nombre-signalements → déclenche flow_nombre_signalements
  POST /flows/transmis            → déclenche flow_transmis_global
  POST /flows/lus-reponse         → déclenche flow_signalements_lus_reponse
  GET  /flows/status/{run_id}     → statut d'un flow run
  GET  /flows/latest-kpis         → derniers KPIs publiés depuis GCS
"""

from __future__ import annotations

import os
import uuid
from datetime import date
from typing import Any

from fastapi import APIRouter, HTTPException, Query
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


# ══════════════════════════════════════════════════════════════════════════════
# HELPER : déclencher un flow via Prefect API
# ══════════════════════════════════════════════════════════════════════════════


async def _trigger_deployment(deployment_name: str, parameters: dict) -> dict[str, Any]:
    """
    Déclenche un flow run via le client Prefect asynchrone.
    Retourne le flow_run créé.
    """
    try:
        from prefect.client.orchestration import get_client
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail=(
                "Prefect non installé dans l'environnement API. "
                "Ajoutez 'prefect>=3.0,<4' aux dépendances."
            ),
        ) from None

    try:
        async with get_client() as client:
            # 1. On récupère les déploiements
            deployments = await client.read_deployments()

            # 2. Correction ici : On cherche le déploiement correspondant
            deployment = None
            for d in deployments:
                # On vérifie si le nom du déploiement correspond au format "flow-name/deployment-name"
                # Ou simplement au nom du déploiement seul.
                if d.name == deployment_name.split("/")[-1]:
                    deployment = d
                    break

            if deployment is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Déploiement '{deployment_name}' introuvable.",
                )

            # 3. Création du flow run
            flow_run = await client.create_flow_run_from_deployment(
                deployment_id=deployment.id,
                parameters=parameters,
            )

            return {
                "flow_run_id": str(flow_run.id),
                "flow_run_name": flow_run.name,
                "deployment_name": deployment_name,
                "state": str(flow_run.state.type.value) if flow_run.state else "SCHEDULED",
                "message": f"Flow run '{flow_run.name}' déclenché avec succès.",
            }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur Prefect : {exc}") from exc


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════


@router.post(
    "/pipeline",
    response_model=FlowRunResponse,
    summary="Déclenche le pipeline KPI complet",
    description=(
        "Lance le flow Prefect `kpi-pipeline-flow` : extraction GCS → filtrage → "
        "calcul des 4 KPIs (signalements, transmis, lus, réponse) → publication artifact."
    ),
)
async def trigger_pipeline(body: PipelineRequest) -> FlowRunResponse:
    params = body.model_dump(exclude_none=True)
    if "reference_date" in params and params["reference_date"] is not None:
        params["reference_date"] = params["reference_date"].isoformat()

    result = await _trigger_deployment(PREFECT_DEPLOYMENT_PIPELINE, params)
    return FlowRunResponse(**result)


@router.post(
    "/nombre-signalements",
    response_model=FlowRunResponse,
    summary="Déclenche le KPI : nombre de signalements",
    description="Lance uniquement le flow calculant le nombre total de signalements.",
)
async def trigger_nombre_signalements(
    bucket_name: str = Query(default=GCS_BUCKET_NAME),
    prefix: str = Query(default="processed/"),
    period: str = Query(default="Depuis le début du mois"),
    region: str | None = Query(default=None),
) -> FlowRunResponse:
    params = {
        "bucket_name": bucket_name,
        "prefix": prefix,
        "period": period,
    }
    if region:
        params["region"] = region

    result = await _trigger_deployment(PREFECT_DEPLOYMENT_NOMBRE, params)
    return FlowRunResponse(**result)


@router.post(
    "/transmis",
    response_model=FlowRunResponse,
    summary="Déclenche les KPIs : signalements transmis et/ou transmis lus",
    description=(
        "Lance le flow `flow-transmis-global`. "
        "Paramètre `kpi_type` : 'transmis' | 'transmis_lus' | 'both'."
    ),
)
async def trigger_transmis(body: TransmisRequest) -> FlowRunResponse:
    params = body.model_dump(exclude_none=True)
    result = await _trigger_deployment(PREFECT_DEPLOYMENT_TRANSMIS, params)
    return FlowRunResponse(**result)


@router.post(
    "/lus-reponse",
    response_model=FlowRunResponse,
    summary="Déclenche le KPI : signalements lus ayant une réponse",
    description="Lance uniquement le flow calculant le taux de réponse aux signalements lus.",
)
async def trigger_lus_reponse(
    bucket_name: str = Query(default=GCS_BUCKET_NAME),
    prefix: str = Query(default="processed/"),
    period: str = Query(default="Depuis le début du mois"),
    region: str | None = Query(default=None),
) -> FlowRunResponse:
    params = {
        "bucket_name": bucket_name,
        "prefix": prefix,
        "period": period,
    }
    if region:
        params["region"] = region

    result = await _trigger_deployment(PREFECT_DEPLOYMENT_REPONSE, params)
    return FlowRunResponse(**result)


# ── GET status ────────────────────────────────────────────────────────────────


@router.get(
    "/status/{run_id}",
    summary="Statut d'un flow run Prefect",
    description="Retourne l'état courant d'un flow run (SCHEDULED, RUNNING, COMPLETED, FAILED…).",
)
async def get_flow_run_status(run_id: str) -> dict[str, Any]:
    try:
        from prefect.client.orchestration import get_client

        async with get_client() as client:
            flow_run = await client.read_flow_run(uuid.UUID(run_id))
            return {
                "flow_run_id": run_id,
                "flow_run_name": flow_run.name,
                "state": flow_run.state.type.value if flow_run.state else "UNKNOWN",
                "state_message": flow_run.state.message if flow_run.state else None,
                "start_time": flow_run.start_time.isoformat() if flow_run.start_time else None,
                "end_time": flow_run.end_time.isoformat() if flow_run.end_time else None,
            }
    except ImportError as err:
        raise HTTPException(status_code=503, detail="Prefect non disponible.") from err
    except ValueError as err:
        raise HTTPException(status_code=400, detail=f"UUID invalide : {run_id}") from err
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err
