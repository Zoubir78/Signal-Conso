"""
Routes FastAPI qui déclenchent les flows Prefect via l'API Prefect Cloud.

Architecture :
  POST /flows/pipeline            → déclenche kpi_pipeline_flow complet
  POST /flows/nombre-signalements → déclenche flow_nombre_signalements
  POST /flows/transmis            → déclenche flow_transmis_global
  POST /flows/lus-reponse         → déclenche flow_signalements_lus_reponse
  GET  /flows/runs                → liste les derniers runs depuis Prefect Cloud API
  GET  /flows/status/{run_id}     → statut d'un flow run
  GET  /flows/latest-kpis         → derniers KPIs publiés depuis GCS
"""

from __future__ import annotations

import json
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

# Noms des déploiements surveillés (auto-scheduled)
WATCHED_DEPLOYMENT_NAMES = [
    "signal-conso-pipeline",
    "kpi-nombre-signalements",
    "kpi-transmis-global",
    "kpi-signalements-lus-reponse",
    "signal-conso-daily-report",
]


# ══════════════════════════════════════════════════════════════════════════════
# SCHÉMAS
# ══════════════════════════════════════════════════════════════════════════════


class PipelineRequest(BaseModel):
    bucket_name: str = Field(default=GCS_BUCKET_NAME)
    prefix: str = Field(default="processed/")
    reference_date: date | None = Field(default=None)
    period: str = Field(default="Depuis le début du mois")
    region: str | None = Field(default=None)
    department_label: str | None = Field(default=None)

    model_config = {
        "json_schema_extra": {"example": {"period": "7 derniers jours", "region": "Île-de-France"}}
    }


class TransmisRequest(BaseModel):
    kpi_type: str = Field(default="both")
    bucket_name: str = Field(default=GCS_BUCKET_NAME)
    prefix: str = Field(default="processed/")
    period: str = Field(default="Depuis le début du mois")
    region: str | None = None
    department_label: str | None = None


class FlowRunResponse(BaseModel):
    flow_run_id: str
    flow_run_name: str
    deployment_name: str
    state: str
    message: str


class FlowRunSummary(BaseModel):
    """Résumé d'un run Prefect Cloud pour le dashboard."""

    flow_run_id: str
    flow_run_name: str
    deployment_name: str
    flow_name: str
    state: str
    state_type: str
    created: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    duration_seconds: float | None = None
    scheduled: bool = False


class KpiSummaryResponse(BaseModel):
    status: str
    source: str | None = None
    computed_at: str | None = None
    kpis: list[dict[str, Any]] = []


# ══════════════════════════════════════════════════════════════════════════════
# HELPER : déclencher un flow via Prefect API
# ══════════════════════════════════════════════════════════════════════════════


async def _trigger_deployment(deployment_name: str, parameters: dict) -> dict[str, Any]:
    try:
        from prefect.client.orchestration import get_client
    except ImportError:
        raise HTTPException(status_code=503, detail="Prefect non installé.") from None

    try:
        async with get_client() as client:
            deployments = await client.read_deployments()
            deployment = next(
                (d for d in deployments if d.name == deployment_name.split("/")[-1]),
                None,
            )
            if deployment is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Déploiement '{deployment_name}' introuvable.",
                )
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
# ROUTES — DÉCLENCHEMENT
# ══════════════════════════════════════════════════════════════════════════════


@router.post(
    "/pipeline", response_model=FlowRunResponse, summary="Déclenche le pipeline KPI complet"
)
async def trigger_pipeline(body: PipelineRequest) -> FlowRunResponse:
    params = body.model_dump(exclude_none=True)
    if params.get("reference_date") is not None:
        params["reference_date"] = params["reference_date"].isoformat()
    result = await _trigger_deployment(PREFECT_DEPLOYMENT_PIPELINE, params)
    return FlowRunResponse(**result)


@router.post(
    "/nombre-signalements", response_model=FlowRunResponse, summary="KPI : nombre de signalements"
)
async def trigger_nombre_signalements(
    bucket_name: str = Query(default=GCS_BUCKET_NAME),
    prefix: str = Query(default="processed/"),
    period: str = Query(default="Depuis le début du mois"),
    region: str | None = Query(default=None),
) -> FlowRunResponse:
    params = {"bucket_name": bucket_name, "prefix": prefix, "period": period}
    if region:
        params["region"] = region
    result = await _trigger_deployment(PREFECT_DEPLOYMENT_NOMBRE, params)
    return FlowRunResponse(**result)


@router.post(
    "/transmis", response_model=FlowRunResponse, summary="KPIs : transmis et/ou transmis lus"
)
async def trigger_transmis(body: TransmisRequest) -> FlowRunResponse:
    params = body.model_dump(exclude_none=True)
    result = await _trigger_deployment(PREFECT_DEPLOYMENT_TRANSMIS, params)
    return FlowRunResponse(**result)


@router.post("/lus-reponse", response_model=FlowRunResponse, summary="KPI : lus ayant une réponse")
async def trigger_lus_reponse(
    bucket_name: str = Query(default=GCS_BUCKET_NAME),
    prefix: str = Query(default="processed/"),
    period: str = Query(default="Depuis le début du mois"),
    region: str | None = Query(default=None),
) -> FlowRunResponse:
    params = {"bucket_name": bucket_name, "prefix": prefix, "period": period}
    if region:
        params["region"] = region
    result = await _trigger_deployment(PREFECT_DEPLOYMENT_REPONSE, params)
    return FlowRunResponse(**result)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE — LISTE DES RUNS PREFECT CLOUD (auto-scheduled + manuels)
# ══════════════════════════════════════════════════════════════════════════════


@router.get(
    "/runs",
    response_model=list[FlowRunSummary],
    summary="Derniers runs Prefect Cloud",
    description=(
        "Interroge Prefect Cloud API et retourne les derniers runs "
        "(auto-schedulés et manuels) pour tous les déploiements Signal Conso."
    ),
)
async def get_flow_runs(
    limit: int = Query(default=20, ge=1, le=100, description="Nombre de runs à retourner"),
    deployment_name: str | None = Query(default=None, description="Filtrer par déploiement"),
) -> list[FlowRunSummary]:
    try:
        from prefect.client.orchestration import get_client
        from prefect.client.schemas.filters import (
            FlowRunFilter,
            FlowRunFilterDeploymentId,
        )
        from prefect.client.schemas.sorting import FlowRunSort
    except ImportError:
        raise HTTPException(status_code=503, detail="Prefect non disponible.") from None

    try:
        async with get_client() as client:
            # Récupérer les déploiements Signal Conso
            all_deployments = await client.read_deployments()
            watched = [
                d
                for d in all_deployments
                if d.name in WATCHED_DEPLOYMENT_NAMES
                or (deployment_name and d.name == deployment_name)
            ]

            if not watched:
                return []

            deployment_ids = [d.id for d in watched]
            dep_id_to_name = {str(d.id): d.name for d in watched}

            # Lire les flow runs filtrés par deployment_id
            flow_runs = await client.read_flow_runs(
                flow_run_filter=FlowRunFilter(
                    deployment_id=FlowRunFilterDeploymentId(any_=deployment_ids)
                ),
                sort=FlowRunSort.EXPECTED_START_TIME_DESC,
                limit=limit,
            )

            results: list[FlowRunSummary] = []
            for run in flow_runs:
                dep_name = dep_id_to_name.get(str(run.deployment_id), "—")
                state_type = run.state.type.value if run.state else "UNKNOWN"
                state_name = run.state.name if run.state else "Unknown"

                # Calcul durée
                duration = None
                if run.start_time and run.end_time:
                    duration = (run.end_time - run.start_time).total_seconds()

                results.append(
                    FlowRunSummary(
                        flow_run_id=str(run.id),
                        flow_run_name=run.name,
                        deployment_name=dep_name,
                        flow_name=run.flow_id and str(run.flow_id) or "—",
                        state=state_name,
                        state_type=state_type,
                        created=run.created.isoformat() if run.created else None,
                        start_time=run.start_time.isoformat() if run.start_time else None,
                        end_time=run.end_time.isoformat() if run.end_time else None,
                        duration_seconds=duration,
                        scheduled=run.auto_scheduled if hasattr(run, "auto_scheduled") else False,
                    )
                )

            return results

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur Prefect API : {exc}") from exc


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE — STATUT D'UN RUN
# ══════════════════════════════════════════════════════════════════════════════


@router.get("/status/{run_id}", summary="Statut d'un flow run Prefect")
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


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE — DERNIERS KPIs DEPUIS GCS
# ══════════════════════════════════════════════════════════════════════════════


@router.get("/latest-kpis", response_model=KpiSummaryResponse, summary="Derniers KPIs calculés")
def get_latest_kpis() -> KpiSummaryResponse:
    try:
        from datetime import UTC, datetime

        from google.cloud import storage as gcs

        client = gcs.Client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blobs = list(bucket.list_blobs(prefix=GCS_RESULTS_PREFIX))

        if not blobs:
            raise HTTPException(
                status_code=404,
                detail=f"Aucun résultat dans gs://{GCS_BUCKET_NAME}/{GCS_RESULTS_PREFIX}.",
            )

        fallback = datetime.min.replace(tzinfo=UTC)
        latest = max(blobs, key=lambda b: b.updated or fallback)
        data = json.loads(latest.download_as_bytes().decode("utf-8"))

        return KpiSummaryResponse(
            status=data.get("status", "unknown"),
            source=data.get("source"),
            computed_at=data.get("computed_at"),
            kpis=data.get("kpis", []),
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur GCS : {exc}") from exc
