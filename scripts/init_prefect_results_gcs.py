"""
Deux modes d'utilisation :

  1. Script one-shot (CLI) :
       python init_prefect_results_gcs.py
       → Crée le dossier prefect-results/ dans GCS si absent.

  2. Module importé par le dashboard (à chaque chargement) :
       from app.scripts.init_prefect_results_gcs import sync_prefect_runs_to_gcs
       sync_prefect_runs_to_gcs()
       → Interroge Prefect Cloud API, récupère les derniers runs complétés
         et écrit/met à jour les artefacts JSON dans GCS prefect-results/.
"""

from __future__ import annotations

import json
import os
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from google.cloud import storage

GCS_BUCKET_NAME: str = os.getenv("GCS_BUCKET_NAME") or "clean_complaints"
GCS_RESULTS_PREFIX: str = os.getenv("GCS_RESULTS_PREFIX") or "prefect-results/"

WATCHED_DEPLOYMENT_NAMES = [
    "signal-conso-pipeline",
    "kpi-nombre-signalements",
    "kpi-transmis-global",
    "kpi-signalements-lus-reponse",
    "signal-conso-daily-report",
]


# ══════════════════════════════════════════════════════════════════════════════
# GCS HELPER
# ══════════════════════════════════════════════════════════════════════════════


def _gcs_client() -> storage.Client:
    return storage.Client()


# ══════════════════════════════════════════════════════════════════════════════
# 1. INITIALISATION (one-shot)
# ══════════════════════════════════════════════════════════════════════════════


def init_prefect_results_prefix() -> None:
    """
    Crée le préfixe GCS prefect-results/ avec un fichier seed si absent.
    À appeler une seule fois au bootstrap du projet.
    """
    print(f"Connexion au bucket gs://{GCS_BUCKET_NAME}...")
    client = _gcs_client()
    bucket = client.bucket(GCS_BUCKET_NAME)

    existing = list(bucket.list_blobs(prefix=GCS_RESULTS_PREFIX, max_results=1))
    if existing:
        print(f"Le préfixe gs://{GCS_BUCKET_NAME}/{GCS_RESULTS_PREFIX} existe déjà.")
        print(f"  → Premier fichier trouvé : {existing[0].name}")
        return

    now = datetime.now(UTC)
    seed = {
        "status": "init",
        "source": None,
        "computed_at": now.isoformat(),
        "flow_name": "kpi-pipeline-flow",
        "deployment_name": "signal-conso-pipeline",
        "deployment_id": None,
        "flow_run_id": None,
        "kpis": [
            {
                "kpi": "nombre_signalements",
                "label": "Nombre de signalements",
                "value": 0,
                "unit": "signalements",
                "computed_at": now.isoformat(),
            },
            {
                "kpi": "signalements_transmis",
                "label": "Part de signalements transmis",
                "value": 0.0,
                "value_pct": "0.00%",
                "numerator": 0,
                "denominator": 0,
                "computed_at": now.isoformat(),
            },
            {
                "kpi": "signalements_transmis_lus",
                "label": "Part des signalements transmis lus",
                "value": 0.0,
                "value_pct": "0.00%",
                "numerator": 0,
                "denominator": 0,
                "computed_at": now.isoformat(),
            },
            {
                "kpi": "signalements_lus_reponse",
                "label": "Part des signalements lus ayant une réponse",
                "value": 0.0,
                "value_pct": "0.00%",
                "numerator": 0,
                "denominator": 0,
                "computed_at": now.isoformat(),
            },
        ],
        "rows": [],
        "note": "Fichier d'initialisation — sera écrasé par le premier flow Prefect.",
    }

    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    blob_name = f"{GCS_RESULTS_PREFIX.rstrip('/')}/prefect_summary_{timestamp}_init.json"
    blob = bucket.blob(blob_name)
    blob.upload_from_string(
        json.dumps(seed, ensure_ascii=False, indent=2),
        content_type="application/json",
    )

    print("Préfixe initialisé avec succès !")
    print(f"  → Fichier créé : gs://{GCS_BUCKET_NAME}/{blob_name}")
    print()
    print("Prochaine étape : lancer le pipeline Prefect.")
    print("  prefect deployment run 'kpi-pipeline-flow/signal-conso-pipeline'")


# ══════════════════════════════════════════════════════════════════════════════
# 2. SYNCHRONISATION (appelée à chaque chargement du dashboard)
# ══════════════════════════════════════════════════════════════════════════════


def sync_prefect_runs_to_gcs(limit: int = 10) -> list[dict[str, Any]]:
    """
    Interroge Prefect Cloud API, récupère les derniers runs COMPLETED
    des déploiements Signal Conso et écrit les artefacts JSON dans GCS.

    Appelée à chaque chargement du dashboard (via @st.cache_data(ttl=60)).

    Returns:
        list[dict] : runs synchronisés (vide si Prefect API indisponible)
    """
    try:
        import asyncio

        return asyncio.run(_async_sync_prefect_runs_to_gcs(limit=limit))
    except Exception as exc:
        print(f"[sync_prefect_runs_to_gcs] Erreur : {exc}")
        return []


async def _async_sync_prefect_runs_to_gcs(limit: int = 10) -> list[dict[str, Any]]:
    """Version async de sync_prefect_runs_to_gcs."""
    try:
        from prefect.client.orchestration import get_client
        from prefect.client.schemas.filters import (
            FlowRunFilter,
            FlowRunFilterDeploymentId,
            FlowRunFilterStateName,
        )
        from prefect.client.schemas.sorting import FlowRunSort
    except ImportError:
        print("[sync] prefect non installé — synchronisation ignorée.")
        return []

    synced: list[dict[str, Any]] = []

    try:
        async with get_client() as client:
            # ── 1. Récupérer les déploiements Signal Conso ────────────────
            all_deployments = await client.read_deployments()
            watched = [d for d in all_deployments if d.name in WATCHED_DEPLOYMENT_NAMES]

            if not watched:
                print("[sync] Aucun déploiement Signal Conso trouvé dans Prefect Cloud.")
                return []

            deployment_ids = [d.id for d in watched]
            dep_id_to_name = {str(d.id): d.name for d in watched}

            # ── 2. Lire les derniers runs COMPLETED ───────────────────────
            flow_runs = await client.read_flow_runs(
                flow_run_filter=FlowRunFilter(
                    deployment_id=FlowRunFilterDeploymentId(any_=deployment_ids),
                    state=FlowRunFilterStateName(any_=["Completed", "Failed", "Crashed"]),
                ),
                sort=FlowRunSort.EXPECTED_START_TIME_DESC,
                limit=limit,
            )

            # ── 3. Pour chaque run, construire un artefact JSON et l'écrire dans GCS ──
            gcs = _gcs_client()
            bucket = gcs.bucket(GCS_BUCKET_NAME)

            for run in flow_runs:
                dep_name = dep_id_to_name.get(str(run.deployment_id), "unknown")
                state_type = run.state.type.value if run.state else "UNKNOWN"
                state_name = run.state.name if run.state else "Unknown"

                duration = None
                if run.start_time and run.end_time:
                    duration = (run.end_time - run.start_time).total_seconds()

                summary: dict[str, Any] = {
                    "status": "success" if state_type == "COMPLETED" else state_type.lower(),
                    "state": state_name,
                    "state_type": state_type,
                    "flow_run_id": str(run.id),
                    "flow_run_name": run.name,
                    "flow_name": "kpi-pipeline-flow",
                    "deployment_name": dep_name,
                    "deployment_id": str(run.deployment_id),
                    "computed_at": (
                        run.end_time or run.start_time or datetime.now(UTC)
                    ).isoformat(),
                    "started_at": run.start_time.isoformat() if run.start_time else None,
                    "finished_at": run.end_time.isoformat() if run.end_time else None,
                    "duration_seconds": duration,
                    "scheduled": getattr(run, "auto_scheduled", False),
                    "kpis": [],  # Les KPIs sont écrits par publish_kpi_results_task
                    "source": None,
                }

                # Nom du fichier : évite les doublons en utilisant le flow_run_id
                blob_name = f"{GCS_RESULTS_PREFIX.rstrip('/')}/prefect_run_{run.id}.json"
                blob = bucket.blob(blob_name)

                # N'écraser que si le fichier n'existe pas encore (évite les requêtes inutiles)
                if not blob.exists():
                    summary["results_blob"] = blob_name
                    blob.upload_from_string(
                        json.dumps(summary, ensure_ascii=False, indent=2),
                        content_type="application/json",
                    )
                    print(
                        f"[sync] Run {run.name} ({dep_name}) → gs://{GCS_BUCKET_NAME}/{blob_name}"
                    )
                else:
                    # Relire le fichier existant pour retourner ses données
                    with suppress(Exception):
                        summary = json.loads(blob.download_as_text())

                    synced.append(summary)

    except Exception as exc:
        print(f"[sync] Erreur Prefect Cloud API : {exc}")

    return synced


# ══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if "--sync" in sys.argv:
        print("Mode synchronisation — récupération des derniers runs Prefect Cloud...")
        results = sync_prefect_runs_to_gcs(limit=20)
        print(f"  → {len(results)} run(s) synchronisé(s) dans GCS.")
    else:
        print("Mode initialisation — création du préfixe GCS...")
        init_prefect_results_prefix()
