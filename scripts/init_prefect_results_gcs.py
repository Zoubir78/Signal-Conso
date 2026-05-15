"""
Initialise le dossier prefect-results/ dans GCS avec un fichier JSON seed.

Ce script est à lancer UNE SEULE FOIS pour créer le préfixe GCS
que le dashboard et les flows Prefect attendent.

"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime

from google.cloud import storage

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME") or "clean_complaints"
GCS_RESULTS_PREFIX = os.getenv("GCS_RESULTS_PREFIX", "prefect-results/")


def init_prefect_results_prefix() -> None:
    print(f"Connexion au bucket gs://{GCS_BUCKET_NAME}...")
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)

    # Vérifier si le préfixe existe déjà
    existing = list(bucket.list_blobs(prefix=GCS_RESULTS_PREFIX, max_results=1))
    if existing:
        print(f"Le préfixe gs://{GCS_BUCKET_NAME}/{GCS_RESULTS_PREFIX} existe déjà.")
        print(f"  → Premier fichier trouvé : {existing[0].name}")
        return

    # Créer un fichier seed JSON valide (structure attendue par le dashboard)
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
    print("  → Fichier créé : gs://{GCS_BUCKET_NAME}/{blob_name}")
    print()
    print("Prochaine étape : lancer le pipeline Prefect pour générer de vrais résultats.")
    print("  prefect deployment run 'kpi-pipeline-flow/signal-conso-pipeline'")


if __name__ == "__main__":
    init_prefect_results_prefix()
