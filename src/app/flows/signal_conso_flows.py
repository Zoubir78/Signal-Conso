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

# -- Config --------------------------------------------------------------------
GCS_BUCKET_NAME: str = os.getenv("GCS_BUCKET_NAME", "clean_complaints")
GCS_PROCESSED_PREFIX: str = os.getenv("GCS_PROCESSED_PREFIX", "processed/")
GCS_RESULTS_PREFIX: str = os.getenv("GCS_RESULTS_PREFIX", "prefect-results/")
BOOL_TRUE_VALUES: frozenset[str] = frozenset({"1", "true", "t", "yes", "y", "oui", "vrai", "on"})
