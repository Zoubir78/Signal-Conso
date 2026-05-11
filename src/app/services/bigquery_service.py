from __future__ import annotations

import pandas as pd
from google.cloud import bigquery

# ── Constantes projet ────────────────────────────────────────────────────────
PROJECT_ID = "tri-demandes-clients"
SOURCE_DATASET = "Complaints"
SOURCE_TABLE = "Signal_Conso"
MART_DATASET = "signalconso_dev_marts"  # signalconso_prod_marts en prod
MART_TABLE = "mart_signalconso"
GCS_BUCKET = "clean_complaints"


# ── Client ───────────────────────────────────────────────────────────────────


def get_client(project_id: str = PROJECT_ID) -> bigquery.Client:
    return bigquery.Client(project=project_id)


# ── Lecture ──────────────────────────────────────────────────────────────────


def read_source_table(
    limit: int | None = None,
    project_id: str = PROJECT_ID,
) -> pd.DataFrame:
    """
    Lit la table source SignalConso depuis BigQuery et retourne un DataFrame.

    Args:
        limit: Nombre maximum de lignes (None = tout lire).
        project_id: ID du projet GCP.
    """
    client = get_client(project_id)
    table_ref = f"{project_id}.{SOURCE_DATASET}.{SOURCE_TABLE}"

    query = f"SELECT * FROM `{table_ref}`"
    if limit:
        query += f" LIMIT {limit}"

    df = client.query(query).to_dataframe()
    print(f"Lu {len(df)} lignes depuis {table_ref}")
    return df


def read_mart_table(
    project_id: str = PROJECT_ID,
    dataset_id: str = MART_DATASET,
    table_id: str = MART_TABLE,
    filters: str | None = None,
) -> pd.DataFrame:
    """
    Lit le mart dbt final depuis BigQuery.

    Args:
        project_id: ID du projet GCP.
        dataset_id: Dataset du mart dbt (dev ou prod).
        table_id: Nom de la table mart.
        filters: Clause WHERE optionnelle (ex: "year = 2024 AND dep_code = '75'").
    """
    client = get_client(project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    query = f"SELECT * FROM `{table_ref}`"
    if filters:
        query += f" WHERE {filters}"

    df = client.query(query).to_dataframe()
    print(f"Lu {len(df)} lignes depuis {table_ref}")
    return df
