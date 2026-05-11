import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from google.cloud import bigquery

import app.services.bigquery_service as bq_svc


@pytest.fixture
def mock_bq_client():
    """Fixture pour simuler le client BigQuery."""
    with patch("app.services.bigquery_service.bigquery.Client") as mock:
        yield mock


@pytest.fixture
def sample_df():
    """Fixture fournissant un DataFrame de test."""
    return pd.DataFrame(
        {
            "id": [1, 2],
            "data": [{"key": "val"}, ["item1", "item2"]],
            "date_str": ["2023-01-01", "2023-01-02"],
        }
    )


# --- Tests des fonctions de lecture ---


def test_read_source_table(mock_bq_client):
    # Configuration du mock
    mock_instance = mock_bq_client.return_value
    mock_query_job = MagicMock()
    mock_instance.query.return_value = mock_query_job
    mock_query_job.to_dataframe.return_value = pd.DataFrame({"col1": [1, 2]})

    df = bq_svc.read_source_table(limit=10)

    # Vérifications
    assert len(df) == 2
    mock_instance.query.assert_called_once()
    query_sent = mock_instance.query.call_args[0][0]
    assert "LIMIT 10" in query_sent
    assert bq_svc.SOURCE_TABLE in query_sent


# --- Tests de la logique interne d'alignement de schéma ---


def test_align_to_bq_schema():
    # Simulation d'un schéma BigQuery
    schema = [
        bigquery.SchemaField("id", "INTEGER"),
        bigquery.SchemaField("tags", "STRING"),  # On veut stocker du JSON ici
        bigquery.SchemaField("created_at", "DATE"),
    ]

    df = pd.DataFrame(
        {
            "id": ["1", "2"],
            "tags": [["a", "b"], {"key": "value"}],
            "created_at": ["2023-01-01", "2023-01-02"],
        }
    )

    aligned_df = bq_svc._align_to_bq_schema(df, schema)

    # Vérification des types
    assert aligned_df["id"].dtype == "Int64"
    assert isinstance(aligned_df["tags"].iloc[0], str)
    assert json.loads(aligned_df["tags"].iloc[0]) == ["a", "b"]
    assert (
        isinstance(aligned_df["created_at"].iloc[0], (pd.Timestamp, datetime)).date == "2023-01-01"
    )


# --- Tests de l'upload ---


@patch("app.services.bigquery_service.get_client")
@patch("app.services.bigquery_service._align_to_bq_schema")
def test_upload_dataframe_to_bigquery(mock_align, mock_get_client, sample_df):
    # Setup mocks
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    # Mock de la table existante pour le schéma
    mock_table = MagicMock()
    mock_table.schema = [bigquery.SchemaField("id", "INTEGER")]
    mock_client.get_table.return_value = mock_table

    mock_align.return_value = sample_df.copy()

    bq_svc.upload_dataframe_to_bigquery(sample_df)

    # Vérifications
    mock_client.get_table.assert_called_once()
    mock_client.load_table_from_dataframe.assert_called_once()

    # Vérification que la colonne technique a été ajoutée
    called_df = mock_client.load_table_from_dataframe.call_args[0][0]
    assert "_ingested_at" in called_df.columns


# --- Tests de l'export GCS ---


def test_export_mart_to_gcs(mock_bq_client):
    mock_instance = mock_bq_client.return_value
    mock_extract_job = MagicMock()
    mock_instance.extract_table.return_value = mock_extract_job

    uri = bq_svc.export_mart_to_gcs(file_format="JSON")

    # Vérifications
    assert uri.startswith("gs://")
    assert ".csv" in uri  # Note: votre code force l'extension .csv dans l'URI
    mock_instance.extract_table.assert_called_once()

    # Vérification du format passé à la config
    args, kwargs = mock_instance.extract_table.call_args
    job_config = kwargs.get("job_config")
    assert job_config.destination_format == bigquery.DestinationFormat.NEWLINE_DELIMITED_JSON


def test_export_table_to_gcs_alias(mock_bq_client):
    mock_instance = mock_bq_client.return_value

    bq_svc.export_table_to_gcs("proj", "dataset", "table", "bucket", "prefix")

    mock_instance.extract_table.assert_called_once()
    uri_called = mock_instance.extract_table.call_args[0][1]
    assert uri_called == "gs://bucket/prefix_*.csv"
