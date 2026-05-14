from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════════════════════════


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "OK"


# ══════════════════════════════════════════════════════════════════════════════
# TICKETS
# ══════════════════════════════════════════════════════════════════════════════


@patch("app.api.routes.tickets.model")
def test_predict_ticket_returns_category(mock_model):
    mock_model.model = MagicMock()
    mock_model.predict_with_proba.return_value = ("Alimentation", 0.92)

    response = client.post("/tickets/predict", json={"text": "produit avarié"})

    assert response.status_code == 200
    data = response.json()
    assert data["predicted_category"] == "Alimentation"
    assert data["confidence"] == pytest.approx(0.92)


@patch("app.api.routes.tickets.model")
def test_predict_ticket_loads_model_if_none(mock_model):
    mock_model.model = None
    mock_model.predict_with_proba.return_value = ("Hygiène", 0.80)

    response = client.post("/tickets/predict", json={"text": "restaurant sale"})

    mock_model.load.assert_called_once()
    assert response.status_code == 200


@patch("app.api.routes.tickets.model")
def test_predict_ticket_returns_503_on_file_not_found(mock_model):
    mock_model.model = MagicMock()
    mock_model.predict_with_proba.side_effect = FileNotFoundError("model not found")

    response = client.post("/tickets/predict", json={"text": "test"})

    assert response.status_code == 503


@patch("app.api.routes.tickets.model")
def test_predict_ticket_returns_500_on_unexpected_error(mock_model):
    mock_model.model = MagicMock()
    mock_model.predict_with_proba.side_effect = RuntimeError("crash")

    response = client.post("/tickets/predict", json={"text": "test"})

    assert response.status_code == 500


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════

FAKE_PREDICTION = {
    "id": "pred-123",
    "input_text": "produit périmé",
    "clean_text": "produit perime",
    "predicted_category": "Alimentation",
    "confidence": 0.91,
    "model_version": "logreg-v1",
    "created_at": "2024-01-01T00:00:00",
}


@patch("app.api.routes.predictions.upload_json_to_gcs")
@patch("app.api.routes.predictions.get_model")
def test_create_prediction_returns_201(mock_get_model, mock_upload):
    mock_model = MagicMock()
    mock_model.predict_with_proba.return_value = ("Alimentation", 0.91)
    mock_get_model.return_value = mock_model

    response = client.post("/predictions", json={"text": "produit périmé"})

    assert response.status_code == 200
    data = response.json()
    assert data["predicted_category"] == "Alimentation"
    assert "id" in data
    mock_upload.assert_called_once()


@patch("app.api.routes.predictions.find_prediction_in_bucket")
def test_get_prediction_returns_data(mock_find):
    mock_find.return_value = FAKE_PREDICTION

    response = client.get("/predictions/pred-123")

    assert response.status_code == 200
    assert response.json()["id"] == "pred-123"


@patch("app.api.routes.predictions.find_prediction_in_bucket")
def test_get_prediction_returns_404_when_not_found(mock_find):
    mock_find.return_value = None

    response = client.get("/predictions/unknown-id")

    assert response.status_code == 404


@patch("app.api.routes.predictions.get_model")
def test_create_prediction_returns_503_on_missing_model(mock_get_model):
    mock_get_model.side_effect = FileNotFoundError("model.joblib not found")

    response = client.post("/predictions", json={"text": "test"})

    assert response.status_code == 503


# ══════════════════════════════════════════════════════════════════════════════
# FLOWS
# ══════════════════════════════════════════════════════════════════════════════

FAKE_FLOW_RUN_ID = str(uuid.uuid4())

FAKE_TRIGGER_RESULT = {
    "flow_run_id": FAKE_FLOW_RUN_ID,
    "flow_run_name": "prudent-seagull",
    "deployment_name": "kpi-pipeline-flow/signal-conso-pipeline",
    "state": "SCHEDULED",
    "message": "Flow run 'prudent-seagull' déclenché avec succès.",
}


@patch("app.api.routes.flows._trigger_deployment", new_callable=AsyncMock)
def test_trigger_pipeline_returns_flow_run(mock_trigger):
    mock_trigger.return_value = FAKE_TRIGGER_RESULT

    response = client.post("/flows/pipeline", json={"period": "7 derniers jours"})

    assert response.status_code == 200
    data = response.json()
    assert data["state"] == "SCHEDULED"
    assert data["flow_run_id"] == FAKE_FLOW_RUN_ID
    mock_trigger.assert_awaited_once()


@patch("app.api.routes.flows._trigger_deployment", new_callable=AsyncMock)
def test_trigger_nombre_signalements(mock_trigger):
    mock_trigger.return_value = {
        **FAKE_TRIGGER_RESULT,
        "deployment_name": "flow-nombre-signalements/kpi-nombre-signalements",
    }

    response = client.post("/flows/nombre-signalements")

    assert response.status_code == 200
    mock_trigger.assert_awaited_once()


@patch("app.api.routes.flows._trigger_deployment", new_callable=AsyncMock)
def test_trigger_transmis_with_kpi_type(mock_trigger):
    mock_trigger.return_value = {
        **FAKE_TRIGGER_RESULT,
        "deployment_name": "flow-transmis-global/kpi-signalements-transmis",
    }

    response = client.post("/flows/transmis", json={"kpi_type": "transmis"})

    assert response.status_code == 200
    _, call_params = mock_trigger.call_args
    # kpi_type doit être transmis dans les paramètres
    assert mock_trigger.await_args.args[1].get("kpi_type") == "transmis"


@patch("app.api.routes.flows._trigger_deployment", new_callable=AsyncMock)
def test_trigger_lus_reponse(mock_trigger):
    mock_trigger.return_value = {
        **FAKE_TRIGGER_RESULT,
        "deployment_name": "flow-signalements-lus-reponse/kpi-signalements-lus-reponse",
    }

    response = client.post("/flows/lus-reponse")

    assert response.status_code == 200
    mock_trigger.assert_awaited_once()


@patch("app.api.routes.flows._trigger_deployment", new_callable=AsyncMock)
def test_trigger_pipeline_propagates_404(mock_trigger):
    from fastapi import HTTPException

    mock_trigger.side_effect = HTTPException(status_code=404, detail="Déploiement introuvable.")

    response = client.post("/flows/pipeline", json={})

    assert response.status_code == 404


def test_get_flow_run_status_invalid_uuid():
    response = client.get("/flows/status/not-a-uuid")
    assert response.status_code == 400


@patch("app.api.routes.flows._trigger_deployment", new_callable=AsyncMock)
def test_get_flow_run_status_valid_uuid(mock_trigger):
    # ✅ Patcher prefect.client.orchestration.get_client, pas le module flows
    with patch("prefect.client.orchestration.get_client") as mock_get_client:
        mock_flow_run = MagicMock()
        mock_flow_run.name = "prudent-seagull"
        mock_flow_run.state.type.value = "COMPLETED"
        mock_flow_run.state.message = None
        mock_flow_run.start_time = datetime(2024, 1, 1, tzinfo=UTC)
        mock_flow_run.end_time = datetime(2024, 1, 1, 1, tzinfo=UTC)

        mock_client = AsyncMock()
        mock_client.read_flow_run = AsyncMock(return_value=mock_flow_run)
        mock_get_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_get_client.return_value.__aexit__ = AsyncMock(return_value=None)

        response = client.get(f"/flows/status/{FAKE_FLOW_RUN_ID}")

    assert response.status_code == 200
    assert response.json()["state"] == "COMPLETED"


def test_get_latest_kpis_returns_summary():
    # ✅ Mocker google.cloud.storage.Client, pas la fonction route
    fake_blob = MagicMock()
    fake_blob.updated = datetime(2024, 1, 1, tzinfo=UTC)
    fake_blob.download_as_bytes.return_value = json.dumps(
        {
            "status": "success",
            "source": "processed/data.csv",
            "computed_at": "2024-01-01T00:00:00Z",
            "kpis": [{"kpi": "nombre_signalements", "value": 42}],
        }
    ).encode()

    mock_bucket = MagicMock()
    mock_bucket.list_blobs.return_value = [fake_blob]

    mock_gcs_client = MagicMock()
    mock_gcs_client.bucket.return_value = mock_bucket

    with patch("google.cloud.storage.Client", return_value=mock_gcs_client):
        response = client.get("/flows/latest-kpis")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["kpis"]) == 1
