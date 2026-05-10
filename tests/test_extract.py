from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from app.ingestion.extract import extract_from_signalconso_api

API_URL = "https://data.economie.gouv.fr/api/explore/v2.1/catalog/datasets/signalconso/records"


def _make_response(records: list, status_code: int = 200) -> MagicMock:
    """Construit un faux objet requests.Response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = {"results": records}
    mock_resp.raise_for_status = MagicMock()
    if status_code >= 400:
        mock_resp.raise_for_status.side_effect = requests.HTTPError(response=mock_resp)
    return mock_resp


def _make_record(i: int) -> dict:
    return {"id": i, "category": f"cat_{i}", "status": "open"}


# ══════════════════════════════════════════════════════════════════════════════
# Cas nominaux
# ══════════════════════════════════════════════════════════════════════════════


@patch("app.pipeline.extract.requests.get")
def test_returns_dataframe(mock_get):
    """Doit retourner un DataFrame quand l'API renvoie des résultats."""
    records = [_make_record(i) for i in range(3)]
    mock_get.return_value = _make_response(records)

    df = extract_from_signalconso_api(API_URL, limit=3)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == ["id", "category", "status"]


@patch("app.pipeline.extract.requests.get")
def test_columns_match_api_keys(mock_get):
    """Les colonnes du DataFrame doivent correspondre aux clés des records."""
    records = [{"id": 1, "title": "spam", "score": 0.9}]
    mock_get.return_value = _make_response(records)

    df = extract_from_signalconso_api(API_URL, limit=10)

    assert set(df.columns) == {"id", "title", "score"}


@patch("app.pipeline.extract.requests.get")
def test_respects_limit(mock_get):
    """Ne doit pas retourner plus de lignes que `limit`."""
    # L'API renvoie 100 records par page, mais limit=5
    records = [_make_record(i) for i in range(100)]
    mock_get.return_value = _make_response(records)

    df = extract_from_signalconso_api(API_URL, limit=5)

    assert len(df) == 5


@patch("app.pipeline.extract.requests.get")
def test_paginates_until_limit(mock_get):
    """Doit paginer et accumuler les records jusqu'à atteindre `limit`."""
    page = [_make_record(i) for i in range(100)]
    # Chaque appel retourne 100 records
    mock_get.return_value = _make_response(page)

    df = extract_from_signalconso_api(API_URL, limit=250)

    assert len(df) == 250
    # Doit avoir fait au moins 3 appels (100 + 100 + 50)
    assert mock_get.call_count >= 3


@patch("app.pipeline.extract.requests.get")
def test_passes_correct_pagination_params(mock_get):
    """Vérifie que offset et limit sont bien transmis à l'API."""
    records = [_make_record(i) for i in range(100)]
    # Premier appel : 100 records ; deuxième : vide pour stopper
    mock_get.side_effect = [
        _make_response(records),
        _make_response([]),
    ]

    extract_from_signalconso_api(API_URL, limit=10000)

    first_call_params = mock_get.call_args_list[0].kwargs["params"]
    assert first_call_params["limit"] == 100
    assert first_call_params["offset"] == 0

    second_call_params = mock_get.call_args_list[1].kwargs["params"]
    assert second_call_params["offset"] == 100


@patch("app.pipeline.extract.requests.get")
def test_stops_when_api_returns_empty_results(mock_get):
    """Doit s'arrêter dès que l'API retourne une liste vide."""
    records = [_make_record(i) for i in range(100)]
    mock_get.side_effect = [
        _make_response(records),
        _make_response([]),  # ← stop
    ]

    df = extract_from_signalconso_api(API_URL, limit=10000)

    assert len(df) == 100
    assert mock_get.call_count == 2


@patch("app.pipeline.extract.requests.get")
def test_returns_empty_dataframe_when_no_results(mock_get):
    """Doit retourner un DataFrame vide si l'API ne renvoie aucun record."""
    mock_get.return_value = _make_response([])

    df = extract_from_signalconso_api(API_URL, limit=100)

    assert isinstance(df, pd.DataFrame)
    assert df.empty


@patch("app.pipeline.extract.requests.get")
def test_uses_60s_timeout(mock_get):
    """Le timeout doit être fixé à 60 secondes."""
    mock_get.return_value = _make_response([_make_record(0)])

    extract_from_signalconso_api(API_URL, limit=1)

    _, kwargs = mock_get.call_args
    assert kwargs["timeout"] == 60


# ══════════════════════════════════════════════════════════════════════════════
# Cas d'erreur
# ══════════════════════════════════════════════════════════════════════════════


@patch("app.pipeline.extract.requests.get")
def test_raises_on_http_error(mock_get):
    """Doit propager l'HTTPError levée par raise_for_status."""
    mock_get.return_value = _make_response([], status_code=500)

    with pytest.raises(requests.HTTPError):
        extract_from_signalconso_api(API_URL, limit=10)


@patch("app.pipeline.extract.requests.get")
def test_raises_on_connection_error(mock_get):
    """Doit propager une ConnectionError si l'API est inaccessible."""
    mock_get.side_effect = requests.ConnectionError("unreachable")

    with pytest.raises(requests.ConnectionError):
        extract_from_signalconso_api(API_URL, limit=10)


@patch("app.pipeline.extract.requests.get")
def test_raises_on_timeout(mock_get):
    """Doit propager un Timeout si l'API met trop longtemps à répondre."""
    mock_get.side_effect = requests.Timeout("timeout")

    with pytest.raises(requests.Timeout):
        extract_from_signalconso_api(API_URL, limit=10)
