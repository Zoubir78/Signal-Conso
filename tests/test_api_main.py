from fastapi.testclient import TestClient

from app.api.main import app


def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Bienvenue sur la plateforme intelligente de Signal Conso !"
    }


def test_openapi_title():
    with TestClient(app) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json()["info"]["title"] == "Signal Conso API"
