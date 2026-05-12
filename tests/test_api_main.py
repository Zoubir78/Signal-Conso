from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Bienvenue sur la plateforme intelligente de Signal Conso !"
    }


def test_openapi_title():
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json()["info"]["title"] == "Signal Conso API"


def test_routes_are_registered():
    routes = {route.path for route in app.routes}

    assert "/" in routes
    assert "/health" in routes
    assert "/tickets" in routes
    assert "/predictions" in routes
