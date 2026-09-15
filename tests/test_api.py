"""Integration tests for the FastAPI app against the real trained pipeline.

These are skipped (not failed) when ``ml/artifacts/model.joblib`` is absent, so
a fresh clone can run the unit tests before training.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ml import config

pytestmark = pytest.mark.skipif(
    not config.MODEL_PATH.exists(),
    reason="no trained model; run `python -m ml.train` first",
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    from backend.app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def schema(client) -> dict:
    return client.get("/schema").json()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_name"]


def test_schema_is_renderable(schema):
    assert schema["classes"]
    assert schema["disclaimer"]
    names = [f["name"] for f in schema["fields"]]
    assert "symptoms" in names

    symptom_field = next(f for f in schema["fields"] if f["name"] == "symptoms")
    assert symptom_field["type"] == "multiselect"
    assert len(symptom_field["options"]) > 10
    # Every field the frontend renders must declare a type it knows about.
    assert {f["type"] for f in schema["fields"]} <= {
        "select",
        "multiselect",
        "number",
        "text",
        "yesno",
    }


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["model_name"]
    assert body["trained_at"]
    assert 0 <= body["test_metrics"]["accuracy"] <= 1
    assert 0 <= body["test_metrics"]["macro_f1"] <= 1
    assert len(body["comparison"]) >= 5
    assert body["confusion_matrix"]["matrix"]
    assert body["disclaimer"]


def test_predict_returns_ranked_probabilities(client, schema):
    top_n = min(2, len(schema["classes"]))
    response = client.post(
        "/predict",
        json={
            "species": "dog",
            "symptoms": ["fever", "diarrhea", "vomiting"],
            "top_n": top_n,
        },
    )
    assert response.status_code == 200
    body = response.json()

    assert len(body["predictions"]) == top_n
    probs = [p["probability"] for p in body["predictions"]]
    assert probs == sorted(probs, reverse=True)
    assert all(0 <= p <= 1 for p in probs)
    assert [p["rank"] for p in body["predictions"]] == list(range(1, top_n + 1))
    assert body["top_prediction"] == body["predictions"][0]
    assert body["top_prediction"]["label"] in schema["classes"]
    assert body["n_symptoms_used"] == 3
    assert body["disclaimer"]
    assert response.headers["X-Request-ID"]


def test_predict_explains_only_present_features(client):
    body = client.post(
        "/predict",
        json={"species": "dog", "symptoms": ["fever", "diarrhea", "coughing"]},
    ).json()

    assert body["explanation"], "expected at least one contributing feature"
    for item in body["explanation"]:
        assert item["direction"] in {"increases", "decreases"}
        assert item["label"]
        assert isinstance(item["contribution"], float)
    # The explanation must reference signs the caller actually reported.
    mentioned = " ".join(i["feature"] for i in body["explanation"])
    assert any(s in mentioned for s in ("fever", "diarrhea", "coughing", "species"))


def test_predict_flags_unknown_symptoms(client):
    body = client.post(
        "/predict",
        json={"species": "dog", "symptoms": ["fever", "glows in the dark"]},
    ).json()
    assert body["unrecognized_symptoms"] == ["glows in the dark"]
    assert body["top_prediction"]["probability"] > 0


def test_predict_accepts_unseen_species(client):
    response = client.post(
        "/predict", json={"species": "capybara", "symptoms": ["fever"]}
    )
    assert response.status_code == 200


def test_predict_canonicalises_messy_input(client):
    """"Pains"/"Anorexia" must hit the same features as their canonical terms."""
    messy = client.post(
        "/predict", json={"species": "Dogs", "symptoms": ["Pains", "Anorexia", "FEVER"]}
    ).json()
    canonical = client.post(
        "/predict",
        json={"species": "dog", "symptoms": ["pain", "loss of appetite", "fever"]},
    ).json()
    assert messy["predictions"] == canonical["predictions"]
    assert messy["unrecognized_symptoms"] == []


@pytest.mark.parametrize(
    "payload",
    [
        {"species": "dog", "symptoms": []},
        {"species": "dog"},
        {"symptoms": ["fever"]},
        {"species": "  ", "symptoms": ["fever"]},
        {"species": "dog", "symptoms": ["fever"], "age_years": -5},
        {"species": "dog", "symptoms": ["fever"], "top_n": 0},
        {"species": "dog", "symptoms": ["fever"], "unexpected_field": 1},
    ],
)
def test_predict_rejects_malformed_input(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_cors_headers_for_frontend_origin(client):
    response = client.options(
        "/predict",
        headers={
            "Origin": config.CORS_ORIGINS[0],
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == config.CORS_ORIGINS[0]
