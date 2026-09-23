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


# ---------------------------------------------------------------------------
# Routing: the API must not answer a browser with a bare {"detail":"Not Found"}
# ---------------------------------------------------------------------------
def test_root_serves_a_service_descriptor(client):
    """Hitting the API root in a browser must not look like a broken backend."""
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "VetDx API"
    assert body["status"] == "running"
    # It must say plainly that this is not the web interface.
    assert "not the VetDx web interface" in body["note"]
    assert set(body["endpoints"]) >= {"health", "schema", "model_info", "predict", "docs"}


@pytest.mark.parametrize("path", ["/", "/health", "/schema", "/model-info"])
def test_routes_are_served_under_the_api_prefix_too(client, path):
    """Both deployment shapes work: own domain, or mounted under /api."""
    prefixed = "/api" if path == "/" else f"/api{path}"
    assert client.get(path).status_code == 200
    assert client.get(prefixed).status_code == 200


def test_predict_is_identical_under_both_prefixes(client):
    payload = {"species": "dog", "symptoms": ["fever", "diarrhea"], "top_n": 2}
    bare = client.post("/predict", json=payload)
    prefixed = client.post("/api/predict", json=payload)
    assert bare.status_code == prefixed.status_code == 200
    assert bare.json()["predictions"] == prefixed.json()["predictions"]


def test_unknown_paths_still_404(client):
    """The /api mount must not turn every URL into a 200."""
    assert client.get("/definitely-not-a-route").status_code == 404
    assert client.get("/api/definitely-not-a-route").status_code == 404


# ---------------------------------------------------------------------------
# Chat triage
# ---------------------------------------------------------------------------
def test_chat_extracts_and_assesses_in_one_turn(client):
    body = client.post(
        "/chat",
        json={"message": "my buffalo has had fever for 2 days and won't eat"},
    ).json()
    assert body["intent"] == "assessed"
    assert body["ready"] is True
    assert body["case"]["species"] == "buffaloes"
    assert set(body["case"]["symptoms"]) >= {"fever", "loss of appetite"}
    assert body["case"]["duration_days"] == 2.0
    assert body["prediction"]["top_prediction"]["label"]
    assert body["disclaimer"]


def test_chat_carries_state_across_turns(client):
    """Later turns add to the case rather than replacing it."""
    first = client.post("/chat", json={"message": "my goat is off her feed"}).json()
    # Species plus one sign is already enough to assess -- a triage tool should
    # not withhold an answer to collect a fuller form.
    assert first["ready"] is True
    assert first["case"]["symptoms"] == ["loss of appetite"]

    second = client.post(
        "/chat", json={"message": "also fever and loose motions", "case": first["case"]}
    ).json()
    assert second["case"]["species"] == "goat"
    assert set(second["case"]["symptoms"]) == {"loss of appetite", "fever", "diarrhea"}
    assert second["prediction"] is not None


def test_chat_honours_negation_across_turns(client):
    first = client.post("/chat", json={"message": "cow with fever and cough"}).json()
    assert "coughing" in first["case"]["symptoms"]
    second = client.post(
        "/chat", json={"message": "actually no cough", "case": first["case"]}
    ).json()
    assert "coughing" not in second["case"]["symptoms"]
    assert "coughing" in second["case"]["negated"]


def test_chat_asks_for_the_missing_species(client):
    body = client.post("/chat", json={"message": "it has a fever"}).json()
    assert body["ready"] is False
    assert "which animal" in body["reply"].lower()


def test_chat_reports_unrecognised_words_without_inventing(client):
    body = client.post("/chat", json={"message": "my dog has purple spangles"}).json()
    assert body["case"]["symptoms"] == []
    assert "spangles" in body["extracted"]["unmatched_terms"]


def test_chat_reset_clears_the_case(client):
    first = client.post("/chat", json={"message": "buffalo with fever"}).json()
    assert first["case"]["symptoms"]
    second = client.post("/chat", json={"message": "reset", "case": first["case"]}).json()
    assert second["case"]["symptoms"] == []
    assert second["case"]["species"] == ""


def test_chat_suggestions_come_from_cooccurrence(client):
    body = client.post("/chat", json={"message": "buffalo with fever"}).json()
    assert body["suggestions"]
    assert not set(body["suggestions"]) & set(body["case"]["symptoms"])


def test_chat_agrees_with_the_predict_endpoint(client):
    """Chat must not be a second, divergent path to the model."""
    chat = client.post(
        "/chat", json={"message": "buffalo with fever, diarrhea and lethargy"}
    ).json()
    direct = client.post(
        "/predict",
        json={
            "species": chat["case"]["species"],
            "symptoms": chat["case"]["symptoms"],
            "top_n": 3,
        },
    ).json()
    assert chat["prediction"]["predictions"] == direct["predictions"]


@pytest.mark.parametrize(
    "payload",
    [{"message": ""}, {}, {"message": "hi", "unexpected": 1}, {"message": "x" * 2001}],
)
def test_chat_rejects_malformed_input(client, payload):
    assert client.post("/chat", json=payload).status_code == 422


# ---------------------------------------------------------------------------
# Breed advisor
# ---------------------------------------------------------------------------
def test_breeds_catalogue(client):
    body = client.get("/breeds").json()
    assert len(body["breeds"]) >= 40
    assert body["facets"]["n_breeds"] == len(body["breeds"])
    assert body["facets"]["types"]
    assert body["facets"]["caveat"]


def test_breed_recommend_respects_climate(client):
    body = client.post(
        "/breeds/recommend", json={"climate": "hot and humid", "purpose": "dairy", "top_n": 3}
    ).json()
    names = [b["name"] for b in body["recommendations"]]
    assert "Holstein Friesian" not in names
    assert body["recommendations"][0]["suitable"]
    assert body["recommendations"][0]["reasons"]


def test_breed_profile(client):
    body = client.get("/breeds/Gir").json()
    assert body["name"] == "Gir"
    assert body["ideal_conditions"]
    assert body["guidance"]
    assert body["caveat"]


def test_unknown_breed_404s(client):
    assert client.get("/breeds/Tyrannosaurus").status_code == 404


@pytest.mark.parametrize(
    "payload", [{"top_n": 0}, {"top_n": 99}, {"climate": "x" * 201}, {"nope": 1}]
)
def test_breed_recommend_rejects_malformed_input(client, payload):
    assert client.post("/breeds/recommend", json=payload).status_code == 422
