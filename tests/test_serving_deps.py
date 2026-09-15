"""The serving path must not need the training-only dependencies.

`shap`, `xgboost` and `matplotlib` are in requirements-dev.txt, not
requirements.txt, because together they are ~330 MB and would push the Vercel
function bundle over its limit. Nothing in `backend/` may import them, directly
or transitively.

Run in a subprocess: the import guard is process-global, so doing this in-process
would corrupt module state for every other test.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from ml import config

pytestmark = pytest.mark.skipif(
    not config.MODEL_PATH.exists(),
    reason="no trained model; run `python -m ml.train` first",
)

TRAINING_ONLY = ("shap", "xgboost", "matplotlib", "numba", "llvmlite")

SCRIPT = textwrap.dedent(
    f"""
    import builtins, json, sys

    BLOCKED = set({TRAINING_ONLY!r})
    _real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"blocked training-only dependency: {{name}}")
        return _real_import(name, *args, **kwargs)

    builtins.__import__ = guarded
    for module in list(sys.modules):
        if module.split(".")[0] in BLOCKED:
            del sys.modules[module]

    from fastapi.testclient import TestClient
    from backend.app.main import app

    client = TestClient(app)
    health = client.get("/health").json()
    response = client.post(
        "/predict",
        json={{"species": "buffaloes", "symptoms": ["fever", "lethargy", "diarrhea"]}},
    )
    body = response.json()
    print(json.dumps({{
        "health_status": health["status"],
        "predict_status": response.status_code,
        "top": body["top_prediction"],
        "methods": sorted({{e["method"] for e in body["explanation"]}}),
        "n_explanations": len(body["explanation"]),
        "schema_ok": client.get("/schema").status_code == 200,
        "model_info_ok": client.get("/model-info").status_code == 200,
    }}))
    """
)


@pytest.fixture(scope="module")
def served_without_training_deps() -> dict:
    import json

    result = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        capture_output=True,
        text=True,
        cwd=str(config.ROOT),
    )
    assert result.returncode == 0, (
        "the API failed to serve without the training-only dependencies:\n"
        f"{result.stdout}\n{result.stderr}"
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_api_serves_without_training_only_dependencies(served_without_training_deps):
    payload = served_without_training_deps
    assert payload["health_status"] == "ok"
    assert payload["predict_status"] == 200
    assert payload["schema_ok"] and payload["model_info_ok"]


def test_explanations_are_still_exact_without_shap(served_without_training_deps):
    """Explanations must stay exact, not silently degrade to global importance."""
    payload = served_without_training_deps
    assert payload["n_explanations"] > 0
    assert payload["methods"] == ["shap_linear_exact"]


def test_predictions_match_the_full_environment(served_without_training_deps):
    """Dropping the training deps must not change a single probability."""
    from backend.app.model_service import get_service

    reference = get_service().predict(
        {"species": "buffaloes", "symptoms": ["fever", "lethargy", "diarrhea"]}
    )
    assert served_without_training_deps["top"] == reference["top_prediction"]
