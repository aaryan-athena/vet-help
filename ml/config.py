"""Central configuration for the VetDx pipeline.

Everything here is intentionally data-driven: the pipeline discovers the real
schema of ``data/data.csv`` at runtime (see :mod:`ml.inspect_data`) rather than
hardcoding column names, so dropping in a richer CSV (with breed/age/weight/
duration/disease columns) requires no code changes.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_CSV = ROOT / "data" / "data.csv"
BREED_CSV = ROOT / "data" / "breedinfo.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "ml" / "artifacts"
REPORT_DIR = ROOT / "reports"
DOCS_DIR = ROOT / "docs"

MODEL_PATH = ARTIFACT_DIR / "model.joblib"
FEATURE_SCHEMA_PATH = ARTIFACT_DIR / "feature_schema.json"
EVAL_REPORT_PATH = ARTIFACT_DIR / "evaluation_report.json"
MODEL_INFO_PATH = ARTIFACT_DIR / "model_info.json"
SHAP_GLOBAL_PATH = ARTIFACT_DIR / "shap_global.json"
RUN_LOG_PATH = ARTIFACT_DIR / "runs.jsonl"
BREED_ARTIFACT_PATH = ARTIFACT_DIR / "breeds.json"

RANDOM_SEED = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20  # fraction of the post-test-split remainder

# --- Schema discovery ------------------------------------------------------
# Candidate column names, checked case-insensitively, first match wins.
SPECIES_CANDIDATES = ["animalname", "animal", "animal_type", "species", "animaltype"]
BREED_CANDIDATES = ["breed", "breed_name"]
AGE_CANDIDATES = ["age", "age_years", "animal_age"]
WEIGHT_CANDIDATES = ["weight", "weight_kg", "body_weight"]
DURATION_CANDIDATES = ["duration", "symptom_duration", "duration_of_symptoms"]

# Target preference order: a disease label is the project's ideal target; the
# shipped dataset only carries a binary danger flag, so that is the fallback.
TARGET_CANDIDATES = [
    "disease",
    "diagnosis",
    "diseasename",
    "disease_name",
    "label",
    "dangerous",
]

# Any column matching this prefix (case-insensitive) is treated as a symptom slot.
SYMPTOM_PREFIXES = ("symptom",)

# Binary Yes/No behavioural indicators are auto-detected: object columns whose
# values are a subset of this vocabulary.
YES_VALUES = {"yes", "y", "true", "1", "present"}
NO_VALUES = {"no", "n", "false", "0", "absent"}

# --- Feature engineering knobs --------------------------------------------
# A canonical symptom term must appear at least this many times to earn a
# dedicated binary feature. Rarer terms still contribute via `n_symptoms` and
# the character n-gram fallback, but do not blow up dimensionality.
MIN_SYMPTOM_FREQ = 3
# Number of most-informative symptom pairs to add as co-occurrence features.
N_COOCCURRENCE_PAIRS = 25
AGE_BINS = [0, 1, 3, 7, 12, 200]
AGE_BIN_LABELS = ["infant", "young", "adult", "mature", "senior"]
DURATION_BINS = [0, 2, 7, 30, 10_000]
DURATION_BIN_LABELS = ["acute", "short", "subacute", "chronic"]

TOP_N_PREDICTIONS = 3
TOP_N_EXPLANATIONS = 5

# Frontend origins allowed to call the API.
#
# Local development defaults are always allowed. In production set
# VETDX_CORS_ORIGINS to a comma-separated list of your deployed frontend
# origins, e.g. "https://vetdx.vercel.app,https://vetdx.example.org".
_DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://localhost:3000",
]

CORS_ORIGINS = _DEFAULT_CORS_ORIGINS + [
    origin.strip().rstrip("/")
    for origin in os.environ.get("VETDX_CORS_ORIGINS", "").split(",")
    if origin.strip()
]

# Vercel gives every preview deployment its own generated *.vercel.app origin,
# which cannot be enumerated ahead of time. Set VETDX_CORS_ALLOW_VERCEL_PREVIEWS=0
# to turn this off and rely solely on the explicit list above.
CORS_ORIGIN_REGEX = (
    r"https://.*\.vercel\.app"
    if os.environ.get("VETDX_CORS_ALLOW_VERCEL_PREVIEWS", "1") != "0"
    else None
)

BREED_CAVEAT = (
    "Breed guidance is a starting point drawn from a 41-breed reference table, "
    "not a farm plan. Real yield depends on feed, water, housing, herd health "
    "and management far more than on breed alone, and local availability, "
    "market access and extension support matter just as much. Confirm any "
    "choice with your local veterinary or animal husbandry department."
)

DISCLAIMER = (
    "VetDx is a triage and decision-support aid, not a diagnostic tool. Its "
    "predictions are statistical associations learned from a small, noisy, "
    "publicly sourced dataset and are not a substitute for examination and "
    "diagnosis by a qualified veterinarian. Always seek professional care for "
    "a sick animal."
)
