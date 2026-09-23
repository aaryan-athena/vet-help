"""Pydantic request/response models for the VetDx API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ml import config


class PredictRequest(BaseModel):
    """One triage case.

    Only ``species`` and at least one symptom are required; every other field
    is optional so a request stays valid whether or not the training CSV
    carried breed/age/weight/duration columns.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "species": "dog",
                "symptoms": ["fever", "diarrhea", "vomiting"],
                "age_years": 3,
                "weight_kg": 12.5,
                "duration": "2 days",
                "top_n": 3,
            }
        },
    )

    species: str = Field(..., min_length=1, max_length=80, description="Animal type")
    symptoms: list[str] = Field(
        ..., min_length=1, max_length=40, description="Observed clinical signs"
    )
    breed: str | None = Field(None, max_length=80)
    age_years: float | None = Field(None, ge=0, le=200)
    weight_kg: float | None = Field(None, ge=0, le=20_000)
    duration: str | None = Field(
        None, max_length=60, description='Free text, e.g. "2 days" or "1 week"'
    )
    duration_days: float | None = Field(None, ge=0, le=10_000)
    indicators: dict[str, str] | None = Field(
        None, description="Yes/No behavioural indicators keyed by field name"
    )
    top_n: int = Field(config.TOP_N_PREDICTIONS, ge=1, le=20)

    @field_validator("symptoms")
    @classmethod
    def _non_empty_symptoms(cls, value: list[str]) -> list[str]:
        cleaned = [s for s in (str(v).strip() for v in value) if s]
        if not cleaned:
            raise ValueError("provide at least one non-empty symptom")
        return cleaned

    @field_validator("species")
    @classmethod
    def _non_blank_species(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("species must not be blank")
        return value.strip()


class Prediction(BaseModel):
    label: str
    probability: float
    rank: int


class Contribution(BaseModel):
    feature: str
    label: str
    contribution: float
    direction: Literal["increases", "decreases"]
    method: str


class ModelStamp(BaseModel):
    name: str
    target_column: str
    target_kind: str
    trained_at: str | None = None


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    top_prediction: Prediction
    explanation: list[Contribution]
    unrecognized_symptoms: list[str]
    n_symptoms_used: int
    model: ModelStamp
    disclaimer: str


class HealthResponse(BaseModel):
    # "model_*" is a protected pydantic namespace by default; these are plain
    # response fields, not model settings.
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_name: str | None = None
    detail: str | None = None


class SchemaResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    version: int
    target_column: str
    target_kind: str
    classes: list[str]
    fields: list[dict[str, Any]]
    symptom_vocabulary: list[dict[str, Any]]
    disclaimer: str


class ErrorResponse(BaseModel):
    detail: str


# ---------------------------------------------------------------------------
# Chat triage
# ---------------------------------------------------------------------------
class ChatCase(BaseModel):
    """Accumulated case state, round-tripped by the client (the API is stateless)."""

    model_config = ConfigDict(extra="ignore")

    species: str = ""
    symptoms: list[str] = Field(default_factory=list, max_length=40)
    negated: list[str] = Field(default_factory=list, max_length=40)
    age_years: float | None = Field(None, ge=0, le=200)
    duration: str = Field("", max_length=60)
    duration_days: float | None = Field(None, ge=0, le=10_000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "message": "my buffalo has had fever for 2 days and won't eat",
                "case": {"species": "", "symptoms": []},
            }
        },
    )

    message: str = Field(..., min_length=1, max_length=2000)
    case: ChatCase | None = None


class Extraction(BaseModel):
    model_config = ConfigDict(extra="allow")

    species: str = ""
    symptoms: list[str] = Field(default_factory=list)
    negated: list[str] = Field(default_factory=list)
    unmatched_terms: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    intent: str
    case: ChatCase
    ready: bool
    extracted: Extraction
    prediction: PredictResponse | None = None
    suggestions: list[str] = Field(default_factory=list)
    disclaimer: str


# ---------------------------------------------------------------------------
# Breed advisor
# ---------------------------------------------------------------------------
class BreedRecommendRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "climate": "hot and humid",
                "purpose": "dairy",
                "type": "Buffalo",
                "top_n": 5,
            }
        },
    )

    climate: str = Field("", max_length=200)
    purpose: str = Field("", max_length=100)
    type: str = Field("", max_length=60)
    region: str = Field("", max_length=100)
    top_n: int = Field(5, ge=1, le=41)
