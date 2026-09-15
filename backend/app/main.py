"""VetDx FastAPI application.

Run locally with:

    uvicorn backend.app.main:app --reload --port 8000

Interactive docs at http://localhost:8000/docs
"""
from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ml import config
from backend.app.model_service import ModelNotTrainedError, ModelService, get_service
from backend.app.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    SchemaResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
)
logger = logging.getLogger("vetdx.api")

app = FastAPI(
    title="VetDx API",
    version="1.0.0",
    summary="ML-driven early-warning triage for animal disease",
    description=(
        "Predicts likely outcomes from an animal profile and observed clinical "
        "signs, with a SHAP-based explanation of the drivers.\n\n**"
        + config.DISCLAIMER
        + "**"
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_origin_regex=config.CORS_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Structured access log with a request id for correlating predictions."""
    request_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    request.state.request_id = request_id
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "request_id=%s method=%s path=%s status=500 unhandled error",
            request_id,
            request.method,
            request.url.path,
        )
        raise
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    response.headers["X-Request-ID"] = request_id
    return response


# Routes live on a router, which is then mounted twice: at the root (the
# two-project Vercel setup, where the API has its own domain) and under /api
# (a single-domain setup, or a frontend whose VITE_API_BASE already ends in
# /api). Serving both spellings removes an entire category of 404.
router = APIRouter()


def _service() -> ModelService:
    try:
        return get_service()
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.on_event("startup")
def warm_up() -> None:
    """Load the model at boot so the first prediction is not the slow one."""
    try:
        get_service()
    except ModelNotTrainedError as exc:
        logger.warning("starting without a model: %s", exc)


@router.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    try:
        service = get_service()
    except ModelNotTrainedError as exc:
        return HealthResponse(status="degraded", model_loaded=False, detail=str(exc))
    return HealthResponse(
        status="ok", model_loaded=True, model_name=service.model_name
    )


@router.get(
    "/schema",
    response_model=SchemaResponse,
    responses={503: {"model": ErrorResponse}},
    tags=["meta"],
)
def input_schema() -> dict:
    """The input contract, so the frontend form is generated, not hardcoded."""
    return _service().feature_schema


@router.get("/model-info", responses={503: {"model": ErrorResponse}}, tags=["meta"])
def model_info() -> dict:
    """Deployed model, its evaluation metrics, and when it was trained."""
    return _service().info()


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["inference"],
)
def predict(payload: PredictRequest, request: Request) -> dict:
    """Rank the likely outcomes for one case and explain the top one."""
    service = _service()
    try:
        result = service.predict(payload.model_dump(exclude_none=True), top_n=payload.top_n)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("prediction failed")
        raise HTTPException(status_code=500, detail="prediction failed") from exc

    logger.info(
        "request_id=%s prediction species=%s n_symptoms=%d top=%s p=%.3f unknown=%s",
        getattr(request.state, "request_id", "-"),
        payload.species,
        result["n_symptoms_used"],
        result["top_prediction"]["label"],
        result["top_prediction"]["probability"],
        result["unrecognized_symptoms"],
    )
    return result


@router.get("/", include_in_schema=False)
def index() -> dict:
    """Service descriptor.

    Without this, hitting the API's root URL in a browser returns a bare
    ``{"detail": "Not Found"}``, which reads like the backend is broken when it
    is simply an API with no landing page. This is NOT the user interface --
    that is the separate frontend deployment.
    """
    try:
        service = get_service()
        model = {"name": service.model_name, "loaded": True}
    except ModelNotTrainedError as exc:
        model = {"loaded": False, "detail": str(exc)}
    return {
        "service": "VetDx API",
        "version": app.version,
        "status": "running",
        "note": (
            "This is the JSON API, not the VetDx web interface. Open /docs for "
            "interactive documentation."
        ),
        "model": model,
        "endpoints": {
            "health": "/health",
            "schema": "/schema",
            "model_info": "/model-info",
            "predict": "POST /predict",
            "docs": "/docs",
        },
        "disclaimer": config.DISCLAIMER,
    }


app.include_router(router)
# Same routes under /api, so both deployment shapes work. Hidden from the
# OpenAPI schema to keep /docs showing one canonical path per endpoint.
app.include_router(router, prefix="/api", include_in_schema=False)


@app.exception_handler(ModelNotTrainedError)
def model_missing_handler(request: Request, exc: ModelNotTrainedError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})
