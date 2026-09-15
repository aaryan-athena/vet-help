"""Loads the trained artifacts once and serves predictions + explanations."""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from ml import config
from ml.cleaning import DatasetSchema, canonical_symptom
from ml.explain import explain_prediction
from ml.features import request_to_record

logger = logging.getLogger("vetdx.model")


class ModelNotTrainedError(RuntimeError):
    """Raised when the API starts before ``python -m ml.train`` has been run."""


class ModelService:
    """Thin wrapper around the persisted pipeline.

    Holds the fitted feature builder + classifier, the input contract served by
    ``/schema``, and the feature means that exact linear SHAP needs.
    """

    def __init__(self, artifact_dir: Path | None = None):
        artifact_dir = artifact_dir or config.ARTIFACT_DIR
        model_path = artifact_dir / config.MODEL_PATH.name

        if not model_path.exists():
            raise ModelNotTrainedError(
                f"No model at {model_path}. Run `python -m ml.train` first."
            )

        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        # 1-D array of training-set feature means: all that exact linear SHAP
        # needs. Keeps the serving path free of both the CSV and the `shap`
        # package (which pulls in ~150 MB of numba/llvmlite).
        self.background_means = bundle.get("background_means")
        self.classes: list[str] = list(bundle["classes"])
        self.model_name: str = bundle["model_name"]
        self.dataset_schema = DatasetSchema(**bundle["dataset_schema"])
        self.feature_names: list[str] = list(bundle["feature_names"])

        self.feature_schema = _read_json(artifact_dir / config.FEATURE_SCHEMA_PATH.name, {})
        self.model_info = _read_json(artifact_dir / config.MODEL_INFO_PATH.name, {})
        self.evaluation_report = _read_json(artifact_dir / config.EVAL_REPORT_PATH.name, {})
        self.global_importance = _read_json(artifact_dir / config.SHAP_GLOBAL_PATH.name, {})

        self.known_symptoms = {
            s["value"] for s in self.feature_schema.get("symptom_vocabulary", [])
        }
        if self.background_means is None:
            logger.warning(
                "bundle has no background_means; retrain to enable exact "
                "per-prediction explanations"
            )
        logger.info(
            "loaded model=%s classes=%s features=%d explain=%s",
            self.model_name,
            self.classes,
            len(self.feature_names),
            "exact" if self.background_means is not None else "global-fallback",
        )

    # -- inference ---------------------------------------------------------
    def predict(self, payload: dict, top_n: int = config.TOP_N_PREDICTIONS) -> dict:
        record = request_to_record(payload, self.dataset_schema.binary_columns)

        proba = np.asarray(self.pipeline.predict_proba(record))[0]
        order = np.argsort(proba)[::-1][: max(1, min(top_n, len(self.classes)))]
        predictions = [
            {
                "label": self.classes[i],
                "probability": round(float(proba[i]), 4),
                "rank": rank + 1,
            }
            for rank, i in enumerate(order)
        ]

        top_index = int(order[0])
        explanation = explain_prediction(
            self.pipeline,
            record,
            class_index=top_index,
            background_means=self.background_means,
        )

        submitted = [canonical_symptom(s) for s in payload.get("symptoms") or []]
        unrecognised = sorted(
            {s for s in submitted if s and s not in self.known_symptoms}
        )

        return {
            "predictions": predictions,
            "top_prediction": predictions[0],
            "explanation": explanation,
            "unrecognized_symptoms": unrecognised,
            "n_symptoms_used": int(len([s for s in submitted if s])),
            "model": {
                "name": self.model_name,
                "target_column": self.dataset_schema.target,
                "target_kind": self.feature_schema.get("target_kind", "class"),
                "trained_at": self.model_info.get("trained_at"),
            },
            "disclaimer": config.DISCLAIMER,
        }

    # -- metadata ----------------------------------------------------------
    def info(self) -> dict:
        report = self.evaluation_report
        return {
            **self.model_info,
            "dataset": report.get("dataset", {}),
            "split": report.get("split", {}),
            "confusion_matrix": report.get("confusion_matrix", {}),
            "test_per_class": report.get("test_per_class", {}),
            "comparison": [
                {
                    "model": r["model"],
                    "accuracy": r["val"]["accuracy"],
                    "macro_f1": r["val"]["macro_f1"],
                    "balanced_accuracy": r["val"]["balanced_accuracy"],
                    "roc_auc": r["val"].get("roc_auc"),
                    "cv_macro_f1_mean": r.get("cv_macro_f1_mean"),
                }
                for r in report.get("comparison", [])
            ],
            "global_importance": self.global_importance.get("features", [])[:20],
            "explanation_method": self.global_importance.get("method", "unavailable"),
        }


def _read_json(path: Path, default):
    if not path.exists():
        logger.warning("missing artifact %s", path)
        return default
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def get_service() -> ModelService:
    """Process-wide singleton; loading the pipeline is the expensive part."""
    return ModelService()
