"""SHAP explainability for the deployed VetDx model.

    python -m ml.explain                       # global feature importance
    python -m ml.explain --case '{"species": "dog", "symptoms": ["fever"]}'

``compute_global_importance`` is called at the end of training and writes
``ml/artifacts/shap_global.json``. ``explain_prediction`` is what the API calls
per request to answer "which symptoms drove *this* prediction?".
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ml import config

# `shap` is a TRAINING-ONLY dependency: it drags in numba + llvmlite (~150 MB),
# which would dominate a serverless bundle. Linear models -- including the
# currently deployed one -- are explained exactly in numpy by
# `linear_contributions` below, so the serving path never imports it.
try:
    import shap

    HAS_SHAP = True
except ImportError:  # pragma: no cover - optional at serve time by design
    shap = None
    HAS_SHAP = False

# Cached explainer, keyed by id() of the pipeline it was built for.
_EXPLAINER_CACHE: dict[int, object] = {}

# Pretty labels for engineered (non-symptom) features.
_PREFIX_LABELS = {
    "sym__": "Symptom: {}",
    "pair__": "Symptom pair: {}",
    "species__": "Species: {}",
    "breed__": "Breed: {}",
    "age_bin__": "Age group: {}",
    "duration_bin__": "Symptom duration: {}",
    "bin__": "Indicator: {}",
}

_PLAIN_LABELS = {
    "n_symptoms": "Number of symptoms reported",
    "n_rare_symptoms": "Number of uncommon symptoms reported",
    "age_years": "Age (years)",
    "weight_kg": "Weight (kg)",
    "duration_days": "Symptom duration (days)",
}


def humanize_feature(name: str) -> str:
    """Turn ``sym__loss of appetite`` into ``Symptom: Loss Of Appetite``."""
    if name in _PLAIN_LABELS:
        return _PLAIN_LABELS[name]
    for prefix, template in _PREFIX_LABELS.items():
        if name.startswith(prefix):
            value = name[len(prefix) :].replace("+", " + ").replace("_", " ")
            return template.format(value.title())
    return name.replace("_", " ").title()


def _split_classifier(classifier):
    """Return (final estimator, leading pipeline steps or None)."""
    if hasattr(classifier, "steps"):
        return classifier.steps[-1][1], Pipeline(classifier.steps[:-1])
    return classifier, None


def linear_contributions(
    pipeline,
    X: pd.DataFrame,
    class_index: int,
    background_means: np.ndarray | None = None,
    background: pd.DataFrame | None = None,
) -> np.ndarray | None:
    """Exact SHAP values for a linear model, in numpy.

    For ``f(x) = w.x + b`` the Shapley value of feature *i* has a closed form::

        phi_i = w_i * (x_i - E[x_i])

    so no sampling, no approximation and no `shap` package is needed -- this is
    what `shap.LinearExplainer` computes, and `tests/test_explain.py` asserts
    the two agree. Contributions are in log-odds (margin) space.

    Returns ``None`` when the estimator is not linear, so callers can fall back.
    """
    classifier = pipeline.named_steps["classifier"]
    estimator, pre = _split_classifier(classifier)
    coef = getattr(estimator, "coef_", None)
    if coef is None:
        return None
    coef = np.asarray(coef, dtype=float)

    builder = pipeline.named_steps["features"]
    columns = list(builder.feature_names_)

    if background_means is None:
        if background is None:
            return None
        background_means = builder.transform(background).to_numpy(dtype=float).mean(axis=0)
    means = np.asarray(background_means, dtype=float).reshape(1, -1)

    # StandardScaler and friends are affine, so transforming the mean equals the
    # mean of the transformed background.
    x = np.asarray(X[columns] if list(X.columns) != columns else X, dtype=float)
    if pre is not None:
        x = np.asarray(pre.transform(pd.DataFrame(x, columns=columns)), dtype=float)
        means = np.asarray(pre.transform(pd.DataFrame(means, columns=columns)), dtype=float)

    if coef.shape[0] == 1:
        # Binary: sklearn stores one weight vector, for the positive class.
        weights = coef[0] if class_index == 1 else -coef[0]
    else:
        weights = coef[min(class_index, coef.shape[0] - 1)]

    return weights * (x - means)


def _get_explainer(pipeline, background: pd.DataFrame):
    """Build (and cache) a SHAP explainer for the pipeline's classifier."""
    key = id(pipeline)
    if key in _EXPLAINER_CACHE:
        return _EXPLAINER_CACHE[key]

    builder = pipeline.named_steps["features"]
    classifier = pipeline.named_steps["classifier"]
    bg = builder.transform(background)
    # Keep the background small: TreeExplainer is exact, the kernel/permutation
    # fallbacks are O(background) per prediction.
    if len(bg) > 100:
        bg = bg.sample(100, random_state=config.RANDOM_SEED)

    # Unwrap a scaler-wrapped estimator so tree/linear fast paths still apply.
    inner, inner_bg = classifier, bg
    if hasattr(classifier, "steps"):
        inner = classifier.steps[-1][1]
        inner_bg = pd.DataFrame(
            Pipeline(classifier.steps[:-1]).transform(bg), columns=bg.columns
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = None
        for build in (
            lambda: shap.TreeExplainer(inner),
            lambda: shap.LinearExplainer(inner, inner_bg),
        ):
            try:
                explainer = build()
                break
            except Exception:
                continue
        if explainer is None:
            predict = getattr(classifier, "predict_proba", classifier.predict)
            explainer = shap.Explainer(predict, bg, silent=True)

    # A tree/linear explainer operates on the *inner* feature space, so record
    # the leading pipeline steps that a caller must apply first.
    pre = (
        Pipeline(classifier.steps[:-1])
        if hasattr(classifier, "steps") and explainer is not None and not _is_generic(explainer)
        else None
    )
    _EXPLAINER_CACHE[key] = (explainer, pre)
    return _EXPLAINER_CACHE[key]


def _is_generic(explainer) -> bool:
    return type(explainer).__name__ in {"Explainer", "PermutationExplainer", "Permutation"}


def _shap_values(pipeline, background: pd.DataFrame, X: pd.DataFrame):
    """Run the cached explainer on ``X``, applying any pre-transform first."""
    explainer, pre = _get_explainer(pipeline, background)
    data = pd.DataFrame(pre.transform(X), columns=X.columns) if pre is not None else X
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            raw = explainer(data)
        except TypeError:
            raw = explainer.shap_values(data)
    return getattr(raw, "values", raw)


def _shap_matrix(values, class_index: int, n_features: int) -> np.ndarray:
    """Normalise SHAP output (list / 2-D / 3-D) to (n_samples, n_features)."""
    if isinstance(values, list):
        arr = np.asarray(values[min(class_index, len(values) - 1)])
    else:
        arr = np.asarray(values)
    if arr.ndim == 3:  # (samples, features, classes)
        idx = min(class_index, arr.shape[2] - 1)
        arr = arr[:, :, idx]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != n_features and arr.shape[0] == n_features:
        arr = arr.T
    return arr


def compute_global_importance(pipeline, background: pd.DataFrame, classes: list[str]) -> dict:
    """Mean |SHAP| per feature over the training data -> shap_global.json."""
    builder = pipeline.named_steps["features"]
    feature_names = list(builder.feature_names_)

    sample = background
    if len(sample) > 300:
        sample = sample.sample(300, random_state=config.RANDOM_SEED)
    X = builder.transform(sample)
    background_means = builder.transform(background).to_numpy(dtype=float).mean(axis=0)

    # Exact closed form for linear models -- no sampling, no `shap` import.
    linear = linear_contributions(pipeline, X, class_index=1, background_means=background_means)
    if linear is not None:
        return _write_global(
            np.abs(linear).mean(axis=0), feature_names, classes, len(X), "shap_linear_exact"
        )

    if not HAS_SHAP:
        print("[!] shap not installed and model is not linear; using feature importances")
        payload = _fallback_importance(pipeline, feature_names, classes)
        config.SHAP_GLOBAL_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    try:
        values = _shap_values(pipeline, background, X)
        # Average absolute contribution across every class.
        if isinstance(values, list):
            mean_abs = np.mean([np.abs(np.asarray(v)) for v in values], axis=0).mean(axis=0)
        else:
            arr = np.asarray(values)
            if arr.ndim == 3:
                mean_abs = np.abs(arr).mean(axis=(0, 2))
            else:
                mean_abs = np.abs(arr).mean(axis=0)
        mean_abs = np.asarray(mean_abs).ravel()[: len(feature_names)]
    except Exception as exc:  # pragma: no cover - explainer-specific failures
        print(f"[!] SHAP failed ({exc}); falling back to model feature importances")
        payload = _fallback_importance(pipeline, feature_names, classes)
        config.SHAP_GLOBAL_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    return _write_global(mean_abs, feature_names, classes, len(X), "shap")


def _write_global(mean_abs, feature_names, classes, n_rows, method) -> dict:
    mean_abs = np.asarray(mean_abs, dtype=float).ravel()[: len(feature_names)]
    order = np.argsort(mean_abs)[::-1]
    features = [
        {
            "feature": feature_names[i],
            "label": humanize_feature(feature_names[i]),
            "importance": round(float(mean_abs[i]), 6),
        }
        for i in order[:60]
        if mean_abs[i] > 0
    ]
    payload = {
        "method": method,
        "n_background_rows": int(n_rows),
        "classes": classes,
        "features": features,
    }
    config.SHAP_GLOBAL_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"global importance ({method}) -> {config.SHAP_GLOBAL_PATH}")
    for f in features[:10]:
        print(f"  {f['importance']:.4f}  {f['label']}")
    return payload


def _fallback_importance(pipeline, feature_names: list[str], classes: list[str]) -> dict:
    classifier = pipeline.named_steps["classifier"]
    estimator = getattr(classifier, "steps", None)
    estimator = classifier if estimator is None else classifier.steps[-1][1]

    if hasattr(estimator, "feature_importances_"):
        weights = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        weights = np.abs(np.asarray(estimator.coef_, dtype=float)).mean(axis=0)
    else:
        weights = np.zeros(len(feature_names))

    order = np.argsort(weights)[::-1]
    return {
        "method": "model_feature_importance",
        "classes": classes,
        "features": [
            {
                "feature": feature_names[i],
                "label": humanize_feature(feature_names[i]),
                "importance": round(float(weights[i]), 6),
            }
            for i in order[:60]
            if weights[i] > 0
        ],
    }


def explain_prediction(
    pipeline,
    record: pd.DataFrame,
    class_index: int,
    background: pd.DataFrame | None = None,
    background_means: np.ndarray | None = None,
    top_n: int = config.TOP_N_EXPLANATIONS,
) -> list[dict]:
    """Top contributing features for one prediction, signed and ranked.

    Only features that are actually *present* for this case are reported --
    telling a user that "not having seizures" pushed the score around is noise
    in a triage context.
    """
    builder = pipeline.named_steps["features"]
    feature_names = list(builder.feature_names_)
    X = builder.transform(record)
    present = X.iloc[0].to_numpy() != 0

    contributions: np.ndarray | None = None
    method = "none"

    # 1. Exact, numpy-only closed form -- the serving path for linear models.
    linear = linear_contributions(
        pipeline, X, class_index, background_means=background_means, background=background
    )
    if linear is not None:
        contributions = linear[0]
        method = "shap_linear_exact"

    # 2. Sampling-based SHAP, only for non-linear models (training-time dep).
    if contributions is None and HAS_SHAP and background is not None:
        try:
            values = _shap_values(pipeline, background, X)
            contributions = _shap_matrix(values, class_index, len(feature_names))[0]
            method = "shap"
        except Exception:  # pragma: no cover - explainer-specific failures
            contributions = None

    if contributions is None:
        cached = _load_global_importance()
        lookup = {f["feature"]: f["importance"] for f in cached.get("features", [])}
        contributions = np.array([lookup.get(n, 0.0) for n in feature_names])
        method = cached.get("method", "model_feature_importance")

    contributions = np.asarray(contributions, dtype=float).ravel()[: len(feature_names)]
    ranked = sorted(
        (
            {
                "feature": feature_names[i],
                "label": humanize_feature(feature_names[i]),
                "contribution": round(float(contributions[i]), 6),
                "direction": "increases" if contributions[i] >= 0 else "decreases",
                "method": method,
            }
            for i in range(len(feature_names))
            if present[i] and abs(contributions[i]) > 0
        ),
        key=lambda d: abs(d["contribution"]),
        reverse=True,
    )
    return ranked[:top_n]


_GLOBAL_CACHE: dict | None = None


def _load_global_importance() -> dict:
    global _GLOBAL_CACHE
    if _GLOBAL_CACHE is None:
        if config.SHAP_GLOBAL_PATH.exists():
            _GLOBAL_CACHE = json.loads(config.SHAP_GLOBAL_PATH.read_text(encoding="utf-8"))
        else:
            _GLOBAL_CACHE = {"features": [], "method": "unavailable"}
    return _GLOBAL_CACHE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="JSON payload to explain, as sent to /predict")
    parser.add_argument("--top", type=int, default=config.TOP_N_EXPLANATIONS)
    args = parser.parse_args()

    if args.case:
        from backend.app.model_service import get_service

        service = get_service()
        result = service.predict(json.loads(args.case), top_n=len(service.classes))
        print(json.dumps(result, indent=2))
        return

    import joblib

    from ml.cleaning import clean_dataframe, discover_schema
    from ml.features import build_record_frame

    bundle = joblib.load(config.MODEL_PATH)
    raw = pd.read_csv(config.RAW_CSV)
    schema = discover_schema(raw)
    records = build_record_frame(clean_dataframe(raw, schema), schema)
    compute_global_importance(bundle["pipeline"], records, bundle["classes"])


if __name__ == "__main__":
    main()
