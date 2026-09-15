"""Train, compare and persist the VetDx model.

    python -m ml.train [--csv data/data.csv] [--no-shap]

Trains five classifiers on identical folds, compares them on a held-out
validation split, re-scores the winner on an untouched test split, then writes
everything the API needs:

    ml/artifacts/model.joblib            fitted feature builder + classifier
    ml/artifacts/feature_schema.json     input contract for /schema
    ml/artifacts/evaluation_report.json  full comparison + per-class metrics
    ml/artifacts/model_info.json         deployed model summary for /model-info
    ml/artifacts/shap_global.json        global SHAP importance (see ml.explain)
    ml/artifacts/runs.jsonl              append-only experiment log
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.base import clone
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from ml import config
from ml.cleaning import clean_dataframe, discover_schema
from ml.features import BIN_PREFIX, SymptomFeatureBuilder, build_record_frame

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_XGB = False


# --------------------------------------------------------------------------
# Model zoo
# --------------------------------------------------------------------------
# Small, deliberately coarse grids -- the dataset has ~840 rows, so a wide
# search would just overfit the validation split. Scored by macro F1.
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "logistic_regression": {"clf__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]},
    "naive_bayes": {"alpha": [0.1, 0.5, 1.0, 2.0]},
    "random_forest": {
        "max_depth": [None, 6, 12],
        "min_samples_leaf": [1, 2, 5],
    },
    "neural_net": {
        "clf__alpha": [1e-4, 1e-3, 1e-2],
        "clf__hidden_layer_sizes": [(64, 32), (32,)],
    },
    "xgboost": {
        "max_depth": [2, 3, 4],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [200, 400],
    },
}


def build_models(n_classes: int) -> dict[str, Any]:
    """Baselines, tree ensembles and a small neural net, all class-weighted."""
    seed = config.RANDOM_SEED
    models: dict[str, Any] = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler(with_mean=False)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        C=1.0,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "naive_bayes": BernoulliNB(alpha=1.0),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
        "neural_net": Pipeline(
            [
                ("scale", StandardScaler(with_mean=False)),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        alpha=1e-3,
                        max_iter=600,
                        early_stopping=True,
                        n_iter_no_change=25,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }
    if HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic" if n_classes == 2 else "multi:softprob",
            eval_metric="logloss" if n_classes == 2 else "mlogloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )
    return models


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def score(y_true, y_pred, y_proba, classes: list[str]) -> dict:
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
    }
    try:
        if len(classes) == 2:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_proba[:, 1])), 4)
        else:
            metrics["roc_auc"] = round(
                float(
                    roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="macro"
                    )
                ),
                4,
            )
    except (ValueError, IndexError):
        metrics["roc_auc"] = None
    return metrics


def per_class_report(y_true, y_pred, classes: list[str]) -> dict:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    return {
        k: {m: round(float(v), 4) for m, v in vals.items()} if isinstance(vals, dict) else vals
        for k, vals in report.items()
    }


# --------------------------------------------------------------------------
# Feature schema for the API / frontend
# --------------------------------------------------------------------------
def build_feature_schema(
    records: pd.DataFrame,
    builder: SymptomFeatureBuilder,
    schema,
    classes: list[str],
) -> dict:
    """The contract ``GET /schema`` serves so the form is never hardcoded."""
    term_counts = Counter(t for s in records["symptoms"] for t in s)
    symptoms = [
        {"value": t, "label": t.title(), "count": int(term_counts.get(t, 0))}
        for t in builder.vocabulary_
    ]
    symptoms.sort(key=lambda s: (-s["count"], s["value"]))

    species_levels = builder.categorical_levels_.get("species", [])
    fields: list[dict] = []

    if species_levels:
        fields.append(
            {
                "name": "species",
                "label": "Animal type",
                "type": "select",
                "required": True,
                "options": [{"value": v, "label": v.title()} for v in species_levels],
                "allow_other": True,
                "help": "Species recorded in the training data. Anything else is scored as 'other'.",
            }
        )

    for name, levels_key, label in (("breed", "breed", "Breed"),):
        levels = builder.categorical_levels_.get(levels_key)
        if levels:
            fields.append(
                {
                    "name": name,
                    "label": label,
                    "type": "select",
                    "required": False,
                    "options": [{"value": v, "label": v.title()} for v in levels],
                    "allow_other": True,
                }
            )

    numeric_meta = {
        "age_years": ("Age (years)", 0, 200, 0.1),
        "weight_kg": ("Weight (kg)", 0, 20000, 0.1),
    }
    for col in builder.active_numeric_columns_:
        if col in numeric_meta:
            label, lo, hi, step = numeric_meta[col]
            fields.append(
                {
                    "name": col,
                    "label": label,
                    "type": "number",
                    "required": False,
                    "min": lo,
                    "max": hi,
                    "step": step,
                }
            )

    fields.append(
        {
            "name": "symptoms",
            "label": "Observed symptoms",
            "type": "multiselect",
            "required": True,
            "min_items": 1,
            "options": symptoms,
            "allow_other": True,
            "help": "Select every clinical sign you can observe. Unrecognised terms still count towards symptom burden.",
        }
    )

    if "duration_days" in builder.active_numeric_columns_ or schema.duration:
        fields.append(
            {
                "name": "duration",
                "label": "How long have symptoms been present?",
                "type": "text",
                "required": False,
                "placeholder": "e.g. 2 days, 1 week",
                "help": "Free text; parsed into a day count.",
            }
        )

    indicator_fields = [
        {
            "name": col,
            "label": str(col).replace("_", " ").title(),
            "type": "yesno",
            "required": False,
        }
        for col in schema.binary_columns
    ]
    fields.extend(indicator_fields)

    absent = [
        name
        for name, col in (
            ("breed", schema.breed),
            ("age", schema.age),
            ("weight", schema.weight),
            ("symptom_duration", schema.duration),
        )
        if col is None
    ]

    return {
        "version": 1,
        "target_column": schema.target,
        "target_kind": "disease" if schema.target.lower() != "dangerous" else "risk",
        "classes": classes,
        "fields": fields,
        "symptom_vocabulary": symptoms,
        "binary_columns": list(schema.binary_columns),
        "fields_absent_from_dataset": absent,
        "n_features": len(builder.feature_names_),
        "disclaimer": config.DISCLAIMER,
    }


# --------------------------------------------------------------------------
# Training entry point
# --------------------------------------------------------------------------
def train(csv_path=None, run_shap: bool = True) -> dict:
    csv_path = csv_path or config.RAW_CSV
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(csv_path)
    schema = discover_schema(raw)
    clean = clean_dataframe(raw, schema)
    records = build_record_frame(clean, schema)

    encoder = LabelEncoder()
    y = encoder.fit_transform(clean[schema.target].astype(str))
    classes = [str(c) for c in encoder.classes_]
    class_counts = Counter(clean[schema.target].astype(str))

    print(f"target column : {schema.target}")
    print(f"rows          : {len(raw)} raw -> {len(clean)} clean")
    print(f"classes       : {dict(class_counts)}")

    # Stratified split: a class with a single member cannot be stratified, so
    # guard rather than crash on a long-tailed disease label.
    rare = [c for c, n in class_counts.items() if n < 3]
    if rare:
        print(f"[!] classes with <3 samples are kept but cannot be split evenly: {rare}")

    idx = np.arange(len(clean))
    idx_trainval, idx_test = train_test_split(
        idx,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y if min(class_counts.values()) >= 2 else None,
    )
    y_trainval = y[idx_trainval]
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=config.VAL_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y_trainval if min(Counter(y_trainval).values()) >= 2 else None,
    )
    print(
        f"split         : train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}"
    )

    # The feature builder learns its vocabulary from the TRAIN split only.
    builder = SymptomFeatureBuilder()
    X_train = builder.fit_transform(records.iloc[idx_train])
    X_val = builder.transform(records.iloc[idx_val])
    X_test = builder.transform(records.iloc[idx_test])
    X_trainval = builder.transform(records.iloc[idx_trainval])
    print(f"features      : {X_train.shape[1]} ({len(builder.vocabulary_)} symptom terms)")

    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

    # Persist the processed matrices so the run is reproducible / inspectable.
    processed = X_trainval.copy()
    processed[schema.target] = clean[schema.target].to_numpy()[idx_trainval]
    processed.to_csv(config.PROCESSED_DIR / "features_trainval.csv", index=False)
    X_test.assign(**{schema.target: clean[schema.target].to_numpy()[idx_test]}).to_csv(
        config.PROCESSED_DIR / "features_test.csv", index=False
    )
    clean.to_csv(config.PROCESSED_DIR / "cases_clean.csv", index=False)

    # -- compare -----------------------------------------------------------
    results: list[dict] = []
    fitted: dict[str, Any] = {}
    cv = StratifiedKFold(
        n_splits=min(5, int(min(Counter(y_trainval).values()))),
        shuffle=True,
        random_state=config.RANDOM_SEED,
    )

    best_params: dict[str, dict] = {}

    for name, model in build_models(len(classes)).items():
        started = time.perf_counter()
        fit_kwargs = {}
        if name == "xgboost":
            fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y_train)

        grid = PARAM_GRIDS.get(name)
        if grid and cv.get_n_splits() >= 2:
            search = GridSearchCV(
                model, grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True
            )
            search.fit(X_train, y_train, **fit_kwargs)
            model = search.best_estimator_
            best_params[name] = {k: str(v) for k, v in search.best_params_.items()}
        else:
            model.fit(X_train, y_train, **fit_kwargs)
        elapsed = time.perf_counter() - started

        val_pred = model.predict(X_val)
        val_proba = _safe_proba(model, X_val, len(classes))
        val_metrics = score(y_val, val_pred, val_proba, classes)

        cv_macro_f1 = None
        if cv.get_n_splits() >= 2:
            try:
                cv_scores = cross_val_score(
                    model, X_trainval, y_trainval, cv=cv, scoring="f1_macro"
                )
                cv_macro_f1 = [round(float(s), 4) for s in cv_scores]
            except Exception as exc:  # pragma: no cover - model-specific failures
                print(f"    (cv skipped for {name}: {exc})")

        results.append(
            {
                "model": name,
                "val": val_metrics,
                "cv_macro_f1": cv_macro_f1,
                "cv_macro_f1_mean": round(float(np.mean(cv_macro_f1)), 4)
                if cv_macro_f1
                else None,
                "fit_seconds": round(elapsed, 2),
                "tuned_params": best_params.get(name),
                "params": _describe_params(model),
            }
        )
        fitted[name] = model
        print(
            f"  {name:<20} val acc={val_metrics['accuracy']:.3f} "
            f"macroF1={val_metrics['macro_f1']:.3f} "
            f"balAcc={val_metrics['balanced_accuracy']:.3f} "
            f"({elapsed:.1f}s)"
        )

    # -- pick the winner ---------------------------------------------------
    # Macro F1 is the selection metric: with a 97/3 class split, accuracy would
    # crown the majority-class baseline. Ties break on balanced accuracy.
    candidates = [r for r in results if r["model"] != "majority_baseline"]
    best = max(
        candidates,
        key=lambda r: (r["val"]["macro_f1"], r["val"]["balanced_accuracy"]),
    )
    best_name = best["model"]
    print(f"\nwinner: {best_name} (val macro F1 {best['val']['macro_f1']:.3f})")

    # Refit the winner on train+val before the final untouched-test scoring,
    # reusing the hyperparameters the grid search selected.
    final_model = clone(fitted[best_name])
    fit_kwargs = {}
    if best_name == "xgboost":
        fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y_trainval)
    final_model.fit(X_trainval, y_trainval, **fit_kwargs)

    test_pred = final_model.predict(X_test)
    test_proba = _safe_proba(final_model, X_test, len(classes))
    test_metrics = score(y_test, test_pred, test_proba, classes)
    test_per_class = per_class_report(y_test, test_pred, classes)
    cm = confusion_matrix(y_test, test_pred, labels=list(range(len(classes)))).tolist()

    print(
        f"test: acc={test_metrics['accuracy']:.3f} "
        f"macroF1={test_metrics['macro_f1']:.3f} "
        f"balAcc={test_metrics['balanced_accuracy']:.3f}"
    )

    # -- persist -----------------------------------------------------------
    pipeline = Pipeline([("features", builder), ("classifier", final_model)])
    # Exact linear SHAP needs only the mean of each feature over the training
    # data, so ship those 1-D means instead of making the API re-read the CSV.
    background_means = X_trainval.to_numpy(dtype=float).mean(axis=0)
    joblib.dump(
        {
            "pipeline": pipeline,
            "background_means": background_means,
            "classes": classes,
            "label_encoder": encoder,
            "dataset_schema": schema.to_dict(),
            "feature_names": list(builder.feature_names_),
            "model_name": best_name,
        },
        config.MODEL_PATH,
    )

    feature_schema = build_feature_schema(records, builder, schema, classes)
    config.FEATURE_SCHEMA_PATH.write_text(
        json.dumps(feature_schema, indent=2), encoding="utf-8"
    )

    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report = {
        "trained_at": trained_at,
        "dataset": {
            "path": _relative_to_root(csv_path),
            "n_raw_rows": int(len(raw)),
            "n_clean_rows": int(len(clean)),
            "duplicates_dropped": int(clean.attrs.get("duplicates_dropped", 0)),
            "target_column": schema.target,
            "class_counts": {k: int(v) for k, v in class_counts.items()},
            "imbalance_ratio": round(
                max(class_counts.values()) / max(min(class_counts.values()), 1), 2
            ),
            "columns": list(raw.columns),
            "fields_absent_from_dataset": feature_schema["fields_absent_from_dataset"],
        },
        "split": {
            "train": int(len(idx_train)),
            "val": int(len(idx_val)),
            "test": int(len(idx_test)),
            "stratified": True,
            "test_class_counts": {
                classes[i]: int(n) for i, n in zip(*np.unique(y_test, return_counts=True))
            },
        },
        "n_features": int(X_train.shape[1]),
        "selection_metric": "macro_f1 (validation split)",
        "comparison": results,
        "best_model": best_name,
        "test_metrics": test_metrics,
        "test_per_class": test_per_class,
        "confusion_matrix": {"labels": classes, "matrix": cm},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    config.EVAL_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    model_info = {
        "model_name": best_name,
        "trained_at": trained_at,
        "target_column": schema.target,
        "target_kind": feature_schema["target_kind"],
        "classes": classes,
        "n_features": int(X_train.shape[1]),
        "n_training_rows": int(len(idx_trainval)),
        "selection_metric": "macro_f1",
        "test_metrics": test_metrics,
        "class_counts": {k: int(v) for k, v in class_counts.items()},
        "comparison_summary": [
            {
                "model": r["model"],
                "val_accuracy": r["val"]["accuracy"],
                "val_macro_f1": r["val"]["macro_f1"],
                "val_balanced_accuracy": r["val"]["balanced_accuracy"],
                "tuned_params": r.get("tuned_params"),
            }
            for r in results
        ],
        "disclaimer": config.DISCLAIMER,
    }
    config.MODEL_INFO_PATH.write_text(json.dumps(model_info, indent=2), encoding="utf-8")

    # Append-only experiment log (a lightweight stand-in for MLflow).
    with config.RUN_LOG_PATH.open("a", encoding="utf-8") as fh:
        for r in results:
            fh.write(
                json.dumps(
                    {
                        "run_at": trained_at,
                        "model": r["model"],
                        "params": r["params"],
                        "val_metrics": r["val"],
                        "cv_macro_f1_mean": r["cv_macro_f1_mean"],
                        "selected": r["model"] == best_name,
                        "test_metrics": test_metrics if r["model"] == best_name else None,
                    }
                )
                + "\n"
            )

    print(f"\nartifacts -> {config.ARTIFACT_DIR}")

    if run_shap:
        from ml.explain import compute_global_importance

        compute_global_importance(pipeline, records.iloc[idx_trainval], classes)
    elif not config.SHAP_GLOBAL_PATH.exists():
        # /model-info surfaces this, so never leave it missing.
        from ml.explain import _fallback_importance

        config.SHAP_GLOBAL_PATH.write_text(
            json.dumps(
                _fallback_importance(pipeline, list(builder.feature_names_), classes),
                indent=2,
            ),
            encoding="utf-8",
        )

    return report


def _relative_to_root(path) -> str:
    """Repo-relative path, so reports are not machine-specific."""
    from pathlib import Path

    try:
        return Path(path).resolve().relative_to(config.ROOT).as_posix()
    except ValueError:
        return str(path)


def _safe_proba(model, X, n_classes: int) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        if proba.ndim == 2 and proba.shape[1] == n_classes:
            return proba
    pred = np.asarray(model.predict(X))
    out = np.zeros((len(pred), n_classes))
    out[np.arange(len(pred)), pred.astype(int)] = 1.0
    return out


def _describe_params(model) -> dict:
    estimator = model
    if isinstance(model, Pipeline):
        estimator = model.steps[-1][1]
    params = estimator.get_params()
    return {
        k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
        for k, v in params.items()
        if k in {
            "C", "alpha", "n_estimators", "max_depth", "learning_rate",
            "min_samples_leaf", "hidden_layer_sizes", "class_weight",
            "subsample", "colsample_bytree", "strategy", "max_iter",
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--no-shap", action="store_true", help="skip SHAP computation")
    args = parser.parse_args()
    train(csv_path=args.csv, run_shap=not args.no_shap)


if __name__ == "__main__":
    main()
