"""The numpy closed form must agree with the `shap` package, exactly.

`ml.explain.linear_contributions` is what the deployed API uses, precisely so
that serving does not need `shap` (which pulls in numba + llvmlite, ~150 MB).
That is only safe if it computes the same numbers, so this test pins the two
implementations together. It skips when `shap` is not installed, since `shap`
is deliberately a training-only dependency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml import config
from ml.explain import linear_contributions

shap = pytest.importorskip("shap", reason="shap is a training-only dependency")

pytestmark = pytest.mark.skipif(
    not config.MODEL_PATH.exists(),
    reason="no trained model; run `python -m ml.train` first",
)


@pytest.fixture(scope="module")
def trained():
    import joblib

    from ml.cleaning import clean_dataframe, discover_schema
    from ml.features import build_record_frame

    bundle = joblib.load(config.MODEL_PATH)
    raw = pd.read_csv(config.RAW_CSV)
    schema = discover_schema(raw)
    records = build_record_frame(clean_dataframe(raw, schema), schema)
    return bundle, records


def test_linear_contributions_match_shap_linear_explainer(trained):
    bundle, records = trained
    pipeline = bundle["pipeline"]
    builder = pipeline.named_steps["features"]
    classifier = pipeline.named_steps["classifier"]

    if not hasattr(classifier.steps[-1][1] if hasattr(classifier, "steps") else classifier, "coef_"):
        pytest.skip("deployed model is not linear")

    background = records.iloc[:200]
    X = builder.transform(records.iloc[200:230])
    bg = builder.transform(background)
    means = bg.to_numpy(dtype=float).mean(axis=0)

    mine = linear_contributions(pipeline, X, class_index=1, background_means=means)
    assert mine is not None

    # Same model, same background, via the reference implementation. The masker
    # is pinned to the FULL background: shap's default Independent masker
    # subsamples to 100 rows, which shifts the base value and would make this a
    # comparison of two different background distributions rather than of two
    # implementations.
    from sklearn.pipeline import Pipeline

    estimator = classifier.steps[-1][1]
    pre = Pipeline(classifier.steps[:-1])
    ref_bg = pd.DataFrame(pre.transform(bg), columns=bg.columns)
    ref_X = pd.DataFrame(pre.transform(X), columns=X.columns)
    masker = shap.maskers.Independent(ref_bg, max_samples=len(ref_bg))
    theirs = np.asarray(shap.LinearExplainer(estimator, masker).shap_values(ref_X))
    if theirs.ndim == 3:
        theirs = theirs[:, :, -1]

    # The feature matrix is float32, so ~1e-7 is round-off, not disagreement.
    np.testing.assert_allclose(mine, theirs, rtol=1e-4, atol=1e-6)


def test_contributions_reconstruct_the_margin(trained):
    """SHAP values are additive: sum(phi) + base == the model's logit."""
    bundle, records = trained
    pipeline = bundle["pipeline"]
    builder = pipeline.named_steps["features"]
    classifier = pipeline.named_steps["classifier"]

    background = records.iloc[:200]
    bg = builder.transform(background)
    means = bg.to_numpy(dtype=float).mean(axis=0)
    X = builder.transform(records.iloc[200:210])

    phi = linear_contributions(pipeline, X, class_index=1, background_means=means)
    if phi is None:
        pytest.skip("deployed model is not linear")

    margin = classifier.decision_function(X)
    base = classifier.decision_function(pd.DataFrame(means.reshape(1, -1), columns=bg.columns))
    np.testing.assert_allclose(phi.sum(axis=1) + base[0], margin, rtol=1e-4, atol=1e-5)


def test_class_zero_contributions_are_negated(trained):
    """For a binary model, flipping the class flips every contribution."""
    bundle, records = trained
    pipeline = bundle["pipeline"]
    builder = pipeline.named_steps["features"]
    bg = builder.transform(records.iloc[:200])
    means = bg.to_numpy(dtype=float).mean(axis=0)
    X = builder.transform(records.iloc[200:205])

    pos = linear_contributions(pipeline, X, class_index=1, background_means=means)
    neg = linear_contributions(pipeline, X, class_index=0, background_means=means)
    if pos is None:
        pytest.skip("deployed model is not linear")
    np.testing.assert_allclose(pos, -neg, rtol=1e-9, atol=1e-12)


def test_returns_none_for_non_linear_model(trained):
    """Callers rely on None to fall back to the sampling explainer."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

    bundle, records = trained
    builder = bundle["pipeline"].named_steps["features"]
    X = builder.transform(records.iloc[:20])
    forest = RandomForestClassifier(n_estimators=3, random_state=0).fit(
        X, [0, 1] * 10
    )
    pipeline = Pipeline([("features", builder), ("classifier", forest)])
    assert linear_contributions(pipeline, X, 1, background_means=np.zeros(X.shape[1])) is None
