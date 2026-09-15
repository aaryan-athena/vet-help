"""Feature engineering for VetDx.

The pipeline funnels both training rows and live API requests through the same
intermediate representation -- a "record frame" with one row per case and these
columns:

    species        str            canonical species name ("" if unknown)
    breed          str            canonical breed name (only if the CSV has one)
    symptoms       set[str]       canonical symptom terms observed
    age_years      float          NaN if unknown / column absent
    weight_kg      float          NaN if unknown / column absent
    duration_days  float          parsed from free text, NaN if unknown
    bin__<name>    float          0/1 behavioural indicators (Yes/No columns)

``SymptomFeatureBuilder`` is a plain scikit-learn transformer over that frame,
so it serialises with the model and guarantees train/serve parity.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ml import config
from ml.cleaning import (
    DatasetSchema,
    canonical_animal,
    canonical_symptom,
    parse_duration_to_days,
    symptom_sets,
)

BIN_PREFIX = "bin__"


# --------------------------------------------------------------------------
# Record frame construction
# --------------------------------------------------------------------------
def build_record_frame(df: pd.DataFrame, schema: DatasetSchema) -> pd.DataFrame:
    """Turn a *cleaned* dataframe into the canonical record frame."""
    records = pd.DataFrame(index=df.index)
    records["species"] = df["species"] if "species" in df else ""
    if "breed" in df:
        records["breed"] = df["breed"]
    records["symptoms"] = symptom_sets(df, schema)
    records["age_years"] = df["age_years"] if "age_years" in df else np.nan
    records["weight_kg"] = df["weight_kg"] if "weight_kg" in df else np.nan
    records["duration_days"] = df["duration_days"] if "duration_days" in df else np.nan
    for col in schema.binary_columns:
        records[f"{BIN_PREFIX}{col}"] = df[f"{col}__bin"]
    return records


def request_to_record(payload: dict, binary_columns: list[str]) -> pd.DataFrame:
    """Build a one-row record frame from an API request payload.

    Applies exactly the same canonicalisation as the training path.
    """
    symptoms = {
        term
        for term in (canonical_symptom(s) for s in payload.get("symptoms") or [])
        if term
    }
    row: dict = {
        "species": canonical_animal(payload.get("species") or payload.get("animal_type")),
        "symptoms": symptoms,
        "age_years": _as_float(payload.get("age_years", payload.get("age"))),
        "weight_kg": _as_float(payload.get("weight_kg", payload.get("weight"))),
        "duration_days": _duration_from_payload(payload),
    }
    if payload.get("breed") is not None:
        from ml.cleaning import normalize_text

        row["breed"] = normalize_text(payload.get("breed"))
    indicators = {
        str(k).strip().lower(): v for k, v in (payload.get("indicators") or {}).items()
    }
    for col in binary_columns:
        from ml.cleaning import encode_yes_no

        row[f"{BIN_PREFIX}{col}"] = encode_yes_no(indicators.get(str(col).strip().lower()))
    return pd.DataFrame([row])


def _as_float(value) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _duration_from_payload(payload: dict) -> float:
    if payload.get("duration_days") is not None:
        return _as_float(payload["duration_days"])
    return parse_duration_to_days(payload.get("duration"))


# --------------------------------------------------------------------------
# Transformer
# --------------------------------------------------------------------------
class SymptomFeatureBuilder(BaseEstimator, TransformerMixin):
    """Vectorise a record frame into a dense numeric feature matrix.

    Produces, in a stable column order:

    * ``sym__<term>``  -- multi-hot over the symptom vocabulary
    * ``pair__<a>+<b>`` -- co-occurrence flags for the most common symptom pairs
    * ``n_symptoms`` / ``n_rare_symptoms`` -- symptom burden and out-of-vocabulary count
    * ``species__<name>`` / ``breed__<name>`` -- one-hot with an ``__other`` bucket
    * ``age_years`` / ``weight_kg`` / ``duration_days`` -- median-imputed numerics
    * ``age_bin__*`` / ``duration_bin__*`` -- binned versions of the above
    * ``bin__<col>`` -- Yes/No behavioural indicators, median-imputed
    """

    def __init__(
        self,
        min_symptom_freq: int = config.MIN_SYMPTOM_FREQ,
        n_pairs: int = config.N_COOCCURRENCE_PAIRS,
        min_category_freq: int = 3,
    ):
        self.min_symptom_freq = min_symptom_freq
        self.n_pairs = n_pairs
        self.min_category_freq = min_category_freq

    # -- fit ---------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None) -> "SymptomFeatureBuilder":
        sets = list(X["symptoms"])

        term_counts = Counter(t for s in sets for t in s)
        self.vocabulary_ = sorted(
            t for t, n in term_counts.items() if n >= self.min_symptom_freq
        )
        vocab_set = set(self.vocabulary_)

        pair_counts: Counter = Counter()
        for s in sets:
            known = sorted(s & vocab_set)
            pair_counts.update(combinations(known, 2))
        self.pairs_ = [p for p, _ in pair_counts.most_common(self.n_pairs)]

        self.categorical_levels_: dict[str, list[str]] = {}
        for col in ("species", "breed"):
            if col in X.columns:
                counts = Counter(X[col].fillna("").astype(str))
                self.categorical_levels_[col] = sorted(
                    v for v, n in counts.items() if v and n >= self.min_category_freq
                )

        self.binary_columns_ = [c for c in X.columns if c.startswith(BIN_PREFIX)]

        self.numeric_columns_ = [
            c for c in ("age_years", "weight_kg", "duration_days") if c in X.columns
        ]
        self.numeric_medians_ = {}
        self.active_numeric_columns_ = []
        for col in self.numeric_columns_:
            series = pd.to_numeric(X[col], errors="coerce")
            if series.notna().any():
                self.numeric_medians_[col] = float(series.median())
                self.active_numeric_columns_.append(col)

        self.binary_medians_ = {}
        for col in self.binary_columns_:
            series = pd.to_numeric(X[col], errors="coerce")
            self.binary_medians_[col] = (
                float(series.median()) if series.notna().any() else 0.0
            )

        self.feature_names_ = self._feature_names()
        return self

    def _feature_names(self) -> list[str]:
        names = [f"sym__{t}" for t in self.vocabulary_]
        names += [f"pair__{a}+{b}" for a, b in self.pairs_]
        names += ["n_symptoms", "n_rare_symptoms"]
        for col, levels in self.categorical_levels_.items():
            names += [f"{col}__{v}" for v in levels] + [f"{col}__other"]
        for col in self.active_numeric_columns_:
            names.append(col)
        if "age_years" in self.active_numeric_columns_:
            names += [f"age_bin__{b}" for b in config.AGE_BIN_LABELS]
        if "duration_days" in self.active_numeric_columns_:
            names += [f"duration_bin__{b}" for b in config.DURATION_BIN_LABELS]
        names += list(self.binary_columns_)
        return names

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.feature_names_, dtype=object)

    # -- transform ---------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SymptomFeatureBuilder expects a record DataFrame")

        n = len(X)
        vocab_set = set(self.vocabulary_)
        sets = [set(s) if isinstance(s, (set, list, tuple)) else set() for s in X["symptoms"]]
        blocks: list[pd.DataFrame] = []

        multi_hot = np.zeros((n, len(self.vocabulary_)), dtype=np.float32)
        index = {t: i for i, t in enumerate(self.vocabulary_)}
        for r, s in enumerate(sets):
            for term in s:
                col = index.get(term)
                if col is not None:
                    multi_hot[r, col] = 1.0
        blocks.append(
            pd.DataFrame(multi_hot, columns=[f"sym__{t}" for t in self.vocabulary_])
        )

        if self.pairs_:
            pair_arr = np.zeros((n, len(self.pairs_)), dtype=np.float32)
            for r, s in enumerate(sets):
                for c, (a, b) in enumerate(self.pairs_):
                    if a in s and b in s:
                        pair_arr[r, c] = 1.0
            blocks.append(
                pd.DataFrame(
                    pair_arr, columns=[f"pair__{a}+{b}" for a, b in self.pairs_]
                )
            )

        counts = pd.DataFrame(
            {
                "n_symptoms": [float(len(s)) for s in sets],
                "n_rare_symptoms": [float(len(s - vocab_set)) for s in sets],
            }
        )
        blocks.append(counts)

        for col, levels in self.categorical_levels_.items():
            values = (
                X[col].fillna("").astype(str).tolist()
                if col in X.columns
                else [""] * n
            )
            arr = np.zeros((n, len(levels) + 1), dtype=np.float32)
            level_index = {v: i for i, v in enumerate(levels)}
            for r, v in enumerate(values):
                arr[r, level_index.get(v, len(levels))] = 1.0
            blocks.append(
                pd.DataFrame(
                    arr, columns=[f"{col}__{v}" for v in levels] + [f"{col}__other"]
                )
            )

        numeric_frame = {}
        for col in self.active_numeric_columns_:
            series = (
                pd.to_numeric(X[col], errors="coerce")
                if col in X.columns
                else pd.Series([np.nan] * n)
            )
            numeric_frame[col] = series.fillna(self.numeric_medians_[col]).to_numpy(
                dtype=np.float32
            )
        if numeric_frame:
            blocks.append(pd.DataFrame(numeric_frame))

        if "age_years" in self.active_numeric_columns_:
            blocks.append(
                _binned(
                    numeric_frame["age_years"],
                    config.AGE_BINS,
                    config.AGE_BIN_LABELS,
                    "age_bin__",
                )
            )
        if "duration_days" in self.active_numeric_columns_:
            blocks.append(
                _binned(
                    numeric_frame["duration_days"],
                    config.DURATION_BINS,
                    config.DURATION_BIN_LABELS,
                    "duration_bin__",
                )
            )

        if self.binary_columns_:
            bin_frame = {}
            for col in self.binary_columns_:
                series = (
                    pd.to_numeric(X[col], errors="coerce")
                    if col in X.columns
                    else pd.Series([np.nan] * n)
                )
                bin_frame[col] = series.fillna(self.binary_medians_[col]).to_numpy(
                    dtype=np.float32
                )
            blocks.append(pd.DataFrame(bin_frame))

        out = pd.concat([b.reset_index(drop=True) for b in blocks], axis=1)
        return out.reindex(columns=self.feature_names_, fill_value=0.0).astype(np.float32)


def _binned(values, bins, labels, prefix: str) -> pd.DataFrame:
    cats = pd.cut(pd.Series(values), bins=bins, labels=labels, include_lowest=True)
    dummies = pd.get_dummies(cats).astype(np.float32)
    dummies = dummies.reindex(columns=labels, fill_value=0.0)
    dummies.columns = [f"{prefix}{c}" for c in labels]
    return dummies
