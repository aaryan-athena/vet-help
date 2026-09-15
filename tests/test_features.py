"""Unit tests for the feature-engineering transformer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.features import BIN_PREFIX, SymptomFeatureBuilder, build_record_frame, request_to_record
from ml.cleaning import clean_dataframe, discover_schema


@pytest.fixture
def records() -> pd.DataFrame:
    """A record frame with a clear frequency split in the symptom vocabulary."""
    rows = []
    for _ in range(6):
        rows.append({"species": "dog", "symptoms": {"fever", "diarrhea"}})
    for _ in range(4):
        rows.append({"species": "cat", "symptoms": {"fever", "coughing"}})
    rows.append({"species": "horse", "symptoms": {"fever", "a very rare sign"}})
    frame = pd.DataFrame(rows)
    frame["age_years"] = np.nan
    frame["weight_kg"] = np.nan
    frame["duration_days"] = np.nan
    return frame


def test_vocabulary_respects_min_frequency(records):
    builder = SymptomFeatureBuilder(min_symptom_freq=3).fit(records)
    assert set(builder.vocabulary_) == {"fever", "diarrhea", "coughing"}
    assert "a very rare sign" not in builder.vocabulary_


def test_multi_hot_and_counts(records):
    builder = SymptomFeatureBuilder(min_symptom_freq=3, n_pairs=5).fit(records)
    X = builder.transform(records)

    assert list(X.columns) == builder.feature_names_
    assert X.shape[0] == len(records)
    assert X["sym__fever"].tolist() == [1.0] * 11
    assert X["sym__diarrhea"].tolist() == [1.0] * 6 + [0.0] * 5
    # Every row has two symptoms; only the last has an out-of-vocabulary one.
    assert X["n_symptoms"].tolist() == [2.0] * 11
    assert X["n_rare_symptoms"].tolist() == [0.0] * 10 + [1.0]


def test_cooccurrence_pairs(records):
    builder = SymptomFeatureBuilder(min_symptom_freq=3, n_pairs=5).fit(records)
    X = builder.transform(records)
    assert "pair__diarrhea+fever" in X.columns
    assert X["pair__diarrhea+fever"].tolist() == [1.0] * 6 + [0.0] * 5


def test_rare_species_falls_into_other_bucket(records):
    builder = SymptomFeatureBuilder(min_symptom_freq=3, min_category_freq=3).fit(records)
    X = builder.transform(records)
    assert "species__dog" in X.columns and "species__cat" in X.columns
    assert "species__horse" not in X.columns  # seen once, below min_category_freq
    assert X["species__other"].tolist() == [0.0] * 10 + [1.0]


def test_transform_is_stable_for_unseen_values(records):
    builder = SymptomFeatureBuilder(min_symptom_freq=3).fit(records)
    unseen = pd.DataFrame(
        [{"species": "capybara", "symptoms": {"glowing", "fever"}}]
    )
    X = builder.transform(unseen)
    assert list(X.columns) == builder.feature_names_
    assert X["species__other"].iloc[0] == 1.0
    assert X["sym__fever"].iloc[0] == 1.0
    assert X["n_rare_symptoms"].iloc[0] == 1.0


def test_numeric_columns_are_dropped_when_entirely_missing(records):
    builder = SymptomFeatureBuilder().fit(records)
    assert builder.active_numeric_columns_ == []
    assert not any(c.startswith("age_bin__") for c in builder.feature_names_)


def test_numeric_columns_are_binned_when_present(records):
    frame = records.copy()
    frame["age_years"] = [0.5, 2, 5, 9, 15, 3, 4, 6, 8, 11, 2]
    frame["duration_days"] = [1, 3, 10, 40, 2, 5, 1, 7, 20, 60, 3]
    builder = SymptomFeatureBuilder().fit(frame)
    X = builder.transform(frame)

    assert "age_years" in X.columns
    assert X["age_bin__infant"].iloc[0] == 1.0
    assert X["age_bin__senior"].iloc[4] == 1.0
    assert X["duration_bin__acute"].iloc[0] == 1.0
    assert X["duration_bin__chronic"].iloc[3] == 1.0


def test_missing_numerics_are_median_imputed(records):
    frame = records.copy()
    frame["age_years"] = [1, 2, 3, np.nan, 5, 6, 7, 8, 9, 10, 11]
    builder = SymptomFeatureBuilder().fit(frame)
    X = builder.transform(frame)
    assert X["age_years"].iloc[3] == pytest.approx(builder.numeric_medians_["age_years"])
    assert not X.isna().any().any()


def test_request_to_record_matches_training_canonicalisation():
    record = request_to_record(
        {
            "species": "Dogs",
            "symptoms": ["Pains", "Anorexia", "Fever", "  "],
            "duration": "1 week",
            "age_years": 4,
        },
        binary_columns=["Appetite_Loss"],
    )
    assert record.loc[0, "species"] == "dog"
    assert record.loc[0, "symptoms"] == {"pain", "loss of appetite", "fever"}
    assert record.loc[0, "duration_days"] == 7.0
    assert record.loc[0, "age_years"] == 4.0
    assert f"{BIN_PREFIX}Appetite_Loss" in record.columns


def test_end_to_end_on_shipped_layout():
    """The full cleaning -> record -> features path on a realistic frame."""
    raw = pd.DataFrame(
        {
            "AnimalName": ["Dog", "Dog", "Sheep", "Sheep", "cat", "cat"],
            "symptoms1": ["Fever"] * 6,
            "symptoms2": ["Diarrhea", "Diarrhea", "Coughing", "Coughing", "Pains", "Pain"],
            "Dangerous": ["Yes", "No", "Yes", "Yes", "No", "Yes"],
        }
    )
    schema = discover_schema(raw)
    records = build_record_frame(clean_dataframe(raw, schema), schema)
    X = SymptomFeatureBuilder(min_symptom_freq=2).fit_transform(records)

    # Rows 2 and 3 are identical, so cleaning dedupes them: 6 raw -> 5 cases.
    assert len(X) == 5
    assert X.dtypes.unique().tolist() == [np.dtype("float32")]
    assert X["sym__fever"].sum() == 5
    # "Pains" and "Pain" collapse onto one feature rather than two.
    assert "sym__pain" in X.columns
    assert X["sym__pain"].sum() == 2
