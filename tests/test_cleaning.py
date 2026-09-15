"""Unit tests for schema discovery and the data-cleaning primitives."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from ml.cleaning import (
    canonical_animal,
    canonical_symptom,
    clean_dataframe,
    discover_schema,
    encode_yes_no,
    normalize_text,
    parse_duration_to_days,
    symptom_sets,
)


# --------------------------------------------------------------------------
# normalize_text / canonicalisation
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Loss of  appetite", "loss of appetite"),
        ("  FEVER ", "fever"),
        ("Pneumonia�", "pneumonia"),
        ("Weight-loss", "weight loss"),
        (None, ""),
        (float("nan"), ""),
        ("", ""),
    ],
)
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Pains", "pain"),
        ("Anorexia", "loss of appetite"),
        ("Poor Appetite", "loss of appetite"),
        ("Difficulty breathing", "difficulty in breathing"),
        ("Dyspnea", "difficulty in breathing"),
        ("Tiredness", "lethargy"),
        ("Sudden death", "death"),
        ("Teeth griding", "grinding of teeth"),
        ("Some Novel Sign", "some novel sign"),
    ],
)
def test_canonical_symptom_merges_variants(raw, expected):
    assert canonical_symptom(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [("Dogs", "dog"), ("cow", "cattle"), ("Other birds", "birds"), ("Sheep", "sheep")],
)
def test_canonical_animal(raw, expected):
    assert canonical_animal(raw) == expected


# --------------------------------------------------------------------------
# Yes/No encoding
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected", [("Yes", 1.0), ("y", 1.0), ("TRUE", 1.0), ("No", 0.0), ("0", 0.0)]
)
def test_encode_yes_no(raw, expected):
    assert encode_yes_no(raw) == expected


@pytest.mark.parametrize("raw", ["maybe", "", None, "n/a"])
def test_encode_yes_no_unknown_is_nan(raw):
    assert math.isnan(encode_yes_no(raw))


# --------------------------------------------------------------------------
# Duration parsing
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2 days", 2.0),
        ("1 week", 7.0),
        ("three weeks", 21.0),
        ("36 hours", 1.5),
        ("2-3 days", 2.5),
        ("2 to 4 days", 3.0),
        ("1 week 2 days", 9.0),
        ("1 month", 30.0),
        ("half a day", 0.5),
        ("5", 5.0),  # bare number means days
        ("10 d", 10.0),
    ],
)
def test_parse_duration_to_days(raw, expected):
    assert parse_duration_to_days(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["", None, "chronic", "unknown"])
def test_parse_duration_unparseable_is_nan(raw):
    assert math.isnan(parse_duration_to_days(raw))


# --------------------------------------------------------------------------
# Schema discovery + frame cleaning
# --------------------------------------------------------------------------
@pytest.fixture
def shipped_shape_df() -> pd.DataFrame:
    """Mirrors the real data/data.csv layout."""
    return pd.DataFrame(
        {
            "AnimalName": ["Dog", "Dogs", "cow", "Sheep", "Dog"],
            "symptoms1": ["Fever", "Fever", "Fever", "wasting", "Fever"],
            "symptoms2": ["Diarrhea", "Diarrhea", "Chills", "Depression", "Diarrhea"],
            "symptoms3": ["Vomiting", "Vomiting", "Coughing", "Lethargy", "Vomiting"],
            "symptoms4": ["Weight loss", "Weight loss", "Pains", "Anorexia", "Weight loss"],
            "symptoms5": ["Dehydration", "Dehydration", "Pain", "Death", "Dehydration"],
            "Dangerous": ["Yes", "Yes", "No", "Yes", None],
        }
    )


@pytest.fixture
def rich_df() -> pd.DataFrame:
    """The richer schema the project proposal describes."""
    return pd.DataFrame(
        {
            "Animal_Type": ["Dog", "Cat"],
            "Breed": ["Labrador", "Siamese"],
            "Age": [3, 7],
            "Weight": [28.0, 4.2],
            "Symptom_1": ["Fever", "Coughing"],
            "Symptom_2": ["Vomiting", "Sneezing"],
            "Duration": ["2 days", "1 week"],
            "Appetite_Loss": ["Yes", "No"],
            "Disease": ["Parvovirus", "Kennel Cough"],
        }
    )


def test_discover_schema_on_shipped_layout(shipped_shape_df):
    schema = discover_schema(shipped_shape_df)
    assert schema.target == "Dangerous"
    assert schema.species == "AnimalName"
    assert schema.symptom_columns == [f"symptoms{i}" for i in range(1, 6)]
    assert schema.breed is None and schema.age is None and schema.duration is None


def test_discover_schema_prefers_disease_over_binary_flag(rich_df):
    schema = discover_schema(rich_df)
    assert schema.target == "Disease"
    assert schema.species == "Animal_Type"
    assert schema.breed == "Breed"
    assert schema.age == "Age"
    assert schema.weight == "Weight"
    assert schema.duration == "Duration"
    assert schema.symptom_columns == ["Symptom_1", "Symptom_2"]
    assert schema.binary_columns == ["Appetite_Loss"]


def test_discover_schema_without_target_raises():
    with pytest.raises(ValueError, match="target column"):
        discover_schema(pd.DataFrame({"AnimalName": ["Dog"], "symptoms1": ["Fever"]}))


def test_clean_dataframe_drops_null_targets_and_duplicates(shipped_shape_df):
    schema = discover_schema(shipped_shape_df)
    clean = clean_dataframe(shipped_shape_df, schema)

    # Row 4 has a null target; rows 0 and 1 differ only by "Dog"/"Dogs" so both
    # survive (dedup is on raw values), leaving 4 rows.
    assert len(clean) == 4
    assert clean[schema.target].tolist() == ["Yes", "Yes", "No", "Yes"]
    assert clean["species"].tolist() == ["dog", "dog", "cattle", "sheep"]


def test_clean_dataframe_removes_exact_duplicates(shipped_shape_df):
    doubled = pd.concat([shipped_shape_df, shipped_shape_df], ignore_index=True)
    schema = discover_schema(doubled)
    clean = clean_dataframe(doubled, schema)
    assert clean.attrs["duplicates_dropped"] == 4
    assert len(clean) == 4


def test_clean_dataframe_parses_rich_columns(rich_df):
    schema = discover_schema(rich_df)
    clean = clean_dataframe(rich_df, schema)
    assert clean["duration_days"].tolist() == [2.0, 7.0]
    assert clean["age_years"].tolist() == [3.0, 7.0]
    assert clean["Appetite_Loss__bin"].tolist() == [1.0, 0.0]
    assert clean["breed"].tolist() == ["labrador", "siamese"]
    assert clean["Disease"].tolist() == ["Parvovirus", "Kennel Cough"]


def test_symptom_sets_canonicalises_and_dedupes(shipped_shape_df):
    schema = discover_schema(shipped_shape_df)
    clean = clean_dataframe(shipped_shape_df, schema)
    sets = symptom_sets(clean, schema)

    assert sets.iloc[0] == {"fever", "diarrhea", "vomiting", "weight loss", "dehydration"}
    # "Pains" and "Pain" both canonicalise to "pain", so the set collapses them.
    assert sets.iloc[2] == {"fever", "chills", "coughing", "pain"}
    # "Anorexia" -> "loss of appetite", "wasting" -> "weight loss".
    assert "loss of appetite" in sets.iloc[3]
    assert "weight loss" in sets.iloc[3]
