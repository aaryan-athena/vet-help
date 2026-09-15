"""Schema discovery and data cleaning for the VetDx pipeline.

The functions here are deliberately small and pure so they can be unit tested
without touching disk (see ``tests/test_cleaning.py``).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

import pandas as pd

from ml import config


# --------------------------------------------------------------------------
# Schema discovery
# --------------------------------------------------------------------------
@dataclass
class DatasetSchema:
    """Which real columns of the CSV play which role in the pipeline."""

    target: str
    species: str | None = None
    breed: str | None = None
    age: str | None = None
    weight: str | None = None
    duration: str | None = None
    symptom_columns: list[str] = field(default_factory=list)
    binary_columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _find_column(columns: Sequence[str], candidates: Iterable[str]) -> str | None:
    """Return the first column whose lowercased name matches a candidate."""
    lookup = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        if cand in lookup:
            return lookup[cand]
    return None


def _looks_binary(series: pd.Series) -> bool:
    values = {str(v).strip().lower() for v in series.dropna().unique()}
    if not values or len(values) > 3:
        return False
    return values <= (config.YES_VALUES | config.NO_VALUES)


def discover_schema(df: pd.DataFrame) -> DatasetSchema:
    """Infer the role of every column of ``df`` without hardcoding names."""
    cols = list(df.columns)

    target = _find_column(cols, config.TARGET_CANDIDATES)
    if target is None:
        raise ValueError(
            "Could not find a target column. Expected one of "
            f"{config.TARGET_CANDIDATES} (case-insensitive); got {cols}."
        )

    symptom_columns = [
        c
        for c in cols
        if str(c).strip().lower().startswith(config.SYMPTOM_PREFIXES) and c != target
    ]

    species = _find_column(cols, config.SPECIES_CANDIDATES)
    breed = _find_column(cols, config.BREED_CANDIDATES)
    age = _find_column(cols, config.AGE_CANDIDATES)
    weight = _find_column(cols, config.WEIGHT_CANDIDATES)
    duration = _find_column(cols, config.DURATION_CANDIDATES)

    structural = {target, species, breed, age, weight, duration, *symptom_columns}
    binary_columns = [c for c in cols if c not in structural and _looks_binary(df[c])]

    return DatasetSchema(
        target=target,
        species=species,
        breed=breed,
        age=age,
        weight=weight,
        duration=duration,
        symptom_columns=symptom_columns,
        binary_columns=binary_columns,
    )


# --------------------------------------------------------------------------
# Text normalisation
# --------------------------------------------------------------------------
_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]")


def normalize_text(value) -> str:
    """Lowercase, strip punctuation/mojibake and collapse whitespace.

    The shipped dataset contains U+FFFD replacement characters, double spaces
    and inconsistent casing ("Loss of  appetite" vs "loss of appetite"), all of
    which would otherwise become distinct categories.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).replace("�", " ").replace(" ", " ")
    text = _NON_ALNUM_RE.sub(" ", text.lower())
    return _WS_RE.sub(" ", text).strip()


# Clinically equivalent phrasings collapsed onto one canonical term. Keys are
# already-normalised strings; anything not listed keeps its normalised form.
SYMPTOM_SYNONYMS: dict[str, str] = {
    "pains": "pain",
    "painful": "pain",
    "poor appetite": "loss of appetite",
    "loss of apetite": "loss of appetite",
    "lack of appetite": "loss of appetite",
    "reduced appetite": "loss of appetite",
    "stop eating": "loss of appetite",
    "not eating": "loss of appetite",
    "anorexia": "loss of appetite",
    "inappetence": "loss of appetite",
    "loss of weight": "weight loss",
    "wasting": "weight loss",
    "emaciation": "weight loss",
    "difficulty breathing": "difficulty in breathing",
    "difficult breathing": "difficulty in breathing",
    "breathing difficulty": "difficulty in breathing",
    "laboured breathing": "difficulty in breathing",
    "labored breathing": "difficulty in breathing",
    "dyspnea": "difficulty in breathing",
    "dyspnoea": "difficulty in breathing",
    "cough": "coughing",
    "coughs": "coughing",
    "dull ness": "dullness",
    "dull": "dullness",
    "listlessness": "lethargy",
    "tiredness": "lethargy",
    "fatique": "lethargy",
    "fatigue": "lethargy",
    "weakness": "lethargy",
    "sudden death": "death",
    "diarrhoea": "diarrhea",
    "loose stool": "diarrhea",
    "loose stools": "diarrhea",
    "watery faeces": "diarrhea",
    "watery feces": "diarrhea",
    "vomit": "vomiting",
    "vomitting": "vomiting",
    "nasal discharges": "nasal discharge",
    "runny nose": "nasal discharge",
    "sneeze": "sneezing",
    "swollen": "swelling",
    "teeth griding": "grinding of teeth",
    "teeth grinding": "grinding of teeth",
    "excessive drooling": "drooling",
    "salivation": "drooling",
    "excessive salivation": "drooling",
    "staggering": "stumbling",
    "convulsions": "convulsion",
    "seizure": "seizures",
    "fits": "seizures",
    "high fever": "fever",
    "high temperature": "fever",
    "pyrexia": "fever",
    "depressed": "depression",
    "anaemia": "anemia",
    "paleness": "anemia",
    "lameness in affected leg": "lameness",
    "difficulty in walking": "lameness",
    "aversion to light": "photophobia",
    "anversion to light": "photophobia",
}


def canonical_symptom(value) -> str:
    """Normalise then map a raw symptom string to its canonical term."""
    term = normalize_text(value)
    return SYMPTOM_SYNONYMS.get(term, term)


ANIMAL_SYNONYMS: dict[str, str] = {
    "dogs": "dog",
    "cats": "cat",
    "pigs": "pig",
    "goats": "goat",
    "buffalo": "buffaloes",
    "buffalos": "buffaloes",
    "cow": "cattle",
    "cows": "cattle",
    "cattles": "cattle",
    "ox": "cattle",
    "bull": "cattle",
    "mules": "mule",
    "wolves": "wolf",
    "hyaenas": "hyaena",
    "hyenas": "hyaena",
    "moos": "moose",
    "other birds": "birds",
    "bird": "birds",
    "fowls": "fowl",
    "hen": "chicken",
    "chickens": "chicken",
    "mule deer": "deer",
    "sika deer": "deer",
    "white tailed deer": "deer",
    "black tailed deer": "deer",
    "reindeer": "deer",
    "elk": "deer",
    "wapiti": "deer",
}


def canonical_animal(value) -> str:
    term = normalize_text(value)
    return ANIMAL_SYNONYMS.get(term, term)


# --------------------------------------------------------------------------
# Yes/No and duration parsing
# --------------------------------------------------------------------------
def encode_yes_no(value) -> float:
    """Map a Yes/No-ish value to 1.0 / 0.0. Unknowns become NaN."""
    term = normalize_text(value)
    if term in config.YES_VALUES:
        return 1.0
    if term in config.NO_VALUES:
        return 0.0
    return float("nan")


_DURATION_UNITS = {
    "hour": 1 / 24,
    "hr": 1 / 24,
    "h": 1 / 24,
    "day": 1.0,
    "d": 1.0,
    "night": 1.0,
    "week": 7.0,
    "wk": 7.0,
    "w": 7.0,
    "fortnight": 14.0,
    "month": 30.0,
    "mo": 30.0,
    "mon": 30.0,
    "year": 365.0,
    "yr": 365.0,
    "y": 365.0,
}

_NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "half": 0.5,
    "couple": 2,
    "few": 3,
    "several": 4,
}

_UNIT_ALTERNATION = "|".join(sorted(_DURATION_UNITS, key=len, reverse=True))
_QTY_ALTERNATION = "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))

_DURATION_RE = re.compile(
    r"(?:(?P<qty>\d+(?:\.\d+)?|" + _QTY_ALTERNATION + r")\s*)?"
    r"(?P<unit>" + _UNIT_ALTERNATION + r")s?\b"
)

# normalize_text() has already stripped hyphens, so "2-3 days" arrives as
# "2 3 days"; the separator is therefore optional here.
_RANGE_RE = re.compile(
    r"^(\d+(?:\.\d+)?)\s*(?:-|to|or)?\s+(\d+(?:\.\d+)?)\s*(?P<unit>[a-z]+)"
)


def parse_duration_to_days(value) -> float:
    """Parse a free-text symptom duration into a day count.

    "2 days" -> 2.0, "1 week" -> 7.0, "three weeks" -> 21.0, "36 hours" -> 1.5,
    "2-3 days" -> 2.5, "1 week 2 days" -> 9.0. Unparseable input -> NaN.
    """
    text = normalize_text(value)
    if not text:
        return float("nan")

    # "half a day" / "half an hour" -> a plain 0.5 quantity.
    text = re.sub(r"\bhalf\s+an?\b", "0.5", text)

    # A bare number with no unit is interpreted as days.
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)

    # Ranges such as "2-3 days" / "2 to 3 days" -> midpoint of the range.
    range_match = _RANGE_RE.match(text)
    if range_match:
        unit = _DURATION_UNITS.get(range_match.group("unit").rstrip("s"))
        if unit is not None:
            lo, hi = float(range_match.group(1)), float(range_match.group(2))
            return (lo + hi) / 2 * unit

    total = 0.0
    found = False
    for match in _DURATION_RE.finditer(text):
        unit = _DURATION_UNITS[match.group("unit")]
        raw_qty = match.group("qty")
        if raw_qty is None:
            qty = 1.0
        elif raw_qty in _NUMBER_WORDS:
            qty = float(_NUMBER_WORDS[raw_qty])
        else:
            qty = float(raw_qty)
        total += qty * unit
        found = True

    return total if found else float("nan")


# --------------------------------------------------------------------------
# Frame-level cleaning
# --------------------------------------------------------------------------
def clean_dataframe(df: pd.DataFrame, schema: DatasetSchema) -> pd.DataFrame:
    """Apply every cleaning rule and return a tidy frame.

    Steps: drop rows with a missing target, drop exact duplicates, canonicalise
    species and symptom text, encode Yes/No indicators to 0/1, parse duration
    strings into ``duration_days``, and coerce age/weight to numeric.
    """
    out = df.copy()

    out = out[out[schema.target].notna()]
    out[schema.target] = out[schema.target].map(normalize_text).str.title()
    out = out[out[schema.target] != ""]

    n_before = len(out)
    out = out.drop_duplicates()
    dropped = n_before - len(out)

    if schema.species:
        out["species"] = out[schema.species].map(canonical_animal).replace("", "unknown")
    if schema.breed:
        out["breed"] = out[schema.breed].map(normalize_text).replace("", "unknown")

    for col in schema.symptom_columns:
        out[col] = out[col].map(canonical_symptom)

    for col in schema.binary_columns:
        out[f"{col}__bin"] = out[col].map(encode_yes_no)

    if schema.age:
        out["age_years"] = pd.to_numeric(out[schema.age], errors="coerce")
    if schema.weight:
        out["weight_kg"] = pd.to_numeric(out[schema.weight], errors="coerce")
    if schema.duration:
        out["duration_days"] = out[schema.duration].map(parse_duration_to_days)

    out = out.reset_index(drop=True)
    out.attrs["duplicates_dropped"] = dropped
    return out


def symptom_sets(df: pd.DataFrame, schema: DatasetSchema) -> pd.Series:
    """Collapse the per-row symptom slot columns into a set of canonical terms."""
    if not schema.symptom_columns:
        return pd.Series([set() for _ in range(len(df))], index=df.index)

    def row_to_set(row) -> set[str]:
        return {t for t in (canonical_symptom(v) for v in row) if t}

    return df[schema.symptom_columns].apply(row_to_set, axis=1)
