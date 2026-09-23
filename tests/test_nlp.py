"""Unit tests for the chat interface's language understanding."""
from __future__ import annotations

import pytest

from ml.nlp import CaseState, SymptomExtractor, detect_intent, merge

VOCAB = [
    "fever", "diarrhea", "coughing", "vomiting", "lethargy", "loss of appetite",
    "weight loss", "nasal discharge", "lameness", "sneezing", "swelling",
    "difficulty in breathing", "dullness", "limping",
]
SPECIES = ["buffaloes", "cattle", "dog", "cat", "goat", "sheep", "chicken", "pig"]


@pytest.fixture(scope="module")
def extractor() -> SymptomExtractor:
    return SymptomExtractor(VOCAB, SPECIES)


# --------------------------------------------------------------------------
# Symptom extraction
# --------------------------------------------------------------------------
def test_extracts_plain_symptoms(extractor):
    r = extractor.extract("the cow has fever and diarrhea")
    assert r["species"] == "cattle"
    assert set(r["symptoms"]) == {"fever", "diarrhea"}
    assert r["negated"] == []


def test_maps_farmer_phrasing_to_clinical_terms(extractor):
    r = extractor.extract("my goat is off her feed, loose motions and throwing up")
    assert set(r["symptoms"]) == {"loss of appetite", "diarrhea", "vomiting"}


def test_synonym_table_is_shared_with_training(extractor):
    """'Anorexia' must land on the same feature as 'not eating'."""
    a = extractor.extract("dog with anorexia")
    b = extractor.extract("dog is not eating")
    assert a["symptoms"] == b["symptoms"] == ["loss of appetite"]


def test_longest_match_beats_negation(extractor):
    """'not eating' is a symptom, not a denial of 'eating'."""
    r = extractor.extract("she is not eating")
    assert r["symptoms"] == ["loss of appetite"]
    assert r["negated"] == []


def test_fuzzy_matches_typos(extractor):
    r = extractor.extract("dog has diarhea and coughin")
    assert "diarrhea" in r["symptoms"]
    assert any(m["match"] == "fuzzy" for m in r["matches"])


def test_fuzzy_matches_misspelled_species(extractor):
    assert extractor.extract("my bufalo has fever")["species"] == "buffaloes"


# --------------------------------------------------------------------------
# Negation
# --------------------------------------------------------------------------
def test_simple_negation(extractor):
    r = extractor.extract("cow has fever but no cough")
    assert r["symptoms"] == ["fever"]
    assert r["negated"] == ["coughing"]


def test_negation_does_not_cross_a_comma(extractor):
    """'no vomiting, limping' must not negate the limping."""
    r = extractor.extract("cow with fever, no vomiting, limping on back leg")
    assert "limping" in r["symptoms"]
    assert r["negated"] == ["vomiting"]


def test_negation_does_not_cross_but(extractor):
    r = extractor.extract("no cough but fever")
    assert r["symptoms"] == ["fever"]
    assert r["negated"] == ["coughing"]


@pytest.mark.parametrize(
    "text", ["no fever", "not coughing", "without fever", "she doesn't have fever"]
)
def test_negation_cue_variants(extractor, text):
    r = extractor.extract(text)
    assert r["symptoms"] == []
    assert len(r["negated"]) == 1


def test_multiple_negations(extractor):
    r = extractor.extract("no fever, no diarrhea")
    assert set(r["negated"]) == {"fever", "diarrhea"}
    assert r["symptoms"] == []


# --------------------------------------------------------------------------
# Age vs duration
# --------------------------------------------------------------------------
def test_duration_is_captured(extractor):
    r = extractor.extract("buffalo with fever for 2 days")
    assert r["duration_days"] == 2.0


def test_bare_last_week_is_a_duration(extractor):
    r = extractor.extract("goat off feed since last week")
    assert r["duration_days"] == 7.0


def test_age_and_duration_are_not_confused(extractor):
    r = extractor.extract("my 3 year old goat has had fever for 2 days")
    assert r["age_years"] == 3.0
    assert r["duration_days"] == 2.0


def test_hours_become_fractional_days(extractor):
    assert extractor.extract("dog fever for 36 hours")["duration_days"] == 1.5


# --------------------------------------------------------------------------
# Robustness
# --------------------------------------------------------------------------
def test_empty_input_is_safe(extractor):
    r = extractor.extract("")
    assert r["symptoms"] == [] and r["species"] == ""


def test_unknown_words_are_reported_not_invented(extractor):
    r = extractor.extract("my dog has purple spangles")
    assert r["symptoms"] == []
    assert "spangles" in r["unmatched_terms"]


def test_never_invents_a_term_outside_the_vocabulary(extractor):
    """Whatever comes out must be a feature the model actually knows."""
    r = extractor.extract("cow with fever, diarrhea, moon blindness and zoomies")
    assert set(r["symptoms"]) <= set(VOCAB)
    assert set(r["negated"]) <= set(VOCAB)


def test_species_is_not_reported_as_unrecognised(extractor):
    r = extractor.extract("my buffalo has fever")
    assert "buffalo" not in r["unmatched_terms"]


# --------------------------------------------------------------------------
# State merging
# --------------------------------------------------------------------------
def test_merge_accumulates_across_turns(extractor):
    state = CaseState()
    state = merge(state, extractor.extract("my cow has fever"))
    state = merge(state, extractor.extract("also diarrhea"))
    assert state.species == "cattle"
    assert state.symptoms == ["fever", "diarrhea"]
    assert state.ready


def test_later_denial_removes_an_earlier_symptom(extractor):
    state = merge(CaseState(), extractor.extract("cow with fever and cough"))
    assert "coughing" in state.symptoms
    state = merge(state, extractor.extract("actually no cough"))
    assert "coughing" not in state.symptoms
    assert "coughing" in state.negated


def test_reasserting_clears_the_denial(extractor):
    state = merge(CaseState(), extractor.extract("cow, no cough"))
    assert state.negated == ["coughing"]
    state = merge(state, extractor.extract("she is coughing now"))
    assert state.symptoms == ["coughing"]
    assert state.negated == []


def test_case_state_round_trips(extractor):
    state = merge(CaseState(), extractor.extract("buffalo fever for 2 days"))
    assert CaseState.from_dict(state.to_dict()).to_dict() == state.to_dict()


def test_not_ready_without_species_or_symptoms(extractor):
    assert not merge(CaseState(), extractor.extract("fever")).ready
    assert not merge(CaseState(), extractor.extract("my cow")).ready


# --------------------------------------------------------------------------
# Intents
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("reset", "reset"),
        ("start over", "reset"),
        ("help", "help"),
        ("hello", "greeting"),
        ("that's all", "assess"),
        ("my cow has fever", "describe"),
        ("", "empty"),
    ],
)
def test_detect_intent(text, expected):
    assert detect_intent(text) == expected
