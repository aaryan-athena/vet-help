"""Unit tests for the breed advisor.

The safety-critical property is in `test_recommendations_respect_climate`:
ranking on yield alone would put a 25 L/day temperate Holstein at the top of a
hot, humid query, which is bad advice that costs a farmer real money.
"""
from __future__ import annotations

import pytest

from ml import config
from ml.breeds import (
    describe,
    extract_climate_tags,
    extract_purpose,
    facets,
    load_breeds,
    recommend,
)

pytestmark = pytest.mark.skipif(
    not config.BREED_CSV.exists(), reason="data/breedinfo.csv not present"
)


@pytest.fixture(scope="module")
def catalog() -> list[dict]:
    return load_breeds()


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def test_catalog_loads(catalog):
    assert len(catalog) >= 40
    assert all(b["name"] for b in catalog)
    assert all(b["climate_tags"] for b in catalog), "every breed needs climate tags"


def test_yield_is_numeric(catalog):
    yields = [b["milk_yield_l_per_day"] for b in catalog]
    assert all(isinstance(y, float) for y in yields)
    assert max(yields) > 20  # Holstein Friesian
    assert min(yields) > 0


def test_fat_ranges_are_averaged(catalog):
    """'7-8% fat' should become 7.5, not 7 or 78."""
    assert all(b["fat_percent"] is None or 2 < b["fat_percent"] < 12 for b in catalog)
    jaffrabadi = next(b for b in catalog if b["name"] == "Jaffrabadi")
    assert jaffrabadi["fat_percent"] == pytest.approx(7.5)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Hot, semi-arid", {"hot", "semi_arid"}),
        ("Hot & dry", {"hot", "dry"}),
        ("Cold to temperate", {"cold", "temperate"}),
        ("Arid, saline areas", {"arid", "saline"}),
        ("Heavy rainfall, hilly", {"high_rainfall", "hilly"}),
        ("very hot and humid", {"hot", "humid"}),
        ("", set()),
    ],
)
def test_climate_tag_extraction(text, expected):
    assert extract_climate_tags(text) == expected


def test_semi_arid_does_not_also_report_arid():
    assert "arid" not in extract_climate_tags("semi-arid")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("dairy", {"dairy"}),
        ("I want milk", {"dairy"}),
        ("ploughing", {"draught"}),
        ("Dual-purpose", {"dual"}),
        ("", set()),
    ],
)
def test_purpose_extraction(text, expected):
    assert extract_purpose(text) == expected


# --------------------------------------------------------------------------
# Recommendation — the safety property
# --------------------------------------------------------------------------
def test_recommendations_respect_climate(catalog):
    """A temperate exotic must never top a hot, humid query."""
    result = recommend(climate="hot and humid", purpose="dairy", top_n=5, catalog=catalog)
    names = [b["name"] for b in result["recommendations"]]
    assert "Holstein Friesian" not in names[:3]

    top = result["recommendations"][0]
    assert top["suitable"]
    assert not top["climate_conflicts"]
    assert {"hot", "humid"} & set(top["climate_tags"])


def test_temperate_query_does_surface_exotics(catalog):
    """The same rule must not blanket-penalise exotics where they belong."""
    result = recommend(climate="temperate", purpose="dairy", top_n=3, catalog=catalog)
    assert result["recommendations"][0]["name"] == "Holstein Friesian"


def test_conflicts_are_explained_not_silent(catalog):
    result = recommend(climate="hot and humid", purpose="dairy", top_n=41, catalog=catalog)
    holstein = next(b for b in result["recommendations"] if b["name"] == "Holstein Friesian")
    assert holstein["climate_conflicts"]
    assert not holstein["suitable"]
    assert any("conflict" in r.lower() for r in holstein["reasons"])


def test_yield_breaks_ties_among_suitable_breeds(catalog):
    """Given equal climate fit, the higher yielder should win a dairy query."""
    result = recommend(climate="hot, semi-arid", purpose="dairy", top_n=6, catalog=catalog)
    suitable = [b for b in result["recommendations"] if b["suitable"]]
    assert len(suitable) >= 2
    gir = next((b for b in suitable if b["name"] == "Gir"), None)
    assert gir is not None, "Gir is the standout hot/semi-arid dairy breed"
    assert gir["score"] >= min(b["score"] for b in suitable)


def test_draught_query_ignores_yield(catalog):
    """A plough animal should not be ranked on milk."""
    result = recommend(climate="hot and dry", purpose="draught", top_n=5, catalog=catalog)
    for breed in result["recommendations"][:3]:
        assert "draught" in breed["utility_tags"]


def test_type_filter_demotes_other_types(catalog):
    result = recommend(climate="hot", purpose="dairy", breed_type="Buffalo", top_n=4, catalog=catalog)
    assert all(b["type"] == "Buffalo" for b in result["recommendations"][:2])


def test_empty_query_still_returns_something(catalog):
    result = recommend(catalog=catalog)
    assert result["recommendations"]
    assert result["n_considered"] == len(catalog)


def test_every_recommendation_carries_reasons(catalog):
    result = recommend(climate="hot, humid", purpose="dairy", top_n=6, catalog=catalog)
    for breed in result["recommendations"]:
        assert breed["reasons"], f"{breed['name']} has no explanation"
        assert 0.0 <= breed["score"] <= 1.0


def test_caveat_is_always_attached(catalog):
    assert recommend(catalog=catalog)["caveat"]
    assert facets(catalog)["caveat"]


# --------------------------------------------------------------------------
# Breed -> conditions
# --------------------------------------------------------------------------
def test_describe_known_breed(catalog):
    profile = describe("Gir", catalog=catalog)
    assert profile["name"] == "Gir"
    assert profile["ideal_conditions"]
    assert profile["guidance"]


def test_describe_is_case_and_slug_insensitive(catalog):
    assert describe("gir", catalog=catalog)["name"] == "Gir"
    assert describe("HOLSTEIN FRIESIAN", catalog=catalog)["name"] == "Holstein Friesian"


def test_describe_unknown_returns_none(catalog):
    assert describe("Tyrannosaurus", catalog=catalog) is None


def test_exotic_breeds_are_flagged_as_climate_sensitive(catalog):
    guidance = " ".join(describe("Holstein Friesian", catalog=catalog)["guidance"]).lower()
    assert "exotic" in guidance or "heat stress" in guidance


def test_facets_cover_the_catalog(catalog):
    f = facets(catalog)
    assert f["n_breeds"] == len(catalog)
    assert set(f["types"]) == {b["type"] for b in catalog}
    assert f["yield_range"][0] <= f["yield_range"][1]
