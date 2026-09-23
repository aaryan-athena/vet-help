"""Cattle & buffalo breed advisor.

Two directions, both over ``data/breedinfo.csv``:

* **conditions -> breed**  ``recommend()`` ranks breeds for a farm's climate,
  purpose and preferences, optimising for yield *within* what is actually
  suitable.
* **breed -> conditions**  ``describe()`` returns the husbandry envelope a
  chosen breed wants.

This is a **reference lookup with transparent scoring, not a trained model**.
The file holds 41 curated rows; there is nothing to learn from that and a model
would only launder a lookup table into false confidence. Every number a user
sees here traces back to a row in the CSV, and every recommendation carries the
reasons that produced its score.

    python -m ml.breeds                     # write ml/artifacts/breeds.json
    python -m ml.breeds --climate "hot humid" --purpose dairy
"""
from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache

import pandas as pd

from ml import config
from ml.cleaning import normalize_text

# --------------------------------------------------------------------------
# Climate vocabulary
# --------------------------------------------------------------------------
# The CSV writes climate as free text ("Hot, dry, semi-arid", "Cold to
# temperate", "Arid, saline areas"). Both the CSV column and whatever the user
# types are parsed by the SAME function into these tags, so a farmer typing
# "very hot and dry" is scored against exactly the vocabulary the data uses.
CLIMATE_TAGS: dict[str, tuple[str, ...]] = {
    "hot": ("hot", "warm", "high temperature", "heat", "tropical", "summer"),
    "cold": ("cold", "chilly", "freezing", "snow", "winter"),
    "cool": ("cool", "mild"),
    "temperate": ("temperate", "moderate climate"),
    "arid": ("arid", "desert"),
    "semi_arid": ("semi arid", "semiarid", "semi-arid"),
    "dry": ("dry", "low rainfall", "drought"),
    "humid": ("humid", "humidity", "muggy", "sticky"),
    "high_rainfall": ("heavy rainfall", "high rainfall", "monsoon", "rainy", "wet"),
    "saline": ("saline", "salty", "coastal"),
    "hilly": ("hilly", "hills", "mountain", "highland", "altitude"),
    "forest": ("forest", "wooded", "jungle"),
    "plateau": ("plateau", "tableland"),
    "rocky": ("rocky", "stony", "rugged"),
}

# Tags that partially satisfy one another (half credit in scoring).
RELATED_TAGS: dict[str, tuple[str, ...]] = {
    "arid": ("dry", "semi_arid"),
    "dry": ("arid", "semi_arid"),
    "semi_arid": ("dry", "arid"),
    "humid": ("high_rainfall",),
    "high_rainfall": ("humid",),
    "cool": ("temperate", "cold"),
    "temperate": ("cool",),
    "cold": ("cool",),
    "hilly": ("forest", "plateau"),
    "forest": ("hilly",),
    "plateau": ("rocky",),
    "rocky": ("plateau",),
}

# Pairs that cannot both be true of one farm. A breed bred for temperate
# Scotland is not merely "less ideal" in hot, humid Kerala -- it is the wrong
# animal, and heat stress will cost far more than the extra litres promise.
CONFLICTING_TAGS: tuple[tuple[str, str], ...] = (
    ("hot", "cold"),
    ("hot", "cool"),
    ("hot", "temperate"),
    ("arid", "humid"),
    ("arid", "high_rainfall"),
    ("dry", "humid"),
    ("dry", "high_rainfall"),
    ("semi_arid", "high_rainfall"),
)

PURPOSE_TAGS: dict[str, tuple[str, ...]] = {
    "dairy": ("dairy", "milk", "milking", "yield", "lactation"),
    "draught": ("draught", "draft", "plough", "plow", "ploughing", "cart", "tillage", "work"),
    "dual": ("dual", "both", "dual-purpose", "dual purpose", "mixed"),
}


def extract_climate_tags(text) -> set[str]:
    """Parse free text into canonical climate tags.

    Used for both the CSV's ``Climate Suitability`` column and user input, so
    the two are always compared in the same vocabulary.
    """
    normalized = normalize_text(text)
    if not normalized:
        return set()
    padded = f" {normalized} "
    tags = set()
    for tag, phrases in CLIMATE_TAGS.items():
        for phrase in phrases:
            if f" {normalize_text(phrase)} " in padded:
                tags.add(tag)
                break
    # "semi arid" also contains "arid"; keep the more specific tag only.
    if "semi_arid" in tags:
        tags.discard("arid")
    return tags


def extract_purpose(text) -> set[str]:
    normalized = f" {normalize_text(text)} "
    found = set()
    for purpose, phrases in PURPOSE_TAGS.items():
        if any(f" {p} " in normalized for p in phrases):
            found.add(purpose)
    return found


def _parse_fat_percent(value) -> float | None:
    """'7–8% fat' -> 7.5, '4% fat' -> 4.0. Handles the en-dash mojibake."""
    text = str(value).replace("�", "-").replace("–", "-").replace("—", "-")
    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return None
    return round(sum(numbers) / len(numbers), 2)


def _utility_tags(value) -> set[str]:
    """'Draught & Dairy' / 'Dual-purpose' -> {'draught','dairy'} / {'dual',...}."""
    tags = extract_purpose(value)
    if "dual" in tags:
        tags |= {"dairy", "draught"}
    return tags or {"dairy"}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
COLUMN_ALIASES = {
    "name": ("breed name", "breed", "name"),
    "type": ("type", "species"),
    "region": ("region of origin (india/world)", "region of origin", "region", "origin"),
    "climate": ("climate suitability", "climate"),
    "yield": ("avg. milk yield (l/day)", "avg milk yield (l/day)", "milk yield", "yield"),
    "fat": ("milk type (fat %)", "fat %", "fat"),
    "traits": ("physical traits", "traits"),
    "utility": ("utility", "use"),
    "programs": ("crossbreeding / programs", "crossbreeding", "programs"),
    "features": ("special features", "features", "notes"),
}


def _resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    resolved = {}
    for key, candidates in COLUMN_ALIASES.items():
        for candidate in candidates:
            if candidate in lookup:
                resolved[key] = lookup[candidate]
                break
    missing = {"name", "climate", "yield"} - set(resolved)
    if missing:
        raise ValueError(
            f"breedinfo.csv is missing required column(s) {sorted(missing)}; "
            f"found {list(df.columns)}"
        )
    return resolved


def load_breeds(csv_path=None) -> list[dict]:
    """Parse the breed CSV into structured records."""
    csv_path = csv_path or config.BREED_CSV
    df = pd.read_csv(csv_path)
    cols = _resolve_columns(df)

    records = []
    for _, row in df.iterrows():
        name = str(row[cols["name"]]).strip()
        if not name:
            continue
        climate_raw = str(row[cols["climate"]]).strip()
        utility_raw = str(row[cols.get("utility", cols["name"])]).strip() if "utility" in cols else ""
        milk_yield = pd.to_numeric(row[cols["yield"]], errors="coerce")
        records.append(
            {
                "name": name,
                "slug": normalize_text(name).replace(" ", "-"),
                "type": str(row[cols["type"]]).strip() if "type" in cols else "Cattle",
                "region": str(row[cols["region"]]).strip() if "region" in cols else "",
                "climate": _clean_text(climate_raw),
                "climate_tags": sorted(extract_climate_tags(climate_raw)),
                "milk_yield_l_per_day": None if pd.isna(milk_yield) else float(milk_yield),
                "fat_percent": _parse_fat_percent(row[cols["fat"]]) if "fat" in cols else None,
                "fat_raw": _clean_text(row[cols["fat"]]) if "fat" in cols else "",
                "traits": _clean_text(row[cols["traits"]]) if "traits" in cols else "",
                "utility": _clean_text(utility_raw),
                "utility_tags": sorted(_utility_tags(utility_raw)),
                "programs": _clean_text(row[cols["programs"]]) if "programs" in cols else "",
                "features": _clean_text(row[cols["features"]]) if "features" in cols else "",
            }
        )
    return records


def _clean_text(value) -> str:
    """Repair the en-dash mojibake without destroying the original wording."""
    text = str(value).replace("�", "-").strip()
    return "" if text.lower() == "nan" else re.sub(r"\s+", " ", text)


@lru_cache(maxsize=1)
def get_catalog() -> list[dict]:
    """Breed records, preferring the generated artifact over the raw CSV."""
    if config.BREED_ARTIFACT_PATH.exists():
        payload = json.loads(config.BREED_ARTIFACT_PATH.read_text(encoding="utf-8"))
        return payload["breeds"]
    return load_breeds()


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
# Weights depend on what the farmer actually asked for. Someone optimising for
# milk should have yield count for a lot; someone buying a plough animal should
# have it count for nothing at all.
WEIGHTS: dict[str, tuple[float, float, float]] = {
    #                climate, purpose, yield
    "dairy": (0.45, 0.20, 0.35),
    "draught": (0.60, 0.40, 0.00),
    "default": (0.55, 0.25, 0.20),
}


def _weights_for(wanted_purpose: set[str]) -> tuple[float, float, float]:
    if wanted_purpose == {"draught"}:
        return WEIGHTS["draught"]
    if "dairy" in wanted_purpose:
        return WEIGHTS["dairy"]
    return WEIGHTS["default"]


def _climate_fit(breed_tags: set[str], wanted: set[str]) -> tuple[float, list[str], list[str]]:
    """Return (0..1 fit, matched reasons, conflict reasons)."""
    if not wanted:
        return 0.5, [], []  # no stated conditions: neutral, let purpose/yield decide

    matched: list[str] = []
    conflicts: list[str] = []
    score = 0.0

    for tag in wanted:
        if tag in breed_tags:
            score += 1.0
            matched.append(tag)
        elif any(rel in breed_tags for rel in RELATED_TAGS.get(tag, ())):
            score += 0.5
            matched.append(f"{tag}~")

    for a, b in CONFLICTING_TAGS:
        if (a in wanted and b in breed_tags) or (b in wanted and a in breed_tags):
            conflicts.append(f"{a} vs {b}")

    fit = score / len(wanted)
    # Each conflict is a hard penalty, not a rounding error.
    fit -= 0.5 * len(conflicts)
    return max(0.0, min(1.0, fit)), matched, conflicts


def recommend(
    climate: str = "",
    purpose: str = "",
    breed_type: str = "",
    region: str = "",
    top_n: int = 5,
    catalog: list[dict] | None = None,
) -> dict:
    """Rank breeds for a stated set of farm conditions.

    Yield is deliberately the *smallest* weight and is only allowed to matter
    among breeds that actually suit the climate. Ranking on yield alone would
    put Holstein Friesian (25 L/day, temperate) at the top of every query,
    including hot and humid ones, where heat stress makes it a poor and
    expensive choice.
    """
    breeds = catalog if catalog is not None else get_catalog()
    wanted_climate = extract_climate_tags(climate)
    wanted_purpose = extract_purpose(purpose)
    type_query = normalize_text(breed_type)
    region_query = normalize_text(region)

    # Pass 1: climate fit, which decides who is even in the running.
    fits = {}
    for breed in breeds:
        fits[breed["name"]] = _climate_fit(set(breed["climate_tags"]), wanted_climate)

    # Yield is normalised against the best *suitable* breed, not the best breed
    # overall. Otherwise Holstein Friesian's 25 L/day compresses every tropical
    # breed into the bottom of the scale and a 12 L/day Gir looks no better than
    # a 1.5 L/day Vechur, when for a dairy farmer that gap is the whole decision.
    suitable_yields = [
        b["milk_yield_l_per_day"] or 0.0
        for b in breeds
        if not fits[b["name"]][2] and fits[b["name"]][0] >= 0.34
    ]
    all_yields = [b["milk_yield_l_per_day"] or 0.0 for b in breeds]
    max_yield = max(suitable_yields or all_yields or [1.0]) or 1.0

    climate_w, purpose_w, yield_w = _weights_for(wanted_purpose)

    scored = []
    for breed in breeds:
        fit, matched, conflicts = fits[breed["name"]]

        purpose_score = 1.0
        if wanted_purpose:
            overlap = wanted_purpose & set(breed["utility_tags"])
            purpose_score = 1.0 if overlap else 0.0

        yield_score = min((breed["milk_yield_l_per_day"] or 0.0) / max_yield, 1.0)

        total = climate_w * fit + purpose_w * purpose_score + yield_w * yield_score

        if type_query and type_query not in normalize_text(breed["type"]):
            total *= 0.35
        if region_query and region_query in normalize_text(breed["region"]):
            total += 0.05

        scored.append(
            {
                **breed,
                "score": round(min(total, 1.0), 4),
                "climate_fit": round(fit, 3),
                "purpose_match": bool(not wanted_purpose or purpose_score),
                "matched_climate_tags": matched,
                "climate_conflicts": conflicts,
                "reasons": _reasons(breed, matched, conflicts, wanted_purpose, wanted_climate),
                "suitable": not conflicts and fit >= 0.34,
            }
        )

    scored.sort(key=lambda b: (b["suitable"], b["score"]), reverse=True)
    top = scored[: max(1, top_n)]

    return {
        "query": {
            "climate": climate,
            "climate_tags": sorted(wanted_climate),
            "purpose": purpose,
            "purpose_tags": sorted(wanted_purpose),
            "type": breed_type,
            "region": region,
        },
        "recommendations": top,
        "n_considered": len(breeds),
        "n_suitable": sum(1 for b in scored if b["suitable"]),
        "caveat": config.BREED_CAVEAT,
    }


def _reasons(
    breed: dict,
    matched: list[str],
    conflicts: list[str],
    wanted_purpose: set[str],
    wanted_climate: set[str],
) -> list[str]:
    """Plain-language justification for the score, good and bad."""
    out: list[str] = []
    pretty = lambda t: t.replace("_", "-").rstrip("~")

    if matched:
        exact = [pretty(t) for t in matched if not t.endswith("~")]
        near = [pretty(t) for t in matched if t.endswith("~")]
        if exact:
            out.append(f"Suited to {', '.join(exact)} conditions")
        if near:
            out.append(f"Tolerates {', '.join(near)}-like conditions")
    elif wanted_climate:
        out.append(f"Bred for {breed['climate'].lower()}, which does not match your conditions")

    for conflict in conflicts:
        a, b = conflict.split(" vs ")
        out.append(
            f"⚠ Climate conflict: you described {pretty(a)}, this breed needs "
            f"{pretty(b)} — expect heat/moisture stress and lower real-world yield"
        )

    if wanted_purpose and not (wanted_purpose & set(breed["utility_tags"])):
        out.append(f"Kept for {breed['utility'].lower()}, not {'/'.join(sorted(wanted_purpose))}")

    if breed["milk_yield_l_per_day"]:
        out.append(f"About {breed['milk_yield_l_per_day']:g} L/day at {breed['fat_raw'] or 'n/a'}")
    return out


def describe(name: str, catalog: list[dict] | None = None) -> dict | None:
    """The husbandry envelope for one breed: what conditions it wants."""
    breeds = catalog if catalog is not None else get_catalog()
    target = normalize_text(name)
    match = next(
        (b for b in breeds if normalize_text(b["name"]) == target or b["slug"] == target),
        None,
    )
    if match is None:
        match = next((b for b in breeds if target and target in normalize_text(b["name"])), None)
    if match is None:
        return None

    tags = set(match["climate_tags"])
    guidance: list[str] = []
    if "hot" in tags:
        guidance.append("Provide shade and ample drinking water; avoid midday work in summer.")
    if "humid" in tags or "high_rainfall" in tags:
        guidance.append("Keep housing well ventilated and dry underfoot; watch for foot rot and parasites.")
    if {"arid", "dry", "semi_arid"} & tags:
        guidance.append("Plan fodder and water storage for the dry season; this breed tolerates scarcity better than exotics.")
    if {"cold", "cool", "temperate"} & tags:
        guidance.append("Needs shelter from cold winds; in hot regions expect heat stress without cooling.")
    if "hilly" in tags:
        guidance.append("Sure-footed on slopes; suited to grazing on uneven terrain.")
    if "saline" in tags:
        guidance.append("Tolerates brackish water and saline grazing that would not suit most breeds.")
    if match["type"].lower().startswith("exotic"):
        guidance.append(
            "Exotic breed: high yield but climate-sensitive. In hot regions it is usually "
            "kept as a cross with a local breed rather than pure."
        )

    return {
        **match,
        "ideal_conditions": match["climate"],
        "guidance": guidance,
        "caveat": config.BREED_CAVEAT,
    }


def facets(catalog: list[dict] | None = None) -> dict:
    """Distinct values the frontend needs to build its filters."""
    breeds = catalog if catalog is not None else get_catalog()
    yields = [b["milk_yield_l_per_day"] for b in breeds if b["milk_yield_l_per_day"]]
    return {
        "types": sorted({b["type"] for b in breeds if b["type"]}),
        "climates": sorted({b["climate"] for b in breeds if b["climate"]}),
        "climate_tags": sorted({t for b in breeds for t in b["climate_tags"]}),
        "purposes": sorted({t for b in breeds for t in b["utility_tags"]}),
        "regions": sorted({b["region"] for b in breeds if b["region"]}),
        "n_breeds": len(breeds),
        "yield_range": [min(yields), max(yields)] if yields else [0, 0],
        "caveat": config.BREED_CAVEAT,
    }


def build_artifact() -> dict:
    """Freeze the parsed catalog to ml/artifacts/breeds.json."""
    breeds = load_breeds()
    payload = {"source": config.BREED_CSV.name, "n_breeds": len(breeds), "breeds": breeds}
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    config.BREED_ARTIFACT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--climate", default="")
    parser.add_argument("--purpose", default="")
    parser.add_argument("--type", dest="breed_type", default="")
    parser.add_argument("--describe", default="")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    if args.describe:
        result = describe(args.describe, catalog=load_breeds())
        print(json.dumps(result, indent=2) if result else f"No breed matching {args.describe!r}")
        return

    if args.climate or args.purpose or args.breed_type:
        result = recommend(
            args.climate, args.purpose, args.breed_type, top_n=args.top, catalog=load_breeds()
        )
        print(f"conditions -> {result['query']['climate_tags']} {result['query']['purpose_tags']}")
        print(f"{result['n_suitable']}/{result['n_considered']} breeds suitable\n")
        for i, b in enumerate(result["recommendations"], 1):
            flag = "" if b["suitable"] else "  [not recommended]"
            print(f"{i}. {b['name']} ({b['type']}) score={b['score']:.3f}{flag}")
            for reason in b["reasons"]:
                print(f"     - {reason}")
        return

    payload = build_artifact()
    print(f"wrote {config.BREED_ARTIFACT_PATH} ({payload['n_breeds']} breeds)")


if __name__ == "__main__":
    main()
