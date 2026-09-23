"""Natural-language understanding for the VetDx chat interface.

Turns free text like

    "my buffalo has had fever for 2 days, loose motions and she's not eating,
     but no cough"

into the structured case the model already expects::

    species   buffaloes
    symptoms  fever, diarrhea, loss of appetite
    negated   coughing
    duration  2 days

**Deliberately not an LLM.** Everything here is deterministic, runs in the
existing serving dependencies (`difflib` is stdlib), costs nothing per message,
works offline, and never sends a user's animal data to a third party. Those are
real advantages for the rural, low-connectivity setting this tool targets. The
trade-off is that it understands *descriptions of symptoms*, not open-ended
conversation, so the reply generator steers users toward describing what they
can see.

The vocabulary is the model's own 146 canonical symptom terms, so anything
extracted here is guaranteed to be a feature the model actually knows -- the
same train/serve parity rule the rest of the pipeline follows.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import get_close_matches

from ml.cleaning import (
    ANIMAL_SYNONYMS,
    SYMPTOM_SYNONYMS,
    canonical_animal,
    canonical_symptom,
    normalize_text,
    parse_duration_to_days,
)

# --------------------------------------------------------------------------
# Negation
# --------------------------------------------------------------------------
NEGATION_CUES = frozenset(
    {
        "no", "not", "never", "without", "denies", "denied", "negative",
        "nothing", "none", "absent", "lacks", "lacking", "free",
        "isnt", "isn", "arent", "aren", "hasnt", "hasn", "havent", "haven",
        "doesnt", "doesn", "didnt", "didn", "wasnt", "wasn", "cant", "cannot",
    }
)

# Words that end a negation's reach: "no cough but fever" must not negate fever.
# CLAUSE_MARK stands in for punctuation, which normalize_text() would otherwise
# discard -- without it, "no vomiting, limping" reads the comma as nothing and
# wrongly negates the limping.
CLAUSE_MARK = "clausebreak"
CLAUSE_BREAKS = frozenset(
    {"but", "however", "though", "although", "yet", "while", "and", CLAUSE_MARK}
)
_PUNCT_RE = re.compile(r"[,;.!?:\n\r]+")
_APOSTROPHE_RE = re.compile(r"[’']")


def _pre_normalize(text) -> str:
    """Drop apostrophes so contractions survive as one token.

    normalize_text() maps every non-alphanumeric character to a space, which
    would split "doesn't" into "doesn t" and "that's all" into "that s all".
    That function is deliberately left alone: it defines the *trained* symptom
    vocabulary, so changing it would silently alter the model's features. This
    is the chat layer's own pre-step instead.
    """
    return _APOSTROPHE_RE.sub("", str(text))

# How many tokens back a negation cue can reach.
NEGATION_WINDOW = 4

STOPWORDS = frozenset(
    {
        "the", "a", "an", "my", "our", "his", "her", "its", "their", "is", "are",
        "was", "were", "has", "have", "had", "been", "being", "be", "she", "he",
        "it", "they", "them", "i", "we", "you", "this", "that", "these", "those",
        "of", "in", "on", "at", "to", "for", "with", "from", "by", "as", "and",
        "or", "also", "very", "really", "quite", "bit", "little", "some", "any",
        "there", "here", "since", "ago", "day", "days", "week", "weeks", "month",
        "months", "year", "years", "old", "please", "help", "think", "seems",
        "looks", "like", "getting", "got", "been", "lot", "much", "too", "still",
        "now", "today", "yesterday", "animal", "last", "past", "hour", "hours",
        "morning", "evening", "night", "leg", "back", "side", "one", "two",
        "shes", "hes", "thats", "theres", "weve", "ive", "weve", "also",
    }
)

# --------------------------------------------------------------------------
# Intents
# --------------------------------------------------------------------------
ASSESS_CUES = ("assess", "diagnose", "check now", "thats all", "that is all",
               "done", "finished", "nothing else", "no more", "go ahead",
               "whats wrong", "what is wrong", "result", "predict")
RESET_CUES = ("reset", "start over", "start again", "clear", "new case", "another animal")
HELP_CUES = ("help", "how do i", "how does this work", "what can you do", "what do you do")
GREETING_CUES = ("hi", "hello", "hey", "good morning", "good evening", "namaste")

_TIME_RE = re.compile(
    r"(\d+(?:\.\d+)?|a|an|one|two|three|four|five|six|seven|eight|nine|ten|half|few|"
    r"couple|several|last|past|this)\s*"
    r"(hour|hr|day|week|wk|month|mon|year|yr)s?\b"
)

# "since last week" / "over the past month" mean one of that unit.
_BARE_QUANTIFIERS = frozenset({"last", "past", "this"})


@dataclass
class CaseState:
    """Accumulated case details across a conversation.

    The API is stateless (it runs as a serverless function), so this is carried
    by the client and echoed back on each turn.
    """

    species: str = ""
    symptoms: list[str] = field(default_factory=list)
    negated: list[str] = field(default_factory=list)
    age_years: float | None = None
    duration: str = ""
    duration_days: float | None = None

    @classmethod
    def from_dict(cls, data: dict | None) -> "CaseState":
        data = data or {}
        return cls(
            species=str(data.get("species") or ""),
            symptoms=[str(s) for s in (data.get("symptoms") or [])],
            negated=[str(s) for s in (data.get("negated") or [])],
            age_years=data.get("age_years"),
            duration=str(data.get("duration") or ""),
            duration_days=data.get("duration_days"),
        )

    def to_dict(self) -> dict:
        return {
            "species": self.species,
            "symptoms": list(self.symptoms),
            "negated": list(self.negated),
            "age_years": self.age_years,
            "duration": self.duration,
            "duration_days": self.duration_days,
        }

    @property
    def ready(self) -> bool:
        return bool(self.species and self.symptoms)


class SymptomExtractor:
    """Longest-match phrase extraction over the model's symptom vocabulary."""

    def __init__(self, vocabulary: list[str], species_options: list[str] | None = None):
        self.vocabulary = sorted(set(vocabulary))
        self.species_options = sorted(set(species_options or []))

        # Surface form -> canonical term. Includes the synonym table, so a user
        # typing "anorexia" lands on the same feature as one typing
        # "not eating", exactly as the training pipeline would have mapped them.
        self.phrase_index: dict[tuple[str, ...], str] = {}
        for term in self.vocabulary:
            self._add_phrase(term, term)
        for surface, canonical in SYMPTOM_SYNONYMS.items():
            if canonical in set(self.vocabulary):
                self._add_phrase(surface, canonical)
        for surface, canonical in EXTRA_SYMPTOM_PHRASES.items():
            target = canonical_symptom(canonical)
            if target in set(self.vocabulary):
                self._add_phrase(surface, target)

        self.species_index: dict[tuple[str, ...], str] = {}
        for option in self.species_options:
            self._add_species(option, option)
        for surface, canonical in ANIMAL_SYNONYMS.items():
            resolved = canonical_animal(canonical)
            if resolved in set(self.species_options) or not self.species_options:
                self._add_species(surface, resolved)

        self.max_phrase_len = max((len(p) for p in self.phrase_index), default=1)
        self.max_species_len = max((len(p) for p in self.species_index), default=1)
        self._single_words = [t for t in self.vocabulary if " " not in t]

    def _add_phrase(self, surface: str, canonical: str) -> None:
        tokens = tuple(normalize_text(surface).split())
        if tokens:
            self.phrase_index.setdefault(tokens, canonical)

    def _add_species(self, surface: str, canonical: str) -> None:
        tokens = tuple(normalize_text(surface).split())
        if tokens:
            self.species_index.setdefault(tokens, canonical)

    # -- the main entry point ---------------------------------------------
    def extract(self, text: str) -> dict:
        marked = _PUNCT_RE.sub(f" {CLAUSE_MARK} ", _pre_normalize(text))
        tokens = normalize_text(marked).split()
        if not tokens:
            return _empty_extraction()

        matches: list[dict] = []
        consumed = [False] * len(tokens)

        # Longest-match first. This is what makes "not eating" resolve to
        # `loss of appetite` instead of being read as a negated "eating".
        i = 0
        while i < len(tokens):
            hit = None
            for length in range(min(self.max_phrase_len, len(tokens) - i), 0, -1):
                span = tuple(tokens[i : i + length])
                canonical = self.phrase_index.get(span)
                if canonical:
                    hit = (length, canonical, span)
                    break
            if hit:
                length, canonical, span = hit
                matches.append(
                    {
                        "term": canonical,
                        "start": i,
                        "end": i + length,
                        "surface": " ".join(span),
                        "match": "exact",
                    }
                )
                for j in range(i, i + length):
                    consumed[j] = True
                i += length
            else:
                i += 1

        # Fuzzy pass over leftover content words, for typos like "diarhea".
        for idx, token in enumerate(tokens):
            if consumed[idx] or token in STOPWORDS or len(token) < 4:
                continue
            if token in NEGATION_CUES or token in CLAUSE_BREAKS:
                continue
            close = get_close_matches(token, self._single_words, n=1, cutoff=0.82)
            if close:
                matches.append(
                    {
                        "term": close[0],
                        "start": idx,
                        "end": idx + 1,
                        "surface": token,
                        "match": "fuzzy",
                    }
                )
                consumed[idx] = True

        # Negation.
        affirmed: list[dict] = []
        negated: list[dict] = []
        for match in matches:
            if self._is_negated(tokens, match):
                negated.append(match)
            else:
                affirmed.append(match)

        species, species_span = self._extract_species(tokens)
        for j in range(*species_span) if species_span else ():
            consumed[j] = True
        age_years, duration_text = self._extract_time(tokens)

        unmatched = [
            t
            for idx, t in enumerate(tokens)
            if not consumed[idx] and t not in STOPWORDS and t not in NEGATION_CUES
            and t not in CLAUSE_BREAKS and len(t) > 3 and not t.isdigit()
        ]

        return {
            "species": species,
            "symptoms": _dedupe([m["term"] for m in affirmed]),
            "negated": _dedupe([m["term"] for m in negated]),
            "matches": affirmed + negated,
            "age_years": age_years,
            "duration": duration_text,
            "duration_days": parse_duration_to_days(duration_text) if duration_text else None,
            "unmatched_terms": _dedupe(unmatched)[:8],
        }

    def _is_negated(self, tokens: list[str], match: dict) -> bool:
        """Look back for a negation cue, stopping at clause boundaries.

        A cue inside the matched phrase itself does not count -- "not eating"
        is a symptom, not a denial of one.
        """
        start = match["start"]
        for offset in range(1, NEGATION_WINDOW + 1):
            idx = start - offset
            if idx < 0:
                break
            token = tokens[idx]
            if token in CLAUSE_BREAKS:
                break
            if token in NEGATION_CUES:
                return True
        return False

    def _extract_species(self, tokens: list[str]) -> tuple[str, tuple[int, int] | None]:
        """Return the species and the token span it came from."""
        for i in range(len(tokens)):
            for length in range(min(self.max_species_len, len(tokens) - i), 0, -1):
                span = tuple(tokens[i : i + length])
                found = self.species_index.get(span)
                if found:
                    return found, (i, i + length)
        # Fuzzy, for "bufalo" / "chiken".
        flat = {" ".join(k): v for k, v in self.species_index.items()}
        for idx, token in enumerate(tokens):
            if len(token) < 4 or token in STOPWORDS:
                continue
            close = get_close_matches(token, list(flat), n=1, cutoff=0.85)
            if close:
                return flat[close[0]], (idx, idx + 1)
        return "", None

    def _extract_time(self, tokens: list[str]) -> tuple[float | None, str]:
        """Separate "3 years old" (age) from "for 2 days" (duration)."""
        text = " ".join(tokens)
        age_years: float | None = None
        duration_text = ""

        for m in _TIME_RE.finditer(text):
            phrase = m.group(0)
            before = text[: m.start()].split()
            after = text[m.end() :].split()
            prev = before[-1] if before else ""
            nxt = after[0] if after else ""

            is_age = nxt in {"old", "age", "aged"} or prev in {"aged", "age"}
            is_duration = prev in {"for", "since", "past", "last", "over"}

            quantity = m.group(1)
            if quantity in _BARE_QUANTIFIERS:
                phrase = f"1 {m.group(2)}"
                is_duration = True

            if is_age and not is_duration:
                days = parse_duration_to_days(phrase)
                if days is not None and days == days:  # not NaN
                    age_years = round(days / 365.0, 2)
            elif not duration_text:
                duration_text = phrase
        return age_years, duration_text


# Phrasings a farmer would use that are not in the clinical synonym table.
EXTRA_SYMPTOM_PHRASES: dict[str, str] = {
    "loose motion": "diarrhea",
    "loose motions": "diarrhea",
    "watery dung": "diarrhea",
    "scouring": "diarrhea",
    "running stomach": "diarrhea",
    "off feed": "loss of appetite",
    "off her feed": "loss of appetite",
    "off his feed": "loss of appetite",
    "wont eat": "loss of appetite",
    "refusing food": "loss of appetite",
    "refusing to eat": "loss of appetite",
    "eating less": "loss of appetite",
    "high temperature": "fever",
    "running temperature": "fever",
    "hot to touch": "fever",
    "dull": "lethargy",
    "sluggish": "lethargy",
    "weak": "lethargy",
    "tired": "lethargy",
    "no energy": "lethargy",
    "not moving": "lethargy",
    "lying down": "lethargy",
    "throwing up": "vomiting",
    "puking": "vomiting",
    "runny nose": "nasal discharge",
    "snot": "nasal discharge",
    "limping": "lameness",
    "cant walk": "lameness",
    "hard to breathe": "difficulty in breathing",
    "panting": "difficulty in breathing",
    "breathing fast": "difficulty in breathing",
    "losing weight": "weight loss",
    "gone thin": "weight loss",
    "thin": "weight loss",
    "drooling": "drooling",
    "swollen": "swelling",
    "sneezing": "sneezing",
}


def _dedupe(items) -> list[str]:
    seen, out = set(), []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _empty_extraction() -> dict:
    return {
        "species": "",
        "symptoms": [],
        "negated": [],
        "matches": [],
        "age_years": None,
        "duration": "",
        "duration_days": None,
        "unmatched_terms": [],
    }


# --------------------------------------------------------------------------
# Intent detection + state merging
# --------------------------------------------------------------------------
def detect_intent(text: str) -> str:
    normalized = normalize_text(_pre_normalize(text))
    if not normalized:
        return "empty"
    padded = f" {normalized} "
    if any(cue in padded for cue in RESET_CUES):
        return "reset"
    if any(f" {cue} " in padded or padded.strip() == cue for cue in HELP_CUES):
        return "help"
    if any(cue in padded for cue in ASSESS_CUES):
        return "assess"
    if normalized in GREETING_CUES or any(padded.strip() == cue for cue in GREETING_CUES):
        return "greeting"
    return "describe"


def merge(state: CaseState, extraction: dict) -> CaseState:
    """Fold a turn's extraction into the running case.

    A symptom the user denies is removed if it was previously asserted, so
    "actually no, she isn't coughing" corrects an earlier turn.
    """
    if extraction["species"]:
        state.species = extraction["species"]
    if extraction["age_years"] is not None:
        state.age_years = extraction["age_years"]
    if extraction["duration"]:
        state.duration = extraction["duration"]
        state.duration_days = extraction["duration_days"]

    for term in extraction["symptoms"]:
        if term not in state.symptoms:
            state.symptoms.append(term)
        if term in state.negated:
            state.negated.remove(term)

    for term in extraction["negated"]:
        if term in state.symptoms:
            state.symptoms.remove(term)
        if term not in state.negated:
            state.negated.append(term)

    return state
