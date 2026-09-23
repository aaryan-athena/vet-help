"""Conversation orchestration for the chat triage interface.

Stateless by design: the serverless function holds nothing between requests, so
the client sends the accumulated :class:`~ml.nlp.CaseState` back with every
turn and receives the updated one in the reply.

The bot's job is narrow and it says so — collect species and clinical signs,
then hand them to the same model the manual form uses. It does not chat about
anything else, and it never invents a symptom the model does not know.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from ml import config
from ml.nlp import CaseState, SymptomExtractor, detect_intent, merge

logger = logging.getLogger("vetdx.chat")

MAX_SUGGESTIONS = 4

GREETING = (
    "Hello. Tell me about the animal and what you are seeing — for example "
    "\"my buffalo has had a fever for 2 days and won't eat\". I'll work out "
    "which signs the model recognises and give you an urgency assessment."
)

HELP = (
    "Describe the animal in your own words and I'll pick out the clinical signs. "
    "You can say things like:\n"
    "• \"my goat is dull, off her feed, loose motions since yesterday\"\n"
    "• \"cow has fever but no cough\" — I understand what you rule out too\n"
    "• \"reset\" to start a new case, or \"assess\" when you're done adding signs.\n"
    "I am a triage aid, not a vet, and I only recognise signs that appear in the "
    "training data."
)


class ChatService:
    """Turns a message plus prior state into a reply, an updated case and a prediction."""

    def __init__(self, model_service):
        self.model = model_service
        schema = model_service.feature_schema
        vocabulary = [s["value"] for s in schema.get("symptom_vocabulary", [])]
        species_options = [
            option["value"]
            for field in schema.get("fields", [])
            if field.get("name") == "species"
            for option in field.get("options", [])
        ]
        self.extractor = SymptomExtractor(vocabulary, species_options)
        self.cooccurrence: dict[str, list[str]] = schema.get("symptom_cooccurrence", {})
        self.species_options = species_options

    # -- suggestions -------------------------------------------------------
    def _suggestions(self, state: CaseState) -> list[str]:
        """Signs that commonly accompany what has been reported already."""
        seen = set(state.symptoms) | set(state.negated)
        ranked: dict[str, int] = {}
        for symptom in state.symptoms:
            for rank, partner in enumerate(self.cooccurrence.get(symptom, [])):
                if partner not in seen:
                    ranked[partner] = ranked.get(partner, 0) + (10 - rank)
        return [s for s, _ in sorted(ranked.items(), key=lambda kv: -kv[1])][:MAX_SUGGESTIONS]

    # -- main turn ---------------------------------------------------------
    def reply(self, message: str, case: dict | None) -> dict:
        state = CaseState.from_dict(case)
        intent = detect_intent(message)

        if intent == "reset":
            return self._response(CaseState(), GREETING, intent="reset")

        extraction = self.extractor.extract(message)
        had_content = bool(
            extraction["symptoms"] or extraction["negated"] or extraction["species"]
        )

        if intent == "help" and not had_content:
            return self._response(state, HELP, intent="help")
        if intent == "greeting" and not had_content:
            return self._response(state, GREETING, intent="greeting")

        state = merge(state, extraction)

        # Nothing understood, and nothing known from earlier turns.
        if not had_content and not state.ready:
            if intent == "assess":
                return self._response(
                    state,
                    "I still need a bit more before I can assess this case. "
                    + self._missing_prompt(state),
                    intent="need_more",
                    extraction=extraction,
                )
            unknown = extraction["unmatched_terms"]
            hint = (
                f" I didn't recognise: {', '.join(unknown)}."
                if unknown
                else ""
            )
            return self._response(
                state,
                "I couldn't pick out any clinical signs from that."
                + hint
                + " Try describing what you can see — temperature, appetite, "
                "droppings, breathing, movement.",
                intent="unrecognised",
                extraction=extraction,
            )

        if not state.ready:
            return self._response(
                state,
                self._acknowledge(extraction) + " " + self._missing_prompt(state),
                intent="need_more",
                extraction=extraction,
            )

        # Enough to assess. Do it immediately rather than making the user ask —
        # this is a triage tool and the answer is the point.
        prediction = self.model.predict(
            {
                "species": state.species,
                "symptoms": state.symptoms,
                "age_years": state.age_years,
                "duration_days": state.duration_days,
            },
            top_n=config.TOP_N_PREDICTIONS,
        )
        suggestions = self._suggestions(state)
        top = prediction["top_prediction"]

        summary = self._acknowledge(extraction)
        verdict = self._verdict_sentence(top, state)
        nudge = (
            f" Anything else — {', '.join(suggestions[:3])}? "
            "Tell me and I'll update the assessment."
            if suggestions
            else " Add any other signs and I'll update the assessment."
        )

        return self._response(
            state,
            f"{summary} {verdict}{nudge}",
            intent="assessed",
            extraction=extraction,
            prediction=prediction,
            suggestions=suggestions,
        )

    # -- reply fragments ---------------------------------------------------
    def _acknowledge(self, extraction: dict) -> str:
        bits = []
        if extraction["species"]:
            bits.append(f"{extraction['species']}")
        if extraction["symptoms"]:
            bits.append(", ".join(extraction["symptoms"]))
        if extraction["negated"]:
            bits.append(f"ruling out {', '.join(extraction['negated'])}")
        if not bits:
            return "Noted."
        return f"Got it — {'; '.join(bits)}."

    def _missing_prompt(self, state: CaseState) -> str:
        if not state.species:
            examples = ", ".join(self.species_options[:6])
            return f"Which animal is it? For example: {examples}."
        if not state.symptoms:
            return "What signs are you seeing? Temperature, appetite, droppings, breathing, movement."
        return "Tell me more when you're ready."

    def _verdict_sentence(self, top: dict, state: CaseState) -> str:
        """Phrase the model's output for a non-technical reader."""
        pct = f"{top['probability'] * 100:.0f}%"
        n = len(state.symptoms)
        kind = self.model.feature_schema.get("target_kind", "class")

        if kind == "risk":
            if str(top["label"]).lower() == "yes":
                return (
                    f"Based on {n} sign{'s' if n != 1 else ''}, this case scores "
                    f"**{pct} likely to need veterinary attention**. Treat it as "
                    "urgent and contact a vet."
                )
            return (
                f"Based on {n} sign{'s' if n != 1 else ''}, this case scores "
                f"**{pct} lower concern** — but keep watching, and see a vet if "
                "anything worsens."
            )
        return f"Most likely: **{top['label']}** ({pct})."

    def _response(
        self,
        state: CaseState,
        reply: str,
        intent: str,
        extraction: dict | None = None,
        prediction: dict | None = None,
        suggestions: list[str] | None = None,
    ) -> dict:
        return {
            "reply": reply,
            "intent": intent,
            "case": state.to_dict(),
            "ready": state.ready,
            "extracted": extraction
            or {"species": "", "symptoms": [], "negated": [], "unmatched_terms": []},
            "prediction": prediction,
            "suggestions": suggestions or self._suggestions(state),
            "disclaimer": config.DISCLAIMER,
        }


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    from backend.app.model_service import get_service

    return ChatService(get_service())
