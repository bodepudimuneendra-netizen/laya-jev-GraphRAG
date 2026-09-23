"""
graphrag/models/jev.py

TypeSafe Jev API client implementing BaseDecisionModel.

REAL API FORMAT (from official docs):
  POST https://api.typesafe.ai/v1/systemone
  {
    "model": "jev-1.13",
    "state": "<context text>",
    "questions": {
      "q_name": {
        "type":         "noul" | "score" | "choice",
        "instructions": "<directive>",
        "criteria":     { ... }   # only for choice / score
      }
    }
  }

Key advantage: ALL questions in a single request are processed in PARALLEL
server-side — so ask_batch() costs the same as a single noul() call.

Pricing: ~$0.042 per million input tokens. Output tokens are free.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache

import requests

from config.settings import settings
from .base_decision import BaseDecisionModel, DecisionResult

logger = logging.getLogger(__name__)

_JEV_API_URL = "https://api.typesafe.ai/v1/systemone"
_JEV_MODEL   = "jev-1.13"


def _fallback(primitive: str, fallback_score: float) -> DecisionResult:
    return DecisionResult(
        score=fallback_score, confidence=0.0,
        raw_probs={}, latency_ms=0.0,
        backend="jev", primitive=primitive,
    )


class JevModel(BaseDecisionModel):
    """
    TypeSafe Jev API wrapper.

    Uses the correct state+questions dict format. All questions in a batch
    are answered in a single parallel API call.
    """

    def __init__(self) -> None:
        self._api_key = settings.jev_api_key
        if not self._api_key:
            raise ValueError(
                "JEV_API_KEY is not set. "
                "Get yours at https://typesafe.ai and set it in .env"
            )
        self._timeout        = settings.jev_timeout_seconds
        self._fallback_score = settings.jev_fallback_score
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        })
        logger.info(
            "Jev API client ready (model=%s, timeout=%ss, fallback=%.2f)",
            _JEV_MODEL, self._timeout, self._fallback_score,
        )

    # ── Core API call ─────────────────────────────────────────────────────────

    def _call(self, state: str, questions: dict) -> tuple[dict, float]:
        """
        Send a multi-question request to the Jev API.
        Returns (response_body, latency_ms).
        """
        payload = {"model": _JEV_MODEL, "state": state, "questions": questions}
        t0 = time.perf_counter()
        resp = self._session.post(_JEV_API_URL, json=payload, timeout=self._timeout)
        latency_ms = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        return resp.json(), latency_ms

    def _single(
        self,
        context: str,
        instruction: str,
        q_type: str,
        criteria: dict | list | None = None,
    ) -> tuple[dict, float]:
        """Build a single-question Jev call and return the question result + latency."""
        question: dict = {"type": q_type, "instructions": instruction}
        if criteria:
            question["criteria"] = criteria
        body, latency_ms = self._call(context, {"q": question})
        return body["questions"]["q"], latency_ms

    # ── Score Primitive ───────────────────────────────────────────────────────

    def score(self, context: str, instruction: str) -> float:
        try:
            result, _ = self._single(
                context, instruction, "score",
                criteria=["Irrelevant", "Tangential", "Relevant", "Critical"],
            )
            # Jev score: weighted position on the ordinal scale → normalise to [0, 1]
            # raw_score is 0-3 over the 4-level rubric
            raw = float(result.get("score", self._fallback_score * 3))
            return min(raw / 3.0, 1.0)
        except Exception as e:
            logger.warning("Jev score() fallback: %s", e)
            return self._fallback_score

    def score_detailed(self, context: str, instruction: str) -> DecisionResult:
        try:
            result, latency_ms = self._single(
                context, instruction, "score",
                criteria=["Irrelevant", "Tangential", "Relevant", "Critical"],
            )
            raw = float(result.get("score", self._fallback_score * 3))
            score = min(raw / 3.0, 1.0)
            probs = result.get("probabilities", {})
            return DecisionResult(
                score=score,
                confidence=float(result.get("confidence", 0.0)),
                raw_probs=probs,
                latency_ms=round(latency_ms, 2),
                backend="jev",
                primitive="score",
            )
        except Exception as e:
            logger.warning("Jev score_detailed() fallback: %s", e)
            return _fallback("score", self._fallback_score)

    # ── Noul Primitive ────────────────────────────────────────────────────────

    def noul(self, context: str, instruction: str) -> float:
        try:
            result, _ = self._single(context, instruction, "noul")
            return float(result.get("probability", self._fallback_score))
        except Exception as e:
            logger.warning("Jev noul() fallback: %s", e)
            return self._fallback_score

    def noul_detailed(self, context: str, instruction: str) -> DecisionResult:
        try:
            result, latency_ms = self._single(context, instruction, "noul")
            p = float(result.get("probability", self._fallback_score))
            return DecisionResult(
                score=p,
                confidence=p,       # noul: probability IS the confidence
                raw_probs={"yes": round(p, 4), "no": round(1 - p, 4)},
                latency_ms=round(latency_ms, 2),
                backend="jev",
                primitive="noul",
            )
        except Exception as e:
            logger.warning("Jev noul_detailed() fallback: %s", e)
            return _fallback("noul", self._fallback_score)

    # ── Choice Primitive ──────────────────────────────────────────────────────

    def choice(
        self, context: str, instruction: str, options: dict[str, str],
    ) -> str:
        try:
            result, _ = self._single(context, instruction, "choice", criteria=options)
            return str(result.get("selected", next(iter(options))))
        except Exception as e:
            logger.warning("Jev choice() fallback: %s", e)
            return next(iter(options))   # default: first option

    def choice_detailed(
        self, context: str, instruction: str, options: dict[str, str],
    ) -> DecisionResult:
        try:
            result, latency_ms = self._single(context, instruction, "choice", criteria=options)
            selected = str(result.get("selected", next(iter(options))))
            probs = result.get("probabilities", {})
            conf  = float(result.get("confidence", probs.get(selected, 0.0)))
            return DecisionResult(
                score=conf,
                confidence=conf,
                raw_probs=probs,
                latency_ms=round(latency_ms, 2),
                backend="jev",
                primitive="choice",
                selected=selected,
            )
        except Exception as e:
            logger.warning("Jev choice_detailed() fallback: %s", e)
            return DecisionResult(
                score=0.0, confidence=0.0, raw_probs={},
                latency_ms=0.0, backend="jev", primitive="choice",
                selected=next(iter(options)),
            )

    # ── Batch (true parallel — single API call) ───────────────────────────────

    def ask_batch(
        self, state: str, questions: dict[str, dict],
    ) -> dict[str, DecisionResult]:
        """
        Send ALL questions in ONE API call — processed in parallel server-side.
        This is the key advantage over Laya's serial implementation.
        """
        api_questions: dict[str, dict] = {}
        for name, spec in questions.items():
            q: dict = {"type": spec["type"], "instructions": spec.get("instruction", "")}
            if "options" in spec:
                q["criteria"] = spec["options"]
            api_questions[name] = q

        try:
            t0 = time.perf_counter()
            body, latency_ms = self._call(state, api_questions)
            results: dict[str, DecisionResult] = {}
            q_results = body.get("questions", {})

            for name, spec in questions.items():
                raw = q_results.get(name, {})
                q_type = spec["type"]
                if q_type == "noul":
                    p = float(raw.get("probability", self._fallback_score))
                    results[name] = DecisionResult(
                        score=p, confidence=p,
                        raw_probs={"yes": round(p, 4), "no": round(1-p, 4)},
                        latency_ms=round(latency_ms, 2),
                        backend="jev", primitive="noul",
                    )
                elif q_type == "score":
                    raw_s = float(raw.get("score", self._fallback_score * 3))
                    s = min(raw_s / 3.0, 1.0)
                    results[name] = DecisionResult(
                        score=s, confidence=float(raw.get("confidence", 0.0)),
                        raw_probs=raw.get("probabilities", {}),
                        latency_ms=round(latency_ms, 2),
                        backend="jev", primitive="score",
                    )
                elif q_type == "choice":
                    opts = spec.get("options", {})
                    sel = str(raw.get("selected", next(iter(opts), "")))
                    probs = raw.get("probabilities", {})
                    conf  = float(raw.get("confidence", probs.get(sel, 0.0)))
                    results[name] = DecisionResult(
                        score=conf, confidence=conf,
                        raw_probs=probs,
                        latency_ms=round(latency_ms, 2),
                        backend="jev", primitive="choice", selected=sel,
                    )
            return results

        except Exception as e:
            logger.warning("Jev ask_batch() fallback: %s", e)
            fallbacks: dict[str, DecisionResult] = {}
            for name, spec in questions.items():
                q_type = spec["type"]
                sel = next(iter(spec.get("options", {".": "."})))
                fallbacks[name] = DecisionResult(
                    score=self._fallback_score, confidence=0.0, raw_probs={},
                    latency_ms=0.0, backend="jev", primitive=q_type,
                    selected=sel if q_type == "choice" else None,
                )
            return fallbacks


@lru_cache(maxsize=1)
def get_jev() -> JevModel:
    """Return the global Jev singleton (lazy-loaded on first call)."""
    return JevModel()
