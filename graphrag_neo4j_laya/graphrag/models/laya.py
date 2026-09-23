"""
graphrag/models/laya.py

Thin, singleton-aware wrapper around the Laya typed-decisions classifier.

Model:  convaiinnovations/laya-typed-decisions
        Apache 2.0 · ModernBERT-large backbone · 421M params
        ~1.2 GB VRAM in FP16 on CUDA

All three decision primitives are implemented:

  score(context, instruction)               → float [0,1]  ordinal relevance
  noul(context, instruction)                → float [0,1]  P(yes)
  choice(context, instruction, options)     → str   selected option key

S_Laya formula (idea.md §2.1):
  S_Laya = P(Critical)·1.0 + P(Tangential)·0.5 + P(Irrelevant)·0.0
"""

from __future__ import annotations

import logging
import threading
import time
from functools import lru_cache
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config.settings import settings
from .base_decision import BaseDecisionModel, DecisionResult

logger = logging.getLogger(__name__)

# ── Label weight maps ─────────────────────────────────────────────────────────
# Score primitive — three-class
_SCORE_WEIGHTS = {"irrelevant": 0.0, "tangential": 0.5, "critical": 1.0}
# Noul primitive — binary
_NOUL_WEIGHTS  = {"false": 0.0, "no": 0.0, "true": 1.0, "yes": 1.0,
                  "irrelevant": 0.0, "tangential": 0.0, "critical": 1.0}


def _apply_score_weights(probs: torch.Tensor, id2label: dict[int, str]) -> tuple[float, float, dict[str, float]]:
    """Compute (score, confidence, raw_probs) from a softmax tensor."""
    total = 0.0
    raw: dict[str, float] = {}
    for idx, label in id2label.items():
        p = probs[idx].item()
        raw[label] = round(p, 4)
        w = _SCORE_WEIGHTS.get(label, _NOUL_WEIGHTS.get(label, 0.0))
        total += w * p
    return float(total), float(probs.max().item()), raw


def _apply_noul_weights(probs: torch.Tensor, id2label: dict[int, str]) -> tuple[float, float, dict[str, float]]:
    """P(yes) = P(critical) + 0.5·P(tangential) for three-class; P(yes) for binary."""
    total = 0.0
    raw: dict[str, float] = {}
    for idx, label in id2label.items():
        p = probs[idx].item()
        raw[label] = round(p, 4)
        w = _NOUL_WEIGHTS.get(label, 0.0)
        total += w * p
    return float(total), float(probs.max().item()), raw


class LayaModel(BaseDecisionModel):
    """
    Singleton wrapper for the Laya typed-decisions classifier.
    Thread-safe: the internal lock prevents duplicate model loads.
    """

    _instance: "LayaModel | None" = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "LayaModel":
        with cls._lock:
            if cls._instance is None:
                obj = object.__new__(cls)
                obj._initialised = False
                cls._instance = obj
        return cls._instance

    def __init__(self) -> None:
        if self._initialised:
            return
        self._initialised = True
        self._load()

    def _load(self) -> None:
        model_id = settings.laya_model_id
        logger.info("Loading Laya model: %s", model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            logger.warning("CUDA not available — Laya will run on CPU (slow).")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()
        self._id2label: dict[int, str] = {
            int(k): v.lower() for k, v in self.model.config.id2label.items()
        }
        logger.info("Laya ready on %s | labels: %s", self.device, list(self._id2label.values()))

    # ── Internal tokenise + forward ───────────────────────────────────────────

    @torch.inference_mode()
    def _forward(self, context: str, text_pair: str) -> tuple[torch.Tensor, float]:
        """Tokenise, run forward pass, return (softmax_probs, latency_ms)."""
        inputs = self.tokenizer(
            context, text_pair=text_pair,
            return_tensors="pt", truncation=True, max_length=512, padding=True,
        ).to(self.device)
        t0 = time.perf_counter()
        outputs = self.model(**inputs)
        latency_ms = (time.perf_counter() - t0) * 1000
        probs = torch.softmax(outputs.logits, dim=1)[0]
        return probs, latency_ms

    # ── Score Primitive ───────────────────────────────────────────────────────

    def score(self, context: str, instruction: str) -> float:
        probs, _ = self._forward(context, instruction)
        s, _, _ = _apply_score_weights(probs, self._id2label)
        return s

    def score_detailed(self, context: str, instruction: str) -> DecisionResult:
        probs, latency_ms = self._forward(context, instruction)
        s, conf, raw = _apply_score_weights(probs, self._id2label)
        return DecisionResult(score=s, confidence=conf, raw_probs=raw,
                              latency_ms=round(latency_ms, 2), backend="laya", primitive="score")

    # ── Noul Primitive ────────────────────────────────────────────────────────

    def noul(self, context: str, instruction: str) -> float:
        """P(yes) — maps P(critical) → yes, P(irrelevant) → no."""
        probs, _ = self._forward(context, instruction)
        p, _, _ = _apply_noul_weights(probs, self._id2label)
        return p

    def noul_detailed(self, context: str, instruction: str) -> DecisionResult:
        probs, latency_ms = self._forward(context, instruction)
        p, conf, raw = _apply_noul_weights(probs, self._id2label)
        return DecisionResult(score=p, confidence=conf, raw_probs=raw,
                              latency_ms=round(latency_ms, 2), backend="laya", primitive="noul")

    # ── Choice Primitive ──────────────────────────────────────────────────────

    def choice(self, context: str, instruction: str, options: dict[str, str]) -> str:
        """
        Zero-shot multi-class selection.
        Scores each option description against the context+instruction
        and returns the key with the highest score.
        """
        if not options:
            raise ValueError("choice() requires at least one option.")
        best_key, best_score = "", -1.0
        for key, description in options.items():
            pair = f"{instruction}\n\nOption: {description}"
            s = self.score(context, pair)
            if s > best_score:
                best_score, best_key = s, key
        return best_key

    def choice_detailed(
        self, context: str, instruction: str, options: dict[str, str]
    ) -> DecisionResult:
        if not options:
            raise ValueError("choice_detailed() requires at least one option.")
        scores: dict[str, float] = {}
        t0 = time.perf_counter()
        for key, description in options.items():
            pair = f"{instruction}\n\nOption: {description}"
            scores[key] = self.score(context, pair)
        latency_ms = (time.perf_counter() - t0) * 1000

        best_key = max(scores, key=scores.__getitem__)
        # Normalise scores to a probability distribution
        total = sum(scores.values()) or 1.0
        raw_probs = {k: round(v / total, 4) for k, v in scores.items()}

        return DecisionResult(
            score=scores[best_key],
            confidence=scores[best_key] / total,
            raw_probs=raw_probs,
            latency_ms=round(latency_ms, 2),
            backend="laya",
            primitive="choice",
            selected=best_key,
        )

    # ── Batch Score (GPU-optimised) ───────────────────────────────────────────

    def batch_score(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 16,
    ) -> list[float]:
        results: list[float] = []
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            contexts, instructions = zip(*chunk)
            inputs = self.tokenizer(
                list(contexts), text_pair=list(instructions),
                return_tensors="pt", truncation=True, max_length=512, padding=True,
            ).to(self.device)
            with torch.inference_mode():
                outputs = self.model(**inputs)
                probs_batch = torch.softmax(outputs.logits, dim=1)
            for row in probs_batch:
                s, _, _ = _apply_score_weights(row, self._id2label)
                results.append(s)
        return results


@lru_cache(maxsize=1)
def get_laya() -> LayaModel:
    """Return the global Laya singleton (lazy-loaded on first call)."""
    return LayaModel()
