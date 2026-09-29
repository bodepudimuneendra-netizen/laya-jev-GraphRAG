"""
graphrag/models/ablation.py

Ablation harness that runs BOTH Laya and Jev simultaneously on each
(context, instruction) pair and records a side-by-side comparison.

Usage
-----
Set DECISION_MODEL_BACKEND=ablation in your .env.

The ablation model:
  1. Scores with Laya (local, ~33 ms)
  2. Scores with Jev  (API, ~70-500 ms)
  3. Logs both results to a JSONL file for analysis
  4. Returns the score from the DECISION_MODEL_PRIMARY backend for the
     live pipeline (so traversal continues normally)

Ablation log format (ablation_log.jsonl)
-----------------------------------------
{
  "timestamp": "2026-09-23T16:52:00Z",
  "context":    "...",
  "instruction": "...",
  "laya": {"score": 0.82, "confidence": 0.91, "latency_ms": 31.2, "raw_probs": {...}},
  "jev":  {"score": 0.79, "confidence": 0.94, "latency_ms": 183.0, "raw_probs": {...}},
  "delta": 0.03,          # abs(laya.score - jev.score)
  "primary": "laya"       # which backend drives the live pipeline
}
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config.settings import settings
from .base_decision import BaseDecisionModel, DecisionResult

logger = logging.getLogger(__name__)


class AblationModel(BaseDecisionModel):
    """
    Side-by-side evaluation harness for Laya vs. Jev.

    Runs both models in parallel threads to minimise wall-clock overhead.
    The primary backend (DECISION_MODEL_PRIMARY) determines the score
    returned to the live A* and BFS traversal.

    Parameters
    ----------
    laya_model:
        Pre-constructed Laya instance (injected for testability).
    jev_model:
        Pre-constructed Jev instance (injected for testability).
    log_path:
        Path to the JSONL ablation log. Defaults to ablation_log.jsonl.
    primary:
        Which backend drives the live pipeline score ("laya" or "jev").
    """

    def __init__(
        self,
        laya_model: Optional[BaseDecisionModel] = None,
        jev_model: Optional[BaseDecisionModel] = None,
        log_path: Optional[Path] = None,
        primary: str = "laya",
    ) -> None:
        # Lazy-import to avoid loading models at import time
        if laya_model is None:
            from .laya import get_laya
            laya_model = get_laya()
        if jev_model is None:
            from .jev import get_jev
            jev_model = get_jev()

        self._laya = laya_model
        self._jev = jev_model
        self._primary = primary.lower()
        self._log_path = log_path or Path(settings.ablation_log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_lock = threading.Lock()

        logger.info(
            "AblationModel ready — primary=%s | log=%s",
            self._primary, self._log_path
        )

    def _log(
        self,
        context: str,
        instruction: str,
        laya_result: DecisionResult,
        jev_result: DecisionResult,
    ) -> None:
        """Append one comparison record to the JSONL log (thread-safe)."""
        record = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "context": context[:200],        # truncate for log compactness
            "instruction": instruction[:200],
            "laya": {
                "score": laya_result.score,
                "confidence": laya_result.confidence,
                "latency_ms": laya_result.latency_ms,
                "raw_probs": laya_result.raw_probs,
            },
            "jev": {
                "score": jev_result.score,
                "confidence": jev_result.confidence,
                "latency_ms": jev_result.latency_ms,
                "raw_probs": jev_result.raw_probs,
            },
            "delta": round(abs(laya_result.score - jev_result.score), 4),
            "primary": self._primary,
        }
        with self._file_lock:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    def _run_both(
        self, context: str, instruction: str
    ) -> tuple[DecisionResult, DecisionResult]:
        """Run Laya and Jev score in parallel threads."""
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_laya = ex.submit(self._laya.score_detailed, context, instruction)
            fut_jev = ex.submit(self._jev.score_detailed, context, instruction)
            laya_result = fut_laya.result()
            jev_result = fut_jev.result()
        return laya_result, jev_result

    def _run_both_noul(
        self, context: str, instruction: str
    ) -> tuple[DecisionResult, DecisionResult]:
        """Run Laya and Jev noul in parallel threads."""
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_laya = ex.submit(self._laya.noul_detailed, context, instruction)
            fut_jev = ex.submit(self._jev.noul_detailed, context, instruction)
            laya_result = fut_laya.result()
            jev_result = fut_jev.result()
        return laya_result, jev_result

    def _run_both_choice(
        self, context: str, instruction: str, options: dict[str, str]
    ) -> tuple[DecisionResult, DecisionResult]:
        """Run Laya and Jev choice in parallel threads."""
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_laya = ex.submit(self._laya.choice_detailed, context, instruction, options)
            fut_jev = ex.submit(self._jev.choice_detailed, context, instruction, options)
            laya_result = fut_laya.result()
            jev_result = fut_jev.result()
        return laya_result, jev_result

    # ── BaseDecisionModel interface ───────────────────────────────────────────

    def score(self, context: str, instruction: str) -> float:
        """Run both models, log results, return primary backend's score."""
        laya_result, jev_result = self._run_both(context, instruction)
        self._log(context, instruction, laya_result, jev_result)

        primary_result = laya_result if self._primary == "laya" else jev_result
        logger.debug(
            "Ablation | laya=%.3f (%.0fms) | jev=%.3f (%.0fms) | delta=%.3f | primary=%s",
            laya_result.score, laya_result.latency_ms,
            jev_result.score, jev_result.latency_ms,
            abs(laya_result.score - jev_result.score),
            self._primary,
        )
        return primary_result.score

    def score_detailed(self, context: str, instruction: str) -> DecisionResult:
        """
        Return a combined DecisionResult from the primary backend.
        The ablation log always captures both regardless.
        """
        laya_result, jev_result = self._run_both(context, instruction)
        self._log(context, instruction, laya_result, jev_result)

        primary_result = laya_result if self._primary == "laya" else jev_result
        return DecisionResult(
            score=primary_result.score,
            confidence=primary_result.confidence,
            raw_probs=primary_result.raw_probs,
            latency_ms=max(laya_result.latency_ms, jev_result.latency_ms),  # wall-clock (parallel)
            backend="ablation",
            primitive="score",
        )

    def noul(self, context: str, instruction: str) -> float:
        """Run both models for noul, log, return primary backend's P(yes)."""
        laya_result, jev_result = self._run_both_noul(context, instruction)
        self._log(context, instruction, laya_result, jev_result)
        primary_result = laya_result if self._primary == "laya" else jev_result
        return primary_result.score

    def noul_detailed(self, context: str, instruction: str) -> DecisionResult:
        """Full DecisionResult for a noul query — both backends logged."""
        laya_result, jev_result = self._run_both_noul(context, instruction)
        self._log(context, instruction, laya_result, jev_result)
        primary_result = laya_result if self._primary == "laya" else jev_result
        return DecisionResult(
            score=primary_result.score,
            confidence=primary_result.confidence,
            raw_probs=primary_result.raw_probs,
            latency_ms=max(laya_result.latency_ms, jev_result.latency_ms),
            backend="ablation",
            primitive="noul",
        )

    def choice(
        self, context: str, instruction: str, options: dict[str, str]
    ) -> str:
        """Run both models for choice, log, return primary backend's selection."""
        laya_result, jev_result = self._run_both_choice(context, instruction, options)
        self._log(context, instruction, laya_result, jev_result)
        primary_result = laya_result if self._primary == "laya" else jev_result
        return primary_result.selected or next(iter(options))

    def choice_detailed(
        self, context: str, instruction: str, options: dict[str, str]
    ) -> DecisionResult:
        """Full DecisionResult for a choice query — both backends logged."""
        laya_result, jev_result = self._run_both_choice(context, instruction, options)
        self._log(context, instruction, laya_result, jev_result)
        primary_result = laya_result if self._primary == "laya" else jev_result
        return DecisionResult(
            score=primary_result.score,
            confidence=primary_result.confidence,
            raw_probs=primary_result.raw_probs,
            latency_ms=max(laya_result.latency_ms, jev_result.latency_ms),
            backend="ablation",
            primitive="choice",
            selected=primary_result.selected,
        )

    def batch_score(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 16,
    ) -> list[float]:
        return [self.score(ctx, instr) for ctx, instr in pairs]

    # ── Analysis helpers ──────────────────────────────────────────────────────

    def analyze_log(self) -> dict:
        """
        Read the ablation log and compute summary statistics.

        Returns
        -------
        dict with keys: n_samples, mean_delta, max_delta, mean_laya_ms, mean_jev_ms,
                        agreement_rate (delta < 0.05)
        """
        if not self._log_path.exists():
            return {"error": "No ablation log found"}

        records = []
        with open(self._log_path, encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line.strip()))

        if not records:
            return {"error": "Empty ablation log"}

        deltas = [r["delta"] for r in records]
        laya_ms = [r["laya"]["latency_ms"] for r in records]
        jev_ms = [r["jev"]["latency_ms"] for r in records]
        agreed = sum(1 for d in deltas if d < 0.05)

        return {
            "n_samples": len(records),
            "mean_delta": round(sum(deltas) / len(deltas), 4),
            "max_delta": round(max(deltas), 4),
            "mean_laya_ms": round(sum(laya_ms) / len(laya_ms), 1),
            "mean_jev_ms": round(sum(jev_ms) / len(jev_ms), 1),
            "agreement_rate": round(agreed / len(records), 3),
            "log_path": str(self._log_path),
        }


# ── AblationHarness — public API shown in README quickstart ──────────────────

class AblationHarness:
    """
    High-level ablation API shown in the README quickstart.

    Runs both Laya and Jev on a single primitive call and returns a
    structured comparison dict ready for RLCD training data inspection.

    Usage
    -----
    >>> from graphrag.models.ablation import AblationHarness
    >>> harness = AblationHarness()
    >>> result = harness.compare(
    ...     context="Isaac Newton published Principia Mathematica in 1687.",
    ...     instruction="Score the historical significance of this event.",
    ... )
    >>> print(f"Laya: {result['laya']['score']:.3f} @ {result['laya']['latency_ms']:.0f}ms")
    >>> print(f"Jev:  {result['jev']['score']:.3f}  @ {result['jev']['latency_ms']:.0f}ms")
    >>> print(f"delta:    {result['delta']:.4f}")
    """

    def __init__(
        self,
        laya_model: Optional[BaseDecisionModel] = None,
        jev_model: Optional[BaseDecisionModel] = None,
        log_path: Optional[Path] = None,
    ) -> None:
        if laya_model is None:
            from .laya import get_laya
            laya_model = get_laya()
        if jev_model is None:
            from .jev import get_jev
            jev_model = get_jev()

        self._laya = laya_model
        self._jev = jev_model
        self._log_path = log_path or Path(settings.ablation_log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_lock = threading.Lock()

    def compare(
        self,
        context: str,
        instruction: str,
        primitive: str = "score",
        options: dict[str, str] | None = None,
    ) -> dict:
        """
        Run Laya and Jev on the same input and return a comparison dict.

        Parameters
        ----------
        context:
            The state/text to evaluate.
        instruction:
            The question/directive.
        primitive:
            Which primitive to compare: "score" | "noul" | "choice".
        options:
            Required for 'choice' primitive.

        Returns
        -------
        dict
            {
              "laya":  {"score": float, "latency_ms": float, "confidence": float},
              "jev":   {"score": float, "latency_ms": float, "confidence": float},
              "delta": float,
              "primitive": str,
            }
        """
        with ThreadPoolExecutor(max_workers=2) as ex:
            if primitive == "noul":
                fut_laya = ex.submit(self._laya.noul_detailed, context, instruction)
                fut_jev = ex.submit(self._jev.noul_detailed, context, instruction)
            elif primitive == "choice":
                if not options:
                    raise ValueError("options dict is required for 'choice' primitive")
                fut_laya = ex.submit(self._laya.choice_detailed, context, instruction, options)
                fut_jev = ex.submit(self._jev.choice_detailed, context, instruction, options)
            else:  # score (default)
                fut_laya = ex.submit(self._laya.score_detailed, context, instruction)
                fut_jev = ex.submit(self._jev.score_detailed, context, instruction)
            laya_r = fut_laya.result()
            jev_r = fut_jev.result()

        delta = round(abs(laya_r.score - jev_r.score), 4)

        record = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "primitive": primitive,
            "context": context[:200],
            "instruction": instruction[:200],
            "laya": {
                "score": laya_r.score,
                "latency_ms": laya_r.latency_ms,
                "confidence": laya_r.confidence,
            },
            "jev": {
                "score": jev_r.score,
                "latency_ms": jev_r.latency_ms,
                "confidence": jev_r.confidence,
            },
            "delta": delta,
        }
        with self._file_lock:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        return record
