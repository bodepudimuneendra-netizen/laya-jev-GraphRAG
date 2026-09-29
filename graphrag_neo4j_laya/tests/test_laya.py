"""
tests/test_laya.py — Unit tests for the Laya model wrapper and decision primitives.
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import torch
import pytest


# ── Tokenizer mock that supports .to(device) ─────────────────────────────────

class _TensorBatch(dict):
    """A dict that implements .to(device) so _forward() doesn't fail."""

    def to(self, device):
        return self


def _make_tensor_inputs(batch_size: int = 1) -> "_TensorBatch":
    return _TensorBatch(
        input_ids=torch.zeros(batch_size, 10, dtype=torch.long),
        attention_mask=torch.ones(batch_size, 10, dtype=torch.long),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_laya(logits: list[float], labels: dict[int, str]):
    """Build a LayaModel with mocked tokenizer and model."""
    from graphrag.models.laya import LayaModel

    laya = LayaModel.__new__(LayaModel)
    laya._initialised = True
    laya.device = "cpu"
    laya._id2label = {k: v.lower() for k, v in labels.items()}

    mock_tok = MagicMock()
    mock_tok.return_value = _make_tensor_inputs(1)
    laya.tokenizer = mock_tok

    mock_model = MagicMock()
    mock_model.return_value.logits = torch.tensor([logits])
    laya.model = mock_model

    return laya


# ── LayaModel.score / noul / choice tests ─────────────────────────────────────

class TestLayaModelScore:

    def test_critical_label_returns_high_score(self):
        """With all probability on 'critical', score should be close to 1.0."""
        laya = _make_laya(
            logits=[0.0, 0.0, 10.0],
            labels={0: "irrelevant", 1: "tangential", 2: "critical"},
        )
        score = laya.score("context", "instruction")
        assert score > 0.9, f"Expected score close to 1.0, got {score}"

    def test_irrelevant_label_returns_low_score(self):
        """With all probability on 'irrelevant', score should be ~0.0."""
        laya = _make_laya(
            logits=[10.0, 0.0, 0.0],
            labels={0: "irrelevant", 1: "tangential", 2: "critical"},
        )
        score = laya.score("context", "instruction")
        assert score < 0.1, f"Expected score close to 0.0, got {score}"

    def test_tangential_label_returns_mid_score(self):
        """With all probability on 'tangential', score should be ~0.5."""
        laya = _make_laya(
            logits=[0.0, 10.0, 0.0],
            labels={0: "irrelevant", 1: "tangential", 2: "critical"},
        )
        score = laya.score("context", "instruction")
        assert 0.4 < score < 0.6, f"Expected score ~0.5, got {score}"

    def test_score_is_float_in_unit_interval(self):
        """score() must return a float in [0, 1]."""
        laya = _make_laya(
            logits=[5.0, 5.0],
            labels={0: "irrelevant", 1: "critical"},
        )
        score = laya.score("context", "instruction")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_noul_critical_maps_to_high_pyes(self):
        """noul() should return high P(yes) when 'critical' has all probability."""
        laya = _make_laya(
            logits=[0.0, 0.0, 10.0],
            labels={0: "irrelevant", 1: "tangential", 2: "critical"},
        )
        p_yes = laya.noul("context", "instruction")
        assert p_yes > 0.9

    def test_score_detailed_has_all_fields(self):
        """score_detailed() must return a DecisionResult with all required fields."""
        from graphrag.models.base_decision import DecisionResult

        laya = _make_laya(
            logits=[1.0, 2.0, 7.0],
            labels={0: "irrelevant", 1: "tangential", 2: "critical"},
        )
        result = laya.score_detailed("context", "instruction")
        assert isinstance(result, DecisionResult)
        assert 0.0 <= result.score <= 1.0
        assert result.backend == "laya"
        assert result.primitive == "score"
        assert isinstance(result.raw_probs, dict)
        assert result.latency_ms >= 0

    def test_choice_returns_highest_scoring_key(self):
        """choice() must return the key whose option scores highest."""
        from graphrag.models.laya import LayaModel

        call_count = [0]

        def _tok_side_effect(*args, **kwargs):
            return _make_tensor_inputs(1)

        def _model_side_effect(**kwargs):
            mock_out = MagicMock()
            idx = call_count[0]
            # option_a first (low score), option_b second (high score)
            mock_out.logits = (
                torch.tensor([[10.0, 0.0, 0.0]]) if idx == 0  # irrelevant → score~0
                else torch.tensor([[0.0, 0.0, 10.0]])           # critical   → score~1
            )
            call_count[0] += 1
            return mock_out

        laya = LayaModel.__new__(LayaModel)
        laya._initialised = True
        laya.device = "cpu"
        laya._id2label = {0: "irrelevant", 1: "tangential", 2: "critical"}
        laya.tokenizer = MagicMock(side_effect=_tok_side_effect)
        laya.model = MagicMock(side_effect=_model_side_effect)

        result = laya.choice(
            "context",
            "instruction",
            {"option_a": "unrelated topic", "option_b": "highly critical match"},
        )
        assert result == "option_b"

    def test_batch_score_returns_correct_length(self):
        """batch_score() must return one score per pair (using batch_size=1)."""
        from graphrag.models.laya import LayaModel

        def _tok_side_effect(*args, **kwargs):
            return _make_tensor_inputs(1)

        def _model_side_effect(**kwargs):
            mock_out = MagicMock()
            mock_out.logits = torch.tensor([[0.0, 0.0, 10.0]])  # 1 item → critical
            return mock_out

        laya = LayaModel.__new__(LayaModel)
        laya._initialised = True
        laya.device = "cpu"
        laya._id2label = {0: "irrelevant", 1: "tangential", 2: "critical"}
        laya.tokenizer = MagicMock(side_effect=_tok_side_effect)
        laya.model = MagicMock(side_effect=_model_side_effect)

        pairs = [("ctx1", "inst1"), ("ctx2", "inst2"), ("ctx3", "inst3")]
        # batch_size=1 → serial processing, one forward pass per pair
        scores = laya.batch_score(pairs, batch_size=1)
        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)
        assert all(s > 0.9 for s in scores)  # all critical → high scores


# ── NoulBoundaryChunker tests ─────────────────────────────────────────────────

class TestNoulBoundaryChunker:
    """Test NoulBoundaryChunker by directly injecting a mock model via __new__."""

    @staticmethod
    def _make_chunker(noul_score: float, threshold: float = 0.85, min_sents: int = 2):
        """Build a NoulBoundaryChunker with mocked decision model."""
        from graphrag.ingestion.chunker import NoulBoundaryChunker
        mock_model = MagicMock()
        mock_model.noul.return_value = noul_score
        chunker = NoulBoundaryChunker.__new__(NoulBoundaryChunker)
        chunker._model = mock_model
        chunker._threshold = threshold
        chunker._min_sents = min_sents
        return chunker

    def test_single_sentence_returns_one_chunk(self):
        """A single-sentence doc should always produce exactly one chunk."""
        chunker = self._make_chunker(noul_score=0.1)  # never a boundary
        result = chunker.chunk("Only one sentence here.")
        assert len(result) == 1

    def test_high_boundary_score_creates_chunks(self):
        """When every boundary scores above threshold, each sentence becomes a chunk."""
        chunker = self._make_chunker(noul_score=0.99, min_sents=1)  # always a boundary
        text = "First sentence. Second sentence. Third sentence."
        result = chunker.chunk(text)
        assert len(result) > 1

    def test_low_boundary_score_keeps_single_chunk(self):
        """When all boundaries score below threshold, the doc stays as one chunk."""
        chunker = self._make_chunker(noul_score=0.1)  # no boundary ever
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        result = chunker.chunk(text)
        assert len(result) == 1

    def test_empty_string_returns_list(self):
        """Empty input should not raise and return a list."""
        chunker = self._make_chunker(noul_score=0.5)
        result = chunker.chunk("")
        assert isinstance(result, list)
