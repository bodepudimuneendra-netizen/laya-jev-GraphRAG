"""
tests/test_laya.py — Unit tests for the Laya model wrapper.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
import pytest


class TestLayaModelScore:
    """Test LayaModel.score() with a mocked HuggingFace model."""

    def _make_mock_model(self, logits: list[list[float]], id2label: dict) -> MagicMock:
        mock_model = MagicMock()
        mock_model.config.id2label = id2label
        mock_output = MagicMock()
        mock_output.logits = torch.tensor(logits)
        mock_model.return_value = mock_output
        return mock_model

    def test_critical_label_returns_high_score(self):
        """With all probability on 'critical', score should be 1.0."""
        from graphrag.models.laya import LayaModel

        laya = LayaModel.__new__(LayaModel)
        laya._initialised = True
        laya.device = "cpu"
        laya._id2label = {0: "irrelevant", 1: "tangential", 2: "critical"}

        mock_tok = MagicMock()
        mock_tok.return_value = {"input_ids": torch.zeros(1, 10, dtype=torch.long)}
        laya.tokenizer = mock_tok

        mock_model = MagicMock()
        # Large logit on index 2 (critical) → ~1.0 probability on critical
        mock_model.return_value.logits = torch.tensor([[0.0, 0.0, 10.0]])
        laya.model = mock_model

        score = laya.score("context", "instruction")
        assert score > 0.9, f"Expected score close to 1.0, got {score}"

    def test_irrelevant_label_returns_low_score(self):
        """With all probability on 'irrelevant', score should be ~0.0."""
        from graphrag.models.laya import LayaModel

        laya = LayaModel.__new__(LayaModel)
        laya._initialised = True
        laya.device = "cpu"
        laya._id2label = {0: "irrelevant", 1: "tangential", 2: "critical"}

        mock_tok = MagicMock()
        mock_tok.return_value = {"input_ids": torch.zeros(1, 10, dtype=torch.long)}
        laya.tokenizer = mock_tok

        mock_model = MagicMock()
        mock_model.return_value.logits = torch.tensor([[10.0, 0.0, 0.0]])
        laya.model = mock_model

        score = laya.score("context", "instruction")
        assert score < 0.1, f"Expected score close to 0.0, got {score}"

    def test_score_is_float(self):
        from graphrag.models.laya import LayaModel

        laya = LayaModel.__new__(LayaModel)
        laya._initialised = True
        laya.device = "cpu"
        laya._id2label = {0: "irrelevant", 1: "critical"}

        mock_tok = MagicMock()
        mock_tok.return_value = {"input_ids": torch.zeros(1, 10, dtype=torch.long)}
        laya.tokenizer = mock_tok

        mock_model = MagicMock()
        mock_model.return_value.logits = torch.tensor([[5.0, 5.0]])
        laya.model = mock_model

        score = laya.score("context", "instruction")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestLayaChunker:
    """Test NoulBoundaryChunker with mocked Laya."""

    def test_single_sentence_returns_one_chunk(self):
        with patch("graphrag.ingestion.chunker.get_laya") as mock_get_laya:
            mock_laya = MagicMock()
            mock_laya.score.return_value = 0.1  # never a boundary
            mock_get_laya.return_value = mock_laya

            from graphrag.ingestion.chunker import NoulBoundaryChunker
            chunker = NoulBoundaryChunker(threshold=0.85)
            result = chunker.chunk("Only one sentence here.")
            assert len(result) == 1

    def test_high_boundary_score_creates_chunks(self):
        with patch("graphrag.ingestion.chunker.get_laya") as mock_get_laya:
            mock_laya = MagicMock()
            mock_laya.score.return_value = 0.95  # always a boundary
            mock_get_laya.return_value = mock_laya

            from graphrag.ingestion.chunker import NoulBoundaryChunker
            chunker = NoulBoundaryChunker(threshold=0.85, min_chunk_sentences=1)
            text = "First sentence. Second sentence. Third sentence."
            result = chunker.chunk(text)
            assert len(result) > 1
