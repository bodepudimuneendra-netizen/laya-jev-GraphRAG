"""
graphrag/models/decision_factory.py

Factory to instantiate the active decision/scoring backend.

Configuration (in .env):
  DECISION_MODEL_BACKEND = laya | jev | ablation
  DECISION_MODEL_PRIMARY = laya | jev   (ablation mode only — which drives live pipeline)

Usage:
  from graphrag.models.decision_factory import get_decision_model
  model = get_decision_model()
  score = model.score(context, instruction)   # always returns float in [0, 1]
"""

from __future__ import annotations

import logging
from functools import lru_cache

from config.settings import settings
from .base_decision import BaseDecisionModel

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_decision_model() -> BaseDecisionModel:
    """
    Instantiate and return the active decision model based on DECISION_MODEL_BACKEND.

    Returns
    -------
    BaseDecisionModel
        One of: LayaModel | JevModel | AblationModel
    """
    backend = settings.decision_model_backend.strip().lower()

    if backend == "laya":
        from .laya import get_laya
        logger.info("Decision backend: Laya (local, ~33ms)")
        return get_laya()

    elif backend == "jev":
        from .jev import get_jev
        logger.info("Decision backend: Jev API (cloud, ~70-500ms)")
        return get_jev()

    elif backend == "ablation":
        from .laya import get_laya
        from .jev import get_jev
        from .ablation import AblationModel
        primary = settings.decision_model_primary.strip().lower()
        logger.info(
            "Decision backend: Ablation (Laya + Jev in parallel, primary=%s)",
            primary,
        )
        return AblationModel(
            laya_model=get_laya(),
            jev_model=get_jev(),
            primary=primary,
        )

    else:
        raise ValueError(
            f"Unknown DECISION_MODEL_BACKEND: '{backend}'. "
            f"Choose one of: laya, jev, ablation"
        )
