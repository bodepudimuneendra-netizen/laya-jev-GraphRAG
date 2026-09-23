"""
graphrag/models/llm.py

Wrapper for the pre-quantized Llama-3.1 8B Instruct (NF4) model.

Model:  hugging-quants/Meta-Llama-3.1-8B-Instruct-BNB-NF4
        Pre-quantized with bitsandbytes NF4 — no on-the-fly quantization.
        ~5.5 GB VRAM, leaving headroom for Laya (~1.2 GB) on RTX 5060 (8 GB).

Usage
-----
    from graphrag.models.llm import get_llm
    answer = get_llm().generate("Summarise these graph paths: ...")
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.settings import settings

logger = logging.getLogger(__name__)


class LLMModel:
    """
    Singleton wrapper for the Llama-3.1 8B Instruct NF4 model.

    The model is loaded once and kept resident in VRAM for the duration of
    the process.  Loading is deferred to the first call to .generate().
    """

    _instance: "LLMModel | None" = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "LLMModel":
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

    # ── Private ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        model_id = settings.llm_model_id
        logger.info("Loading LLM: %s", model_id)

        hf_token = settings.huggingface_token or None

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=hf_token,
        )
        # The NF4 checkpoint already has quantization baked in;
        # device_map="auto" handles multi-GPU / CPU offload automatically.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=hf_token,
        )
        self.model.eval()
        logger.info("LLM ready (device_map=auto).")

    # ── Public API ────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        do_sample: bool = False,
        system_prompt: str = (
            "You are a precise knowledge-graph assistant. "
            "Synthesise the retrieved graph paths into a concise, factual answer."
        ),
    ) -> str:
        """
        Run a single forward pass and return the generated text.

        Parameters
        ----------
        prompt:
            User-facing query or instruction string.
        max_new_tokens:
            Maximum number of new tokens to generate.
        temperature:
            Sampling temperature (only used when do_sample=True).
        do_sample:
            Enable stochastic sampling.  False → greedy decoding.
        system_prompt:
            System-role message prepended to every conversation.

        Returns
        -------
        str
            Decoded assistant response (leading/trailing whitespace stripped).
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(next(self.model.parameters()).device)

        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        # Slice off the prompt tokens
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@lru_cache(maxsize=1)
def get_llm() -> LLMModel:
    """Return the global LLM singleton (lazy-loaded on first call)."""
    return LLMModel()
