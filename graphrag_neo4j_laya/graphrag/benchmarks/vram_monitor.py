"""
graphrag/benchmarks/vram_monitor.py

VRAM Ceiling Verification — idea.md §5.

    "Monitor nvidia-smi while the Laya PyTorch loop passes its final output
     to the 4-bit Llama-3.1 instance.
     Ensure peak consumption remains below 7.5 GB."

Polls `nvidia-smi` every 500 ms in a background thread while a user-supplied
callback runs.  Prints a report and raises if the ceiling is breached.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from config.settings import settings

logger = logging.getLogger(__name__)
POLL_INTERVAL_S = 0.5


@dataclass
class VRAMReport:
    peak_mb:   float = 0.0
    ceiling_mb: float = field(default_factory=lambda: settings.vram_ceiling_gb * 1024)
    samples:   list[float] = field(default_factory=list)
    breached:  bool = False

    @property
    def peak_gb(self) -> float:
        return self.peak_mb / 1024

    @property
    def ceiling_gb(self) -> float:
        return self.ceiling_mb / 1024


def _read_vram_mb() -> float | None:
    """Query GPU 0 VRAM usage via nvidia-smi (returns MB or None)."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
        line = result.stdout.strip().splitlines()[0]
        return float(line)
    except Exception as exc:
        logger.debug("nvidia-smi query failed: %s", exc)
        return None


def monitor(workload: Callable[[], None]) -> VRAMReport:
    """
    Run *workload* while polling VRAM every 500 ms.

    Parameters
    ----------
    workload:
        A zero-argument callable (e.g., a full pipeline call).

    Returns
    -------
    VRAMReport
        Contains peak usage, all samples, and breach flag.
    """
    report  = VRAMReport()
    stop_ev = threading.Event()

    def _poll() -> None:
        while not stop_ev.is_set():
            mb = _read_vram_mb()
            if mb is not None:
                report.samples.append(mb)
                if mb > report.peak_mb:
                    report.peak_mb = mb
            time.sleep(POLL_INTERVAL_S)

    monitor_thread = threading.Thread(target=_poll, daemon=True)
    monitor_thread.start()

    try:
        workload()
    finally:
        stop_ev.set()
        monitor_thread.join(timeout=2)

    report.breached = report.peak_mb > report.ceiling_mb

    # ── Print report ──────────────────────────────────────────────────────────
    status = "❌ BREACHED" if report.breached else "✅ SAFE"
    print(f"\n─── VRAM Monitor Report ───────────────────────────────────")
    print(f"  Peak usage : {report.peak_gb:.2f} GB / {report.ceiling_gb:.2f} GB  {status}")
    print(f"  Samples    : {len(report.samples)}  (every {POLL_INTERVAL_S:.1f}s)")
    print(f"  Mean       : {sum(report.samples)/max(len(report.samples),1)/1024:.2f} GB")
    print(f"────────────────────────────────────────────────────────────\n")

    if report.breached:
        logger.error(
            "VRAM ceiling breached! Peak %.2f GB > ceiling %.2f GB",
            report.peak_gb,
            report.ceiling_gb,
        )
    return report


def main() -> None:  # pragma: no cover
    """CLI demo: load both models and run a trivial forward pass."""
    logging.basicConfig(level=logging.INFO)

    def _demo_workload() -> None:
        from graphrag.models.laya import get_laya  # noqa: PLC0415
        from graphrag.models.llm import get_llm    # noqa: PLC0415

        laya = get_laya()
        llm  = get_llm()

        # Laya forward pass
        laya.score("Node: gravity. Edge: CAUSES. Target: apple falling.", "Score relevance to: 'what causes objects to fall?'")

        # LLM forward pass
        llm.generate("Summarise the retrieved graph paths in one sentence.", max_new_tokens=64)

    report = monitor(_demo_workload)
    if report.breached:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
