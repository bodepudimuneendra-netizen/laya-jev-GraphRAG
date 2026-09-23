"""
graphrag/benchmarks/laya_vs_jev.py

Ablation benchmark: head-to-head comparison of Laya vs. Jev on a set of
representative graph edge-scoring prompts.

Usage
-----
    # First, ensure DECISION_MODEL_BACKEND=ablation in your .env
    # and JEV_API_KEY is set.

    python -m graphrag.benchmarks.laya_vs_jev

    # Or run with a custom prompt file:
    python -m graphraf.benchmarks.laya_vs_jev --prompts my_prompts.jsonl

Output
------
    benchmarks/results/ablation_log.jsonl   — raw JSONL (one record per pair)
    benchmarks/results/laya_vs_jev.png      — latency/score scatter plot
    Terminal table showing summary stats
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Canned test prompts (used if no --prompts file is provided) ───────────────
_DEFAULT_PROMPTS: list[tuple[str, str]] = [
    (
        "Newton discovered gravity when an apple fell from a tree.",
        "Score relevance to: 'What causes objects to fall toward Earth?'",
    ),
    (
        "The Python programming language was created by Guido van Rossum.",
        "Score relevance to: 'What causes objects to fall toward Earth?'",
    ),
    (
        "Einstein's theory of general relativity describes gravity as the curvature of spacetime.",
        "Score relevance to: 'What causes objects to fall toward Earth?'",
    ),
    (
        "The Amazon rainforest covers most of Brazil.",
        "Score relevance to: 'What causes objects to fall toward Earth?'",
    ),
    (
        "Gravitational force between two bodies is F = G·m₁·m₂/r²",
        "Score relevance to: 'What causes objects to fall toward Earth?'",
    ),
    (
        "Source entity: Newton. Relationship type: DISCOVERED. Target entity: Gravity",
        "Is this a logically valid and meaningful relationship between these two entities?",
    ),
    (
        "Source entity: Apple. Relationship type: DISCOVERED. Target entity: Gravity",
        "Is this a logically valid and meaningful relationship between these two entities?",
    ),
    (
        "Source entity: Photosynthesis. Relationship type: CAUSES. Target entity: Gravity",
        "Is this a logically valid and meaningful relationship between these two entities?",
    ),
]


def run_ablation(prompts: list[tuple[str, str]]) -> dict:
    """
    Run both Laya and Jev on all prompts and return summary stats.
    """
    from graphrag.models.laya import get_laya
    from graphrag.models.jev import get_jev
    from graphrag.models.ablation import AblationModel
    from config.settings import settings

    log_path = Path(settings.ablation_log_path)
    ablation = AblationModel(
        laya_model=get_laya(),
        jev_model=get_jev(),
        log_path=log_path,
        primary=settings.decision_model_primary,
    )

    logger.info("Running %d prompts through Laya + Jev...", len(prompts))
    for i, (ctx, instr) in enumerate(prompts):
        result = ablation.score_detailed(ctx, instr)
        logger.info(
            "[%d/%d] score=%.3f (primary=%s, Δ captured in log)",
            i + 1, len(prompts), result.score, ablation._primary
        )

    return ablation.analyze_log()


def print_table(stats: dict) -> None:
    """Print a formatted terminal table of results."""
    print("\n" + "=" * 60)
    print("  LAYA vs. JEV — ABLATION SUMMARY")
    print("=" * 60)
    if "error" in stats:
        print(f"  Error: {stats['error']}")
        return

    rows = [
        ("Samples scored",        stats["n_samples"]),
        ("Mean score delta",      f"{stats['mean_delta']:.4f}"),
        ("Max score delta",       f"{stats['max_delta']:.4f}"),
        ("Agreement rate (Δ<0.05)", f"{stats['agreement_rate']:.1%}"),
        ("Mean Laya latency",     f"{stats['mean_laya_ms']:.1f} ms"),
        ("Mean Jev latency",      f"{stats['mean_jev_ms']:.1f} ms"),
        ("Speedup (Jev/Laya)",    f"{stats['mean_jev_ms']/max(stats['mean_laya_ms'], 0.001):.1f}x slower"),
        ("Log file",              str(stats["log_path"])),
    ]
    for label, val in rows:
        print(f"  {label:<30} {val}")
    print("=" * 60)


def plot_results(log_path: Path) -> None:
    """Generate a scatter plot of Laya score vs Jev score."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot. Run: pip install matplotlib")
        return

    records = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))

    laya_scores = [r["laya"]["score"] for r in records]
    jev_scores  = [r["jev"]["score"]  for r in records]
    laya_ms     = [r["laya"]["latency_ms"] for r in records]
    jev_ms      = [r["jev"]["latency_ms"]  for r in records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Laya vs. Jev: Ablation Analysis", fontsize=14, fontweight="bold")

    # Score correlation
    ax1.scatter(laya_scores, jev_scores, alpha=0.7, color="#6C63FF", edgecolors="white", s=80)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect agreement")
    ax1.set_xlabel("Laya Score")
    ax1.set_ylabel("Jev Score")
    ax1.set_title("Score Correlation")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Latency comparison
    ax2.bar(["Laya (local)", "Jev (API)"],
            [sum(laya_ms)/len(laya_ms), sum(jev_ms)/len(jev_ms)],
            color=["#00C9A7", "#FF6584"])
    ax2.set_ylabel("Mean Latency (ms)")
    ax2.set_title("Latency Comparison")
    ax2.grid(True, axis="y", alpha=0.3)

    out_path = log_path.parent / "laya_vs_jev.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Plot saved: %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Laya vs. Jev ablation benchmark")
    parser.add_argument(
        "--prompts",
        type=Path,
        default=None,
        help="Optional JSONL file with {'context': ..., 'instruction': ...} objects",
    )
    args = parser.parse_args()

    prompts = _DEFAULT_PROMPTS
    if args.prompts and args.prompts.exists():
        with open(args.prompts, encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
        prompts = [(d["context"], d["instruction"]) for d in data]
        logger.info("Loaded %d custom prompts from %s", len(prompts), args.prompts)

    stats = run_ablation(prompts)
    print_table(stats)

    from config.settings import settings
    plot_results(Path(settings.ablation_log_path))


if __name__ == "__main__":
    main()
