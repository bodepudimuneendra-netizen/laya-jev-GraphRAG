"""
graphrag/benchmarks/hop_latency.py

Benchmark: Neo4j Bolt vs Apache AGE hop-latency test.

From idea.md §5:
    "Hop Latency Degradation Test: Measure edge retrieval time
     (_get_neo4j_edges) vs _get_age_edges in Postgres.
     Prove that at Hop 4, Neo4j stays at ~5ms while Apache AGE spikes
     due to recursive JSONB joins."

Output: terminal table + CSV + matplotlib bar chart.
"""

from __future__ import annotations

import csv
import logging
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt

from graphrag.graph.neo4j_client import Neo4jClient
from graphrag.graph.age_client import AGEClient

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Number of timing repetitions per hop depth per database
REPS = 10


def _time_hop(client, node_name: str, hop: int) -> list[float]:
    """
    Time *hop* sequential neighbour lookups starting from *node_name*.
    Returns a list of per-hop wall-clock times in milliseconds.
    """
    times = []
    for _ in range(REPS):
        current = node_name
        lap_start = time.perf_counter()
        for _ in range(hop):
            neighbors = client.get_neighbors(current)
            if neighbors:
                current = neighbors[0]["target_name"]
            else:
                break
        lap_end = time.perf_counter()
        times.append((lap_end - lap_start) * 1000)  # ms
    return times


def run(seed_node: str = "TestNode", max_hops: int = 4) -> dict:
    """
    Execute the latency benchmark and return result dict.

    Parameters
    ----------
    seed_node:
        Starting node name (must exist in both databases).
    max_hops:
        Maximum hop depth to test.

    Returns
    -------
    dict
        {hop: {"neo4j_mean_ms": float, "age_mean_ms": float}}
    """
    neo4j = Neo4jClient()
    age   = AGEClient()
    results: dict[int, dict] = {}

    print(f"\n{'Hop':>4}  {'Neo4j (ms)':>12}  {'AGE (ms)':>10}  {'Speedup':>8}")
    print("-" * 44)

    for hop in range(1, max_hops + 1):
        neo4j_times = _time_hop(neo4j, seed_node, hop)
        age_times   = _time_hop(age,   seed_node, hop)

        neo4j_mean = statistics.mean(neo4j_times)
        age_mean   = statistics.mean(age_times)
        speedup    = age_mean / neo4j_mean if neo4j_mean > 0 else float("inf")

        results[hop] = {
            "neo4j_mean_ms": round(neo4j_mean, 3),
            "age_mean_ms":   round(age_mean,   3),
            "speedup":        round(speedup,    2),
        }
        print(f"{hop:>4}  {neo4j_mean:>12.3f}  {age_mean:>10.3f}  {speedup:>8.2f}x")

    neo4j.close()
    age.close()

    _save_csv(results)
    _plot(results)
    return results


def _save_csv(results: dict) -> None:
    path = OUTPUT_DIR / "hop_latency.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hop", "neo4j_mean_ms", "age_mean_ms", "speedup"])
        for hop, data in results.items():
            writer.writerow([hop, data["neo4j_mean_ms"], data["age_mean_ms"], data["speedup"]])
    logger.info("CSV saved: %s", path)


def _plot(results: dict) -> None:
    hops      = list(results.keys())
    neo4j_ms  = [results[h]["neo4j_mean_ms"] for h in hops]
    age_ms    = [results[h]["age_mean_ms"]   for h in hops]

    x = range(len(hops))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], neo4j_ms, width, label="Neo4j (Bolt)", color="#2563EB")
    ax.bar([i + width / 2 for i in x], age_ms,   width, label="Apache AGE",   color="#DC2626")

    ax.set_xlabel("Hop Depth")
    ax.set_ylabel("Mean Latency (ms)")
    ax.set_title("Hop Latency: Neo4j Bolt vs Apache AGE")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Hop {h}" for h in hops])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    path = OUTPUT_DIR / "hop_latency.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    logger.info("Chart saved: %s", path)
    plt.close(fig)


def main() -> None:  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    run()


if __name__ == "__main__":
    main()
