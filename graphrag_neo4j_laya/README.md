# laya-jev-GraphRAG

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)
![Databases](https://img.shields.io/badge/DB-Neo4j%20%7C%20Memgraph%20%7C%20AGE%20%7C%20Kùzu-018bff.svg)
![Backend](https://img.shields.io/badge/AI%20Backend-Laya%20%7C%20Jev%20%7C%20Ablation-blueviolet.svg)

A production-grade **Agentic GraphRAG** engine that replaces slow, generative LLM routing with **System One typed-decision models**.
Every single graph operation — from ingesting a document to returning a cited answer — is governed by exactly one of three mathematical primitives:

| Primitive | Meaning | Example Usage |
|-----------|---------|---------------|
| **`Score`** | Ordinal relevance `[0, 1]` | A\* edge heuristic, context reranking |
| **`Noul`**  | Binary `P(yes)` `[0, 1]`   | Semantic chunking, hallucination gate, citation verification |
| **`Choice`**| Categorical selection       | Intent routing, ontology alignment, conflict resolution |

Switch the entire decision layer — every primitive in every phase — with **one environment variable**.

---

## 🛑 Why Traditional GraphRAG Fails

Traditional GraphRAG calls a generative LLM at **every hop** to filter and rank graph edges.
Evaluating 5 edges at depth 4 = 20 serial LLM calls = 30–90 seconds of latency and frequent context-window overflow.

## ⚡ The Solution: System One A\* Traversal

This engine decouples traversal from generation. A tiny, fast, typed-decision model evaluates each edge in **~33 ms** (Laya on CUDA) or **~50 ms** (Jev cloud API) using a composite heuristic that blends:

1. **Semantic Relevance**: The edge score evaluated by the AI backend.
2. **Structural Centrality**: The pre-calculated PageRank of the target node.
3. **Depth Penalty**: A regularizer to prevent shallow, local looping.

The LLM (Llama-3.1-8B 4-bit) only runs **once**, at the end, to synthesise the already-filtered subgraph into a final answer.

---

## 🔌 Pluggable Architecture

### AI Backend — Swap with One Line

```bash
# .env
DECISION_MODEL_BACKEND=laya      # Local CUDA, free, ~33ms/call, ~1.2 GB VRAM
DECISION_MODEL_BACKEND=jev       # TypeSafe cloud API, zero-shot ready, ~50ms/call
DECISION_MODEL_BACKEND=ablation  # Run BOTH, log side-by-side for RLCD fine-tuning
```

| Feature | Laya (Local) | Jev (Cloud) |
|---------|-------------|-------------|
| Model | `convaiinnovations/laya-typed-decisions` 421M | `jev-1.13` (TypeSafe) |
| Latency | ~33 ms (RTX 5060 FP16) | ~50 ms (API round-trip) |
| Cost | Free | ~$0.042 / M tokens |
| Privacy | 100% local | API |
| Batch | GPU-batched | True parallel (1 API call) |

**Ablation mode** logs `latency_ms`, `score`, and `delta` for every call — giving you automatic RLCD training data to fine-tune Laya towards Jev-level accuracy.

### Graph Database — Swap with One Line

```bash
# .env
GRAPH_DB_BACKEND=neo4j       # Production: index-free adjacency, native GDS
GRAPH_DB_BACKEND=memgraph    # In-memory Bolt: identical Cypher, low-latency analytics
GRAPH_DB_BACKEND=age         # PostgreSQL + Apache AGE: unified SQL/graph stack
GRAPH_DB_BACKEND=kuzu        # Embedded local: no Docker, zero setup for development
```

All four backends implement the same `BaseGraphClient` interface. Zero code changes needed.

---

## 🗺️ The 19-Function Pipeline

```
┌─────────────────────────────────────────────────────────┐
│  PHASE 1 — Ingestion (Offline, runs once per document)  │
│                                                         │
│  1. Semantic Chunking      → Noul   (boundary detect)   │
│  2. Entity Extraction      → LLM    (Llama-3.1-8B)      │
│  3. Entity Disambiguation  → Noul   (merge duplicates)   │
│  4. Edge Verification      → Score  (prune hallucinated) │
│  5. Ontology Alignment     → Choice (snap to schema)     │
│  6. Community Detection    → Leiden / NetworkX           │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│  PHASE 2 — Pre-Traversal (Per query)                    │
│                                                         │
│  7. Intent Routing         → Choice (local/multi/global) │
│  8. Dense Seed Retrieval   → Embedding cosine           │
│  9. Seed Validation        → Score  (filter bad seeds)   │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│  PHASE 3 — A* Traversal (Per query)                     │
│                                                         │
│  10. Neighborhood Fetch    → Bolt / Cypher              │
│  11. Edge Scoring          → Score  (S_AI heuristic)    │
│  12. Structural Anchoring  → PageRank (C_Graph)         │
│  13. Path Pruning          → Beam cutoff                │
│  14. Early Termination     → Noul   (context sufficient?)│
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│  PHASE 4 — Post-Traversal (Per query)                   │
│                                                         │
│  15. Context Reranking     → Score  (drop bottom 20%)   │
│  16. Conflict Resolution   → Choice (pick credible src) │
│  17. Hallucination Gate    → Noul   (abstain if P<0.5)  │
│  18. Answer Synthesis      → LLM    (Llama-3.1-8B 4-bit)│
│  19. Citation Verification → Noul   (flag if P<0.9)     │
└─────────────────────────────────────────────────────────┘
```

> **📚 Want to see the exact prompts and logic for all 19 functions?**  
> Check out the [Architecture Deep Dive (ARCHITECTURE.md)](ARCHITECTURE.md) for a complete breakdown of every primitive, threshold, and routing decision.

---

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/yourusername/laya-jev-GraphRAG.git
cd laya-jev-GraphRAG/graphrag_neo4j_laya

# Install PyTorch with CUDA 12.4 first (for your GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install all remaining dependencies
pip install -r requirements.txt
```

### Configure your backend

```bash
cp .env.example .env
# Then edit .env to set:
#   DECISION_MODEL_BACKEND=laya
#   GRAPH_DB_BACKEND=kuzu   ← Start here: no Docker required
```

### Run a query

```python
from graphrag.pipeline import GraphRAGPipeline

pipeline = GraphRAGPipeline()
answer = pipeline.query("What caused the 2008 financial crisis?")
print(answer)
```

Or from the CLI:

```bash
python -m graphrag.pipeline --query "What caused the 2008 financial crisis?" --max-depth 4
```

### Run the ablation harness

```python
from graphrag.models.ablation import AblationHarness

harness = AblationHarness()
result = harness.compare(
    context="Isaac Newton published Principia Mathematica in 1687.",
    instruction="Score the historical significance of this event.",
)
print(f"Laya: {result['laya']['score']:.3f} @ {result['laya']['latency_ms']:.0f}ms")
print(f"Jev:  {result['jev']['score']:.3f}  @ {result['jev']['latency_ms']:.0f}ms")
print(f"Δ:    {result['delta']:.4f}")
```

---

## 📁 Project Structure

```
graphrag_neo4j_laya/
├── graphrag/
│   ├── graph/                    # DB abstraction layer
│   │   ├── base.py               # BaseGraphClient ABC
│   │   ├── factory.py            # GRAPH_DB_BACKEND selector
│   │   ├── neo4j_client.py       # Neo4j (Bolt + GDS)
│   │   ├── memgraph_client.py    # Memgraph (Bolt)
│   │   ├── age_client.py         # Apache AGE (PostgreSQL)
│   │   └── kuzu_client.py        # Kùzu (embedded)
│   ├── models/                   # AI decision layer
│   │   ├── base_decision.py      # BaseDecisionModel ABC (Score/Noul/Choice)
│   │   ├── laya.py               # Local CUDA model (421M params)
│   │   ├── jev.py                # TypeSafe Jev API client
│   │   ├── ablation.py           # Side-by-side comparison harness
│   │   └── decision_factory.py   # DECISION_MODEL_BACKEND selector
│   ├── ingestion/                # Phase 1: offline pipeline
│   │   ├── chunker.py            # Noul boundary chunking
│   │   ├── entity_extractor.py   # LLM NER + Noul disambiguation
│   │   ├── edge_verifier.py      # Score-based edge pruning
│   │   ├── ontology_aligner.py   # Choice-based schema alignment
│   │   └── community.py          # Leiden community detection
│   ├── retrieval/                # Phases 2-4: query pipeline
│   │   ├── router.py             # Choice: intent routing
│   │   ├── seed_selector.py      # Score: seed validation
│   │   ├── post_traversal.py     # Score/Choice/Noul: post-processing
│   │   └── traversal/
│   │       ├── astar.py          # A* semantic traversal + early exit
│   │       └── bfs.py            # Score-gated BFS (local intent)
│   ├── benchmarks/               # Performance analysis
│   │   ├── hop_latency.py        # Per-hop DB latency across all backends
│   │   ├── laya_vs_jev.py        # Side-by-side model comparison
│   │   ├── greedy_vs_astar.py    # Search strategy comparison
│   │   └── vram_monitor.py       # GPU memory tracking
│   └── pipeline.py               # Full end-to-end orchestrator
├── config/
│   ├── settings.py               # Pydantic settings (all thresholds)
│   └── .env.example              # Template with all toggles
└── requirements.txt
```

---

## 🔬 The Ablation Mode: RLCD Data Factory

Setting `DECISION_MODEL_BACKEND=ablation` silently runs both Laya and Jev in parallel for every primitive call and logs the results to a JSONL file:

```json
{"primitive": "score", "context": "...", "instruction": "...",
 "laya": {"score": 0.83, "latency_ms": 34.1},
 "jev":  {"score": 0.79, "latency_ms": 48.3},
 "delta": 0.04}
```

This JSONL log is ready-to-use RLCD training data to fine-tune Laya towards Jev-level zero-shot accuracy — the path to a fully local, frontier-quality GraphRAG engine.

---

## 📊 Performance

| Backend | Edge Score Latency | Traversal (4-hop) | VRAM |
|---------|-------------------|-------------------|------|
| Laya (RTX 5060 FP16) | ~33 ms | ~150 ms | ~1.2 GB |
| Jev (cloud API) | ~50 ms | ~220 ms | 0 |
| LLM-per-hop (baseline) | ~2,000 ms | ~15,000 ms | ~6 GB |

---

## 🤝 Contributing

Contributions welcome for:
- Additional graph DB connectors (Nebula, TigerGraph, FalkorDB etc)
- Laya fine-tuning
- Jev


---

## 📄 License

Apache 2.0 — free for personal use only.

Built on:
- [Laya](https://huggingface.co/convaiinnovations/laya-typed-decisions) (Apache 2.0, ModernBERT-large)
- [TypeSafe Jev API](https://typesafe.ai)
- [Neo4j](https://neo4j.com) · [Memgraph](https://memgraph.com) · [Apache AGE](https://age.apache.org) · [Kùzu](https://kuzudb.com)
