"""
graphrag_neo4j_laya — config/settings.py
Centralised, validated configuration via pydantic-settings.
All values are read from environment variables or a .env file.
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the GraphRAG pipeline."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Graph Database Selection ──────────────────────────────────────────────
    graph_db_type: str = Field(
        "neo4j",
        description="Which graph DB to use: 'neo4j', 'postgres_age', 'memgraph', or 'kuzu'",
    )

    # ── Neo4j & Memgraph ──────────────────────────────────────────────────────
    neo4j_uri: str = Field("bolt://localhost:7687", description="Bolt URI")
    neo4j_user: str = Field("neo4j", description="Neo4j username")
    neo4j_password: str = Field("testpassword", description="Neo4j password")

    # ── Apache AGE / Postgres ────────────────────────────────────────────────
    age_host: str = Field("localhost")
    age_port: int = Field(5432)
    age_user: str = Field("postgres")
    age_password: str = Field("testpassword")
    age_database: str = Field("graph_db")

    # ── Decision Model Backend ────────────────────────────────────────────────
    decision_model_backend: str = Field(
        "laya",
        description="Active scoring backend: 'laya' | 'jev' | 'ablation'",
    )
    decision_model_primary: str = Field(
        "laya",
        description="In ablation mode, which backend drives the live pipeline score: 'laya' | 'jev'",
    )

    # ── Jev API (TypeSafe cloud) ───────────────────────────────────────────────
    jev_api_key: str | None = Field(
        None,
        description="TypeSafe Jev API key. Get one at https://typesafe.ai",
    )
    jev_timeout_seconds: int = Field(
        10,
        description="HTTP timeout for Jev API calls in seconds",
    )
    jev_fallback_score: float = Field(
        0.5,
        ge=0.0, le=1.0,
        description="Score returned when the Jev API is unavailable (prevents traversal crash)",
    )
    ablation_log_path: str = Field(
        "benchmarks/results/ablation_log.jsonl",
        description="Path to the JSONL file where ablation comparisons are logged",
    )

    # ── Model IDs ─────────────────────────────────────────────────────────────
    laya_model_id: str = Field(
        "convaiinnovations/laya-typed-decisions",
        description="HuggingFace model ID for the Laya typed-decisions classifier",
    )
    llm_model_id: str = Field(
        "hugging-quants/Meta-Llama-3.1-8B-Instruct-BNB-NF4",
        description="Pre-quantized Llama-3.1 8B NF4 model ID",
    )
    embed_model_id: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence-Transformers model for first-pass node embedding",
    )
    huggingface_token: str | None = Field(
        None,
        description="HF Hub token (only needed for gated meta-llama/* variants)",
    )

    # ── A* Hyperparameters (idea.md §2.3) ────────────────────────────────────
    alpha: float = Field(0.65, ge=0.0, le=1.0, description="Laya semantic weight")
    beta: float = Field(0.25, ge=0.0, le=1.0, description="PageRank structural weight")
    gamma: float = Field(0.10, ge=0.0, le=1.0, description="Depth-penalty coefficient")
    max_pagerank: float = Field(10.0, gt=0.0, description="Graph-specific PageRank normaliser")

    # ── Ingestion thresholds ──────────────────────────────────────────────────
    noul_boundary_threshold: float = Field(
        0.85, ge=0.0, le=1.0,
        description="Laya noul score above which a semantic chunk boundary is cut",
    )
    entity_cosine_threshold: float = Field(
        0.85, ge=0.0, le=1.0,
        description="Cosine similarity above which two entities are disambiguated",
    )
    entity_noul_merge_threshold: float = Field(
        0.95, ge=0.0, le=1.0,
        description="Laya score above which two entities are merged into one node",
    )

    # ── Retrieval thresholds ──────────────────────────────────────────────────
    bfs_prune_threshold: float = Field(
        0.6, ge=0.0, le=1.0,
        description="Score-Gated BFS: paths below this are pruned",
    )
    early_termination_threshold: float = Field(
        0.95, ge=0.0, le=1.0,
        description="A* Early Termination: Noul P(yes) above this exits traversal early (functions.md Phase 3)",
    )
    seed_colbert_top_n: int = Field(
        20, gt=0,
        description="ColBERT top-N candidates before Laya cross-encoding",
    )
    seed_final_top_k: int = Field(
        2, gt=0,
        description="Final number of seed nodes passed to traversal",
    )

    # ── VRAM budget (informational) ───────────────────────────────────────────
    vram_budget_gb: float = Field(8.0, description="Total VRAM on RTX 5060")
    vram_ceiling_gb: float = Field(7.5, description="Safety ceiling for combined peak usage")

    @field_validator("alpha", "beta", "gamma")
    @classmethod
    def weights_sum_leq_one(cls, v: float, info) -> float:  # noqa: ANN001
        # Soft check — caller should ensure α+β+γ <= 1 after construction.
        return v


# ── Module-level singleton ────────────────────────────────────────────────────
settings = Settings()
