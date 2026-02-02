"""Configuration management for the survey agent."""

from __future__ import annotations

import os
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class Configuration(BaseModel):
    """Runtime configuration for survey agent workflows."""

    # LLM config
    llm_model: str = Field(default="Pro/deepseek-ai/DeepSeek-R1")
    llm_provider: str = Field(default="openai")
    llm_base_url: Optional[str] = Field(default=None)
    llm_api_key: Optional[str] = Field(default=None)
    llm_temperature: float = Field(default=0.0)
    llm_max_tokens: int = Field(default=8192)
    max_structured_output_retries: int = Field(default=3)
    allow_clarification: bool = Field(default=True)
    max_concurrent_research_units: int = Field(default=5)
    max_researcher_iterations: int = Field(default=6)
    max_react_tool_calls: int = Field(default=10)

    # Search config
    arxiv_max_results: int = Field(default=10)
    acm_max_results: int = Field(default=10)
    crossref_mailto: str = Field(default="research-agent@example.com")

    # Storage config
    db_path: str = Field(default="research.db")
    pdf_download_dir: str = Field(default="data/pdfs")

    # RAG / Qdrant config
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection: str = Field(default="survey_chunks")
    qdrant_timeout: int = Field(default=30)
    qdrant_prefer_grpc: bool = Field(default=False)

    rag_enabled: bool = Field(default=True)
    rag_chunk_size: int = Field(default=1200)
    rag_chunk_overlap: int = Field(default=200)
    rag_batch_size: int = Field(default=64)

    rag_dense_model: str = Field(default="Qwen/Qwen3-Embedding-8B")
    rag_dense_base_url: Optional[str] = Field(default=None)
    rag_dense_api_key: Optional[str] = Field(default=None)

    rag_sparse_model: str = Field(default="Qdrant/bm25")

    rag_top_k: int = Field(default=20)
    rag_rerank_top_k: int = Field(default=10)
    rag_enable_rerank: bool = Field(default=True)

    # Unstructured parsing
    unstructured_strategy: str = Field(default="auto")
    unstructured_infer_tables: bool = Field(default=True)

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        configurable = config.get("configurable", {}) if config else {}

        def _get(name: str, env: str | None = None) -> Any:
            env_name = env or name.upper()
            return configurable.get(name, os.getenv(env_name))

        values: dict[str, Any] = {
            "llm_model": _get("llm_model", "LLM_MODEL") or cls.model_fields["llm_model"].default,
            "llm_provider": _get("llm_provider", "LLM_PROVIDER") or cls.model_fields["llm_provider"].default,
            "llm_base_url": _get("llm_base_url", "LLM_BASE_URL"),
            "llm_api_key": _get("llm_api_key", "LLM_API_KEY"),
            "llm_temperature": _get("llm_temperature", "LLM_TEMPERATURE") or cls.model_fields["llm_temperature"].default,
            "llm_max_tokens": _get("llm_max_tokens", "LLM_MAX_TOKENS") or cls.model_fields["llm_max_tokens"].default,
            "max_structured_output_retries": _get("max_structured_output_retries", "MAX_STRUCTURED_OUTPUT_RETRIES")
            or cls.model_fields["max_structured_output_retries"].default,
            "allow_clarification": _get("allow_clarification", "ALLOW_CLARIFICATION")
            if _get("allow_clarification", "ALLOW_CLARIFICATION") is not None
            else cls.model_fields["allow_clarification"].default,
            "max_concurrent_research_units": _get(
                "max_concurrent_research_units", "MAX_CONCURRENT_RESEARCH_UNITS"
            )
            or cls.model_fields["max_concurrent_research_units"].default,
            "max_researcher_iterations": _get(
                "max_researcher_iterations", "MAX_RESEARCHER_ITERATIONS"
            )
            or cls.model_fields["max_researcher_iterations"].default,
            "max_react_tool_calls": _get("max_react_tool_calls", "MAX_REACT_TOOL_CALLS")
            or cls.model_fields["max_react_tool_calls"].default,
            "arxiv_max_results": _get("arxiv_max_results", "ARXIV_MAX_RESULTS")
            or cls.model_fields["arxiv_max_results"].default,
            "acm_max_results": _get("acm_max_results", "ACM_MAX_RESULTS")
            or cls.model_fields["acm_max_results"].default,
            "crossref_mailto": _get("crossref_mailto", "CROSSREF_MAILTO")
            or cls.model_fields["crossref_mailto"].default,
            "db_path": _get("db_path", "SURVEY_DB_PATH") or cls.model_fields["db_path"].default,
            "pdf_download_dir": _get("pdf_download_dir", "SURVEY_PDF_DIR")
            or cls.model_fields["pdf_download_dir"].default,
            "qdrant_url": _get("qdrant_url", "QDRANT_URL") or cls.model_fields["qdrant_url"].default,
            "qdrant_collection": _get("qdrant_collection", "QDRANT_COLLECTION")
            or cls.model_fields["qdrant_collection"].default,
            "qdrant_timeout": _get("qdrant_timeout", "QDRANT_TIMEOUT")
            or cls.model_fields["qdrant_timeout"].default,
            "qdrant_prefer_grpc": _get("qdrant_prefer_grpc", "QDRANT_PREFER_GRPC")
            if _get("qdrant_prefer_grpc", "QDRANT_PREFER_GRPC") is not None
            else cls.model_fields["qdrant_prefer_grpc"].default,
            "rag_enabled": _get("rag_enabled", "RAG_ENABLED")
            if _get("rag_enabled", "RAG_ENABLED") is not None
            else cls.model_fields["rag_enabled"].default,
            "rag_chunk_size": _get("rag_chunk_size", "RAG_CHUNK_SIZE")
            or cls.model_fields["rag_chunk_size"].default,
            "rag_chunk_overlap": _get("rag_chunk_overlap", "RAG_CHUNK_OVERLAP")
            or cls.model_fields["rag_chunk_overlap"].default,
            "rag_batch_size": _get("rag_batch_size", "RAG_BATCH_SIZE")
            or cls.model_fields["rag_batch_size"].default,
            "rag_dense_model": _get("rag_dense_model", "RAG_DENSE_MODEL")
            or cls.model_fields["rag_dense_model"].default,
            "rag_dense_base_url": _get("rag_dense_base_url", "RAG_DENSE_BASE_URL"),
            "rag_dense_api_key": _get("rag_dense_api_key", "RAG_DENSE_API_KEY"),
            "rag_sparse_model": _get("rag_sparse_model", "RAG_SPARSE_MODEL")
            or cls.model_fields["rag_sparse_model"].default,
            "rag_top_k": _get("rag_top_k", "RAG_TOP_K") or cls.model_fields["rag_top_k"].default,
            "rag_rerank_top_k": _get("rag_rerank_top_k", "RAG_RERANK_TOP_K")
            or cls.model_fields["rag_rerank_top_k"].default,
            "rag_enable_rerank": _get("rag_enable_rerank", "RAG_ENABLE_RERANK")
            if _get("rag_enable_rerank", "RAG_ENABLE_RERANK") is not None
            else cls.model_fields["rag_enable_rerank"].default,
            "unstructured_strategy": _get("unstructured_strategy", "UNSTRUCTURED_STRATEGY")
            or cls.model_fields["unstructured_strategy"].default,
            "unstructured_infer_tables": _get("unstructured_infer_tables", "UNSTRUCTURED_INFER_TABLES")
            if _get("unstructured_infer_tables", "UNSTRUCTURED_INFER_TABLES") is not None
            else cls.model_fields["unstructured_infer_tables"].default,
        }

        return cls(**values)


__all__ = ["Configuration"]
