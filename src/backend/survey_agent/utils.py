"""Search utilities and helpers for the survey agent."""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Mapping

import arxiv
import requests
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig

from survey_agent.configuration import Configuration
from survey_agent.state import BaseSearchRequest, Paper


# ========= utilities

def get_today_str() -> str:
    """Get current date in a human-readable format."""
    # Avoid %-d for Windows compatibility.
    return datetime.now().strftime("%a %b %d, %Y").replace(" 0", " ")


# ========= tools

@tool("search_arxiv_papers", args_schema=BaseSearchRequest)
def search_arxiv_papers(
    query: str,
    max_results: int = 5,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    config: RunnableConfig | None = None,
) -> List[dict]:
    """
    Search for papers on arXiv.org using the arxiv SDK.
    """
    _ = Configuration.from_runnable_config(config)
    full_query = build_arxiv_query(query, date_from, date_to)

    client = arxiv.Client(page_size=max_results, delay_seconds=3.0, num_retries=3)

    search = arxiv.Search(
        query=full_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )

    results: List[dict] = []
    try:
        for result in client.results(search):
            authors = [author.name for author in result.authors]
            published_year = result.published.year if result.published else None
            venue = f"arXiv:{result.primary_category}" if result.primary_category else "arXiv"
            summary = result.summary.replace("\n", " ") if result.summary else None
            title = result.title.replace("\n", " ") if result.title else None
            source_id = result.get_short_id() if hasattr(result, "get_short_id") else None

            results.append(
                Paper(
                    source="arxiv",
                    source_id=source_id,
                    title=title,
                    authors=authors,
                    abstract=summary,
                    year=published_year,
                    venue=venue,
                    doi=result.doi,
                    url=result.entry_id,
                    pdf_url=result.pdf_url,
                ).model_dump()
            )
    except Exception:
        return []

    return results


@tool("think_tool")
def think_tool(reflection: str) -> str:
    """Strategic reflection tool to force deliberate planning."""
    return f"Reflection recorded: {reflection}"


@tool("rag_search")
def rag_search(query: str, config: RunnableConfig | None = None) -> str:
    """Retrieve RAG context from Qdrant/SQLite for a query."""
    from survey_agent.storage import (
        build_rag_context,
        format_rag_context,
        get_chunk_count,
        get_qdrant_collection_info,
    )

    cfg = Configuration.from_runnable_config(config)
    if not cfg.rag_enabled:
        return "RAG is disabled."

    try:
        blocks = build_rag_context(query, config=config, limit=cfg.rag_top_k)
    except Exception as exc:
        diagnostics = []
        chunk_count = get_chunk_count(cfg.db_path)
        diagnostics.append(f"SQLite chunks: {chunk_count}.")
        try:
            info = get_qdrant_collection_info(cfg)
            vectors = info.config.params.vectors.get("dense")
            vector_size = getattr(vectors, "size", None)
            diagnostics.append(
                f"Qdrant points: {info.points_count}, dense size: {vector_size}."
            )
            try:
                from langchain_openai import OpenAIEmbeddings

                dense_dim = len(
                    OpenAIEmbeddings(
                        model=cfg.rag_dense_model,
                        api_key=cfg.rag_dense_api_key or cfg.llm_api_key,
                        base_url=cfg.rag_dense_base_url or cfg.llm_base_url,
                    ).embed_query("dimension check")
                )
                diagnostics.append(f"Embedding dimension: {dense_dim}.")
            except Exception as inner_exc:
                diagnostics.append(f"Embedding check failed: {inner_exc}")
        except Exception as inner_exc:
            diagnostics.append(f"Qdrant check failed: {inner_exc}")
        return f"RAG search error: {exc}. " + " ".join(diagnostics)

    if not blocks:
        chunk_count = get_chunk_count(cfg.db_path)
        if chunk_count > 0:
            return (
                "No RAG context available for this query. "
                f"SQLite has {chunk_count} chunks, so Qdrant may be empty or unreachable. "
                "Check QDRANT_URL and the collection name, or delete/rebuild the collection."
            )
        return "No RAG context available for this query."
    return format_rag_context(blocks)


def build_arxiv_query(
    query: str,
    date_from: datetime | None,
    date_to: datetime | None,
) -> str:
    """Build arXiv query string with date range filters."""
    full_query = query
    if (date_from or date_to) and "submittedDate:" not in query:
        start = (date_from or datetime(1900, 1, 1)).strftime("%Y%m%d%H%M")
        end = (date_to or datetime.utcnow()).strftime("%Y%m%d%H%M")
        full_query = f"({query}) AND submittedDate:[{start} TO {end}]"
    return full_query


@tool("search_acm_papers", args_schema=BaseSearchRequest)
def search_acm_papers(
    query: str,
    max_results: int = 5,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    config: RunnableConfig | None = None,
) -> List[dict]:
    """
    Search ACM papers via Crossref and return normalized metadata.
    """
    cfg = Configuration.from_runnable_config(config)
    base_url = "https://api.crossref.org/works"
    filters = ["member:320"]

    if date_from:
        filters.append(f"from-pub-date:{date_from.date().isoformat()}")
    if date_to:
        filters.append(f"until-pub-date:{date_to.date().isoformat()}")

    base_params = {
        "filter": ",".join(filters),
        "rows": max_results,
        "sort": "relevance",
        "mailto": cfg.crossref_mailto,
    }

    items = []
    for params in (
        {**base_params, "query.bibliographic": query},
        {**base_params, "query": query},
    ):
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException:
            continue

        items = data.get("message", {}).get("items", [])
        if items:
            break

    if not items:
        return []

    results: List[dict] = []

    for item in items:
        title_list = item.get("title", [])
        title = title_list[0] if title_list else None
        doi = item.get("DOI")

        authors = []
        for author in item.get("author", []):
            given = author.get("given", "")
            family = author.get("family", "")
            full_name = f"{given} {family}".strip()
            if full_name:
                authors.append(full_name)

        published = None
        date_parts = item.get("published-print", {}).get("date-parts") or item.get(
            "published-online", {}
        ).get("date-parts")
        if date_parts and date_parts[0]:
            published = date_parts[0][0]

        venue_list = item.get("container-title", [])
        venue = venue_list[0] if venue_list else None

        landing_page = f"https://dl.acm.org/doi/{doi}" if doi else None
        pdf_link = f"https://dl.acm.org/doi/pdf/{doi}" if doi else None

        results.append(
            Paper(
                source="acm",
                source_id=doi,
                title=title,
                authors=authors,
                abstract=item.get("abstract"),
                year=published,
                venue=venue,
                doi=doi,
                url=landing_page,
                pdf_url=pdf_link,
            ).model_dump()
        )

    return results


# ========= PDF download pipeline (for RAG ingestion)

@dataclass(frozen=True)
class PdfDownloadResult:
    downloaded: int
    skipped: int
    failed: int
    errors: list[str]


def _safe_filename(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    return safe.strip("_") or "document"


def _filename_for_entry(entry: Mapping[str, Any]) -> str:
    doc_id = entry.get("doc_id")
    canonical_id = entry.get("canonical_id")
    if doc_id:
        return _safe_filename(doc_id) + ".pdf"
    if canonical_id:
        return _safe_filename(canonical_id) + ".pdf"
    pdf_url = entry.get("pdf_url") or ""
    return hashlib.sha256(pdf_url.encode("utf-8")).hexdigest() + ".pdf"


def filename_for_entry(entry: Mapping[str, Any]) -> str:
    """Public wrapper for consistent PDF filenames."""
    return _filename_for_entry(entry)


def _stream_download(url: str, target_path: Path, timeout: int = 30) -> str:
    sha256 = hashlib.sha256()
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(target_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                handle.write(chunk)
                sha256.update(chunk)
    return sha256.hexdigest()


def download_pdf_queue(
    pdf_queue: Iterable[Mapping[str, Any]],
    *,
    db_path: str,
    download_dir: str | Path = "data/pdfs",
    overwrite: bool = False,
    timeout: int = 30,
) -> PdfDownloadResult:
    from survey_agent.storage import update_document_file_info

    download_path = Path(download_dir)
    download_path.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = 0
    errors: list[str] = []

    for entry in pdf_queue:
        pdf_url = entry.get("pdf_url")
        if not pdf_url:
            skipped += 1
            continue

        filename = _filename_for_entry(entry)
        target_path = download_path / filename
        if target_path.exists() and not overwrite:
            skipped += 1
            continue

        doc_id = entry.get("doc_id")
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, dir=download_path) as handle:
                temp_file = Path(handle.name)
            content_sha256 = _stream_download(pdf_url, temp_file, timeout=timeout)
            temp_file.replace(target_path)

            if doc_id:
                update_document_file_info(
                    db_path,
                    doc_id=doc_id,
                    raw_uri=str(target_path),
                    content_sha256=content_sha256,
                    status="downloaded",
                )
            downloaded += 1
        except Exception as exc:
            failed += 1
            errors.append(f"{pdf_url}: {exc}")
            if doc_id:
                update_document_file_info(
                    db_path,
                    doc_id=doc_id,
                    status="download_failed",
                )
            if temp_file and temp_file.exists():
                temp_file.unlink(missing_ok=True)

    return PdfDownloadResult(
        downloaded=downloaded,
        skipped=skipped,
        failed=failed,
        errors=errors,
    )
