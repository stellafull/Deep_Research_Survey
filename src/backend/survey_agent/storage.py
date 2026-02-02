"""SQLite persistence for search runs and paper metadata."""

from __future__ import annotations

import hashlib
import logging
import math
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Any

from langchain_core.runnables import RunnableConfig
from survey_agent.configuration import Configuration
from survey_agent.utils import download_pdf_queue, filename_for_entry

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
  doc_id TEXT PRIMARY KEY,
  canonical_id TEXT UNIQUE NOT NULL,
  source_type TEXT,
  title TEXT,
  authors_json TEXT,
  abstract TEXT,
  year INTEGER,
  venue TEXT,
  doi TEXT,
  url TEXT,
  pdf_url TEXT,
  content_sha256 TEXT,
  raw_uri TEXT,
  parsed_uri TEXT,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS document_sources (
  source TEXT NOT NULL,
  source_id TEXT NOT NULL,
  doc_id TEXT NOT NULL,
  source_url TEXT,
  raw_json TEXT,
  fetched_at TEXT NOT NULL,
  PRIMARY KEY (source, source_id),
  FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);

CREATE TABLE IF NOT EXISTS search_runs (
  run_id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  user_query TEXT,
  generated_query TEXT,
  filters_json TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS search_results (
  run_id TEXT NOT NULL,
  doc_id TEXT NOT NULL,
  source TEXT NOT NULL,
  rank INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (run_id, doc_id, source),
  FOREIGN KEY (run_id) REFERENCES search_runs(run_id),
  FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);

CREATE TABLE IF NOT EXISTS document_chunks (
  chunk_id TEXT PRIMARY KEY,
  doc_id TEXT,
  canonical_id TEXT,
  source TEXT,
  title TEXT,
  url TEXT,
  text TEXT NOT NULL,
  metadata_json TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    value = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if value.startswith(prefix):
            value = value[len(prefix):]
    return value or None


def _canonical_id(paper: Mapping[str, Any]) -> str:
    doi = _normalize_doi(paper.get("doi"))
    if doi:
        return f"doi:{doi}"

    source = paper.get("source")
    source_id = paper.get("source_id")
    if source == "arxiv" and source_id:
        return f"arxiv:{source_id}"

    url = paper.get("url")
    if url:
        return f"url:{_sha256(url)}"

    title = (paper.get("title") or "").strip().lower()
    year = paper.get("year") or ""
    if title:
        return f"title:{_sha256(f'{title}|{year}')}"

    fallback = json.dumps(
        {
            "title": paper.get("title"),
            "year": paper.get("year"),
            "authors": paper.get("authors", []),
            "venue": paper.get("venue"),
            "abstract": paper.get("abstract"),
        },
        sort_keys=True,
    )
    return f"meta:{_sha256(fallback)}"


def canonical_id_for_paper(paper: Mapping[str, Any]) -> str:
    """Public helper so other modules can deduplicate consistently."""
    return _canonical_id(paper)


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str) -> None:
    conn = _connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def update_document_file_info(
    db_path: str,
    *,
    doc_id: str,
    raw_uri: str | None = None,
    content_sha256: str | None = None,
    status: str | None = None,
) -> None:
    if not doc_id:
        return
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            UPDATE documents
            SET raw_uri=COALESCE(?, raw_uri),
                content_sha256=COALESCE(?, content_sha256),
                status=COALESCE(?, status),
                updated_at=?
            WHERE doc_id=?
            """,
            (
                raw_uri,
                content_sha256,
                status,
                _utc_now(),
                doc_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _upsert_document(
    conn: sqlite3.Connection,
    paper: Mapping[str, Any],
    canonical_id: str | None = None,
) -> tuple[str, str]:
    canonical_id = canonical_id or _canonical_id(paper)
    now = _utc_now()

    existing = conn.execute(
        "SELECT doc_id FROM documents WHERE canonical_id=?",
        (canonical_id,),
    ).fetchone()
    if existing:
        doc_id = existing[0]
        conn.execute(
            """
            UPDATE documents
            SET title=COALESCE(?, title),
                authors_json=COALESCE(?, authors_json),
                abstract=COALESCE(?, abstract),
                year=COALESCE(?, year),
                venue=COALESCE(?, venue),
                doi=COALESCE(?, doi),
                url=COALESCE(?, url),
                pdf_url=COALESCE(?, pdf_url),
                source_type=COALESCE(?, source_type),
                status=COALESCE(?, status),
                updated_at=?
            WHERE doc_id=?
            """,
            (
                paper.get("title"),
                json.dumps(paper.get("authors", [])),
                paper.get("abstract"),
                paper.get("year"),
                paper.get("venue"),
                paper.get("doi"),
                paper.get("url"),
                paper.get("pdf_url"),
                paper.get("source"),
                paper.get("status") or "fetched",
                now,
                doc_id,
            ),
        )
        return doc_id, canonical_id

    doc_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO documents (
            doc_id,
            canonical_id,
            source_type,
            title,
            authors_json,
            abstract,
            year,
            venue,
            doi,
            url,
            pdf_url,
            content_sha256,
            raw_uri,
            parsed_uri,
            status,
            created_at,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            doc_id,
            canonical_id,
            paper.get("source"),
            paper.get("title"),
            json.dumps(paper.get("authors", [])),
            paper.get("abstract"),
            paper.get("year"),
            paper.get("venue"),
            paper.get("doi"),
            paper.get("url"),
            paper.get("pdf_url"),
            None,
            None,
            None,
            paper.get("status") or "fetched",
            now,
            now,
        ),
    )
    return doc_id, canonical_id


def _upsert_source_record(
    conn: sqlite3.Connection,
    doc_id: str,
    paper: Mapping[str, Any],
) -> None:
    source = paper.get("source")
    source_id = paper.get("source_id")
    if not source or not source_id:
        return

    conn.execute(
        """
        INSERT INTO document_sources (
            source,
            source_id,
            doc_id,
            source_url,
            raw_json,
            fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(source, source_id) DO UPDATE SET
            doc_id=excluded.doc_id,
            source_url=excluded.source_url,
            raw_json=excluded.raw_json,
            fetched_at=excluded.fetched_at
        """,
        (
            source,
            source_id,
            doc_id,
            paper.get("url"),
            json.dumps(paper, sort_keys=True),
            _utc_now(),
        ),
    )


def _create_search_run(
    conn: sqlite3.Connection,
    source: str,
    user_query: str | None,
    generated_query: str | None,
    filters: Mapping[str, Any] | None,
) -> str:
    run_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO search_runs (
            run_id,
            source,
            user_query,
            generated_query,
            filters_json,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            source,
            user_query,
            generated_query,
            json.dumps(filters or {}, sort_keys=True),
            _utc_now(),
        ),
    )
    return run_id


def _insert_search_result(
    conn: sqlite3.Connection,
    run_id: str,
    doc_id: str,
    source: str,
    rank: int,
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO search_results (
            run_id,
            doc_id,
            source,
            rank,
            created_at
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (run_id, doc_id, source, rank, _utc_now()),
    )


@dataclass(frozen=True)
class PersistResult:
    run_id: str
    doc_ids_by_canonical: dict[str, str]


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    text: str
    doc_id: str | None = None
    canonical_id: str | None = None
    source: str | None = None
    title: str | None = None
    url: str | None = None
    metadata: Mapping[str, Any] | None = None


def persist_search_results(
    db_path: str,
    *,
    source: str,
    user_query: str | None,
    generated_query: str | None,
    filters: Mapping[str, Any] | None,
    papers: Iterable[Mapping[str, Any]] | None,
) -> PersistResult | None:
    if not papers:
        return None

    init_db(db_path)
    conn = _connect(db_path)
    try:
        run_id = _create_search_run(conn, source, user_query, generated_query, filters)
        doc_ids_by_canonical: dict[str, str] = {}
        for rank, paper in enumerate(papers, start=1):
            canonical_id = _canonical_id(paper)
            doc_id, _ = _upsert_document(conn, paper, canonical_id=canonical_id)
            _upsert_source_record(conn, doc_id, paper)
            _insert_search_result(conn, run_id, doc_id, source, rank)
            doc_ids_by_canonical[canonical_id] = doc_id
        conn.commit()
        return PersistResult(run_id=run_id, doc_ids_by_canonical=doc_ids_by_canonical)
    finally:
        conn.close()


def insert_chunks(db_path: str, chunks: Iterable[Mapping[str, Any]]) -> None:
    if not chunks:
        return
    init_db(db_path)
    conn = _connect(db_path)
    try:
        now = _utc_now()
        rows = []
        for chunk in chunks:
            rows.append(
                (
                    chunk.get("chunk_id"),
                    chunk.get("doc_id"),
                    chunk.get("canonical_id"),
                    chunk.get("source"),
                    chunk.get("title"),
                    chunk.get("url"),
                    chunk.get("text"),
                    json.dumps(chunk.get("metadata") or {}, sort_keys=True),
                    now,
                )
            )
        conn.executemany(
            """
            INSERT OR REPLACE INTO document_chunks (
                chunk_id,
                doc_id,
                canonical_id,
                source,
                title,
                url,
                text,
                metadata_json,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def get_chunks_by_ids(db_path: str, chunk_ids: list[str]) -> list[dict]:
    if not chunk_ids:
        return []
    conn = _connect(db_path)
    try:
        placeholders = ",".join("?" for _ in chunk_ids)
        query = (
            "SELECT chunk_id, doc_id, canonical_id, source, title, url, text, metadata_json "
            f"FROM document_chunks WHERE chunk_id IN ({placeholders})"
        )
        rows = conn.execute(query, chunk_ids).fetchall()
    finally:
        conn.close()

    results: list[dict] = []
    for row in rows:
        metadata = json.loads(row[7]) if row[7] else {}
        results.append(
            {
                "chunk_id": row[0],
                "doc_id": row[1],
                "canonical_id": row[2],
                "source": row[3],
                "title": row[4],
                "url": row[5],
                "text": row[6],
                "metadata": metadata,
            }
        )
    return results


def get_chunk_count(db_path: str) -> int:
    conn = _connect(db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM document_chunks").fetchone()[0]
    finally:
        conn.close()


def get_qdrant_collection_info(cfg: Configuration):
    client = _get_qdrant_client(cfg)
    return client.get_collection(cfg.qdrant_collection)


# ========= RAG ingestion + retrieval (Qdrant + SQLite)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RagHit:
    chunk_id: str
    score: float
    doc_id: str | None = None
    canonical_id: str | None = None
    source: str | None = None
    title: str | None = None
    url: str | None = None


@dataclass(frozen=True)
class RagContextBlock:
    index: int
    source_index: int
    source_key: str
    chunk_id: str
    title: str | None
    url: str | None
    text: str


def _get_dense_embedder(cfg: Configuration):
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=cfg.rag_dense_model,
        api_key=cfg.rag_dense_api_key or cfg.llm_api_key,
        base_url=cfg.rag_dense_base_url or cfg.llm_base_url,
    )


def _get_sparse_embedder(cfg: Configuration):
    try:
        from fastembed import SparseTextEmbedding
    except ImportError as exc:
        raise ImportError("fastembed is required for sparse BM25 embeddings") from exc
    return SparseTextEmbedding(cfg.rag_sparse_model)


def _get_qdrant_client(cfg: Configuration):
    from qdrant_client import QdrantClient

    return QdrantClient(
        url=cfg.qdrant_url,
        timeout=cfg.qdrant_timeout,
        prefer_grpc=cfg.qdrant_prefer_grpc,
    )


def _qdrant_query(
    client,
    *,
    collection_name: str,
    vector,
    using: str,
    limit: int,
    with_payload: bool,
    with_vectors: bool,
):
    if hasattr(client, "search"):
        return client.search(
            collection_name=collection_name,
            query_vector=(using, vector),
            limit=limit,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
    response = client.query_points(
        collection_name=collection_name,
        query=vector,
        using=using,
        limit=limit,
        with_payload=with_payload,
        with_vectors=with_vectors,
    )
    return getattr(response, "points", response)


def _ensure_collection(client, cfg: Configuration, dense_dim: int) -> None:
    try:
        existing = client.get_collection(cfg.qdrant_collection)
        if existing:
            return
    except Exception:
        pass

    from qdrant_client.models import Distance, VectorParams

    sparse_vectors_config = None
    try:
        from qdrant_client.models import SparseVectorParams, SparseIndexParams

        sparse_vectors_config = {
            "bm25": SparseVectorParams(index=SparseIndexParams(on_disk=False))
        }
    except Exception:
        sparse_vectors_config = None

    client.create_collection(
        collection_name=cfg.qdrant_collection,
        vectors_config={"dense": VectorParams(size=dense_dim, distance=Distance.COSINE)},
        sparse_vectors_config=sparse_vectors_config,
    )


def _parse_pdf_text(path: Path, cfg: Configuration) -> str:
    try:
        from unstructured.partition.pdf import partition_pdf

        elements = partition_pdf(
            filename=str(path),
            strategy=cfg.unstructured_strategy,
            infer_table_structure=cfg.unstructured_infer_tables,
        )
        texts = []
        for el in elements:
            text = getattr(el, "text", None) or str(el)
            if text:
                texts.append(text)
        return "\n\n".join(texts)
    except Exception as exc:
        logger.warning("Unstructured parsing failed for %s: %s", path, exc)

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    except Exception as exc:
        logger.error("PDF fallback parsing failed for %s: %s", path, exc)
        return ""


def _chunk_text(text: str, cfg: Configuration) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.rag_chunk_size,
        chunk_overlap=cfg.rag_chunk_overlap,
    )
    return splitter.split_text(text)


def _to_sparse_vector(sparse_embedding):
    try:
        indices = list(sparse_embedding.indices)
        values = list(sparse_embedding.values)
    except AttributeError:
        indices = list(sparse_embedding.get("indices", []))
        values = list(sparse_embedding.get("values", []))

    from qdrant_client.models import SparseVector

    return SparseVector(indices=indices, values=values)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    if denom == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / denom


def ingest_pdf_queue(
    pdf_queue: Iterable[Mapping[str, Any]],
    *,
    config: RunnableConfig | None = None,
) -> dict:
    cfg = Configuration.from_runnable_config(config)
    if not cfg.rag_enabled:
        return {"skipped": True}

    download_result = download_pdf_queue(
        pdf_queue,
        db_path=cfg.db_path,
        download_dir=cfg.pdf_download_dir,
    )

    download_path = Path(cfg.pdf_download_dir)
    entries_to_index: list[tuple[Mapping[str, Any], Path]] = []
    for entry in pdf_queue:
        filename = filename_for_entry(entry)
        path = download_path / filename
        if path.exists():
            entries_to_index.append((entry, path))

    if not entries_to_index:
        return {
            "downloaded": download_result.downloaded,
            "skipped": download_result.skipped,
            "failed": download_result.failed,
            "indexed": 0,
            "errors": download_result.errors,
        }

    dense_embedder = _get_dense_embedder(cfg)
    sparse_embedder = _get_sparse_embedder(cfg)

    sample_vec = dense_embedder.embed_query("dimension check")
    client = _get_qdrant_client(cfg)
    _ensure_collection(client, cfg, len(sample_vec))

    indexed = 0
    for entry, path in entries_to_index:
        text = _parse_pdf_text(path, cfg)
        if not text:
            continue
        chunks = _chunk_text(text, cfg)
        if not chunks:
            continue

        chunk_records = []
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            chunk_records.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": entry.get("doc_id"),
                    "canonical_id": entry.get("canonical_id"),
                    "source": entry.get("source"),
                    "title": entry.get("title"),
                    "url": entry.get("url") or entry.get("pdf_url"),
                    "text": chunk,
                    "metadata": {
                        "source": entry.get("source"),
                        "source_id": entry.get("source_id"),
                        "pdf_url": entry.get("pdf_url"),
                    },
                }
            )

        from qdrant_client.models import PointStruct

        batch_size = max(1, cfg.rag_batch_size)
        for start in range(0, len(chunk_records), batch_size):
            batch_records = chunk_records[start : start + batch_size]
            dense_vectors = dense_embedder.embed_documents(
                [c["text"] for c in batch_records]
            )
            sparse_vectors = list(
                sparse_embedder.embed([c["text"] for c in batch_records])
            )

            points = []
            for record, dense_vec, sparse_vec in zip(
                batch_records, dense_vectors, sparse_vectors
            ):
                payload = {
                    "chunk_id": record["chunk_id"],
                    "doc_id": record.get("doc_id"),
                    "canonical_id": record.get("canonical_id"),
                    "source": record.get("source"),
                    "title": record.get("title"),
                    "url": record.get("url"),
                }
                points.append(
                    PointStruct(
                        id=record["chunk_id"],
                        vector={
                            "dense": dense_vec,
                            "bm25": _to_sparse_vector(sparse_vec),
                        },
                        payload=payload,
                    )
                )

            client.upsert(collection_name=cfg.qdrant_collection, points=points)
            insert_chunks(cfg.db_path, batch_records)
            indexed += len(batch_records)

    return {
        "downloaded": download_result.downloaded,
        "skipped": download_result.skipped,
        "failed": download_result.failed,
        "indexed": indexed,
        "errors": download_result.errors,
    }


def _rrf_fusion(*ranked_lists, k: int = 60):
    scores: dict[str, float] = {}
    items: dict[str, Any] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            item_id = str(item.id)
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
            items[item_id] = item
    return [items[item_id] for item_id in sorted(scores, key=scores.get, reverse=True)]


def hybrid_search(
    query: str,
    *,
    config: RunnableConfig | None = None,
    limit: int | None = None,
) -> list[RagHit]:
    cfg = Configuration.from_runnable_config(config)
    limit = limit or cfg.rag_top_k

    dense_embedder = _get_dense_embedder(cfg)
    sparse_embedder = _get_sparse_embedder(cfg)
    client = _get_qdrant_client(cfg)

    dense_vec = dense_embedder.embed_query(query)
    sparse_vec = _to_sparse_vector(next(iter(sparse_embedder.embed([query]))))

    dense_hits = _qdrant_query(
        client,
        collection_name=cfg.qdrant_collection,
        vector=dense_vec,
        using="dense",
        limit=limit,
        with_payload=True,
        with_vectors=cfg.rag_enable_rerank,
    )
    sparse_hits = _qdrant_query(
        client,
        collection_name=cfg.qdrant_collection,
        vector=sparse_vec,
        using="bm25",
        limit=limit,
        with_payload=True,
        with_vectors=cfg.rag_enable_rerank,
    )

    fused = _rrf_fusion(dense_hits, sparse_hits)
    if not cfg.rag_enable_rerank:
        top_hits = fused[:limit]
    else:
        scored = []
        for hit in fused[: max(limit, cfg.rag_rerank_top_k)]:
            vector = None
            if isinstance(hit.vector, dict):
                vector = hit.vector.get("dense")
            elif isinstance(hit.vector, list):
                vector = hit.vector
            score = _cosine_similarity(dense_vec, vector or [])
            scored.append((score, hit))
        scored.sort(key=lambda item: item[0], reverse=True)
        top_hits = [hit for _, hit in scored[:limit]]

    results: list[RagHit] = []
    for hit in top_hits:
        payload = hit.payload or {}
        results.append(
            RagHit(
                chunk_id=str(payload.get("chunk_id") or hit.id),
                score=float(hit.score or 0.0),
                doc_id=payload.get("doc_id"),
                canonical_id=payload.get("canonical_id"),
                source=payload.get("source"),
                title=payload.get("title"),
                url=payload.get("url"),
            )
        )
    return results


def build_rag_context(
    query: str,
    *,
    config: RunnableConfig | None = None,
    limit: int | None = None,
) -> list[RagContextBlock]:
    cfg = Configuration.from_runnable_config(config)
    hits = hybrid_search(query, config=config, limit=limit)
    chunk_ids = [hit.chunk_id for hit in hits]
    chunks = {chunk["chunk_id"]: chunk for chunk in get_chunks_by_ids(cfg.db_path, chunk_ids)}

    source_index_by_key: dict[str, int] = {}
    blocks: list[RagContextBlock] = []
    for idx, hit in enumerate(hits, start=1):
        chunk = chunks.get(hit.chunk_id)
        if not chunk:
            continue
        source_key = chunk.get("doc_id") or chunk.get("canonical_id")
        if not source_key:
            url = chunk.get("url") or hit.url
            if url:
                source_key = f"url:{_sha256(url)}"
            else:
                source_key = f"chunk:{hit.chunk_id}"
        if source_key not in source_index_by_key:
            source_index_by_key[source_key] = len(source_index_by_key) + 1
        blocks.append(
            RagContextBlock(
                index=idx,
                source_index=source_index_by_key[source_key],
                source_key=source_key,
                chunk_id=hit.chunk_id,
                title=chunk.get("title") or hit.title,
                url=chunk.get("url") or hit.url,
                text=chunk.get("text"),
            )
        )
    return blocks


def format_rag_context(blocks: list[RagContextBlock]) -> str:
    if not blocks:
        return ""
    sources: dict[str, tuple[str | None, str | None]] = {}
    for block in blocks:
        sources[block.source_key] = (block.title, block.url)

    parts: list[str] = ["Sources (cite using the bracketed key):"]
    for source_key, (title, url) in sorted(sources.items()):
        title = title or "Untitled"
        url = url or ""
        parts.append(f"[{source_key}] {title}\nURL: {url}")

    parts.append("\nContext Chunks:")
    for block in blocks:
        title = block.title or "Untitled"
        url = block.url or ""
        parts.append(
            f"[{block.index}] (Source [{block.source_key}]) {title}\nURL: {url}\nChunk ID: {block.chunk_id}\n\n{block.text}"
        )
    return "\n\n".join(parts)
