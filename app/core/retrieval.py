"""In-memory vector index with real numpy cosine similarity.

Vectors stored in SQLite (BLOB float32) are loaded into a numpy matrix at
startup. Top-K retrieval is a pure numpy dot product over L2-normalized rows,
so similarity is reproducible and auditable.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import List

import numpy as np

from . import db, embedding as emb_mod

_index_vecs: np.ndarray | None = None
_index_meta: List[dict] = []
_lock = threading.Lock()


@dataclass
class Hit:
    score: float
    source_type: str
    source_id: str
    chunk_index: int
    chunk_text: str
    meta: dict = field(default_factory=dict)


def rebuild() -> dict:
    """(Re)load all embeddings from SQLite into memory."""
    global _index_vecs, _index_meta
    with db.get_conn() as conn:
        rows = db.all_embeddings(conn)
    if not rows:
        _index_vecs = np.zeros((0, 0), dtype=np.float32)
        _index_meta = []
        return {"rows": 0, "dim": 0, "backend": emb_mod.get_backend()}

    dims = {r["vector_dim"] for r in rows}
    if len(dims) != 1:
        raise RuntimeError(f"Inconsistent embedding dims in DB: {dims}")
    dim = dims.pop()
    mat = np.zeros((len(rows), dim), dtype=np.float32)
    meta: List[dict] = []
    for i, r in enumerate(rows):
        v = r["vector"]
        n = np.linalg.norm(v)
        mat[i] = v / n if n else v
        meta.append({
            "source_type": r["source_type"],
            "source_id": r["source_id"],
            "chunk_index": r.get("chunk_index", 0),
            "chunk_text": r["chunk_text"],
        })
    _index_vecs = mat
    _index_meta = meta
    return {"rows": len(rows), "dim": dim, "backend": emb_mod.get_backend()}


def _lookup_meta(source_type: str, source_id: str) -> dict:
    """Resolve extra display info (title, url, page ...) for a hit."""
    with db.get_conn() as conn:
        if source_type == "news":
            n = db.get_news(conn, source_id)
            if n:
                title = n.get("title_zh") or n.get("title_en") or source_id
                return {
                    "title": title,
                    "url": n.get("url", ""),
                    "location": f"data/news/{source_id}.json",
                    "domain": n.get("domain", ""),
                }
        elif source_type == "aggregator":
            a = db.get_aggregator(conn, source_id)
            if a:
                title = a.get("title_zh") or a.get("title_en") or source_id
                return {
                    "title": f"[{a.get('source','')}] {title}",
                    "url": a.get("url", ""),
                    "location": f"data/aggregator/{a.get('source','').lower()}_events.json",
                    "domain": a.get("domain", ""),
                    "page": 0,
                }
        elif source_type == "corpus":
            c = db.get_corpus(conn, source_id)
            if c:
                page = int(c.get("page") or 0)
                return {
                    "title": f"平行句对 {source_id} ({c.get('source_lang')}→{c.get('target_lang')})",
                    "url": c.get("source_url", ""),
                    "location": "data/corpus/parallel_corpus_1to1.json",
                    "domain": c.get("domain", ""),
                    "page": page,
                }
        elif source_type == "term":
            t = db.get_term(conn, int(source_id))
            if t:
                return {
                    "title": f"{t.get('zh_term','')} / {t.get('en_term','')}",
                    "url": t.get("source_url", ""),
                    "location": "data/terminology/INSARAG术语库.xlsx",
                    "domain": t.get("domain", ""),
                }
    return {"title": source_id, "url": "", "location": "", "page": 0}


def top_k(query: str, k: int = 5, source_types: list[str] | None = None) -> List[Hit]:
    if _index_vecs is None or _index_vecs.shape[0] == 0:
        return []
    qv = emb_mod.encode([query])[0]
    qv = qv / (np.linalg.norm(qv) or 1.0)
    sims = _index_vecs @ qv  # cosine (both normalized)
    if source_types:
        mask = np.array(
            [1 if m["source_type"] in source_types else 0 for m in _index_meta],
            dtype=bool,
        )
        sims = sims * mask
    k = min(k, len(sims))
    idx = np.argsort(-sims)[:k]
    hits: List[Hit] = []
    for i in idx:
        if sims[i] <= 0:
            continue
        m = _index_meta[i]
        hits.append(Hit(
            score=float(sims[i]),
            source_type=m["source_type"],
            source_id=m["source_id"],
            chunk_index=m.get("chunk_index", 0),
            chunk_text=m["chunk_text"],
            meta=_lookup_meta(m["source_type"], m["source_id"]),
        ))
    return hits


def stats() -> dict:
    if _index_vecs is None:
        return {"rows": 0, "dim": 0, "backend": emb_mod.get_backend()}
    return {
        "rows": int(_index_vecs.shape[0]),
        "dim": int(_index_vecs.shape[1]),
        "backend": emb_mod.get_backend(),
    }
