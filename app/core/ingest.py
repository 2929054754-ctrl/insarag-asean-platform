"""Runtime ingest: read external data files and persist + vectorize.

Design (serverless-friendly):
  * INCREMENTAL: before embedding a row, check which (source_type, source_id,
    chunk_index) slices already exist in the embeddings table. Rows whose slices
    are all present are skipped entirely. Interrupted builds are resumed on the
    next run; no duplicate BigModel calls.
  * COMMIT PER ROW: every row commits its own transaction, so Ctrl-C at any
    point leaves a consistent DB.
  * PROGRESS CALLBACK: callers (main background thread / offline build script)
    receive (done, total, stage) updates.

Reads (never writes):
  data/news/news_*.json, data/aggregator/*.json,
  data/corpus/parallel_corpus_1to1.json, data/terminology/*.xlsx
"""
from __future__ import annotations

import json
import re
import threading
from typing import Callable

from . import db, embedding as emb_mod, retrieval
from .config import (SOURCE_AGG_DIR, SOURCE_CORPUS_FILE, SOURCE_NEWS_DIR,
                     SOURCE_TERM_DIR)

CHUNK_SIZE = 450

# Module-level build state, surfaced by /api/status.
BUILD_STATE = {
    "status": "idle",        # idle | running | done | error
    "done": 0,
    "total": 0,
    "stage": "",
    "backend": "",
    "error": "",
}
_state_lock = threading.Lock()

ProgressFn = Callable[[int, int, str], None]


def _set(**kw) -> None:
    with _state_lock:
        BUILD_STATE.update(kw)


def _progress(pcb: ProgressFn | None, done: int, total: int, stage: str) -> None:
    _set(done=done, total=total, stage=stage, backend=emb_mod.get_backend())
    if pcb:
        try:
            pcb(done, total, stage)
        except Exception:
            pass


def _split_chunks(text: str, size: int = CHUNK_SIZE) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    parts = re.split(r"(?<=[。！？.!?\n])", text)
    chunks, cur = [], ""
    for p in parts:
        if len(cur) + len(p) > size and cur:
            chunks.append(cur.strip())
            cur = p
        else:
            cur += p
    if cur.strip():
        chunks.append(cur.strip())
    return [c for c in chunks if c]


def _news_chunks(n: dict) -> list[str]:
    title = n.get("title_zh") or n.get("title_en") or ""
    title_en = n.get("title_en") or ""
    summary = n.get("summary_zh") or n.get("summary_en") or ""
    content = n.get("content_zh") or n.get("content_en") or ""
    head = f"[标题] {title}\n[Title EN] {title_en}\n[摘要] {summary}"
    out = [head]
    for c in _split_chunks(content):
        out.append(f"[标题] {title}\n{c}")
    return out


def _agg_chunks(a: dict) -> list[str]:
    title = a.get("title_zh") or a.get("title_en") or ""
    summary = a.get("summary_zh") or a.get("summary_en") or ""
    content = a.get("content_zh") or a.get("content_en") or ""
    head = (f"[源] {a.get('source','')} [事件] {a.get('event_type','')}\n"
            f"[标题] {title}\n[摘要] {summary}")
    out = [head]
    for c in _split_chunks(content):
        out.append(f"[标题] {title}\n{c}")
    return out


def _corpus_chunk(c: dict) -> str:
    page = int(c.get("page") or 0)
    pfx = f"[p.{page}] " if page else ""
    return (f"{pfx}{c.get('source_lang','')}: {c.get('source_text','')}\n"
            f"{c.get('target_lang','')}: {c.get('target_text','')}").strip()


def _term_chunk(t: dict) -> str:
    return f"{t.get('zh_term','')} / {t.get('en_term','')}: {t.get('definition','')}".strip()


def _existing_indexed(conn, source_type: str, source_id: str) -> set[int]:
    rows = conn.execute(
        "SELECT chunk_index FROM embeddings WHERE source_type=? AND source_id=?",
        (source_type, source_id),
    ).fetchall()
    return {int(r[0]) for r in rows}


def _row_ready(conn, source_type: str, source_id: str, n_chunks: int) -> bool:
    """Row is already fully embedded iff every chunk_index 0..n-1 exists."""
    have = _existing_indexed(conn, source_type, source_id)
    return have == set(range(n_chunks))


def _embed_one_row(conn, source_type: str, source_id: str, chunks: list[str]) -> None:
    """Embed chunks for one row (if missing) and commit. Existing complete
    rows are skipped so re-runs are cheap and resumable."""
    if not chunks:
        return
    if _row_ready(conn, source_type, source_id, len(chunks)):
        return
    vecs = emb_mod.encode(chunks)
    db.replace_embedding(
        conn, source_type, source_id,
        [(i, t, vecs[i]) for i, t in enumerate(chunks)],
    )


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------
def ingest_news(pcb: ProgressFn | None = None) -> dict:
    added = skipped = 0
    if not SOURCE_NEWS_DIR.exists():
        return {"files": 0, "added": 0, "skipped": 0}
    files = sorted(SOURCE_NEWS_DIR.glob("news_*.json"))
    for i, fp in enumerate(files):
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[ingest] skip {fp.name}: {e}")
            continue
        title = obj.get("title", {}) or {}
        summary = obj.get("summary", {}) or {}
        content = obj.get("content", {}) or {}
        source = obj.get("source", {}) or {}
        row = {
            "id": obj.get("id") or fp.stem,
            "title_en": title.get("en", ""), "title_zh": title.get("zh", ""),
            "url": obj.get("url", ""), "publish_date": obj.get("publish_date", ""),
            "language": obj.get("language", "en"),
            "source_org": source.get("organization", "INSARAG"),
            "source_url": source.get("website", ""),
            "category": obj.get("category", []),
            "summary_en": summary.get("en", ""), "summary_zh": summary.get("zh", ""),
            "content_en": content.get("en", ""), "content_zh": content.get("zh", ""),
            "tags": obj.get("tags", []),
            "crawled_at": obj.get("crawled_at", ""),
            "checksum": obj.get("checksum", ""),
            "domain": obj.get("domain", ""),
            "review_status": obj.get("review_status", "已审核"),
        }
        chunks = _news_chunks(row)
        with db.get_conn() as conn:
            before = _row_ready(conn, "news", row["id"], len(chunks))
            db.upsert_news(conn, row)
            _embed_one_row(conn, "news", row["id"], chunks)
        if before:
            skipped += 1
        else:
            added += 1
        _progress(pcb, i + 1, len(files), f"新闻 {fp.name}")
    return {"files": len(files), "added": added, "skipped": skipped}


def ingest_aggregator(pcb: ProgressFn | None = None) -> dict:
    added = skipped = 0
    if not SOURCE_AGG_DIR.exists():
        return {"files": 0, "added": 0, "skipped": 0}
    files = sorted(p for p in SOURCE_AGG_DIR.glob("*.json")
                   if not p.name.endswith("_index.json"))
    rows = []
    for fp in files:
        try:
            items = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[ingest] skip {fp.name}: {e}")
            continue
        if not isinstance(items, list):
            continue
        for obj in items:
            title = obj.get("title", {}) or {}
            summary = obj.get("summary", {}) or {}
            content = obj.get("content", {}) or {}
            row = {
                "id": obj.get("id") or fp.stem,
                "source": obj.get("source", fp.stem.split("_")[0].upper()),
                "title_en": title.get("en", ""), "title_zh": title.get("zh", ""),
                "url": obj.get("url", ""), "publish_date": obj.get("publish_date", ""),
                "language": obj.get("language", "en"),
                "domain": obj.get("domain", ""),
                "event_type": obj.get("event_type", ""),
                "summary_en": summary.get("en", ""), "summary_zh": summary.get("zh", ""),
                "content_en": content.get("en", ""), "content_zh": content.get("zh", ""),
                "review_status": obj.get("review_status", "待审核"),
                "crawled_at": obj.get("crawled_at", ""),
                "checksum": obj.get("checksum", ""),
            }
            if not row["title_en"] and not row["title_zh"]:
                continue
            rows.append(row)
    for i, row in enumerate(rows):
        chunks = _agg_chunks(row)
        with db.get_conn() as conn:
            before = _row_ready(conn, "aggregator", row["id"], len(chunks))
            db.upsert_aggregator(conn, row)
            _embed_one_row(conn, "aggregator", row["id"], chunks)
        if before:
            skipped += 1
        else:
            added += 1
        _progress(pcb, i + 1, len(rows), f"聚合 {row['id']}")
    return {"files": len(files), "added": added, "skipped": skipped}


def ingest_corpus(pcb: ProgressFn | None = None) -> dict:
    added = skipped = 0
    if not SOURCE_CORPUS_FILE.exists():
        return {"items": 0, "added": 0, "skipped": 0}
    try:
        items = json.loads(SOURCE_CORPUS_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        return {"error": str(e)}
    for i, c in enumerate(items):
        row = {
            "id": c.get("id", ""),
            "source_lang": c.get("source_lang", ""),
            "target_lang": c.get("target_lang", ""),
            "source_text": c.get("source_text", ""),
            "target_text": c.get("target_text", ""),
            "source_url": c.get("source_url", ""),
            "domain": c.get("domain", ""),
            "page": int(c.get("page") or 0),
            "review_status": c.get("review_status", "已审核"),
        }
        if not row["id"]:
            continue
        chunks = [_corpus_chunk(row)]
        with db.get_conn() as conn:
            before = _row_ready(conn, "corpus", row["id"], len(chunks))
            db.upsert_corpus(conn, row)
            _embed_one_row(conn, "corpus", row["id"], chunks)
        if before:
            skipped += 1
        else:
            added += 1
        _progress(pcb, i + 1, len(items), f"语料 {row['id']}")
    return {"items": len(items), "added": added, "skipped": skipped}


def ingest_terminology(pcb: ProgressFn | None = None) -> dict:
    added = skipped = 0
    if not SOURCE_TERM_DIR.exists():
        return {"files": 0, "added": 0, "skipped": 0}
    xlsx_files = sorted(SOURCE_TERM_DIR.glob("*.xlsx"))
    rows = []
    for fp in xlsx_files:
        try:
            import pandas as pd  # lazy: only needed for offline/on-demand ingest
            df = pd.read_excel(fp)
        except Exception as e:
            print(f"[ingest] skip xlsx {fp.name}: {e}")
            continue
        cols = {str(c).strip(): c for c in df.columns}

        def col(*names: str):
            for n in names:
                for k in cols:
                    if n in k:
                        return cols[k]
            return None
        c_zh = col("中文", "术语")
        c_en = col("英文", "English")
        c_def = col("释义", "定义")
        c_chap = col("章节", "chapter", "Chapter")
        c_domain = col("领域", "domain", "Domain")
        c_url = col("来源", "链接", "url", "URL")
        c_rev = col("审核", "review", "状态")
        for _, r in df.iterrows():
            row = {
                "zh_term": str(r[c_zh]).strip() if c_zh is not None else "",
                "en_term": str(r[c_en]).strip() if c_en is not None else "",
                "definition": str(r[c_def]).strip() if c_def is not None else "",
                "chapter": str(r[c_chap]).strip() if c_chap is not None else "",
                "domain": str(r[c_domain]).strip() if c_domain is not None else "",
                "source_url": str(r[c_url]).strip() if c_url is not None else "",
                "review_status": (str(r[c_rev]).strip() if c_rev is not None else "待审核"),
            }
            if not row["zh_term"] or row["zh_term"].lower() == "nan":
                continue
            row["review_status"] = (row["review_status"] if row["review_status"] in db.REVIEW_STATUSES
                                    else "待审核")
            rows.append(row)

    for i, row in enumerate(rows):
        chunks = [_term_chunk(row)]
        with db.get_conn() as conn:
            existing = conn.execute(
                "SELECT id FROM terminology WHERE zh_term=? AND en_term=?",
                (row["zh_term"], row["en_term"]),
            ).fetchone()
            if existing:
                tid = int(existing[0])
                db.update_term(conn, tid, row)
            else:
                tid = db.insert_term(conn, row)
            row["id"] = tid
            if not _row_ready(conn, "term", str(tid), len(chunks)):
                _embed_one_row(conn, "term", str(tid), chunks)
                added += 1
            else:
                skipped += 1
        _progress(pcb, i + 1, len(rows), f"术语 {row['zh_term'][:20]}")
    return {"files": len(xlsx_files), "added": added, "skipped": skipped}


def run_all(pcb: ProgressFn | None = None) -> dict:
    """Ingest all sources incrementally, then rebuild in-memory index."""
    news_res = ingest_news(pcb)
    agg_res = ingest_aggregator(pcb)
    corpus_res = ingest_corpus(pcb)
    term_res = ingest_terminology(pcb)
    idx = retrieval.rebuild()
    return {"news": news_res, "aggregator": agg_res, "corpus": corpus_res,
            "term": term_res, "index": idx}


def ensure_index(pcb: ProgressFn | None = None) -> dict:
    """Idempotent entry: rebuild in memory, then incrementally ingest missing
    slices. Safe to call on every cold start — already-built rows are skipped."""
    retrieval.rebuild()
    return run_all(pcb)
