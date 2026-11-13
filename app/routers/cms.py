"""CMS CRUD: news / aggregator / corpus / terminology.

Supports review-status toggle, domain classification, source/domain/event-type
filtering. SQLite persistence. Every write re-embeds the row's chunks and
rebuilds the in-memory index.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core import db, embedding as emb_mod, retrieval
from core.ingest import _agg_chunks, _corpus_chunk, _news_chunks, _term_chunk

router = APIRouter(prefix="/api/cms", tags=["cms"])


def _reembed(conn, kind: str, row: dict) -> None:
    """Re-embed one row's chunks inside the given open transaction."""
    if kind == "news":
        chunks = _news_chunks(row); sid = row["id"]
    elif kind == "aggregator":
        chunks = _agg_chunks(row); sid = row["id"]
    elif kind == "corpus":
        chunks = [_corpus_chunk(row)]; sid = row["id"]
    elif kind == "term":
        chunks = [_term_chunk(row)]; sid = str(row["id"])
    else:
        raise ValueError(kind)
    vecs = emb_mod.encode(chunks)
    db.replace_embedding(conn, kind, sid,
                        [(i, t, vecs[i]) for i, t in enumerate(chunks)])


# ---------------------------------------------------------------------------
# news
# ---------------------------------------------------------------------------
class NewsIn(BaseModel):
    id: str
    title_en: str = ""
    title_zh: str = ""
    url: str = ""
    publish_date: str = ""
    language: str = "en"
    source_org: str = "INSARAG"
    source_url: str = ""
    category: list[str] = []
    summary_en: str = ""
    summary_zh: str = ""
    content_en: str = ""
    content_zh: str = ""
    tags: list[str] = []
    crawled_at: str = ""
    checksum: str = ""
    review_status: str = "待审核"
    domain: str = ""


@router.get("/news")
def news_list(q: str = "", limit: int = 200, offset: int = 0,
              review_status: str = "", domain: str = ""):
    with db.get_conn() as conn:
        rows, total = db.list_news(conn, q=q, limit=limit, offset=offset,
                                   review_status=review_status, domain=domain)
    return {"items": rows, "total": total}


@router.get("/news/{nid}")
def news_detail(nid: str):
    with db.get_conn() as conn:
        row = db.get_news(conn, nid)
    if not row:
        raise HTTPException(404, "news not found")
    return row


@router.post("/news")
def news_create(n: NewsIn):
    with db.get_conn() as conn:
        if db.get_news(conn, n.id):
            raise HTTPException(409, "news id already exists")
        row = n.model_dump()
        db.upsert_news(conn, row)
        _reembed(conn, "news", row)
    retrieval.rebuild()
    return {"ok": True, "id": n.id}


@router.put("/news/{nid}")
def news_update(nid: str, n: NewsIn):
    if nid != n.id:
        raise HTTPException(400, "id mismatch")
    with db.get_conn() as conn:
        if not db.get_news(conn, nid):
            raise HTTPException(404, "news not found")
        row = n.model_dump()
        db.upsert_news(conn, row)
        _reembed(conn, "news", row)
    retrieval.rebuild()
    return {"ok": True, "id": nid}


@router.delete("/news/{nid}")
def news_delete(nid: str):
    with db.get_conn() as conn:
        if not db.get_news(conn, nid):
            raise HTTPException(404, "news not found")
        db.delete_news(conn, nid)
    retrieval.rebuild()
    return {"ok": True}


# ---------------------------------------------------------------------------
# aggregator
# ---------------------------------------------------------------------------
class AggIn(BaseModel):
    id: str
    source: str = ""
    title_en: str = ""
    title_zh: str = ""
    url: str = ""
    publish_date: str = ""
    language: str = "en"
    domain: str = ""
    event_type: str = ""
    summary_en: str = ""
    summary_zh: str = ""
    content_en: str = ""
    content_zh: str = ""
    review_status: str = "待审核"


@router.get("/aggregator")
def agg_list(q: str = "", limit: int = 300, offset: int = 0,
             source: str = "", domain: str = "", event_type: str = "",
             review_status: str = ""):
    with db.get_conn() as conn:
        rows, total = db.list_aggregator(
            conn, q=q, limit=limit, offset=offset,
            source=source, domain=domain, event_type=event_type,
            review_status=review_status)
    return {"items": rows, "total": total}


@router.get("/aggregator/{aid}")
def agg_detail(aid: str):
    with db.get_conn() as conn:
        row = db.get_aggregator(conn, aid)
    if not row:
        raise HTTPException(404, "aggregator item not found")
    return row


@router.post("/aggregator")
def agg_create(a: AggIn):
    with db.get_conn() as conn:
        if db.get_aggregator(conn, a.id):
            raise HTTPException(409, "aggregator id already exists")
        row = a.model_dump()
        db.upsert_aggregator(conn, row)
        _reembed(conn, "aggregator", row)
    retrieval.rebuild()
    return {"ok": True, "id": a.id}


@router.put("/aggregator/{aid}")
def agg_update(aid: str, a: AggIn):
    if aid != a.id:
        raise HTTPException(400, "id mismatch")
    with db.get_conn() as conn:
        if not db.get_aggregator(conn, aid):
            raise HTTPException(404, "aggregator item not found")
        row = a.model_dump()
        db.upsert_aggregator(conn, row)
        _reembed(conn, "aggregator", row)
    retrieval.rebuild()
    return {"ok": True, "id": aid}


@router.delete("/aggregator/{aid}")
def agg_delete(aid: str):
    with db.get_conn() as conn:
        if not db.get_aggregator(conn, aid):
            raise HTTPException(404, "aggregator item not found")
        db.delete_aggregator(conn, aid)
    retrieval.rebuild()
    return {"ok": True}


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------
class CorpusIn(BaseModel):
    id: str
    source_lang: str
    target_lang: str
    source_text: str
    target_text: str
    source_url: str = ""
    domain: str = ""
    page: int = 0
    review_status: str = "待审核"


@router.get("/corpus")
def corpus_list(q: str = "", limit: int = 200, offset: int = 0,
                domain: str = "", review_status: str = ""):
    with db.get_conn() as conn:
        rows, total = db.list_corpus(conn, q=q, limit=limit, offset=offset,
                                      domain=domain, review_status=review_status)
    return {"items": rows, "total": total}


@router.post("/corpus")
def corpus_create(c: CorpusIn):
    with db.get_conn() as conn:
        if db.get_corpus(conn, c.id):
            raise HTTPException(409, "corpus id already exists")
        row = c.model_dump()
        db.upsert_corpus(conn, row)
        _reembed(conn, "corpus", row)
    retrieval.rebuild()
    return {"ok": True, "id": c.id}


@router.put("/corpus/{cid}")
def corpus_update(cid: str, c: CorpusIn):
    if cid != c.id:
        raise HTTPException(400, "id mismatch")
    with db.get_conn() as conn:
        if not db.get_corpus(conn, cid):
            raise HTTPException(404, "corpus not found")
        row = c.model_dump()
        db.upsert_corpus(conn, row)
        _reembed(conn, "corpus", row)
    retrieval.rebuild()
    return {"ok": True, "id": c.id}


@router.delete("/corpus/{cid}")
def corpus_delete(cid: str):
    with db.get_conn() as conn:
        if not db.get_corpus(conn, cid):
            raise HTTPException(404, "corpus not found")
        db.delete_corpus(conn, cid)
    retrieval.rebuild()
    return {"ok": True}


# ---------------------------------------------------------------------------
# terminology
# ---------------------------------------------------------------------------
class TermIn(BaseModel):
    zh_term: str
    en_term: str = ""
    definition: str = ""
    chapter: str = ""
    domain: str = ""
    source_url: str = ""
    review_status: str = "待审核"


@router.get("/terms")
def terms_list(q: str = "", limit: int = 500, offset: int = 0,
               domain: str = "", review_status: str = ""):
    with db.get_conn() as conn:
        rows, total = db.list_terms(conn, q=q, limit=limit, offset=offset,
                                    domain=domain, review_status=review_status)
    return {"items": rows, "total": total}


@router.post("/terms")
def term_create(t: TermIn):
    with db.get_conn() as conn:
        row = t.model_dump()
        tid = db.insert_term(conn, row)
        row["id"] = tid
        _reembed(conn, "term", row)
    retrieval.rebuild()
    return {"ok": True, "id": tid}


@router.put("/terms/{tid}")
def term_update(tid: int, t: TermIn):
    with db.get_conn() as conn:
        if not db.get_term(conn, tid):
            raise HTTPException(404, "term not found")
        row = t.model_dump()
        row["id"] = tid
        db.update_term(conn, tid, row)
        _reembed(conn, "term", row)
    retrieval.rebuild()
    return {"ok": True, "id": tid}


@router.delete("/terms/{tid}")
def term_delete(tid: int):
    with db.get_conn() as conn:
        if not db.get_term(conn, tid):
            raise HTTPException(404, "term not found")
        db.delete_term(conn, tid)
    retrieval.rebuild()
    return {"ok": True}


# ---------------------------------------------------------------------------
# review status toggle (shared)
# ---------------------------------------------------------------------------
class ReviewIn(BaseModel):
    status: str


@router.post("/review/{table}/{rid}")
def review_toggle(table: str, rid: str, body: ReviewIn):
    if table not in ("news", "aggregator", "corpus", "terminology"):
        raise HTTPException(400, "bad table")
    if body.status not in db.REVIEW_STATUSES:
        raise HTTPException(400, f"status must be one of {db.REVIEW_STATUSES}")
    with db.get_conn() as conn:
        try:
            db.set_review_status(conn, table, rid, body.status)
        except ValueError as e:
            raise HTTPException(400, str(e))
    return {"ok": True, "table": table, "id": rid, "status": body.status}


@router.get("/filters")
def filters():
    """Distinct domain/source/event_type lists for UI dropdowns."""
    out = {"domains": set(), "sources": set(), "event_types": set()}
    with db.get_conn() as conn:
        for tbl in ("news", "aggregator", "corpus", "terminology"):
            for r in conn.execute(f"SELECT DISTINCT domain FROM {tbl} WHERE domain<>''"):
                out["domains"].add(r[0])
        for r in conn.execute("SELECT DISTINCT source FROM aggregator WHERE source<>''"):
            out["sources"].add(r[0])
        for r in conn.execute("SELECT DISTINCT event_type FROM aggregator WHERE event_type<>''"):
            out["event_types"].add(r[0])
    return {k: sorted(v) for k, v in out.items()}
