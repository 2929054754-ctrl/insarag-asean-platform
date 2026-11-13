"""SQLite layer: schema, migrations, CRUD for news / aggregator / corpus /
terminology / embeddings.

All extra columns (review_status, domain, page) are added via idempotent
migrations so existing app.db files keep working.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Iterable

import numpy as np

from .config import (DB_PATH, TURSO_AUTH_TOKEN, TURSO_DATABASE_URL)

SCHEMA = """
CREATE TABLE IF NOT EXISTS news (
    id            TEXT PRIMARY KEY,
    title_en      TEXT NOT NULL DEFAULT '',
    title_zh      TEXT NOT NULL DEFAULT '',
    url           TEXT NOT NULL DEFAULT '',
    publish_date  TEXT NOT NULL DEFAULT '',
    language      TEXT NOT NULL DEFAULT 'en',
    source_org    TEXT NOT NULL DEFAULT 'INSARAG',
    source_url    TEXT NOT NULL DEFAULT '',
    category      TEXT NOT NULL DEFAULT '[]',   -- JSON array
    summary_en    TEXT NOT NULL DEFAULT '',
    summary_zh    TEXT NOT NULL DEFAULT '',
    content_en    TEXT NOT NULL DEFAULT '',
    content_zh    TEXT NOT NULL DEFAULT '',
    tags          TEXT NOT NULL DEFAULT '[]',   -- JSON array
    crawled_at    TEXT NOT NULL DEFAULT '',
    checksum      TEXT NOT NULL DEFAULT '',
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS aggregator (
    id            TEXT PRIMARY KEY,
    source        TEXT NOT NULL DEFAULT '',       -- GDACS | AHA | UNDRR
    title_en      TEXT NOT NULL DEFAULT '',
    title_zh      TEXT NOT NULL DEFAULT '',
    url           TEXT NOT NULL DEFAULT '',
    publish_date  TEXT NOT NULL DEFAULT '',
    language      TEXT NOT NULL DEFAULT 'en',
    domain        TEXT NOT NULL DEFAULT '',
    event_type    TEXT NOT NULL DEFAULT '',
    summary_en    TEXT NOT NULL DEFAULT '',
    summary_zh    TEXT NOT NULL DEFAULT '',
    content_en    TEXT NOT NULL DEFAULT '',
    content_zh    TEXT NOT NULL DEFAULT '',
    review_status TEXT NOT NULL DEFAULT '待审核',
    crawled_at    TEXT NOT NULL DEFAULT '',
    checksum      TEXT NOT NULL DEFAULT '',
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS corpus (
    id           TEXT PRIMARY KEY,
    source_lang  TEXT NOT NULL,
    target_lang  TEXT NOT NULL,
    source_text   TEXT NOT NULL,
    target_text   TEXT NOT NULL,
    source_url    TEXT NOT NULL DEFAULT '',
    domain       TEXT NOT NULL DEFAULT '',
    page         INTEGER NOT NULL DEFAULT 0,
    review_status TEXT NOT NULL DEFAULT '待审核',
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS terminology (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    zh_term     TEXT NOT NULL,
    en_term     TEXT NOT NULL DEFAULT '',
    definition  TEXT NOT NULL DEFAULT '',
    chapter     TEXT NOT NULL DEFAULT '',
    domain      TEXT NOT NULL DEFAULT '',
    source_url  TEXT NOT NULL DEFAULT '',
    review_status TEXT NOT NULL DEFAULT '待审核',
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type   TEXT NOT NULL,                -- news | aggregator | corpus | term
    source_id     TEXT NOT NULL,
    chunk_index   INTEGER NOT NULL DEFAULT 0,   -- chunk ordinal for long docs
    chunk_text    TEXT NOT NULL,
    vector        BLOB NOT NULL,
    vector_dim    INTEGER NOT NULL,
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_emb_source ON embeddings(source_type, source_id);
"""

REVIEW_STATUSES = ("待审核", "已审核")


def _migrate(conn: sqlite3.Connection) -> None:
    """Idempotently add columns introduced after v1."""
    def cols(table: str) -> set[str]:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}

    adds = {
        "news": [("review_status", "TEXT DEFAULT '待审核'"),
                 ("domain", "TEXT DEFAULT ''")],
        "aggregator": [],
        "corpus": [("review_status", "TEXT DEFAULT '待审核'"),
                   ("page", "INTEGER DEFAULT 0")],
        "terminology": [("domain", "TEXT DEFAULT ''"),
                        ("review_status", "TEXT DEFAULT '待审核'")],
        "embeddings": [("chunk_index", "INTEGER DEFAULT 0")],
    }
    for table, spec in adds.items():
        have = cols(table)
        for name, ddl in spec:
            if name not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def _open():
    """Open a connection: libsql/Turso remote when configured, else local sqlite3.

    libsql is an optional best-effort dependency (`pip install libsql`). If the
    import fails we fall back to local sqlite3 and print a warning, so the app
    never breaks on platforms without the extra.
    """
    if TURSO_DATABASE_URL:
        try:
            import libsql  # type: ignore
            conn = libsql.connect(TURSO_DATABASE_URL, auth_token=TURSO_AUTH_TOKEN or None)
            try:
                conn.row_factory = sqlite3.Row
            except Exception:
                pass  # libsql clients return dict-like rows on their own
            return conn
        except Exception as e:  # noqa: BLE001
            print(f"[db] libsql unavailable ({e}); falling back to local sqlite3")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_conn():
    conn = _open()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)


def now() -> float:
    return time.time()


# ---------- vector helpers ----------
def pack_vector(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def unpack_vector(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=dim)


# ---------- row -> dict ----------
def row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for k in ("category", "tags"):
        if k in d and isinstance(d[k], str):
            try:
                d[k] = json.loads(d[k])
            except Exception:
                d[k] = []
    return d


def _filters(allowed: list[str], **kw) -> tuple[str, list]:
    """Build WHERE clause from non-empty kw filters."""
    where, args = [], []
    for k in allowed:
        v = kw.get(k)
        if v not in (None, "", "all"):
            where.append(f"{k} = ?")
            args.append(v)
    return (" WHERE " + " AND ".join(where)) if where else "", args


# ---------- news CRUD ----------
NEWS_FIELDS = [
    "id", "title_en", "title_zh", "url", "publish_date", "language",
    "source_org", "source_url", "category", "summary_en", "summary_zh",
    "content_en", "content_zh", "tags", "crawled_at", "checksum",
    "review_status", "domain",
]


def upsert_news(conn: sqlite3.Connection, n: dict) -> None:
    t = now()
    vals = [n.get(f, "") for f in NEWS_FIELDS]
    if isinstance(vals[8], list):
        vals[8] = json.dumps(vals[8], ensure_ascii=False)
    if isinstance(vals[13], list):
        vals[13] = json.dumps(vals[13], ensure_ascii=False)
    vals[16] = vals[16] or "待审核"
    conn.execute(
        """
        INSERT INTO news (id,title_en,title_zh,url,publish_date,language,
            source_org,source_url,category,summary_en,summary_zh,
            content_en,content_zh,tags,crawled_at,checksum,
            review_status,domain,created_at,updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
            title_en=excluded.title_en, title_zh=excluded.title_zh,
            url=excluded.url, publish_date=excluded.publish_date,
            language=excluded.language, source_org=excluded.source_org,
            source_url=excluded.source_url, category=excluded.category,
            summary_en=excluded.summary_en, summary_zh=excluded.summary_zh,
            content_en=excluded.content_en, content_zh=excluded.content_zh,
            tags=excluded.tags, crawled_at=excluded.crawled_at,
            checksum=excluded.checksum,
            review_status=excluded.review_status, domain=excluded.domain,
            updated_at=excluded.updated_at
        """,
        [*vals, t, t],
    )


def list_news(conn: sqlite3.Connection, q: str = "", limit: int = 200, offset: int = 0,
              **kw) -> tuple[list[dict], int]:
    where, args = _filters(["review_status", "domain"], **kw)
    if q:
        where += (" AND " if where else " WHERE ") + "(title_en LIKE ? OR title_zh LIKE ? OR url LIKE ?)"
        like = f"%{q}%"
        args = [*args, like, like, like]
    total = conn.execute(f"SELECT COUNT(*) c FROM news {where}", args).fetchone()["c"]
    rows = conn.execute(
        f"SELECT * FROM news {where} ORDER BY publish_date DESC LIMIT ? OFFSET ?",
        [*args, limit, offset],
    ).fetchall()
    return [row_to_dict(r) for r in rows], total


def get_news(conn: sqlite3.Connection, nid: str) -> dict | None:
    r = conn.execute("SELECT * FROM news WHERE id=?", (nid,)).fetchone()
    return row_to_dict(r) if r else None


def delete_news(conn: sqlite3.Connection, nid: str) -> None:
    conn.execute("DELETE FROM news WHERE id=?", (nid,))
    conn.execute("DELETE FROM embeddings WHERE source_type='news' AND source_id=?", (nid,))


# ---------- aggregator CRUD ----------
AGG_FIELDS = [
    "id", "source", "title_en", "title_zh", "url", "publish_date", "language",
    "domain", "event_type", "summary_en", "summary_zh", "content_en", "content_zh",
    "review_status", "crawled_at", "checksum",
]


def upsert_aggregator(conn: sqlite3.Connection, a: dict) -> None:
    t = now()
    vals = [a.get(f, "") for f in AGG_FIELDS]
    vals[13] = vals[13] or "待审核"
    conn.execute(
        """
        INSERT INTO aggregator (id,source,title_en,title_zh,url,publish_date,language,
            domain,event_type,summary_en,summary_zh,content_en,content_zh,
            review_status,crawled_at,checksum,created_at,updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
            source=excluded.source, title_en=excluded.title_en, title_zh=excluded.title_zh,
            url=excluded.url, publish_date=excluded.publish_date, language=excluded.language,
            domain=excluded.domain, event_type=excluded.event_type,
            summary_en=excluded.summary_en, summary_zh=excluded.summary_zh,
            content_en=excluded.content_en, content_zh=excluded.content_zh,
            review_status=excluded.review_status, crawled_at=excluded.crawled_at,
            checksum=excluded.checksum, updated_at=excluded.updated_at
        """,
        [*vals, t, t],
    )


def list_aggregator(conn: sqlite3.Connection, q: str = "", limit: int = 300, offset: int = 0,
                    **kw) -> tuple[list[dict], int]:
    where, args = _filters(["source", "domain", "event_type", "review_status"], **kw)
    if q:
        where += (" AND " if where else " WHERE ") + "(title_en LIKE ? OR title_zh LIKE ? OR url LIKE ?)"
        like = f"%{q}%"
        args = [*args, like, like, like]
    total = conn.execute(f"SELECT COUNT(*) c FROM aggregator {where}", args).fetchone()["c"]
    rows = conn.execute(
        f"SELECT * FROM aggregator {where} ORDER BY publish_date DESC LIMIT ? OFFSET ?",
        [*args, limit, offset],
    ).fetchall()
    return [dict(r) for r in rows], total


def get_aggregator(conn: sqlite3.Connection, aid: str) -> dict | None:
    r = conn.execute("SELECT * FROM aggregator WHERE id=?", (aid,)).fetchone()
    return dict(r) if r else None


def delete_aggregator(conn: sqlite3.Connection, aid: str) -> None:
    conn.execute("DELETE FROM aggregator WHERE id=?", (aid,))
    conn.execute("DELETE FROM embeddings WHERE source_type='aggregator' AND source_id=?", (aid,))


# ---------- corpus CRUD ----------
def upsert_corpus(conn: sqlite3.Connection, c: dict) -> None:
    t = now()
    conn.execute(
        """
        INSERT INTO corpus (id,source_lang,target_lang,source_text,target_text,
            source_url,domain,page,review_status,created_at,updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
            source_lang=excluded.source_lang, target_lang=excluded.target_lang,
            source_text=excluded.source_text, target_text=excluded.target_text,
            source_url=excluded.source_url, domain=excluded.domain,
            page=excluded.page, review_status=excluded.review_status,
            updated_at=excluded.updated_at
        """,
        [
            c["id"], c.get("source_lang", ""), c.get("target_lang", ""),
            c.get("source_text", ""), c.get("target_text", ""),
            c.get("source_url", ""), c.get("domain", ""),
            int(c.get("page") or 0), c.get("review_status") or "待审核", t, t,
        ],
    )


def list_corpus(conn: sqlite3.Connection, q: str = "", limit: int = 200, offset: int = 0,
                **kw) -> tuple[list[dict], int]:
    where, args = _filters(["domain", "review_status"], **kw)
    if q:
        where += (" AND " if where else " WHERE ") + "(source_text LIKE ? OR target_text LIKE ?)"
        like = f"%{q}%"
        args = [*args, like, like]
    total = conn.execute(f"SELECT COUNT(*) c FROM corpus {where}", args).fetchone()["c"]
    rows = conn.execute(
        f"SELECT * FROM corpus {where} ORDER BY id LIMIT ? OFFSET ?",
        [*args, limit, offset],
    ).fetchall()
    return [dict(r) for r in rows], total


def get_corpus(conn: sqlite3.Connection, cid: str) -> dict | None:
    r = conn.execute("SELECT * FROM corpus WHERE id=?", (cid,)).fetchone()
    return dict(r) if r else None


def delete_corpus(conn: sqlite3.Connection, cid: str) -> None:
    conn.execute("DELETE FROM corpus WHERE id=?", (cid,))
    conn.execute("DELETE FROM embeddings WHERE source_type='corpus' AND source_id=?", (cid,))


# ---------- terminology CRUD ----------
def insert_term(conn: sqlite3.Connection, t: dict) -> int:
    now_ = now()
    cur = conn.execute(
        """INSERT INTO terminology (zh_term,en_term,definition,chapter,domain,source_url,review_status,created_at,updated_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        [t.get("zh_term", ""), t.get("en_term", ""), t.get("definition", ""),
         t.get("chapter", ""), t.get("domain", ""), t.get("source_url", ""),
         t.get("review_status") or "待审核", now_, now_],
    )
    return cur.lastrowid


def update_term(conn: sqlite3.Connection, tid: int, t: dict) -> None:
    conn.execute(
        """UPDATE terminology SET zh_term=?,en_term=?,definition=?,chapter=?,domain=?,
           source_url=?,review_status=?,updated_at=? WHERE id=?""",
        [t.get("zh_term", ""), t.get("en_term", ""), t.get("definition", ""),
         t.get("chapter", ""), t.get("domain", ""), t.get("source_url", ""),
         t.get("review_status") or "待审核", now(), tid],
    )


def list_terms(conn: sqlite3.Connection, q: str = "", limit: int = 200, offset: int = 0,
               **kw) -> tuple[list[dict], int]:
    where, args = _filters(["domain", "review_status"], **kw)
    if q:
        where += (" AND " if where else " WHERE ") + "(zh_term LIKE ? OR en_term LIKE ?)"
        like = f"%{q}%"
        args = [*args, like, like]
    total = conn.execute(f"SELECT COUNT(*) c FROM terminology {where}", args).fetchone()["c"]
    rows = conn.execute(
        f"SELECT * FROM terminology {where} ORDER BY id LIMIT ? OFFSET ?",
        [*args, limit, offset],
    ).fetchall()
    return [dict(r) for r in rows], total


def get_term(conn: sqlite3.Connection, tid: int) -> dict | None:
    r = conn.execute("SELECT * FROM terminology WHERE id=?", (tid,)).fetchone()
    return dict(r) if r else None


def delete_term(conn: sqlite3.Connection, tid: int) -> None:
    conn.execute("DELETE FROM terminology WHERE id=?", (tid,))
    conn.execute("DELETE FROM embeddings WHERE source_type='term' AND source_id=?", (str(tid),))


# ---------- review status toggle (shared) ----------
def set_review_status(conn: sqlite3.Connection, table: str, rid: str, status: str) -> None:
    if table not in ("news", "aggregator", "corpus", "terminology"):
        raise ValueError("bad table")
    if status not in REVIEW_STATUSES:
        raise ValueError("bad status")
    key = "id" if table != "terminology" else "rowid"
    conn.execute(f"UPDATE {table} SET review_status=?, updated_at=? WHERE {key}=?",
                 (status, now(), rid))


# ---------- embeddings ----------
def replace_embedding(conn: sqlite3.Connection, source_type: str, source_id: str,
                     chunks: list[tuple[int, str, np.ndarray]]) -> None:
    """Replace all chunks of a source row. chunks: [(chunk_index, text, vec)]."""
    conn.execute(
        "DELETE FROM embeddings WHERE source_type=? AND source_id=?",
        (source_type, source_id),
    )
    for idx, text, vec in chunks:
        conn.execute(
            """INSERT INTO embeddings (source_type,source_id,chunk_index,chunk_text,vector,vector_dim,created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (source_type, source_id, int(idx), text, pack_vector(vec), int(vec.shape[-1]), now()),
        )


def all_embeddings(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM embeddings").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["vector"] = unpack_vector(d["vector"], d["vector_dim"])
        out.append(d)
    return out


def count_rows(conn: sqlite3.Connection) -> dict:
    return {
        "news": conn.execute("SELECT COUNT(*) c FROM news").fetchone()["c"],
        "aggregator": conn.execute("SELECT COUNT(*) c FROM aggregator").fetchone()["c"],
        "corpus": conn.execute("SELECT COUNT(*) c FROM corpus").fetchone()["c"],
        "terminology": conn.execute("SELECT COUNT(*) c FROM terminology").fetchone()["c"],
        "embeddings": conn.execute("SELECT COUNT(*) c FROM embeddings").fetchone()["c"],
    }
