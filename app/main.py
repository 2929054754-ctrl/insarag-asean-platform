"""FastAPI entrypoint.

Local:      uvicorn main:app --host 0.0.0.0 --port 8000
Offline build: python scripts/build_index.py
Serverless: api/index.py (Vercel) / netlify/functions/app.py (Netlify).

Startup contract (non-blocking):
  1. db.init_db()
  2. retrieval.rebuild() — loads vectors already persisted in app.db into memory
     (fast). The HTTP server accepts requests immediately.
  3. If the embeddings table is empty OR stored dim != current backend dim,
     spawn a daemon thread that incrementally builds missing slices. Progress
     is exposed at GET /api/status -> build.
"""
from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from core import config, db, embedding as emb_mod, ingest, retrieval
from routers import cms, ingest as ingest_api, rag, translate

app = FastAPI(title="INSARAG ASEAN Multilingual Platform", version="2.0")

STATIC_DIR = Path(__file__).resolve().parent / "static"

app.include_router(cms.router)
app.include_router(rag.router)
app.include_router(translate.router)
app.include_router(ingest_api.router)

_build_lock = threading.Lock()


def _stored_dim() -> int:
    with db.get_conn() as conn:
        rows = conn.execute("SELECT DISTINCT vector_dim FROM embeddings").fetchall()
    return int(rows[0][0]) if len(rows) == 1 else 0


def _needs_build() -> bool:
    with db.get_conn() as conn:
        n = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    if n == 0:
        return True
    return _stored_dim() != emb_mod.dim()


def _build_worker() -> None:
    with _build_lock:
        try:
            ingest._set(status="running", error="", backend=emb_mod.get_backend())
            def pcb(done, total, stage):
                ingest._set(status="running", done=done, total=total, stage=stage,
                            backend=emb_mod.get_backend())
            # dim mismatch (e.g. switched backends) -> wipe stale vectors first
            if _stored_dim() and _stored_dim() != emb_mod.dim():
                with db.get_conn() as conn:
                    conn.execute("DELETE FROM embeddings")
                retrieval.rebuild()
            ingest.run_all(pcb)
            retrieval.rebuild()
            n = retrieval.stats().get("rows", 0)
            ingest._set(status="done", stage=f"index ready: {n} rows",
                        backend=emb_mod.get_backend())
        except Exception as e:  # noqa: BLE001
            ingest._set(status="error", error=f"{type(e).__name__}: {e}")


@app.on_event("startup")
def _startup():
    db.init_db()
    try:
        retrieval.rebuild()
    except Exception as e:
        print(f"[startup] retrieval.rebuild failed: {e}")
    if _needs_build():
        threading.Thread(target=_build_worker, daemon=True, name="index-build").start()


@app.get("/api/status")
def status():
    with db.get_conn() as conn:
        counts = db.count_rows(conn)
    return {
        "llm": config.llm_status(),
        "embedding": retrieval.stats(),
        "embedding_config": config.embedding_status(),
        "counts": counts,
        "build": dict(ingest.BUILD_STATE),
    }


# Static frontend
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def portal():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/aggregator")
def aggregator_page():
    return FileResponse(STATIC_DIR / "aggregator.html")


@app.get("/cms")
def cms_page():
    return FileResponse(STATIC_DIR / "cms.html")


@app.get("/rag")
def rag_page():
    return FileResponse(STATIC_DIR / "rag.html")


@app.get("/translate")
def translate_page():
    return FileResponse(STATIC_DIR / "translate.html")


@app.get("/voice")
def voice_page():
    return FileResponse(STATIC_DIR / "voice.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
