"""Ingest endpoints: incremental rebuild triggers + build progress."""
from __future__ import annotations

import threading

from fastapi import APIRouter

from core import config, db, embedding as emb_mod, ingest, retrieval

router = APIRouter(prefix="/api/ingest", tags=["ingest"])


@router.post("/run")
def run_ingest():
    """Synchronously run incremental ingest (use offline build script for big
    rebuilds). Rows already embedded are skipped."""
    result = ingest.run_all()
    retrieval.rebuild()
    return result


@router.post("/build-async")
def build_async():
    """Kick off a background incremental build (non-blocking)."""
    def worker():
        try:
            ingest._set(status="running", error="", backend=emb_mod.get_backend())
            ingest.run_all(lambda d, t, s: ingest._set(
                status="running", done=d, total=t, stage=s,
                backend=emb_mod.get_backend()))
            retrieval.rebuild()
            ingest._set(status="done",
                        stage=f"index ready: {retrieval.stats().get('rows')} rows",
                        backend=emb_mod.get_backend())
        except Exception as e:  # noqa: BLE001
            ingest._set(status="error", error=f"{type(e).__name__}: {e}")
    threading.Thread(target=worker, daemon=True).start()
    return {"ok": True, "message": "build started (see GET /api/status -> build)"}


@router.get("/status")
def status():
    with db.get_conn() as conn:
        counts = db.count_rows(conn)
    return {
        "llm": config.llm_status(),
        "embedding_backend": retrieval.stats().get("backend"),
        "index": retrieval.stats(),
        "build": dict(ingest.BUILD_STATE),
        "counts": counts,
        "data_dirs": {
            "news": str(config.SOURCE_NEWS_DIR),
            "aggregator": str(config.SOURCE_AGG_DIR),
            "corpus": str(config.SOURCE_CORPUS_FILE),
            "terminology": str(config.SOURCE_TERM_DIR),
        },
    }
