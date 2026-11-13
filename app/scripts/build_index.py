#!/usr/bin/env python3
"""Offline index builder — runs OUTSIDE the web server.

Usage:
    cd app
    python scripts/build_index.py            # incremental build (resumable)
    python scripts/build_index.py --wipe     # delete embeddings, clean rebuild

Behaviour:
  * Reads BigModel/local config from .env (EMBEDDING_BACKEND, BIGMODEL_API_KEY,
    BIGMODEL_EMBEDDING_MODEL, BIGMODEL_EMBEDDING_DIM). Never prints the key.
  * Detects stored vector dim vs current backend dim; if they differ, wipes
    the embeddings table and rebuilds from scratch (or pass --wipe explicitly).
  * Incremental: rows whose slices are already embedded are skipped. Safe to
    Ctrl-C mid-run; re-run resumes and does not re-bill BigModel for done rows.
  * Prints progress: done/total, current stage, backend, any error.
  * At the end prints row counts, vector count, dim, and a cosine recompute
    sanity check on a probe query.

The resulting app.db is the artifact bundled to serverless platforms.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

from core import config, db, embedding as emb_mod, ingest, retrieval  # noqa: E402


def stored_dim() -> int:
    with db.get_conn() as conn:
        rows = conn.execute("SELECT DISTINCT vector_dim FROM embeddings").fetchall()
    return int(rows[0][0]) if len(rows) == 1 else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wipe", action="store_true", help="delete all embeddings first")
    args = ap.parse_args()

    db.init_db()
    print(f"[build] backend requested={config.EMBEDDING_BACKEND} "
          f"bigmodel_key={bool(config.BIGMODEL_API_KEY)} "
          f"model={config.BIGMODEL_EMBEDDING_MODEL} "
          f"dim_override={config.BIGMODEL_EMBEDDING_DIM or 'default'}")

    # Warm probe (resolves backend; raises if bigmodel forced and unreachable)
    _ = emb_mod.encode(["probe"])
    print(f"[build] resolved backend = {emb_mod.get_backend()}, dim = {emb_mod.dim()}")

    want = emb_mod.dim()
    have = stored_dim()
    if args.wipe or (have and have != want):
        print(f"[build] dim mismatch (stored={have} want={want}) or --wipe: "
              f"clearing embeddings")
        with db.get_conn() as conn:
            conn.execute("DELETE FROM embeddings")
        retrieval.rebuild()

    t0 = time.time()
    last = [0.0]

    def pcb(done: int, total: int, stage: str) -> None:
        now = time.time()
        if now - last[0] < 0.5 and done != total:
            return
        last[0] = now
        pct = f"{done}/{total}" if total else f"{done}/?"
        print(f"[build] {pct:<12} {stage}", flush=True)

    result = ingest.run_all(pcb)
    retrieval.rebuild()

    with db.get_conn() as conn:
        counts = db.count_rows(conn)
    stats = retrieval.stats()

    # cosine recompute sanity check
    probe = "OSOCC 是什么"
    hits = retrieval.top_k(probe, k=3)
    recompute = "ok" if hits and hits[0].score > 0 else "no-hit"

    print("=" * 60)
    print(f"[build] DONE in {time.time()-t0:.1f}s")
    print(f"[build] rows      : {counts}")
    print(f"[build] vectors   : {stats.get('rows')}  dim={stats.get('dim')}  backend={stats.get('backend')}")
    print(f"[build] per-source: news={result['news']} aggregator={result['aggregator']} "
          f"corpus={result['corpus']} term={result['term']}")
    print(f"[build] probe '{probe}' top-1: "
          f"{hits[0].meta.get('title') if hits else 'NONE'} "
          f"score={hits[0].score if hits else 0:.3f} ({recompute})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
