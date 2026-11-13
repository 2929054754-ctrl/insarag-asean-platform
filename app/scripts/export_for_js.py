"""Offline export of app.db into JS-friendly artifacts for the Netlify function.

Outputs (function_data/):
  vectors.bin   raw Float32 matrix, row-major, one row per embedding
  meta.json     [{source_type, source_id, chunk_index, chunk_text}]
  tables.json   {news, aggregator, corpus, terminology} full rows
  manifest.json {rows, dim, counts, model}

Run:  python scripts/export_for_js.py
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parent.parent
DB = APP_DIR / "data" / "app.db"
OUT = APP_DIR / "function_data"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    embs = conn.execute(
        "SELECT source_type, source_id, chunk_index, chunk_text, vector, vector_dim "
        "FROM embeddings ORDER BY id"
    ).fetchall()
    if not embs:
        raise SystemExit("no embeddings found")

    dims = {r["vector_dim"] for r in embs}
    assert len(dims) == 1, f"mixed dims: {dims}"
    dim = dims.pop()

    mat = np.zeros((len(embs), dim), dtype="<f4")  # little-endian float32
    meta = []
    for i, r in enumerate(embs):
        vec = np.frombuffer(r["vector"], dtype="<f4", count=dim)
        mat[i] = vec
        meta.append({
            "t": r["source_type"],
            "id": r["source_id"],
            "ci": r["chunk_index"],
            "x": r["chunk_text"],
        })
    mat.tofile(OUT / "vectors.bin")
    (OUT / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )

    def dump(table: str) -> list[dict]:
        return [dict(r) for r in conn.execute(f"SELECT * FROM {table}")]

    tables = {
        "news": dump("news"),
        "aggregator": dump("aggregator"),
        "corpus": dump("corpus"),
        "terminology": dump("terminology"),
    }
    (OUT / "tables.json").write_text(
        json.dumps(tables, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )

    counts = {
        t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in ("news", "aggregator", "corpus", "terminology")
    }
    counts["embeddings"] = len(embs)
    manifest = {"rows": len(embs), "dim": dim, "counts": counts}
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for f in ("vectors.bin", "meta.json", "tables.json", "manifest.json"):
        p = OUT / f
        print(f"{f}: {p.stat().st_size:,} bytes")
    print("manifest:", json.dumps(manifest, ensure_ascii=False))
    conn.close()


if __name__ == "__main__":
    main()
