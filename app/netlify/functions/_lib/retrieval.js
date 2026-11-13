// Pure-JS cosine-similarity retrieval over base + CMS-override vectors.
import { loadBase } from "./data.js";
import { embedOne } from "./embeddings.js";
import { getMergedTables, getOverrideVectors } from "./cms-store.js";

// Build the combined normalized index. Base vectors are already normalized at
// storage time; we normalize defensively and append override vectors.
async function buildIndex() {
  const base = await loadBase();
  const { rows, dim } = base.manifest;
  const overrides = await getOverrideVectors();
  const n = rows + overrides.length;
  const vecs = new Float32Array(n * dim);
  const meta = [];

  for (let i = 0; i < rows; i++) {
    let norm = 0;
    for (let d = 0; d < dim; d++) {
      const v = base.vectors[i * dim + d];
      norm += v * v;
    }
    norm = Math.sqrt(norm) || 1;
    for (let d = 0; d < dim; d++) vecs[i * dim + d] = base.vectors[i * dim + d] / norm;
    meta.push(base.meta[i]);
  }

  overrides.forEach((o, k) => {
    const i = rows + k;
    let norm = 0;
    for (const v of o.v) norm += v * v;
    norm = Math.sqrt(norm) || 1;
    o.v.forEach((v, d) => {
      vecs[i * dim + d] = v / norm;
    });
    meta.push({ t: o.t, id: String(o.id), ci: o.ci, x: o.x });
  });

  return { vecs, meta, dim, n };
}

// Resolve display metadata for a hit (mirrors Python retrieval._lookup_meta).
async function lookupMeta(t, id) {
  const tables = await getMergedTables();
  if (t === "news") {
    const row = tables.news.find((r) => String(r.id) === id);
    if (row) {
      return {
        title: row.title_zh || row.title_en || id,
        url: row.url || "",
        location: `data/news/${id}.json`,
        domain: row.domain || "",
      };
    }
  } else if (t === "aggregator") {
    const row = tables.aggregator.find((r) => String(r.id) === id);
    if (row) {
      return {
        title: `[${row.source || ""}] ${row.title_zh || row.title_en || id}`,
        url: row.url || "",
        location: `data/aggregator/${(row.source || "").toLowerCase()}_events.json`,
        domain: row.domain || "",
        page: 0,
      };
    }
  } else if (t === "corpus") {
    const row = tables.corpus.find((r) => String(r.id) === id);
    if (row) {
      return {
        title: `平行句对 ${id} (${row.source_lang}→${row.target_lang})`,
        url: row.source_url || "",
        location: "data/corpus/parallel_corpus_1to1.json",
        domain: row.domain || "",
        page: parseInt(row.page || 0, 10),
      };
    }
  } else if (t === "term") {
    const row = tables.terminology.find((r) => String(r.id) === id);
    if (row) {
      return {
        title: `${row.zh_term || ""} / ${row.en_term || ""}`,
        url: row.source_url || "",
        location: "data/terminology/INSARAG术语库.xlsx",
        domain: row.domain || "",
      };
    }
  }
  return { title: id, url: "", location: "", page: 0 };
}

export async function topK(query, k = 5, sourceTypes = null) {
  const idx = await buildIndex();
  const qv = await embedOne(query);
  const allow = sourceTypes ? new Set(sourceTypes) : null;

  const scored = [];
  for (let i = 0; i < idx.n; i++) {
    const m = idx.meta[i];
    if (allow && !allow.has(m.t)) continue;
    let dot = 0;
    for (let d = 0; d < idx.dim; d++) dot += idx.vecs[i * idx.dim + d] * qv[d];
    if (dot > 0) scored.push({ score: dot, i });
  }
  scored.sort((a, b) => b.score - a.score);
  const out = [];
  for (const s of scored.slice(0, k)) {
    const m = idx.meta[s.i];
    out.push({
      score: s.score,
      source_type: m.t,
      source_id: String(m.id),
      chunk_index: m.ci || 0,
      chunk_text: m.x,
      meta: await lookupMeta(m.t, String(m.id)),
    });
  }
  return out;
}

export async function indexStats() {
  const base = await loadBase();
  const overrides = await getOverrideVectors();
  return {
    rows: base.manifest.rows + overrides.length,
    dim: base.manifest.dim,
    backend: "bigmodel",
  };
}
