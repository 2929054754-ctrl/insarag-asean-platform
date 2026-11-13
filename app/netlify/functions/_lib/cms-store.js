// CMS override layer backed by Netlify Blobs.
// Baseline rows are read-only in the committed artifacts; every create/update/
// delete is recorded here and merged at request time.
import { getStore } from "@netlify/blobs";
import { TABLES } from "./config.js";

const STORE_NAME = "cms-state";
const KEY = "state";

// table name -> retrieval source_type used by embeddings
const EMB_TYPE = {
  news: "news",
  aggregator: "aggregator",
  corpus: "corpus",
  terminology: "term",
};
export function embType(table) {
  return EMB_TYPE[table];
}

function emptyState() {
  return {
    upserts: { news: {}, aggregator: {}, corpus: {}, terminology: {} },
    deletes: { news: [], aggregator: [], corpus: [], terminology: [] },
    vectors: [],
  };
}

export async function getState() {
  const store = getStore(STORE_NAME);
  const s = await store.get(KEY, { type: "json", consistency: "strong" });
  return s || emptyState();
}

export async function saveState(state) {
  const store = getStore(STORE_NAME);
  await store.setJSON(KEY, state);
}

export async function getMergedTables() {
  const { tables } = await (await import("./data.js")).loadBase();
  const state = await getState();
  const out = {};
  for (const t of TABLES) {
    const deleted = new Set(state.deletes[t].map(String));
    const baseRows = tables[t].filter((r) => !deleted.has(String(r.id)));
    const byId = new Map(baseRows.map((r) => [String(r.id), r]));
    for (const [id, row] of Object.entries(state.upserts[t])) {
      byId.set(String(id), row);
    }
    out[t] = [...byId.values()];
  }
  return out;
}

export async function getOverride(table, id) {
  const state = await getState();
  return state.upserts[table][String(id)] || null;
}

// Record a row upsert and replace its embedded vectors.
export async function putOverride(table, row, chunkTexts, vectors) {
  const state = await getState();
  const id = String(row.id);
  state.upserts[table][id] = row;
  state.deletes[table] = state.deletes[table].filter((d) => String(d) !== id);
  const t = EMB_TYPE[table];
  state.vectors = state.vectors.filter(
    (v) => !(v.t === t && String(v.id) === id)
  );
  chunkTexts.forEach((x, ci) => {
    state.vectors.push({ t, id, ci, x, v: vectors[ci] });
  });
  await saveState(state);
}

export async function markDelete(table, id) {
  const state = await getState();
  const sid = String(id);
  delete state.upserts[table][sid];
  if (!state.deletes[table].map(String).includes(sid)) state.deletes[table].push(sid);
  const t = EMB_TYPE[table];
  state.vectors = state.vectors.filter(
    (v) => !(v.t === t && String(v.id) === sid)
  );
  await saveState(state);
}

// Override vectors as parallel arrays for the retrieval index.
export async function getOverrideVectors() {
  const state = await getState();
  return state.vectors;
}
