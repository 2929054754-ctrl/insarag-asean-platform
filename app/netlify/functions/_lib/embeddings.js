// BigModel (智谱) embedding client, OpenAI-compatible.
// Returns L2-normalized vectors as plain number arrays, batched.
import {
  BIGMODEL_API_KEY,
  BIGMODEL_BASE_URL,
  BIGMODEL_EMBED_DIM,
  BIGMODEL_EMBED_MODEL,
} from "./config.js";

export function normalize(vec) {
  let norm = 0;
  for (const v of vec) norm += v * v;
  norm = Math.sqrt(norm) || 1;
  return vec.map((v) => v / norm);
}

async function embedBatch(texts) {
  const body = { model: BIGMODEL_EMBED_MODEL, input: texts };
  if (BIGMODEL_EMBED_DIM > 0) body.dimensions = BIGMODEL_EMBED_DIM;
  const r = await fetch(`${BIGMODEL_BASE_URL}/embeddings`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${BIGMODEL_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    throw new Error(`BigModel embeddings HTTP ${r.status}: ${(await r.text()).slice(0, 200)}`);
  }
  const data = (await r.json()).data || [];
  if (data.length !== texts.length) throw new Error("BigModel embedding count mismatch");
  data.sort((a, b) => (a.index || 0) - (b.index || 0));
  return data.map((d) => normalize(d.embedding));
}

// Encode one or more texts; always L2-normalized.
export async function embed(texts) {
  if (!texts.length) return [];
  if (!BIGMODEL_API_KEY) throw new Error("BIGMODEL_API_KEY 未配置，无法生成向量");
  const BATCH = 32;
  const out = [];
  for (let i = 0; i < texts.length; i += BATCH) {
    out.push(...(await embedBatch(texts.slice(i, i + BATCH))));
  }
  return out;
}

export async function embedOne(text) {
  return (await embed([text]))[0];
}
