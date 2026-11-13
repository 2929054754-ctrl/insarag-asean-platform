// Netlify single-function entry: routes every /api/* request and streams SSE.
import {
  DEEPSEEK_API_KEY,
  LANGUAGES,
  RAG_MIN_SCORE,
  REVIEW_STATUSES,
  TABLES,
} from "./_lib/config.js";
import { jsonResponse, sse, sseResponse } from "./_lib/sse.js";
import { chatStream } from "./_lib/llm.js";
import { embed } from "./_lib/embeddings.js";
import { loadBase } from "./_lib/data.js";
import { topK, indexStats } from "./_lib/retrieval.js";
import { chunksFor } from "./_lib/chunk.js";
import {
  getMergedTables,
  getOverride,
  getState,
  markDelete,
  putOverride,
  saveState,
} from "./_lib/cms-store.js";
import {
  buildTranslatePrompt,
  matchTerms,
  ragSourcesPayload,
  translateSourcesPayload,
} from "./_lib/translate-logic.js";
import { getFilters, getLibraries, listRows } from "./_lib/cms-logic.js";

// CMS table config: route segment -> table name + id type
const CMS_TABLES = {
  news: { table: "news", numeric: false },
  aggregator: { table: "aggregator", numeric: false },
  corpus: { table: "corpus", numeric: false },
  terms: { table: "terminology", numeric: true },
};

function notFound(msg = "not found") {
  return jsonResponse({ detail: msg }, 404);
}

function badRequest(msg) {
  return jsonResponse({ detail: msg }, 400);
}

// Re-embed a row's chunks and persist the override.
async function persistRow(table, row) {
  const chunkTexts = chunksFor(table, row);
  const vectors = await embed(chunkTexts);
  await putOverride(table, row, chunkTexts, vectors);
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------
async function handleStatus() {
  const base = await loadBase();
  const state = await getState();
  const counts = { ...base.manifest.counts };
  for (const t of TABLES) {
    const merged = (await getMergedTables())[t];
    counts[t] = merged.length;
  }
  counts.embeddings = base.manifest.rows + state.vectors.length;
  return jsonResponse({
    llm: {
      configured: Boolean(DEEPSEEK_API_KEY),
      base_url: process.env.DEEPSEEK_BASE_URL || "https://api.deepseek.com",
      model: process.env.DEEPSEEK_MODEL || "deepseek-flash",
    },
    embedding: await indexStats(),
    embedding_config: {
      requested: process.env.EMBEDDING_BACKEND || "bigmodel",
      bigmodel_key_set: Boolean(process.env.BIGMODEL_API_KEY),
      bigmodel_base_url:
        process.env.BIGMODEL_BASE_URL || "https://open.bigmodel.cn/api/paas/v4",
      bigmodel_model: process.env.BIGMODEL_EMBED_MODEL || "embedding-3",
    },
    counts,
    build: {
      status: "done",
      done: counts.embeddings,
      total: counts.embeddings,
      stage: "index ready (prebuilt artifact)",
      backend: "bigmodel",
      error: "",
    },
  });
}

function handleTranslateLanguages() {
  return jsonResponse(LANGUAGES);
}

function handleTranslateStream(request) {
  return sseResponse(
    (async function* () {
      const body = await request.json().catch(() => null);
      if (!body) {
        yield sse({ type: "error", message: "请求体格式错误" });
        return;
      }
      const text = (body.text || "").trim();
      const tgt = body.target_lang || "zh";
      const src = body.source_lang || "auto";
      const useRag = body.use_rag !== false;

      if (!text) {
        yield sse({ type: "error", message: "待翻译文本为空" });
        return;
      }
      if (!LANGUAGES[tgt]) {
        yield sse({ type: "error", message: `不支持的目标语言: ${tgt}` });
        return;
      }
      if (!DEEPSEEK_API_KEY) {
        yield sse({ type: "error", message: "DEEPSEEK_API_KEY 未配置，无法翻译" });
        return;
      }

      const ragHits = useRag ? await topK(text, 4) : [];
      yield sse({ type: "sources", items: translateSourcesPayload(ragHits) });

      const terms = await matchTerms(text);
      yield sse({ type: "terms", items: terms });

      const messages = buildTranslatePrompt(text, src, tgt, ragHits, terms);
      try {
        for await (const { kind, text: piece } of chatStream(messages, 0.2)) {
          if (kind === "content") yield sse({ type: "token", text: piece });
        }
        yield sse({ type: "done" });
      } catch (err) {
        yield sse({ type: "error", message: `翻译失败: ${err.message}` });
      }
    })()
  );
}

function handleRagAsk(request) {
  return sseResponse(
    (async function* () {
      const body = await request.json().catch(() => null);
      if (!body) {
        yield sse({ type: "error", message: "请求体格式错误" });
        return;
      }
      const q = (body.question || "").trim();
      const k = Math.max(1, Math.min(parseInt(body.top_k || "5", 10), 10));
      const minScore =
        body.min_score !== undefined && body.min_score !== null
          ? Number(body.min_score)
          : RAG_MIN_SCORE;
      const types = body.source_types || null;

      if (!q) {
        yield sse({ type: "error", message: "问题为空" });
        return;
      }
      if (!DEEPSEEK_API_KEY) {
        yield sse({
          type: "error",
          message: "DEEPSEEK_API_KEY 未配置，无法生成答案（检索仍可用）",
        });
        return;
      }

      const hits = await topK(q, k, types);
      yield sse({ type: "sources", items: ragSourcesPayload(hits) });

      const best = hits.length ? hits[0].score : 0;
      if (!hits.length || best < minScore) {
        yield sse({
          type: "refuse",
          text:
            "知识库中未检索到与该问题相关、且可信度足够的上下文" +
            `（最高相关度 ${best.toFixed(2)}，阈值 ${minScore}）。` +
            "为避免误导，我不做臆测回答。请尝试换一个更贴近 INSARAG/东盟搜救" +
            "领域的问题，或先在 CMS 中补充相关资料。",
        });
        yield sse({ type: "done" });
        return;
      }

      const ctxLines = hits.map((h, i) => {
        const page = parseInt(h.meta.page || 0, 10);
        const loc = (h.meta.location || "") + (page ? ` p.${page}` : "");
        return `[资料${i + 1}] 来源：${h.meta.title || ""} (${h.source_type}, ${loc})\n${h.chunk_text}`;
      });
      const context = ctxLines.join("\n\n");
      const system =
        "你是 INSARAG/东盟多语种搜救知识库的问答助手。规则：\n" +
        "1. 严格依据下面给定的上下文回答；上下文没有提到的内容，一律说" +
        "“知识库中没有相关依据”，禁止编造、禁止用常识补全。\n" +
        "2. 回答用与用户问题相同的语言，保持专业、简洁。\n" +
        "3. 答案末尾用“参考：”列出引用的资料编号（资料1/资料2…）。";
      const messages = [
        { role: "system", content: system },
        {
          role: "user",
          content: `知识库上下文：\n${context}\n\n用户问题：${q}`,
        },
      ];
      try {
        for await (const { kind, text } of chatStream(messages, 0.3)) {
          if (kind === "content") yield sse({ type: "token", text });
          else if (kind === "reasoning") yield sse({ type: "reasoning", text });
        }
        yield sse({ type: "done" });
      } catch (err) {
        yield sse({ type: "error", message: `模型调用失败: ${err.message}` });
      }
    })()
  );
}

async function handleRagRetrieve(request) {
  const body = await request.json().catch(() => null);
  if (!body) return badRequest("请求体格式错误");
  const q = (body.question || "").trim();
  const k = Math.max(1, Math.min(parseInt(body.top_k || "5", 10), 10));
  const hits = await topK(q, k, body.source_types || null);
  const stats = await indexStats();
  return jsonResponse({
    backend: stats.backend,
    dim: stats.dim,
    question: q,
    min_score: RAG_MIN_SCORE,
    items: ragSourcesPayload(hits),
  });
}

// CMS list / detail / create / update / delete
async function handleCmsList(seg, url) {
  const cfg = CMS_TABLES[seg];
  if (!cfg) return notFound("bad cms table");
  const { items, total } = await listRows(cfg.table, url.searchParams);
  return jsonResponse({ items, total });
}

async function handleCmsDetail(seg, rid) {
  const cfg = CMS_TABLES[seg];
  const tables = await getMergedTables();
  const row = tables[cfg.table].find((r) => String(r.id) === String(rid));
  return row ? jsonResponse(row) : notFound("row not found");
}

async function handleCmsCreate(seg, request) {
  const cfg = CMS_TABLES[seg];
  if (!cfg) return notFound("bad cms table");
  const body = await request.json();
  const tables = await getMergedTables();
  let row = { ...body };
  if (cfg.numeric) {
    const maxId = tables[cfg.table].reduce((m, r) => Math.max(m, Number(r.id) || 0), 0);
    row.id = maxId + 1;
  } else {
    if (!row.id) return badRequest("id required");
    if (tables[cfg.table].some((r) => String(r.id) === String(row.id))) {
      return jsonResponse({ detail: "id already exists" }, 409);
    }
  }
  row.review_status = row.review_status || "待审核";
  await persistRow(cfg.table, row);
  return jsonResponse({ ok: true, id: row.id });
}

async function handleCmsUpdate(seg, rid, request) {
  const cfg = CMS_TABLES[seg];
  if (!cfg) return notFound("bad cms table");
  const body = await request.json();
  const tables = await getMergedTables();
  if (!tables[cfg.table].some((r) => String(r.id) === String(rid))) {
    return notFound("row not found");
  }
  const row = { ...body, id: cfg.numeric ? Number(rid) : rid };
  if (!cfg.numeric && body.id && String(body.id) !== String(rid)) {
    return badRequest("id mismatch");
  }
  row.review_status = row.review_status || "待审核";
  await persistRow(cfg.table, row);
  return jsonResponse({ ok: true, id: rid });
}

async function handleCmsDelete(seg, rid) {
  const cfg = CMS_TABLES[seg];
  if (!cfg) return notFound("bad cms table");
  const tables = await getMergedTables();
  if (!tables[cfg.table].some((r) => String(r.id) === String(rid))) {
    return notFound("row not found");
  }
  await markDelete(cfg.table, rid);
  return jsonResponse({ ok: true });
}

async function handleReview(table, rid, request) {
  const tableName = table === "term" ? "terminology" : table;
  if (!TABLES.includes(tableName)) return badRequest("bad table");
  const body = await request.json().catch(() => null);
  if (!body || !REVIEW_STATUSES.includes(body.status)) {
    return badRequest(`status must be one of ${REVIEW_STATUSES.join(", ")}`);
  }
  // Apply onto the override layer; baseline rows get an upsert copy.
  const tables = await getMergedTables();
  const row = tables[tableName].find((r) => String(r.id) === String(rid));
  if (!row) return notFound("row not found");
  const updated = { ...row, review_status: body.status };
  await putOverride(tableName, updated, chunksFor(tableName, updated), await embed(
    chunksFor(tableName, updated)
  ));
  return jsonResponse({ ok: true, table: tableName, id: rid, status: body.status });
}

async function handleIngestRun() {
  // Index is prebuilt; report current state without re-embedding.
  const stats = await indexStats();
  return jsonResponse({ ok: true, index: stats, note: "prebuilt artifact; no rebuild needed" });
}

function handleIngestAsync() {
  return jsonResponse({ ok: true, message: "index already built (prebuilt artifact)" });
}

async function handleIngestStatus() {
  const base = await loadBase();
  const state = await getState();
  const merged = await getMergedTables();
  const counts = {};
  for (const t of TABLES) counts[t] = merged[t].length;
  counts.embeddings = base.manifest.rows + state.vectors.length;
  return jsonResponse({
    llm: {
      configured: Boolean(DEEPSEEK_API_KEY),
      model: process.env.DEEPSEEK_MODEL || "deepseek-flash",
    },
    embedding_backend: "bigmodel",
    index: await indexStats(),
    build: { status: "done", backend: "bigmodel", stage: "prebuilt artifact" },
    counts,
  });
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------
export default async (request) => {
  const url = new URL(request.url);
  const p = url.pathname;
  const method = request.method;
  const parts = p.split("/").filter(Boolean); // e.g. ["api","cms","news","123"]

  if (parts[0] !== "api") return notFound("unknown route");
  const resource = parts[1];

  // ---- status ----
  if (resource === "status" && method === "GET") return handleStatus();

  // ---- translate ----
  if (resource === "translate") {
    if (parts[2] === "languages" && method === "GET") return handleTranslateLanguages();
    if (parts[2] === "stream" && method === "POST") return handleTranslateStream(request);
  }

  // ---- rag ----
  if (resource === "rag") {
    if (parts[2] === "ask" && method === "POST") return handleRagAsk(request);
    if (parts[2] === "retrieve" && method === "POST") return handleRagRetrieve(request);
    if (parts[2] === "libraries" && method === "GET") {
      return jsonResponse(await getLibraries());
    }
  }

  // ---- cms ----
  if (resource === "cms") {
    if (parts[2] === "filters" && method === "GET") {
      return jsonResponse(await getFilters());
    }
    const seg = parts[2];
    const rid = parts[3];
    if (seg && CMS_TABLES[seg]) {
      if (!rid && method === "GET") return handleCmsList(seg, url);
      if (!rid && method === "POST") return handleCmsCreate(seg, request);
      if (rid && method === "GET") return handleCmsDetail(seg, rid);
      if (rid && method === "PUT") return handleCmsUpdate(seg, rid, request);
      if (rid && method === "DELETE") return handleCmsDelete(seg, rid);
    }
    if (seg === "review" && parts[3] && parts[4] && method === "POST") {
      return handleReview(parts[3], parts[4], request);
    }
  }

  // ---- ingest ----
  if (resource === "ingest") {
    if (parts[2] === "run" && method === "POST") return handleIngestRun();
    if (parts[2] === "build-async" && method === "POST") return handleIngestAsync();
    if (parts[2] === "status" && method === "GET") return handleIngestStatus();
  }

  return notFound(`no route: ${method} ${p}`);
};
