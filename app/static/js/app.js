// Shared helpers: SSE-over-fetch, nav highlight, source rendering.
// When the frontend is hosted statically (Cloudflare Pages / Vercel static)
// and the API lives elsewhere, set window.API_BASE = "https://api.example.com"
// in an inline script before this file loads.

const API_BASE = window.API_BASE || "";

const LANGS = {
  zh: "中文", en: "英语", id: "印尼语", ms: "马来语", th: "泰语",
  vi: "越南语", fil: "菲律宾语", lo: "老挝语", my: "缅甸语", km: "高棉语",
};

async function api(path, opts = {}) {
  const r = await fetch(API_BASE + path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!r.ok) {
    const t = await r.text();
    throw new Error(`${r.status}: ${t}`);
  }
  return r.json();
}

/** POST an SSE endpoint; callbacks: onToken, onReasoning, onSources, onTerms,
 *  onRefuse, onDone, onError. */
async function ssePost(path, body, cbs) {
  const resp = await fetch(API_BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok || !resp.body) {
    cbs.onError && cbs.onError(`HTTP ${resp.status}`);
    return;
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const line = chunk.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      let evt;
      try { evt = JSON.parse(line.slice(5).trim()); } catch { continue; }
      if (evt.type === "token") cbs.onToken && cbs.onToken(evt.text);
      else if (evt.type === "reasoning") cbs.onReasoning && cbs.onReasoning(evt.text);
      else if (evt.type === "sources") cbs.onSources && cbs.onSources(evt.items || []);
      else if (evt.type === "terms") cbs.onTerms && cbs.onTerms(evt.items || []);
      else if (evt.type === "refuse") cbs.onRefuse && cbs.onRefuse(evt.text);
      else if (evt.type === "done") cbs.onDone && cbs.onDone();
      else if (evt.type === "error") cbs.onError && cbs.onError(evt.message);
    }
  }
}

function renderSources(items) {
  if (!items || !items.length) return '<div class="muted">（无命中来源）</div>';
  return items
    .map((s) => {
      const title = s.title || s.source_id;
      const link = s.url
        ? `<a href="${s.url}" target="_blank" rel="noopener">${escapeHtml(title)}</a>`
        : escapeHtml(title);
      const loc = s.location ? `<span class="muted"> · ${escapeHtml(s.location)}</span>` : "";
      const page = s.page ? `<span class="badge" style="margin-left:6px">p.${escapeHtml(String(s.page))}</span>` : "";
      const score = typeof s.score === "number"
        ? `<span class="score">${s.score.toFixed(3)}</span>` : "";
      return `<div class="src-item">${score}[${s.source_type}] ${link}${page}${loc}</div>`;
    })
    .join("");
}

/** Wrap all occurrences of terms (zh/en) in <mark>. */
function highlightTerms(text, terms) {
  let out = escapeHtml(text);
  const sorted = (terms || []).slice().sort((a, b) => b.matched.length - a.matched.length);
  for (const t of sorted) {
    const need = escapeHtml(t.matched);
    if (!need || need.length < 2) continue;
    const re = new RegExp(need.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "g");
    out = out.replace(re, `<mark title="${escapeHtml(t.target)}">${need}</mark>`);
  }
  return out;
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

const NAV = [
  ["/", "门户"],
  ["/aggregator", "多源聚合"],
  ["/cms", "知识库 CMS"],
  ["/rag", "RAG 问答"],
  ["/translate", "多语翻译"],
  ["/voice", "语音翻译"],
];

function buildNav() {
  const nav = document.querySelector(".nav");
  if (!nav) return;
  const path = location.pathname;
  const links = NAV.map(([href, label]) =>
    `<a class="link${href === path ? " active" : ""}" href="${API_BASE ? API_BASE.replace(/\/$/,"") + href : href}">${label}</a>`
  ).join("");
  const brand = nav.querySelector(".brand");
  nav.innerHTML = brand.outerHTML + links +
    `<div class="spacer"></div><span class="pill" id="llm-pill">检测中…</span>`;
}

async function refreshStatusPill() {
  try {
    const s = await api("/api/status");
    const el = document.getElementById("llm-pill");
    if (!el) return;
    if (s.llm.configured) {
      el.textContent = `DeepSeek ${s.llm.model} · ${s.embedding.backend} · 向量 ${s.embedding.rows}`;
      el.classList.add("ok");
    } else {
      el.textContent = "DeepSeek 未配置";
      el.classList.add("warn");
    }
  } catch (e) { /* ignore */ }
}

document.addEventListener("DOMContentLoaded", () => {
  buildNav();
  refreshStatusPill();
});
