// JS port of core/ingest.py chunking, so CMS-added rows embed identically.
const CHUNK_SIZE = 450;

// Split while keeping the sentence delimiter at the end of each part
// (equivalent to Python re.split(r"(?<=[。！？.!?\n])", text)).
function splitSentences(text) {
  const parts = [];
  let cur = "";
  for (const ch of text) {
    cur += ch;
    if ("。！？.!?\n".includes(ch)) {
      parts.push(cur);
      cur = "";
    }
  }
  if (cur) parts.push(cur);
  return parts;
}

export function splitChunks(text, size = CHUNK_SIZE) {
  text = (text || "").trim();
  if (!text) return [];
  if (text.length <= size) return [text];
  const parts = splitSentences(text);
  const chunks = [];
  let cur = "";
  for (const p of parts) {
    if (cur.length + p.length > size && cur) {
      chunks.push(cur.trim());
      cur = p;
    } else {
      cur += p;
    }
  }
  if (cur.trim()) chunks.push(cur.trim());
  return chunks.filter(Boolean);
}

export function newsChunks(n) {
  const title = n.title_zh || n.title_en || "";
  const titleEn = n.title_en || "";
  const summary = n.summary_zh || n.summary_en || "";
  const content = n.content_zh || n.content_en || "";
  const head = `[标题] ${title}\n[Title EN] ${titleEn}\n[摘要] ${summary}`;
  const out = [head];
  for (const c of splitChunks(content)) out.push(`[标题] ${title}\n${c}`);
  return out;
}

export function aggChunks(a) {
  const title = a.title_zh || a.title_en || "";
  const summary = a.summary_zh || a.summary_en || "";
  const content = a.content_zh || a.content_en || "";
  const head =
    `[源] ${a.source || ""} [事件] ${a.event_type || ""}\n` +
    `[标题] ${title}\n[摘要] ${summary}`;
  const out = [head];
  for (const c of splitChunks(content)) out.push(`[标题] ${title}\n${c}`);
  return out;
}

export function corpusChunk(c) {
  const page = parseInt(c.page || 0, 10);
  const pfx = page ? `[p.${page}] ` : "";
  return (
    `${pfx}${c.source_lang || ""}: ${c.source_text || ""}\n` +
    `${c.target_lang || ""}: ${c.target_text || ""}`
  ).trim();
}

export function termChunk(t) {
  return `${t.zh_term || ""} / ${t.en_term || ""}: ${t.definition || ""}`.trim();
}

// Return the chunk texts for a table row.
export function chunksFor(kind, row) {
  if (kind === "news") return newsChunks(row);
  if (kind === "aggregator") return aggChunks(row);
  if (kind === "corpus") return [corpusChunk(row)];
  if (kind === "term" || kind === "terminology") return [termChunk(row)];
  throw new Error(`bad kind ${kind}`);
}
