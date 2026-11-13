// Translation business logic: glossary matching, prompt assembly, source payload.
import { LANGUAGES } from "./config.js";
import { getMergedTables } from "./cms-store.js";

export function langName(code) {
  return LANGUAGES[code] || code;
}

// Exact-substring glossary match (port of routers/translate._match_terms).
export async function matchTerms(text) {
  const tables = await getMergedTables();
  const low = text.toLowerCase();
  const hits = [];
  for (const t of tables.terminology) {
    const zh = (t.zh_term || "").trim();
    const en = (t.en_term || "").trim();
    const enHead = en.split("(")[0].trim();
    let matched = "";
    for (const cand of [zh, enHead, en]) {
      if (cand && low.includes(cand.toLowerCase())) {
        matched = cand;
        break;
      }
    }
    if (!matched) continue;
    const side = zh && text.includes(zh) ? "zh" : "en";
    hits.push({
      id: t.id,
      source: side === "zh" ? zh : en,
      target: side === "zh" ? en : zh,
      matched,
      definition: t.definition || "",
    });
  }
  hits.sort((a, b) => b.matched.length - a.matched.length);
  const seen = new Set();
  const out = [];
  for (const h of hits) {
    const key = h.matched.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(h);
  }
  return out;
}

export function buildTranslatePrompt(text, src, tgt, ragHits, terms) {
  let sys =
    "You are a professional translation engine for INSARAG / Urban Search and Rescue " +
    "and ASEAN disaster-response domain. Rules:\n" +
    "1. Translate exactly and only the user text; do not add explanations.\n" +
    "2. Preserve numbers, named entities, URLs.\n" +
    `3. Output the translation in ${langName(tgt)}.`;
  if (terms.length) {
    sys +=
      "\n4. The following glossary terms appear in the source text. You MUST use the " +
      "target-side rendering given here, verbatim, and never paraphrase it:";
    for (const h of terms) sys += `\n   - “${h.matched}” -> “${h.target}”`;
  }
  const userParts = [];
  if (ragHits.length) {
    userParts.push("Reference materials (use them for term consistency):");
    ragHits.forEach((h, i) => {
      userParts.push(`[ref${i + 1}] (${h.meta.title || ""}) ${h.chunk_text}`);
    });
    userParts.push("");
  }
  const srcLabel = src === "auto" ? "auto-detected" : langName(src);
  userParts.push(`Translate from ${srcLabel} to ${langName(tgt)}:\n${text}`);
  return [
    { role: "system", content: sys },
    { role: "user", content: userParts.join("\n") },
  ];
}

export function translateSourcesPayload(hits) {
  return hits.map((h) => ({
    score: Math.round(h.score * 10000) / 10000,
    source_type: h.source_type,
    title: h.meta.title || h.source_id,
    url: h.meta.url || "",
    location: h.meta.location || "",
    page: parseInt(h.meta.page || 0, 10),
    chunk: h.chunk_text.slice(0, 300),
  }));
}

export function ragSourcesPayload(hits) {
  return hits.map((h) => ({
    score: Math.round(h.score * 10000) / 10000,
    source_type: h.source_type,
    source_id: h.source_id,
    chunk_index: h.chunk_index,
    title: h.meta.title || h.source_id,
    url: h.meta.url || "",
    location: h.meta.location || "",
    domain: h.meta.domain || "",
    page: parseInt(h.meta.page || 0, 10),
    page_label: h.meta.page ? `p.${h.meta.page}` : "",
    chunk: h.chunk_text.slice(0, 400),
  }));
}
