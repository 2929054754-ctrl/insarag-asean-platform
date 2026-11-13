// CMS list/filter logic and aggregate metadata for UI dropdowns.
import { getMergedTables } from "./cms-store.js";
import { VALID_TYPES } from "./config.js";

function like(haystack, needle) {
  return String(haystack || "").toLowerCase().includes(needle.toLowerCase());
}

export async function listRows(table, params) {
  const tables = await getMergedTables();
  let rows = [...tables[table]];

  const reviewStatus = params.get("review_status") || "";
  const domain = params.get("domain") || "";
  const q = params.get("q") || "";

  if (reviewStatus && reviewStatus !== "all") {
    rows = rows.filter((r) => (r.review_status || "待审核") === reviewStatus);
  }
  if (domain && domain !== "all") rows = rows.filter((r) => r.domain === domain);

  if (table === "news") {
    if (q) {
      rows = rows.filter(
        (r) => like(r.title_en, q) || like(r.title_zh, q) || like(r.url, q)
      );
    }
    rows.sort((a, b) => String(b.publish_date || "").localeCompare(String(a.publish_date || "")));
  } else if (table === "aggregator") {
    const source = params.get("source") || "";
    const eventType = params.get("event_type") || "";
    if (source && source !== "all") rows = rows.filter((r) => r.source === source);
    if (eventType && eventType !== "all") rows = rows.filter((r) => r.event_type === eventType);
    if (q) {
      rows = rows.filter(
        (r) => like(r.title_en, q) || like(r.title_zh, q) || like(r.url, q)
      );
    }
    rows.sort((a, b) => String(b.publish_date || "").localeCompare(String(a.publish_date || "")));
  } else if (table === "corpus") {
    if (q) {
      rows = rows.filter(
        (r) => like(r.source_text, q) || like(r.target_text, q)
      );
    }
    rows.sort((a, b) => String(a.id).localeCompare(String(b.id)));
  } else if (table === "terminology") {
    if (q) {
      rows = rows.filter((r) => like(r.zh_term, q) || like(r.en_term, q));
    }
    rows.sort((a, b) => Number(a.id) - Number(b.id));
  }

  const total = rows.length;
  const limit = parseInt(params.get("limit") || "200", 10);
  const offset = parseInt(params.get("offset") || "0", 10);
  return { items: rows.slice(offset, offset + limit), total };
}

export async function getFilters() {
  const tables = await getMergedTables();
  const domains = new Set();
  const sources = new Set();
  const eventTypes = new Set();
  for (const t of Object.keys(tables)) {
    for (const r of tables[t]) if (r.domain) domains.add(r.domain);
  }
  for (const r of tables.aggregator) {
    if (r.source) sources.add(r.source);
    if (r.event_type) eventTypes.add(r.event_type);
  }
  return {
    domains: [...domains].sort(),
    sources: [...sources].sort(),
    event_types: [...eventTypes].sort(),
  };
}

export async function getLibraries() {
  const tables = await getMergedTables();
  const counts = {
    news: tables.news.length,
    aggregator: tables.aggregator.length,
    corpus: tables.corpus.length,
    term: tables.terminology.length,
  };
  const domains = new Set();
  for (const t of Object.keys(tables)) {
    for (const r of tables[t]) if (r.domain) domains.add(r.domain);
  }
  return { types: VALID_TYPES, counts, domains: [...domains].sort() };
}
