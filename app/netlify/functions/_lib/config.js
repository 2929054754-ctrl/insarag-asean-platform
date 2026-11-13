// Environment-driven configuration. No secrets in source; Netlify env vars.
export const DEEPSEEK_API_KEY = (process.env.DEEPSEEK_API_KEY || "").trim();
export const DEEPSEEK_BASE_URL = (
  process.env.DEEPSEEK_BASE_URL || "https://api.deepseek.com"
).replace(/\/+$/, "");
export const DEEPSEEK_MODEL = (process.env.DEEPSEEK_MODEL || "deepseek-flash").trim();

export const BIGMODEL_API_KEY = (process.env.BIGMODEL_API_KEY || "").trim();
export const BIGMODEL_BASE_URL = (
  process.env.BIGMODEL_BASE_URL || "https://open.bigmodel.cn/api/paas/v4"
).replace(/\/+$/, "");
export const BIGMODEL_EMBED_MODEL = (
  process.env.BIGMODEL_EMBED_MODEL || process.env.BIGMODEL_EMBEDDING_MODEL || "embedding-3"
).trim();
export const BIGMODEL_EMBED_DIM = parseInt(process.env.BIGMODEL_EMBED_DIM || "512", 10);

export const RAG_MIN_SCORE = parseFloat(process.env.RAG_MIN_SCORE || "0.5");

export const LANGUAGES = {
  zh: "中文 (Chinese)",
  en: "英语 (English)",
  id: "印尼语 (Indonesian)",
  ms: "马来语 (Malay)",
  th: "泰语 (Thai)",
  vi: "越南语 (Vietnamese)",
  fil: "菲律宾语 (Filipino)",
  lo: "老挝语 (Lao)",
  my: "缅甸语 (Burmese)",
  km: "高棉语 (Khmer)",
};

export const TABLES = ["news", "aggregator", "corpus", "terminology"];
export const VALID_TYPES = ["news", "aggregator", "corpus", "term"];
export const REVIEW_STATUSES = ["待审核", "已审核"];
