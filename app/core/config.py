"""Central configuration: paths, env, language tables."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Project layout
APP_DIR = Path(__file__).resolve().parent.parent          # .../app
PROJECT_DIR = APP_DIR.parent                             # repo root
DATA_DIR = APP_DIR / "sources"                           # self-contained in repo
APP_DATA_DIR = APP_DIR / "data"                          # sqlite lives here

# Load .env from project root and app dir
load_dotenv(PROJECT_DIR / ".env")
load_dotenv(APP_DIR / ".env")


def _resolve_db_path() -> Path:
    """Pick a writable SQLite location. On read-only serverless FS (Vercel/
    Netlify), copy the bundled app.db to /tmp and use that copy. The bundled
    app.db should be pre-built offline (scripts/build_index.py)."""
    candidate = APP_DATA_DIR / "app.db"
    try:
        APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(candidate, "ab"):
            pass
        return candidate
    except OSError:
        import shutil
        tmp = Path("/tmp/app.db")
        if candidate.exists() and not tmp.exists():
            shutil.copy(candidate, tmp)
        return tmp


DB_PATH = _resolve_db_path()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip().rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash").strip()

# Embedding backend switch: "auto" | "bigmodel" | "local"
#   auto     -> BigModel if BIGMODEL_API_KEY is set, else local (default)
#   bigmodel -> force BigModel; on any failure silently fall back to local
#   local    -> force local sentence-transformers (TF-IDF if model unusable)
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "auto").strip().lower()

# BigModel (智谱) embeddings — OpenAI-compatible POST /v4/embeddings
BIGMODEL_API_KEY = os.getenv("BIGMODEL_API_KEY", "").strip()
BIGMODEL_BASE_URL = os.getenv(
    "BIGMODEL_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"
).strip().rstrip("/")
BIGMODEL_EMBEDDING_MODEL = (
    os.getenv("BIGMODEL_EMBEDDING_MODEL")
    or os.getenv("BIGMODEL_EMBED_MODEL")
    or "embedding-3"
).strip()
# embedding-3 default output dim is 2048; allow override. Keep 0 for "model default".
BIGMODEL_EMBEDDING_DIM = int(os.getenv("BIGMODEL_EMBEDDING_DIM", "0") or 0)

# Local sentence-transformers model
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2"
)

# Serverless behaviour: on cold start, rebuild DB + vector index from bundled
# data files (no persistent disk on Vercel/Netlify/Cloudflare).
REBUILD_ON_START = os.getenv("REBUILD_ON_START", "0").strip().lower() in ("1", "true", "yes")

# Turso / libSQL remote database (optional). When set, all SQLite traffic goes
# through the libsql client instead of local sqlite3.
TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL", "").strip()
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN", "").strip()

# RAG guardrail: below this cosine score we treat the context as "no basis"
# and refuse to answer rather than speculate.
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.5"))

# ASEAN + working languages (schema section 3)
LANGUAGES = {
    "zh": "中文 (Chinese)",
    "en": "英语 (English)",
    "id": "印尼语 (Indonesian)",
    "ms": "马来语 (Malay)",
    "th": "泰语 (Thai)",
    "vi": "越南语 (Vietnamese)",
    "fil": "菲律宾语 (Filipino)",
    "lo": "老挝语 (Lao)",
    "my": "缅甸语 (Burmese)",
    "km": "高棉语 (Khmer)",
}

SOURCE_NEWS_DIR = DATA_DIR / "news"
SOURCE_CORPUS_FILE = DATA_DIR / "corpus" / "parallel_corpus_1to1.json"
SOURCE_TERM_DIR = DATA_DIR / "terminology"
SOURCE_AGG_DIR = DATA_DIR / "aggregator"


def llm_status() -> dict:
    return {
        "configured": bool(DEEPSEEK_API_KEY),
        "base_url": DEEPSEEK_BASE_URL,
        "model": DEEPSEEK_MODEL,
    }


def embedding_status() -> dict:
    return {
        "requested": EMBEDDING_BACKEND,
        "bigmodel_key_set": bool(BIGMODEL_API_KEY),
        "bigmodel_base_url": BIGMODEL_BASE_URL,
        "bigmodel_model": BIGMODEL_EMBEDDING_MODEL,
        "local_model": EMBEDDING_MODEL_NAME,
    }
