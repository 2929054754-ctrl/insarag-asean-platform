"""Embedding adapter with pluggable backends.

Backends (selected via EMBEDDING_BACKEND env):
  - bigmodel : 智谱 BigModel OpenAI-compatible POST /v4/embeddings
               (Key from BIGMODEL_API_KEY, model from BIGMODEL_EMBEDDING_MODEL).
  - local    : sentence-transformers, with TF-IDF char-ngram fallback.
  - auto     : BigModel when BIGMODEL_API_KEY is present, otherwise local.

Unified interface:
  encode(texts: list[str]) -> np.ndarray (float32, L2-normalized rows)
  dim() -> int
  get_backend() -> str
  get_last_error() -> str

Hard rules:
  * No BigModel key -> silently use local, never raise.
  * BigModel call failure -> fall back to local for the same batch.
  * Vectors are always L2-normalized float32 so cosine = dot product.
"""
from __future__ import annotations

import threading
import traceback
from typing import List

import numpy as np

from . import config

_backend: str = "unloaded"
_load_attempted: bool = False
_model = None            # sentence-transformers
_tfidf = None
_lock = threading.Lock()
_last_error: str = ""
_bigmodel_dim: int = 0


# ---------------------------------------------------------------------------
# BigModel backend
# ---------------------------------------------------------------------------
def _bigmodel_encode(texts: List[str]) -> np.ndarray:
    """POST {base}/embeddings, batched. Raises on hard failure."""
    import requests

    url = f"{config.BIGMODEL_BASE_URL}/embeddings"
    headers = {
        "Authorization": f"Bearer {config.BIGMODEL_API_KEY}",
        "Content-Type": "application/json",
    }
    out: List[np.ndarray] = []
    BATCH = 32
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i + BATCH]
        body = {"model": config.BIGMODEL_EMBEDDING_MODEL, "input": chunk}
        if config.BIGMODEL_EMBEDDING_DIM > 0:
            body["dimensions"] = config.BIGMODEL_EMBEDDING_DIM
        r = requests.post(url, headers=headers, json=body, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(
                f"BigModel embeddings HTTP {r.status_code}: {r.text[:200]}"
            )
        data = r.json().get("data", [])
        if len(data) != len(chunk):
            raise RuntimeError("BigModel returned mismatched embedding count")
        # preserve API order
        data = sorted(data, key=lambda d: d.get("index", 0))
        for d in data:
            vec = np.asarray(d["embedding"], dtype=np.float32)
            n = np.linalg.norm(vec)
            out.append(vec / n if n else vec)
    global _bigmodel_dim
    if out:
        _bigmodel_dim = int(out[0].shape[0])
    return np.stack(out, axis=0)


# ---------------------------------------------------------------------------
# Local backends
# ---------------------------------------------------------------------------
def _try_load_st() -> bool:
    global _model, _last_error
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        return True
    except Exception as e:  # pragma: no cover - network dependent
        _last_error = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        return False


def _fit_tfidf(corpus: List[str]) -> None:
    global _tfidf
    from sklearn.feature_extraction.text import TfidfVectorizer

    _tfidf = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), min_df=1, max_features=20000,
        sublinear_tf=True,
    )
    _tfidf.fit(list(corpus))


def _local_encode(texts: List[str]) -> np.ndarray:
    global _model, _tfidf
    if _model is None and _tfidf is None:
        if not _try_load_st():
            _fit_tfidf(texts)
    if _model is not None:
        vecs = _model.encode(
            texts, batch_size=32, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)
    if _tfidf is not None:
        sparse = _tfidf.transform(texts)
        vecs = sparse.toarray().astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms
    raise RuntimeError(f"Local embedding backend unavailable: {_last_error}")


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
def _want_bigmodel() -> bool:
    req = config.EMBEDDING_BACKEND
    if req == "local":
        return False
    if req == "bigmodel":
        return True
    # auto: only use bigmodel when a key is actually configured
    return bool(config.BIGMODEL_API_KEY)


def _select_backend() -> None:
    """Resolve backend once. Never raises; failure -> local."""
    global _backend, _load_attempted, _last_error
    _load_attempted = True
    if _want_bigmodel() and config.BIGMODEL_API_KEY:
        try:
            # probe with a tiny request
            _bigmodel_encode(["ping"])
            _backend = "bigmodel"
            return
        except Exception as e:  # noqa: BLE001 - intentional fallback
            _last_error = f"bigmodel probe failed, using local: {e}"
            print(f"[embedding] {_last_error}")
    # local path (probe ST lazily on first encode)
    _backend = "local"


def get_backend() -> str:
    return _backend


def get_last_error() -> str:
    return _last_error


def encode(texts: List[str]) -> np.ndarray:
    """Return L2-normalized float32 matrix of shape (len(texts), dim)."""
    global _backend, _last_error
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    with _lock:
        if not _load_attempted:
            _select_backend()

        if _backend == "bigmodel":
            try:
                return _bigmodel_encode(texts)
            except Exception as e:  # noqa: BLE001 - graceful per-batch fallback
                _last_error = f"bigmodel encode failed, fell back to local: {e}"
                print(f"[embedding] {_last_error}")
                _backend = "local"

        return _local_encode(texts)


def dim() -> int:
    """Expected output dim of the CURRENT backend. Resolvable without warming
    the model (used by startup checks to avoid false rebuilds)."""
    want_bigmodel = (config.EMBEDDING_BACKEND in ("bigmodel", "auto")
                     and bool(config.BIGMODEL_API_KEY))
    if want_bigmodel or _backend == "bigmodel":
        if config.BIGMODEL_EMBEDDING_DIM > 0:
            return config.BIGMODEL_EMBEDDING_DIM
        return _bigmodel_dim or 2048  # embedding-3 default output dim
    if _model is not None:
        return int(_model.get_sentence_embedding_dimension())
    if _tfidf is not None:
        return int(len(_tfidf.vocabulary_))
    if "MiniLM-L12" in config.EMBEDDING_MODEL_NAME:
        return 384
    return 0
