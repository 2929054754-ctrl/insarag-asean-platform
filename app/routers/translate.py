"""ASEAN multilingual translation with RAG + terminology enforcement.

- 10 languages: zh/en/id/ms/th/vi/fil/lo/my/km (any pair).
- Backend DeepSeek (deepseek-flash), SSE streaming.
- Terminology enforcement: scan the source text against the glossary; matched
  terms are injected as "must use this translation" rules and returned to the
  frontend so it can highlight them in both the source and the output.
- RAG enhancement: retrieve terminology / parallel corpus / news / aggregator
  matches for the source text and return them as clickable sources.
"""
from __future__ import annotations

import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core import config, db, llm, retrieval
from core.config import LANGUAGES

router = APIRouter(prefix="/api/translate", tags=["translate"])


class TranslateIn(BaseModel):
    text: str
    source_lang: str = "auto"
    target_lang: str = "zh"
    use_rag: bool = True


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _lang_name(code: str) -> str:
    return LANGUAGES.get(code, code)


@router.get("/languages")
def languages():
    return LANGUAGES


def _match_terms(text: str) -> list[dict]:
    """Exact-substring glossary match. Returns [{id, source, target, matched}]."""
    low = text.lower()
    hits: list[dict] = []
    with db.get_conn() as conn:
        terms, _ = db.list_terms(conn, limit=5000)
    for t in terms:
        zh = (t.get("zh_term") or "").strip()
        en = (t.get("en_term") or "").strip()
        # canonical canonical English head (before parenthesis) for matching
        en_head = en.split("(")[0].strip()
        matched = ""
        for cand in (zh, en_head, en):
            if cand and cand.lower() in low:
                matched = cand
                break
        if not matched:
            continue
        # decide source->target direction hint: which side appears in text
        side = "zh" if zh and zh in text else "en"
        hits.append({
            "id": t["id"],
            "source": zh if side == "zh" else en,
            "target": en if side == "zh" else zh,
            "matched": matched,
            "definition": t.get("definition", ""),
        })
    # longest-match first, dedup
    hits.sort(key=lambda x: -len(x["matched"]))
    seen, out = set(), []
    for h in hits:
        if h["matched"].lower() in seen:
            continue
        seen.add(h["matched"].lower())
        out.append(h)
    return out


def _build_prompt(text: str, src: str, tgt: str, rag_hits, terms: list[dict]) -> list[dict]:
    sys = (
        "You are a professional translation engine for INSARAG / Urban Search and Rescue "
        "and ASEAN disaster-response domain. Rules:\n"
        "1. Translate exactly and only the user text; do not add explanations.\n"
        "2. Preserve numbers, named entities, URLs.\n"
        f"3. Output the translation in {_lang_name(tgt)}."
    )
    if terms:
        sys += "\n4. The following glossary terms appear in the source text. You MUST use the " \
               "target-side rendering given here, verbatim, and never paraphrase it:"
        for h in terms:
            sys += f"\n   - “{h['matched']}” -> “{h['target']}”"
    user_parts = []
    if rag_hits:
        user_parts.append("Reference materials (use them for term consistency):")
        for i, h in enumerate(rag_hits, 1):
            user_parts.append(f"[ref{i}] ({h.meta.get('title','')}) {h.chunk_text}")
        user_parts.append("")
    src_label = _lang_name(src) if src != "auto" else "auto-detected"
    user_parts.append(f"Translate from {src_label} to {_lang_name(tgt)}:\n{text}")
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


@router.post("/stream")
def translate_stream(body: TranslateIn):
    text = (body.text or "").strip()
    tgt = body.target_lang
    src = body.source_lang or "auto"

    def gen():
        if not text:
            yield _sse({"type": "error", "message": "待翻译文本为空"})
            return
        if tgt not in LANGUAGES:
            yield _sse({"type": "error", "message": f"不支持的目标语言: {tgt}"})
            return
        if not config.DEEPSEEK_API_KEY:
            yield _sse({"type": "error", "message": "DEEPSEEK_API_KEY 未配置，无法翻译"})
            return

        rag_hits = []
        if body.use_rag:
            rag_hits = retrieval.top_k(text, k=4)
        sources = []
        for h in rag_hits:
            page = int(h.meta.get("page") or 0)
            sources.append({
                "score": round(h.score, 4),
                "source_type": h.source_type,
                "title": h.meta.get("title", h.source_id),
                "url": h.meta.get("url", ""),
                "location": h.meta.get("location", ""),
                "page": page,
                "chunk": h.chunk_text[:300],
            })
        yield _sse({"type": "sources", "items": sources})

        terms = _match_terms(text)
        yield _sse({"type": "terms", "items": terms})

        messages = _build_prompt(text, src, tgt, rag_hits, terms)
        try:
            for kind, piece in llm.chat_stream(messages, temperature=0.2):
                if kind == "content":
                    yield _sse({"type": "token", "text": piece})
            yield _sse({"type": "done"})
        except llm.LLMNotConfigured as e:
            yield _sse({"type": "error", "message": str(e)})
        except Exception as e:
            yield _sse({"type": "error", "message": f"翻译失败: {e}"})

    return StreamingResponse(gen(), media_type="text/event-stream")
