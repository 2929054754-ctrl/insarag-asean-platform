"""RAG Q&A with SSE streaming.

Flow:
  question -> embed -> numpy cosine Top-K over four libraries
  (news / aggregator / corpus / term) -> assemble context ->
  DeepSeek stream -> SSE tokens (content/reasoning) -> sources event.

Guardrails:
  * source_types filter lets the caller restrict which libraries are searched.
  * If no hit or the best cosine score < RAG_MIN_SCORE, we explicitly refuse to
    answer ("知识库中没有可依据的内容，不臆测") and do NOT call the LLM.
"""
from __future__ import annotations

import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core import config, db, llm, retrieval

router = APIRouter(prefix="/api/rag", tags=["rag"])

VALID_TYPES = ("news", "aggregator", "corpus", "term")


class AskIn(BaseModel):
    question: str
    top_k: int = 5
    source_types: list[str] | None = None
    min_score: float | None = None


def _sources_payload(hits) -> list[dict]:
    out = []
    for h in hits:
        page = int(h.meta.get("page") or 0)
        out.append({
            "score": round(h.score, 4),
            "source_type": h.source_type,
            "source_id": h.source_id,
            "chunk_index": h.chunk_index,
            "title": h.meta.get("title", h.source_id),
            "url": h.meta.get("url", ""),
            "location": h.meta.get("location", ""),
            "domain": h.meta.get("domain", ""),
            "page": page,
            "page_label": f"p.{page}" if page else "",
            "chunk": h.chunk_text[:400],
        })
    return out


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _norm_types(types: list[str] | None) -> list[str] | None:
    if not types:
        return None
    ok = [t for t in types if t in VALID_TYPES]
    return ok or None


@router.post("/ask")
def ask(body: AskIn):
    q = (body.question or "").strip()
    top_k = max(1, min(body.top_k, 10))
    min_score = body.min_score if body.min_score is not None else config.RAG_MIN_SCORE

    def gen():
        if not q:
            yield _sse({"type": "error", "message": "问题为空"})
            return
        if not config.DEEPSEEK_API_KEY:
            yield _sse({"type": "error", "message": "DEEPSEEK_API_KEY 未配置，无法生成答案（检索仍可用）"})
            return

        hits = retrieval.top_k(q, k=top_k, source_types=_norm_types(body.source_types))
        yield _sse({"type": "sources", "items": _sources_payload(hits)})

        best = hits[0].score if hits else 0.0
        if not hits or best < min_score:
            yield _sse({
                "type": "refuse",
                "text": (
                    "知识库中未检索到与该问题相关、且可信度足够的上下文"
                    f"（最高相关度 {best:.2f}，阈值 {min_score}）。"
                    "为避免误导，我不做臆测回答。请尝试换一个更贴近 INSARAG/东盟搜救"
                    "领域的问题，或先在 CMS 中补充相关资料。"
                ),
            })
            yield _sse({"type": "done"})
            return

        ctx_lines = []
        for i, h in enumerate(hits, 1):
            page = int(h.meta.get("page") or 0)
            loc = h.meta.get("location", "") + (f" p.{page}" if page else "")
            ctx_lines.append(
                f"[资料{i}] 来源：{h.meta.get('title','')} ({h.source_type}, {loc})\n{h.chunk_text}"
            )
        context = "\n\n".join(ctx_lines)

        system = (
            "你是 INSARAG/东盟多语种搜救知识库的问答助手。规则：\n"
            "1. 严格依据下面给定的上下文回答；上下文没有提到的内容，一律说"
            "“知识库中没有相关依据”，禁止编造、禁止用常识补全。\n"
            "2. 回答用与用户问题相同的语言，保持专业、简洁。\n"
            "3. 答案末尾用“参考：”列出引用的资料编号（资料1/资料2…）。"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"知识库上下文：\n{context}\n\n用户问题：{q}"},
        ]
        try:
            for kind, text in llm.chat_stream(messages, temperature=0.3):
                if kind == "content":
                    yield _sse({"type": "token", "text": text})
                elif kind == "reasoning":
                    yield _sse({"type": "reasoning", "text": text})
            yield _sse({"type": "done"})
        except llm.LLMNotConfigured as e:
            yield _sse({"type": "error", "message": str(e)})
        except Exception as e:
            yield _sse({"type": "error", "message": f"模型调用失败: {e}"})

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/retrieve")
def retrieve(body: AskIn):
    """Pure retrieval (no LLM) — for verifying cosine similarity reproducibility."""
    q = (body.question or "").strip()
    hits = retrieval.top_k(
        q, k=max(1, min(body.top_k, 10)), source_types=_norm_types(body.source_types)
    )
    return {
        "backend": retrieval.stats().get("backend"),
        "dim": retrieval.stats().get("dim"),
        "question": q,
        "min_score": config.RAG_MIN_SCORE,
        "items": _sources_payload(hits),
    }


@router.get("/libraries")
def libraries():
    """List available libraries + counts for UI filters."""
    with db.get_conn() as conn:
        counts = {
            "news": conn.execute("SELECT COUNT(*) c FROM news").fetchone()["c"],
            "aggregator": conn.execute("SELECT COUNT(*) c FROM aggregator").fetchone()["c"],
            "corpus": conn.execute("SELECT COUNT(*) c FROM corpus").fetchone()["c"],
            "term": conn.execute("SELECT COUNT(*) c FROM terminology").fetchone()["c"],
        }
        domains = sorted({
            r[0] for tbl in ("news", "aggregator", "corpus", "terminology")
            for r in conn.execute(f"SELECT DISTINCT domain FROM {tbl} WHERE domain <> ''")
        })
    return {"types": VALID_TYPES, "counts": counts, "domains": domains}
