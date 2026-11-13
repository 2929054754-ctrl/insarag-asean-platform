"""DeepSeek chat client (OpenAI-compatible). Streaming SSE passthrough.

The client does NOT hardcode any key; DEEPSEEK_API_KEY comes from .env.
If the key is missing, callers raise LLMNotConfigured; the UI surfaces a
clear '未配置密钥' state.
"""
from __future__ import annotations

import json
from typing import Iterator

import requests

from .config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


class LLMNotConfigured(RuntimeError):
    pass


def ensure_key() -> None:
    if not DEEPSEEK_API_KEY:
        raise LLMNotConfigured("DEEPSEEK_API_KEY 未配置（请在项目根目录 .env 中设置）")


def chat_stream(messages: list[dict], temperature: float = 0.3) -> Iterator[tuple[str, str]]:
    """Yield (kind, text) tuples from the model.

    kind is 'content' for the visible answer, 'reasoning' for the model's
    internal reasoning_content (deepseek-flash may emit both). Callers decide
    how to surface reasoning; the visible answer is built from 'content'.
    """
    ensure_key()
    url = f"{DEEPSEEK_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    with requests.post(url, headers=headers, json=payload, stream=True, timeout=120) as r:
        if r.status_code != 200:
            body = r.text[:500]
            raise RuntimeError(f"DeepSeek API error {r.status_code}: {body}")
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data:"):
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choice = obj.get("choices", [{}])[0]
                delta = choice.get("delta", {}) or {}
                if delta.get("content"):
                    yield "content", delta["content"]
                if delta.get("reasoning_content"):
                    yield "reasoning", delta["reasoning_content"]


def chat_stream_content(messages: list[dict], temperature: float = 0.3) -> Iterator[str]:
    """Backwards-compatible: yield only visible content pieces."""
    for kind, text in chat_stream(messages, temperature=temperature):
        if kind == "content":
            yield text


def chat_once(messages: list[dict], temperature: float = 0.3) -> str:
    return "".join(p for k, p in chat_stream(messages, temperature=temperature) if k == "content")
