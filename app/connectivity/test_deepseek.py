#!/usr/bin/env python3
"""DeepSeek 联通性测试 (requests)。

Key 从 app/.env 读取 (python-dotenv)，不硬编码、不打印到日志。
打印 HTTP 状态码与返回内容，便于核对联通性。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent.parent
load_dotenv(APP_DIR / ".env")

BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()


def mask(headers: dict) -> dict:
    h = dict(headers)
    if "Authorization" in h:
        h["Authorization"] = "Bearer $DEEPSEEK_API_KEY"
    return h


def main() -> int:
    if not KEY:
        print("[ERROR] DEEPSEEK_API_KEY 未配置（请在 app/.env 中设置）")
        return 1

    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a connectivity probe. Reply with one short sentence."},
            {"role": "user", "content": "Reply with exactly: PONG"},
        ],
        "temperature": 0.0,
        "stream": False,
    }

    print(f"==> POST {url}  (model={MODEL})")
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
    except requests.RequestException as e:
        print(f"[ERROR] 请求失败: {e}")
        return 2

    print(f"HTTP {r.status_code}")
    print("Response headers (masked):", json.dumps(mask(dict(r.headers)), ensure_ascii=False, indent=2))
    try:
        body = r.json()
    except Exception:
        print("Raw body:", r.text[:800])
        return 3

    print("Response body:")
    print(json.dumps(body, ensure_ascii=False, indent=2))

    if r.status_code == 200 and body.get("choices"):
        content = body["choices"][0]["message"].get("content", "")
        print(f"\n[OK] 模型回执: {content!r}")
        return 0
    print("\n[FAIL] 未拿到 choices")
    return 4


if __name__ == "__main__":
    sys.exit(main())
