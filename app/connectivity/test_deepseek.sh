#!/usr/bin/env bash
# DeepSeek 联通性测试 (curl)。
# Key 从环境变量 DEEPSEEK_API_KEY 读取（由上层写入 app/.env，本脚本不硬编码、不打印）。
# 用法:
#   set -a; source app/.env; set +a
#   bash app/connectivity/test_deepseek.sh
set -euo pipefail

BASE_URL="${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"
MODEL="${DEEPSEEK_MODEL:-deepseek-flash}"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "[ERROR] DEEPSEEK_API_KEY 未设置，请先 source app/.env" >&2
  exit 1
fi

echo "==> POST ${BASE_URL}/chat/completions  (model=${MODEL})"

curl -sS -i "${BASE_URL}/chat/completions" \
  -H "Authorization: Bearer ${DEEPSEEK_API_KEY}" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "model": "'"${MODEL}"'",
    "messages": [
      {"role": "system", "content": "You are a connectivity probe. Reply with one short sentence."},
      {"role": "user", "content": "Reply with exactly: PONG"}
    ],
    "temperature": 0.0,
    "stream": false
  }' | sed -E 's/(Bearer )[A-Za-z0-9_\-\.]+/\1$DEEPSEEK_API_KEY/g'

echo
echo "==> 完成"
