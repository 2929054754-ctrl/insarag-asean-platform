#!/usr/bin/env bash
# Build the standalone static bundle for Cloudflare Pages / any static host.
# Usage: bash build_static.sh   -> produces ./out/
#
# The static frontend talks to an API deployed separately (Vercel/Netlify).
# Before deploying, set window.API_BASE in out/index-patch.html (or inject the
# API URL via your host's SPA config). See README -> "Serverless deploy".
set -euo pipefail
cd "$(dirname "$0")"

rm -rf out
mkdir -p out
cp -r static/* out/

# Cloudflare Pages: serve index.html for /, /rag, /translate, /cms, /voice, /aggregator
cp out/index.html out/404.html 2>/dev/null || true

cat > out/README.txt <<'TXT'
Static bundle built by build_static.sh.
Set window.API_BASE = "https://your-api.example.com" before /static/js/app.js
when the API is hosted separately from this static site.
TXT

echo "static bundle -> out/ (files: $(ls out | tr '\n' ' '))"
