#!/usr/bin/env bash
# Build the standalone static bundle for Netlify / any static host.
# Usage: bash build_static.sh   -> produces ./out/
#
# HTML pages live at the root; CSS/JS keep the /static prefix referenced by
# the pages, so they land in out/static/. The API is same-origin (/api/*).
set -euo pipefail
cd "$(dirname "$0")"

rm -rf out
mkdir -p out/static

# Pages at root
cp static/*.html out/

# Assets keep the /static prefix
cp -r static/css static/js out/static/

# Pretty-URL fallback: unknown paths serve index.html (SPA-style)
cp out/index.html out/404.html

cat > out/README.txt <<'TXT'
Static bundle built by build_static.sh.
Pages at root; assets under /static. API is same-origin /api/* (JS function).
TXT

echo "static bundle -> out/ (files: $(ls out | tr '\n' ' '))"
