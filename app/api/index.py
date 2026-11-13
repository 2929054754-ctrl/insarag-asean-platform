"""Vercel serverless entrypoint.

Vercel Python functions look for an ASGI/WSGI app named `app` in api/*.py.
This module simply re-exports the FastAPI app built in main.py; all routers
live alongside it in the same `api/` package and are wired by main.py.

Deploy (project root = app/):
  vercel.json rewrites every path to /api/index, so the function serves both
  /api/* and the static frontend (main.py mounts /static and the HTML pages).
"""
import sys
from pathlib import Path

# Ensure project root is importable on the Vercel builder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import app  # noqa: E402,F401
