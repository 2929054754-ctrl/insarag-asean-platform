"""Netlify Functions entrypoint (ASGI via @netlify/plugin-nextjs-style).

Netlify Python functions expect an ASGI app named `app`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from main import app  # noqa: E402,F401
