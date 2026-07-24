"""WSGI entrypoint for production servers (e.g. gunicorn).

Run with:
    gunicorn --bind 0.0.0.0:8050 wsgi:server

Dash builds on Flask; ``app.server`` is the WSGI application object that a
production server serves. The interactive Dash dev server (``app.run``) is for
local development only.
"""
from __future__ import annotations

from src.dashboard.app import create_app

app = create_app()
server = app.server
