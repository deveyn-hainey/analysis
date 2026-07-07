"""Uvicorn entrypoint — kept so `uvicorn app:app` keeps working.

All logic lives in the soccer_vision package; see soccer_vision/__init__.py
for the module map.
"""
from soccer_vision.api import app

__all__ = ["app"]
