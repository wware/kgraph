# Entry point for uvicorn: uvicorn main:app
from query.server import app

__all__ = ["app"]
