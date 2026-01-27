"""
Example GraphQL queries for the Knowledge Graph API.

These queries are displayed in the GraphiQL interface to help users get started.
When a bundle provides its own ``graphql_examples.yml``, that file replaces
the built-in examples at startup.
"""

import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_DEFAULT_PATH = _HERE / "graphql_examples.yml"

EXAMPLE_QUERIES: dict[str, str] = {}
DEFAULT_QUERY: str = ""


def load_examples(path: Path | None = None) -> None:
    """Load (or reload) example queries from a YAML file.

    Args:
        path: Path to a ``graphql_examples.yml`` file.
              When *None*, the built-in default is used.
    """
    global DEFAULT_QUERY
    source = path if path is not None else _DEFAULT_PATH
    data = yaml.safe_load(source.read_text())
    EXAMPLE_QUERIES.clear()
    EXAMPLE_QUERIES.update(data)
    DEFAULT_QUERY = EXAMPLE_QUERIES.get("Search Entities", next(iter(EXAMPLE_QUERIES.values()), ""))
    logger.info("Loaded %d example queries from %s", len(EXAMPLE_QUERIES), source)


def get_examples() -> dict[str, str]:
    """Return the current example queries dict."""
    return EXAMPLE_QUERIES


def get_default_query() -> str:
    """Return the current default query string."""
    return DEFAULT_QUERY


# Load built-in defaults on first import
load_examples()
