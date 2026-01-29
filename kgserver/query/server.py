import logging
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Depends
from fastapi.staticfiles import StaticFiles
import strawberry
from strawberry.fastapi import GraphQLRouter

from .storage_factory import close_storage, get_engine, get_storage
from .bundle_loader import load_bundle_at_startup
from .routers import rest_api
from .routers import graphiql_custom
from .graphql_schema import Query
from storage.interfaces import StorageInterface
from mcp_server import mcp_server

# Let's take this opportunity to do the mkdocs build
logging.basicConfig(format="%(levelname)s:     %(asctime)s - %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger()
result = subprocess.run(["uv", "run", "mkdocs", "build"], check=False)
if result.returncode != 0:
    logger.error("MKDOCS BUILD FAILED")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Initializes storage on startup, loads bundle if configured, and closes on shutdown.
    """
    engine, db_url = get_engine()  # Initialize storage
    load_bundle_at_startup(engine, db_url)  # Load bundle if BUNDLE_PATH is set
    yield
    close_storage()  # Close storage connections


app = FastAPI(
    title="Medical Literature Knowledge Graph API",
    description="A read-only API for querying the medical literature knowledge graph.",
    version="0.1.0",
    lifespan=lifespan,
)

# Mount REST API
app.include_router(rest_api.router)


# Create a context getter that uses your dependency
async def get_context(
    storage: StorageInterface = Depends(get_storage),
):
    return {
        "storage": storage,
    }


# Mount GraphQL with context (no built-in GraphiQL)
graphql_schema = strawberry.Schema(query=Query)
graphql_app = GraphQLRouter(
    graphql_schema,
    graphiql=False,  # Disable built-in GraphiQL
    context_getter=get_context,
)

app.include_router(graphql_app, prefix="/graphql")

# Mount custom GraphiQL interface with example queries
app.include_router(graphiql_custom.router, prefix="/graphiql", tags=["GraphQL"])

# Mount MCP API (Server-Sent Events and HTTP endpoints)
# FastMCP uses http_app() method with transport parameter
mcp_sse_app = mcp_server.http_app(path="/sse", transport="sse")
mcp_http_app = mcp_server.http_app(path="/mcp", transport="streamable-http")
app.mount("/mcp/sse", mcp_sse_app)
app.mount("/mcp", mcp_http_app)


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify that the server is running.
    """
    return {"status": "ok"}


# Mount MkDocs static site at /mkdocs if available
# Built during Docker image creation via: mkdocs build
_mkdocs_site = Path(__file__).parent.parent / "site"
if _mkdocs_site.exists():
    app.mount("/mkdocs", StaticFiles(directory=_mkdocs_site, html=True), name="mkdocs")
