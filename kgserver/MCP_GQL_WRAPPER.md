# Building an MCP Wrapper for the Knowledge Graph GraphQL API

This document describes how to create a **Model Context Protocol (MCP) wrapper** that enables a Large Language Model (LLM) to query a Knowledge Graph GraphQL API. The goal is to make the knowledge graph easily accessible for LLM-based agents, providing entry points, schemas, and guidance for exploration—all exposed via a modern, asynchronous Python web service (FastAPI).

## Table of Contents

- [Overview](#overview)
- [Why Use an MCP Wrapper?](#why-use-an-mcp-wrapper)
- [Architecture](#architecture)
- [Implementation Skeleton](#implementation-skeleton)
- [API Endpoints](#api-endpoints)
- [Suggested LLM Prompts](#suggested-llm-prompts)
- [Example Async GraphQL Query Code](#example-async-graphql-query-code)
- [Security and Rate Limiting](#security-and-rate-limiting)
- [Extending for Traversal (Future)](#extending-for-traversal-future)
- [References](#references)

---

## Overview

The MCP wrapper acts as a translation and orchestration layer between an LLM and the raw GraphQL API. It simplifies certain tasks, offers preconfigured queries, and exposes endpoints that are more "LLM-friendly", e.g., with explicit schema, sample data, and prompt suggestions for common operations.

Features:

- Async HTTP endpoints via FastAPI
- Rich OpenAPI schema for discoverability
- Management of GraphQL endpoint settings and credentials
- Example queries and prompt guidance for LLMs
- Option to constrain or preprocess LLM queries for safety (e.g., limit result set sizes)

---

## Why Use an MCP Wrapper?

- **LLMs benefit from well-scoped, explicit APIs.** You can steer them by exposing clear functions and sample prompts.
- **Security and resource capping.** The wrapper can enforce maximum limits and logging for all queries—important when exposing GraphQL!
- **Schema discovery.** LLMs (and their tools/plugins) often want to discover the available shape of the data; the wrapper can facilitate introspection and snapshots.
- **Central place for prompt suggestions.** Make it easier for prompt designers or agent builders.

---

## Architecture

```
      +---------------+         +-------------------------+
      |   LLM Agent   | <-----> |   MCP FastAPI Wrapper   |
      +---------------+         +-------------------------+
                                              |
                                 GraphQL HTTP Client (Async)
                                              |
                                 +---------------------------+
                                 |   Knowledge Graph Server  |
                                 +---------------------------+
```

---

## Implementation Skeleton

Below is a high-level Python implementation outline. Actual implementations would configure settings for connecting to the GraphQL endpoint, handle credentials, add robust error handling, and log usage metrics.

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import asyncio

app = FastAPI(title="KG MCP Wrapper", description="LLM-oriented wrapper for GraphQL Knowledge Graph API")

GRAPHQL_ENDPOINT = "http://<your_kg_graphql_server>/graphql"  # Set this to your actual endpoint

class GraphQLQuery(BaseModel):
    query: str
    variables: dict = {}

@app.get("/")
async def root():
    return {
        "message": "Knowledge Graph MCP Wrapper. See /docs for OpenAPI schema and try /schema, /example_queries, /llm_prompts.",
    }

@app.post("/query")
async def proxy_graphql_query(payload: GraphQLQuery):
    """General GraphQL query proxy, for use by LLM tools."""
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(
            GRAPHQL_ENDPOINT,
            json=payload.dict(),
            headers={"Content-Type": "application/json"},
        )
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=str(response.text))
        return response.json()

@app.get("/schema")
async def graphql_schema():
    """Return the GraphQL SDL schema as a string (for LLM consumption/discovery)."""
    introspection_query = {
        "query": """
            {
              __schema {
                types { name kind fields { name type { name kind ofType { name kind }}}}
              }
            }
        """
    }
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(GRAPHQL_ENDPOINT, json=introspection_query)
        return response.json()

@app.get("/llm_prompts")
async def suggested_llm_prompts():
    """Pre-made prompt suggestions for using the KG GraphQL API, for LLM tool designers."""
    return {
        "prompts": [
            {
                "task": "Explore the shape of the data",
                "prompt": 'What types and fields are available in this GraphQL schema? List the main entity and relationship fields.'
            },
            {
                "task": "Fetch an entity by ID",
                "prompt": 'Query for an entity by ID: `entity(id: "ENTITY1234") { entityId entityType name }`'
            },
            {
                "task": "Paginate entities",
                "prompt": 'List up to 10 entities: `entities(limit: 10, offset: 0) { items { entityId name } total }`'
            },
            {
                "task": "Filter by exact field",
                "prompt": "Find all entities of type 'Person': `entities(limit: 10, filter: {entityType: \"Person\"}) { items { entityId name } }`"
            },
            {
                "task": "Filter with 'contains'",
                "prompt": "Entities with names containing 'Sherlock': `entities(filter: {nameContains: \"Sherlock\"}) { items { entityId name } }`"
            },
            {
                "task": "Fetch a relationship",
                "prompt": 'Get a relationship between two entities: `relationship(subjectId: "A", predicate: "knows", objectId: "B") { subjectId predicate objectId }`'
            }
        ]
    }

@app.get("/example_queries")
async def example_queries():
    """Example GraphQL queries for clients or LLM to try."""
    return [
        {"description": "Get basic schema info",
         "query": "query { __schema { types { name } } }"},
        {"description": "Fetch entity by ID",
         "query": "query { entity(id: \"ENTITY_ID\") { entityId entityType name } }"},
        {"description": "List entities, paginated",
         "query": "query { entities(limit: 5, offset: 0) { total items { entityId name } } }"},
        {"description": "Filter for type 'Person'",
         "query": "query { entities(filter: {entityType: \"Person\"}) { items { entityId name } } }"},
        {"description": "List relationships for subject",
         "query": "query { relationships(filter: {subjectId: \"ENTITY_ID\"}) { items { subjectId predicate objectId } } }"},
    ]

# For local testing:
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8080)
```

---

## API Endpoints

- `POST /query` — General-purpose GraphQL query proxy; returns raw GraphQL JSON result. The LLM can use this to run arbitrary read-only queries (within safe limits).
- `GET /schema` — Returns the current schema, useful for schema exploration or tools that want to dynamically generate prompts.
- `GET /llm_prompts` — Returns suggested prompt templates and usage patterns.
- `GET /example_queries` — Returns example queries that illustrate common tasks and encourage safe, idiomatic GraphQL usage.
- `GET /` — Welcome/info endpoint.

---

## Suggested LLM Prompts

LLMs (or tools/plugins connected via this MCP) can be provided with the following example prompts to help them access and explore the GraphQL API effectively:

1. **Schema Exploration**

   > "Here is how to explore the shape of the data. Use a GraphQL introspection query to list all entity and relationship fields."

   ```graphql
   {
     __schema {
       types {
         name
         fields { name }
       }
     }
   }
   ```

2. **Fetch Entity By ID**
  
   > "Retrieve an entity by a known ID."

   ```graphql
   query {
     entity(id: "ENTITY_ID") {
       entityId
       entityType
       name
       status
     }
   }
   ```

3. **Paginate Entities**

   > "List up to 5 Person entities supporting pagination."

   ```graphql
   query {
     entities(limit: 5, offset: 0, filter: {entityType: "Person"}) {
       items {
         entityId
         name
       }
       total
       limit
       offset
     }
   }
   ```

4. **Filter by Name**

   > "Find all entities with names containing 'Holmes'."

   ```graphql
   query {
     entities(filter: {nameContains: "Holmes"}) {
       items { entityId name }
     }
   }
   ```

5. **Fetch Relationships**

   > "Get relationships for a specific subject entity."

   ```graphql
   query {
     relationships(filter: {subjectId: "ENTITY_ID"}) {
       items { subjectId predicate objectId }
     }
   }
   ```

---

## Example Async GraphQL Query Code

Here's how the MCP wrapper submits queries to the Knowledge Graph server:

```python
query = """
  query ListEntities($limit: Int, $offset: Int) {
    entities(limit: $limit, offset: $offset) {
      items { entityId name }
      total
    }
  }
"""
variables = {"limit": 10, "offset": 0}
async with httpx.AsyncClient() as client:
    resp = await client.post(GRAPHQL_ENDPOINT, json={"query": query, "variables": variables})
    print(resp.json())
```

---

## Security and Rate Limiting

- All client queries are proxied via `/query` and can be capped for `limit`.
- The wrapper logs any queries exceeding a recommended maximum (e.g., `limit > 100`).
- Only queries, not mutations, are supported. You may block queries with large recursion or unacceptably broad patterns.

---

## Extending for Traversal (Future)

If/when graph traversal is enabled in the backend schema, the wrapper can expose new endpoints, e.g.:

- `/traverse` — Accept traversal-specific queries with configurable depth, result size, and path options.
- Add example traversal queries/prompts: "Find all entities within three hops of ENTITY_X."

Ensure new traversal APIs have strict timeouts and limits to prevent abuse and ensure predictable resource usage.

---

## References

- [GraphQL API Design Principles](GRAPHQL_VIBES.md)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [httpx Async HTTP Client](https://www.python-httpx.org/)
