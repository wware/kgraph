I want to do some work on the MCP server. You'll find the MCP server
in the `kgserver/mcp_server` directory.

As docker-compose.yml is currently written, the MCP server runs on the
same docker container as the fastAPI endpoints. I'd like to move
it to a separate docker container with a dependency on the fastAPI
container. It should serve on port 8001 to avoid conflict with the
fastAPI port 8000 stuff. Let's use SSE as the protocol, not stdio.

The MCP server relies on storage abstractions shared with the fastAPI
server. I expect very little change there but we need to make sure
the MCP server also receives any configuration info, like where to
find the Postgres database.

## Nginx for MCP

An ngingx spec for the MCP server would look something like this.

```
# FastMCP server (SSE transport)
location /mcp/ {
    proxy_pass http://mcpserver:8001/;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    # SSE requires these — disable buffering so events stream immediately
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 3600s;  # SSE connections are long-lived
}
```

A few things worth noting for SSE specifically:

`proxy_buffering` off is critical. Nginx will otherwise buffer the event stream
and clients won't receive events until the buffer fills — effectively breaking
SSE entirely.

`proxy_read_timeout` needs to be long (or 0 for infinite) since SSE connections
stay open. The default 60s will cause nginx to kill idle connections.

The trailing slash on proxy_pass http://localhost:8001/ matters — with the
slash, nginx strips the `/mcp/` prefix before forwarding, so your FastMCP server
sees requests at `/` rather than `/mcp/`. If FastMCP's SSE endpoint is at
`/sse`, it'll be reachable at `/mcp/sse` externally.
