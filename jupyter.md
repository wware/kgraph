# Jupyter notebook

This is a quick easy way to experiment with things. I want to deal
with some quality issues in the ingestion step and this will help.

- Miscategroized entities, like bacteria being classified as diseases.
  You can have an infection of a particular bacterial strain but the
  bacterium is not itself a disease.
- Relationships need more work, too many are `associated_with` meaning
  that we didn't figure out the nature of the relationship and we
  didn't take advantage of the rich collection of different relationship
  types that we have representations for.

```bash
$ docker compose build jupyter
$ docker compose --profile jupyter up -d
```

The Jupyter notebook will appear on port 8888 of a new container.

## Nginx


### Prompt:
2/10/2026, 11:17:07 AM

When I deployed this to a Digital Ocean droplet, I used nginx to map the server's port 8000 endpoint to port 80, but I misplaced my notes about how I did that. Can you help me reconstruct that?  I would also like to take port 8888 from the Jupyter server and map it to an endpoint in the kgserver fastAPI setup.



### Response:
2/10/2026, 11:17:41 AM

````plaintext
Thought process: Synthesized nginx configuration strategy and FastAPI proxy integration approach.

The user wants two things:

1. Reconstruct the nginx configuration that maps the kgserver FastAPI app (running on port 8000) to port 80
2. Add a new endpoint in the FastAPI app that proxies/exposes the Jupyter server running on port 8888

Let me help with both.

For nginx, a typical setup would be:
- Install nginx
- Create a server block/configuration file
- Proxy requests from port 80 to localhost:8000
- Maybe add SSL with Let's Encrypt

For mapping Jupyter to a FastAPI endpoint, there are a few approaches:
1. Use nginx to also proxy /jupyter to port 8888
2. Add a reverse proxy route in FastAPI itself
3. Mount Jupyter at a subpath

I think the user wants:
1. The nginx config to expose port 8000 on port 80
2. Possibly also expose Jupyter on a path like /jupyter or similar through either nginx or FastAPI

Let me provide both the nginx config and show how to add a Jupyter endpoint to FastAPI.
````

I'll help you with both. Here's how to reconstruct your nginx setup and add Jupyter access through your FastAPI server.

### Nginx Configuration for Port 80 → 8000

**Install nginx (if not already installed):**
```bash
sudo apt update
sudo apt install nginx
```

**Create nginx configuration** (`/etc/nginx/sites-available/kgserver`):

```nginx
server {
    listen 80;
    server_name _;
    # server_name 104.131.171.175;

    # Increase client body size for file uploads
    client_max_body_size 100M;

    # Main API
    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;

        # Timeouts for long-running requests
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    # Jupyter Lab (WebSocket support needed)
    location /jupyter/ {
        proxy_pass http://localhost:8888/jupyter/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Scheme $scheme;
        proxy_buffering off;
    }
}
```

**Enable the site:**
```bash
sudo ln -s /etc/nginx/sites-available/kgserver /etc/nginx/sites-enabled/
sudo nginx -t  # Test configuration
sudo systemctl restart nginx
```


### For Production: Add SSL with Let's Encrypt

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

This will automatically update your nginx config to handle HTTPS.

**Updated nginx with SSL:**
```nginx
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    # ... rest of configuration same as above
}
```

### Access URLs

After setup:
- API: `http://104.131.171.175/` (or `https://yourdomain.com/`)
- Docs: `http://104.131.171.175/docs`
- GraphiQL: `http://104.131.171.175/graphiql/`
- Jupyter: `http://104.131.171.175/jupyter/`
- Graph Viz: `http://104.131.171.175/graph-viz/`

### Security Consideration

Since you're exposing Jupyter publicly, you should either:

**Option A: Add authentication to nginx:**
```bash
sudo apt install apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd your_username
```

Then in nginx config:
```nginx
location /jupyter/ {
    auth_basic "Restricted Access";
    auth_basic_user_file /etc/nginx/.htpasswd;

    # ... rest of proxy config
}
```

## Useful commands on the remote server

The user on this server is `root` so we don't really need any `sudo` in front of commands.

```bash
vim /etc/nginx/sites-available/kgraph
docker-compose build api
docker-compose build jupyter
docker-compose down; rm -rf postgres-data/* ; docker-compose --profile api up -d
# save the manual tweaks as a tarball
(cd ..; tar cfz KGRAPH.tgz --exclude kgraph/.venv --exclude kgraph/postgres-data --exclude kgraph/.git kgraph/)
# use scp to fetch it off the server
ln -s /etc/nginx/sites-available/kgserver /etc/nginx/sites-enabled/
nginx -t  # Test configuration
systemctl restart nginx
```
