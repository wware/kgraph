# GPU-Accelerated AI Tinkering

Like me, you may get tired of paying subscription fees to use online LLMs.
Especially when, later, you're told that you've reached the usage limit and you
should "switch to another model" or some such nonsense. The tempation at that
point is to run a model locally using Ollama, but your local machine probably
doesn't have a GPU if you're not a gamer. Then you dream of picking up a cheap
GPU box on eBay and running it locally, and that's not a bad idea but it takes
time and money that you may not want to spend right now.

There is an alternative, services like Lambda Labs, RunPod, and others. Lambda
Labs is what I got when I threw a dart at a dartboard, so I'll be using it
here.

I'm using a LLM to translate medical papers into a graph database of entities
and relationships. I set up GPU-accelerated paper ingestion using Lambda Labs,
and got an **enormous speedup** over CPU-only. The quick turnaround made it
practical to find and fix some bugs discovered during testing.

## GPU Instance Setup

### Lambda Labs Instance
- **Instance:** 1x A10 (24 GB PCIe) @ $0.75/hr
- **Why A10:** Perfect balance of cost and performance for LLM inference
- **Setup time:** ~10 minutes
- **Performance:** 5-10 papers/minute vs 1 paper/20+ minutes on CPU

### Configuration
```bash
export OLLAMA_HOST=http://<LAMBDA_IP>:11434
```

Update your `docker-compose.yml` if you're using one, to read `OLLAMA_HOST` from environment.

### Performance Metrics

- **Embedding generation:** 50-100x faster
- **LLM inference:** 10-30x faster  
- **Overall throughput:** 5-10 papers/minute
- **Cost for typical usage:** ~$1.50 (2 hours @ $0.75/hr)

## Launch and Setup

1. **Sign up:** https://lambdalabs.com/service/gpu-cloud
2. **Launch:** 1x A10 (24 GB PCIe) instance
3. **SSH in** and run setup:

```bash
# Update system
sudo apt-get update

# Install Docker
curl -fsSL https://get.docker.com | sudo sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# you might start here if you already have an existing disk image

sudo systemctl restart docker

# Run Ollama with GPU support
sudo docker run -d \
  --gpus=all \
  --name ollama \
  -p 11434:11434 \
  -v ollama:/root/.ollama \
  --restart unless-stopped \
  ollama/ollama

# Pull models (may take a few minutes)
sudo docker exec ollama ollama pull llama3.1:8b
sudo docker exec ollama ollama pull nomic-embed-text
sudo docker exec ollama ollama pull mxbai-embed-large

# Verify GPU is working
sudo docker exec ollama nvidia-smi
```

## Use from Your Laptop

```bash
# Set remote Ollama server
export OLLAMA_HOST=http://<LAMBDA_IP>:11434

# Test connection
curl $OLLAMA_HOST/api/tags

# Clear any existing entity DB (if you had buggy runs)
rm -rf ./data/entity_db

# Run ingestion with GPU acceleration!
cd ingestion/
docker compose up -d postgres
docker compose run ingest \
  python ingest_papers.py \
  --query "metformin diabetes" \
  --limit 10 \
  --model llama3.1:8b
```

The **A10 for $0.75/hr** is a sweet spot for hobby work - you won't need a second mortgage on your house if you forget to terminate it.

## Seeing logs

### Solution: View Docker Container Logs

#### **From the host (outside container):**

```bash
# View live logs (equivalent to journalctl -f) -- what is Ollama doing RIGHT NOW?
sudo docker logs -f ollama

# View last 100 lines
sudo docker logs --tail 100 ollama

# View logs with timestamps
sudo docker logs -f --timestamps ollama

# View logs since a specific time
sudo docker logs --since 10m ollama  # Last 10 minutes
```

### Understanding Your Docker Setup

You have Ollama running in a Docker container named `ollama`. The container doesn't have systemd, so `journalctl` won't work. Instead, Docker captures all stdout/stderr from the container, which you access via `docker logs`.

### Common Ollama Docker Log Commands

```bash
# See what model is currently loaded
sudo docker logs ollama | grep "llama runner started"

# Check for errors
sudo docker logs ollama | grep -i error

# See recent API requests
sudo docker logs ollama | tail -50

# Monitor live activity
sudo docker logs -f ollama
```

### If You Want More Detailed Logging

If the logs aren't showing enough detail, you can restart the container with debug logging:

```bash
# Stop current container
sudo docker stop ollama

# Start with debug logging
sudo docker run -d \
  --name ollama \
  --gpus all \
  -v ollama:/root/.ollama \
  -p 11434:11434 \
  -e OLLAMA_DEBUG=1 \
  ollama/ollama

# Now follow logs
sudo docker logs -f ollama
```

### Quick Reference

| What you want | Command |
|---------------|---------|
| Live logs (like `journalctl -f`) | `sudo docker logs -f ollama` |
| Last 100 lines | `sudo docker logs --tail 100 ollama` |
| All logs | `sudo docker logs ollama` |
| Logs from last 10 min | `sudo docker logs --since 10m ollama` |
| Logs with timestamps | `sudo docker logs -f --timestamps ollama` |

**TL;DR**: Use `sudo docker logs -f ollama` to see live logs from your Ollama container. This is the Docker equivalent of `journalctl -f -u ollama`.

## Cleanup

It's simplest to terminate the instance from the Lambda Labs dashboard.

## How to set up an SSH TUNNEL for this

To set up an SSH tunnel for port 11434 as a background process, use the  command with the `-f`, `-N`, and `-L` options. [1, 2]  
The general command syntax is: 

```bash
ssh -fN -L <local_port>:localhost:<remote_port> <user>@<remote_host>
```

Specific Command for Port 11434 
For port 11434, the command is: 

```bash
ssh -fN -L 11434:localhost:11434 ubuntu@123.45.67.89
```

Options Explained 

• `-f` : Requests `ssh` to go to the background just before command execution. This is useful for running the tunnel as a daemon. It also waits until the connection and port forwarding have been successfully established before moving to the background. 
• `-N` : Instructs `ssh` to not execute a remote command. This is useful for just forwarding ports. 
• `-L <local_port>:localhost:<remote_port>` : Specifies local port forwarding. Traffic sent to the specified  (e.g.,  on your machine) will be forwarded to the specified  (e.g.,  on the remote server's localhost) through the secure SSH tunnel. 
• Your username and the hostname or IP address of the remote server you are connecting to. [1, 2, 3, 4, 5]  

How to Use 

1. Open your terminal or PowerShell (Windows 10 and later have a built-in OpenSSH client). 
2. Run the command, replacing  and  with your actual credentials. 
3. Enter your SSH password or passphrase when prompted. If you use SSH keys, you won't need to enter a password. 
4. The tunnel will start and run in the background. You will return to your command prompt immediately. [6, 7, 8, 9, 10]  

To Stop the Tunnel 
Since the process runs in the background, you'll need to find its process ID (PID) to terminate it. 

1. List SSH processes to find the PID: 
   ```bash
   pgrep -f "ssh -fN -L 11434:localhost:11434"
   ```
2. Kill the process using its PID [14, 15]  

For easier management, consider using SSH control sockets, especially if you need to manage multiple tunnels within a script. [16]  

[1] https://www.revsys.com/writings/quicktips/ssh-tunnel.html
[2] https://superuser.com/questions/827934/ssh-port-forwarding-without-session
[3] https://4sysops.com/archives/access-remote-ollama-ai-models-through-an-ssh-tunnel/
[4] https://blog.sysxplore.com/p/part-1-ssh-local-port-forwarding
[5] https://gist.github.com/scy/6781836
[6] https://stackoverflow.com/questions/36094597/run-ssh-port-forwarding-in-background-before-connecting-to-mysql-server
[7] https://linuxize.com/post/how-to-setup-ssh-tunneling/
[8] https://pawsey.atlassian.net/wiki/spaces/US/pages/678494209/Use+of+standard+SSH+client+to+connect
[9] https://www.ionos.co.uk/digitalguide/server/configuration/powershell-ssh/
[10] https://docs.oracle.com/cd/E95619_01/html/esbc_ecz810_configuration/GUID-972B466B-D21B-42C4-8CA7-D4D6126AB8B9.htm
[11] https://egghead.io/lessons/bash-configure-local-and-remote-port-forwarding-with-an-ssh-tunnel
[12] https://www.digitalocean.com/community/tutorials/securing-communications-three-tier-rails-application-using-ssh-tunnels
[13] https://ramesh-sahoo.medium.com/accessing-remote-servers-via-ssh-tunnel-reverse-ssh-tunnels-ssh-socks-proxy-ssh-via-a-jump-5e5cfddaf2a3
[14] https://networktocode.com/blog/how-to-create-an-ssh-tunnel-via-command-line/
[15] https://www.digitalocean.com/community/tutorials/how-to-install-and-run-a-node-js-app-on-centos-6-4-64bit
[16] https://gist.github.com/netsensei/834aad8e9b8d503748f2400ce23c50e3


