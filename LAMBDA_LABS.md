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

## Cleanup

It's simplest to terminate the instance from the Lambda Labs dashboard.
