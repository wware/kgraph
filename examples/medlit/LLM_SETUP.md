# Pass 1 LLM backends and configuration

Pass 1 (extraction) requires an LLM to produce per-paper bundle JSON. You can use any of the backends below. Do **not** commit API keys; use `.env` or environment variables and add `.env` to `.gitignore` if present.

## Quick start: Claude with `.env`

1. Put your key in a `.env` file at the **repo root** (e.g. `kgraph/.env`):
   ```bash
   ANTHROPIC_API_KEY=sk-ant-...
   ```
2. From the repo root, run the two-pass pipeline:
   ```bash
   uv run python -m examples.medlit.scripts.pass1_extract \
     --input-dir examples/medlit/pmc_xmls \
     --output-dir pass1_bundles \
     --llm-backend anthropic \
     --papers "PMC127*.xml"
   uv run python -m examples.medlit.scripts.pass2_dedup \
     --bundle-dir pass1_bundles \
     --output-dir medlit_merged
   ```
   Pass 1 loads `.env` automatically (via `python-dotenv`). Use `--papers "PMC127*.xml"` to run a single paper, or a comma-separated list of globs (e.g. `--papers "PMC127*.xml,PMC128*.json"`). Optional: `--limit N` to cap how many papers are processed.

## Backend summary

| Backend    | Env vars / flags                    | When to use                          |
|-----------|--------------------------------------|--------------------------------------|
| **anthropic** | `ANTHROPIC_API_KEY`, optional `ANTHROPIC_MODEL` | You have an Anthropic (Claude) API key |
| **openai**   | `OPENAI_API_KEY`, optional `OPENAI_API_BASE_URL`, `OPENAI_MODEL` | OpenAI API or Lambda Labs (OpenAI-compatible) |
| **ollama**   | optional `OLLAMA_HOST`, `OLLAMA_MODEL`         | Local Ollama server (e.g. `ollama serve`)     |

## 1. Anthropic (Claude) — API key

- Set your API key:
  - **Recommended:** put `ANTHROPIC_API_KEY=sk-ant-...` in a `.env` file in the project root and load it (e.g. `python-dotenv` or export before running). Ensure `.env` is in `.gitignore`.
  - Or export in the shell: `export ANTHROPIC_API_KEY=sk-ant-...`
- Run Pass 1:
  ```bash
  python -m examples.medlit.scripts.pass1_extract --input-dir pmc_xmls/ --output-dir bundles/ --llm-backend anthropic
  ```
- Optional: set model with `ANTHROPIC_MODEL` or `--model` (if the script supports it), e.g. `ANTHROPIC_MODEL=claude-sonnet-4-20250514`.
- See [Anthropic API keys and docs](https://docs.anthropic.com/) for rate limits and usage.

## 2. Lambda Labs (or other OpenAI-compatible endpoint)

- Run your LLM on a Lambda Labs instance (or any server) that exposes an **OpenAI-compatible** API.
- Set:
  - `OPENAI_API_BASE_URL` — base URL of the API (e.g. `https://your-lambda-instance/v1`).
  - `OPENAI_API_KEY` — API key if the endpoint requires one.
- Run Pass 1:
  ```bash
  export OPENAI_API_BASE_URL=https://your-instance/v1
  export OPENAI_API_KEY=your-key
  python -m examples.medlit.scripts.pass1_extract --input-dir pmc_xmls/ --output-dir bundles/ --llm-backend openai
  ```
- This was the previous setup (“as before”) for users who already have a Lambda instance.

## 3. OpenAI (ChatGPT)

- Set `OPENAI_API_KEY` (from [OpenAI API keys](https://platform.openai.com/api-keys)).
- Run Pass 1:
  ```bash
  python -m examples.medlit.scripts.pass1_extract --input-dir pmc_xmls/ --output-dir bundles/ --llm-backend openai
  ```
- Optional: set `OPENAI_MODEL` (e.g. `gpt-4o`).

## 4. Ollama (local)

- Start Ollama and pull a model, e.g. `ollama serve` and `ollama pull llama3.1:8b`.
- Run Pass 1 (default backend is `ollama` if `LLM_BACKEND` is not set):
  ```bash
  python -m examples.medlit.scripts.pass1_extract --input-dir pmc_xmls/ --output-dir bundles/ --llm-backend ollama
  ```
- Optional: `OLLAMA_HOST` (default `http://localhost:11434`), `OLLAMA_MODEL` (default `llama3.1:8b`).

## Dependencies

- **anthropic:** `pip install anthropic`
- **openai:** `pip install openai`
- **ollama:** already in project dependencies (`ollama` package).
