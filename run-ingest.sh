#!/bin/bash -xe

# PAPER="PMC12771675.xml"
PAPER="PMC12756687.xml"
TIMEOUT=300

uv run python -m examples.medlit.scripts.ingest \
  --input-dir examples/medlit/pmc_xmls \
  --input-papers $PAPER \
  --output-dir medlit_bundle \
  --ollama-timeout $TIMEOUT \
  --use-ollama --trace-all --debug
