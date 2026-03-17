#!/bin/bash -xe

# The idea here is that you have a bunch of per-paper JSONs already waiting in extracted/
#     $ ls extracted/
#     paper_PMC10667925.json  paper_PMC11795198.json  paper_PMC12754092.json  ...

PAPER=$(ls extracted/ | sed 's/paper_//' | sed 's/\.json/.xml/' | tr '\n' ',' | sed 's/,$//')
# DEBUG=
DEBUG="--debug"
TIMEOUT=1000

uv run python -m examples.medlit.scripts.ingest \
  --bundle-dir extracted \
  --output-dir merged \
  --synonym-cache merged/synonym_cache.json

uv run python -m examples.medlit.scripts.build_bundle \
  --merged-dir merged \
  --bundles-dir extracted \
  --output-dir bundle
