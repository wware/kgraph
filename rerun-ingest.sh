#!/bin/bash -xe

# The idea here is that you have a bunch of per-paper JSONs already waiting in pass1_bundles/
#     $ ls pass1_bundles/
#     paper_PMC10667925.json  paper_PMC11795198.json  paper_PMC12754092.json  ...

PAPER=$(ls pass1_bundles/ | sed 's/paper_//' | sed 's/\.json/.xml/' | tr '\n' ',' | sed 's/,$//')
# DEBUG=
DEBUG="--debug"
TIMEOUT=1000

uv run python -m examples.medlit.scripts.pass2_dedup \
  --bundle-dir pass1_bundles \
  --output-dir medlit_merged \
  --synonym-cache medlit_merged/synonym_cache.json

uv run python -m examples.medlit.scripts.pass3_build_bundle \
  --merged-dir medlit_merged \
  --bundles-dir pass1_bundles \
  --output-dir medlit_bundle
