#!/bin/bash -xe
# Three-pass medlit ingestion: Pass 1 (LLM extract) → Pass 2 (dedup) → Pass 3 (build kgbundle).
# To ADD more papers to an existing bundle: keep pass1_bundles/ and medlit_merged/ from the first run.
# Then run Pass 1 with --output-dir pass1_bundles and --papers "NEW_PAPER1.xml,NEW_PAPER2.xml" (skips existing),
# then re-run Pass 2 (same --output-dir medlit_merged, same --synonym-cache), then re-run Pass 3.
# See examples/medlit/INGESTION.md § "Adding more papers to an existing bundle".

# a nice short paper
# PAPER="PMC12771675.xml"

# a bit longer
# PAPER="PMC12756687.xml"

# five papers
# PAPER="PMC12757875.xml,PMC12784210.xml,PMC12784773.xml,PMC12788344.xml,PMC12780394.xml"

# ten papers
# PAPER="PMC12757875.xml,PMC12784210.xml,PMC12784773.xml,PMC12788344.xml,PMC12780394.xml,PMC12757429.xml,PMC12784249.xml,PMC12764803.xml,PMC12783088.xml,PMC12775561.xml"

# twenty papers
PAPER="PMC12757875.xml,PMC12784210.xml,PMC12784773.xml,PMC12788344.xml,PMC12780394.xml,PMC12757429.xml,PMC12784249.xml,PMC12764803.xml,PMC12783088.xml,PMC12775561.xml,PMC12766194.xml,PMC12750049.xml,PMC12758042.xml,PMC12780067.xml,PMC12785246.xml,PMC12785631.xml,PMC12753587.xml,PMC12754092.xml,PMC12764813.xml,PMC5487382.xml"

# DEBUG=
DEBUG="--debug"
TIMEOUT=1000

#uv run python -m examples.medlit.scripts.ingest \
#  --input-dir examples/medlit/pmc_xmls \
#  --input-papers $PAPER \
#  --output-dir medlit_bundle \
#  --ollama-timeout $TIMEOUT \
#  --use-ollama --trace-all $DEBUG

uv run python -m examples.medlit.scripts.pass1_extract \
  --input-dir examples/medlit/pmc_xmls \
  --output-dir pass1_bundles \
  --llm-backend anthropic \
  --papers $PAPER

uv run python -m examples.medlit.scripts.pass2_dedup \
  --bundle-dir pass1_bundles \
  --output-dir medlit_merged

uv run python -m examples.medlit.scripts.pass3_build_bundle \
  --merged-dir medlit_merged \
  --bundles-dir pass1_bundles \
  --output-dir medlit_bundle
