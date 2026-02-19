#!/bin/bash -xe

# a nice short paper
# PAPER="PMC12771675.xml"

# a bit longer
# PAPER="PMC12756687.xml"

# ten papers
PAPER="PMC12757875.xml,PMC12784210.xml,PMC12784773.xml,PMC12788344.xml,PMC12780394.xml,PMC12757429.xml,PMC12784249.xml,PMC12764803.xml,PMC12783088.xml,PMC12775561.xml"

# twenty papers
# PAPER="PMC12757875.xml,PMC12784210.xml,PMC12784773.xml,PMC12788344.xml,PMC12780394.xml,PMC12757429.xml,PMC12784249.xml,PMC12764803.xml,PMC12783088.xml,PMC12775561.xml,PMC12766194.xml,PMC12750049.xml,PMC12758042.xml,PMC12780067.xml,PMC12785246.xml,PMC12785631.xml,PMC12753587.xml,PMC12754092.xml,PMC12764813.xml,PMC5487382.xml"

# DEBUG=
DEBUG="--debug"
TIMEOUT=1000

uv run python -m examples.medlit.scripts.ingest \
  --input-dir examples/medlit/pmc_xmls \
  --input-papers $PAPER \
  --output-dir medlit_bundle \
  --ollama-timeout $TIMEOUT \
  --use-ollama --trace-all $DEBUG
