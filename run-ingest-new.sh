#!/bin/bash -e
# Three-pass medlit ingestion using the PostgresIdentityServer for ingest.
# Requires Postgres to be running with the host port exposed (local profile only).
#
# This is the new pipeline counterpart to run-ingest.sh.  The only difference
# is that ingest uses --use-identity-server instead of the legacy file-based
# synonym cache and authority lookup chain.  fetch_vocab, extract, and build_bundle are unchanged.
#
# Prerequisite:
#   export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/kgserver
#   docker-compose --profile local up -d postgres-local   # start Postgres with host port exposed
#
# Usage:
#   ./run-ingest-new.sh --list              Show available paper lists and descriptions
#   ./run-ingest-new.sh --list <name>       Ingest using the named list
#
# To compare against the legacy pipeline, run both scripts on the same list and diff
# the outputs:
#   diff <(jq -r '.[].entity_id' merged/entities.json | sort) \
#        <(jq -r '.[].entity_id' merged_new/entities.json | sort)

# -----------------------------------------------------------------------------
# Paper lists: name -> "description|PMC1.xml,PMC2.xml,..."
# (kept in sync with run-ingest.sh)
# -----------------------------------------------------------------------------
_get_list() {
    case "$1" in
        short)       echo "Single short paper (quick smoke test)|PMC12771675.xml" ;;
        medium)      echo "Single medium-length paper|PMC12756687.xml" ;;
        five)        echo "Five papers|PMC12757875.xml,PMC12784210.xml,PMC12784773.xml,PMC12788344.xml,PMC12780394.xml" ;;
        ten)         echo "Ten papers|PMC12757875.xml,PMC12784210.xml,PMC12784773.xml,PMC12788344.xml,PMC12780394.xml,PMC12757429.xml,PMC12784249.xml,PMC12764803.xml,PMC12783088.xml,PMC12775561.xml" ;;
        ten-hpylori) echo "Ten papers including H. pylori/gastric cancer|PMC12766194.xml,PMC12750049.xml,PMC12758042.xml,PMC12780067.xml,PMC12785246.xml,PMC12785631.xml,PMC12753587.xml,PMC12754092.xml,PMC12764813.xml,PMC5487382.xml" ;;
        twenty)      echo "Twenty papers|PMC12757875.xml,PMC12784210.xml,PMC12784773.xml,PMC12788344.xml,PMC12780394.xml,PMC12757429.xml,PMC12784249.xml,PMC12764803.xml,PMC12783088.xml,PMC12775561.xml,PMC12766194.xml,PMC12750049.xml,PMC12758042.xml,PMC12780067.xml,PMC12785246.xml,PMC12785631.xml,PMC12753587.xml,PMC12754092.xml,PMC12764813.xml,PMC5487382.xml" ;;
        cushing)     echo "Endocrinology/Cushing's syndrome (10 papers)|PMC11560769.xml,PMC11779774.xml,PMC11548364.xml,PMC2386281.xml,PMC12187266.xml,PMC4374115.xml,PMC11128938.xml,PMC11685751.xml,PMC12035109.xml,PMC12055610.xml" ;;
        cushing-6)   echo "Cushing's subset (6 papers)|PMC11560769.xml,PMC11779774.xml,PMC11548364.xml,PMC2386281.xml,PMC12187266.xml,PMC4374115.xml" ;;
        adrenal)     echo "Adrenal topics|PMC10667925.xml,PMC4880116.xml,PMC11795198.xml" ;;
        ferroptosis) echo "Ferroptosis pair (cross-citation test)|PMC12784773.xml,PMC12750178.xml" ;;
        misc)        echo "Misc oncology/other|PMC6727998.xml,PMC4192497.xml,PMC4480270.xml,PMC3607291.xml,PMC4398279.xml,PMC5579818.xml" ;;
        smorgasbord) echo "39 papers across oncology and endocrinology|PMC10667925.xml,PMC11128938.xml,PMC11548364.xml,PMC11560769.xml,PMC11685751.xml,PMC11779774.xml,PMC11795198.xml,PMC12035109.xml,PMC12055610.xml,PMC12187266.xml,PMC12750049.xml,PMC12753587.xml,PMC12754092.xml,PMC12757429.xml,PMC12757875.xml,PMC12758042.xml,PMC12764803.xml,PMC12764813.xml,PMC12766194.xml,PMC12775561.xml,PMC12780067.xml,PMC12780394.xml,PMC12783088.xml,PMC12784210.xml,PMC12784249.xml,PMC12784773.xml,PMC12785246.xml,PMC12785631.xml,PMC12788344.xml,PMC2386281.xml,PMC3607291.xml,PMC4192497.xml,PMC4374115.xml,PMC4398279.xml,PMC4480270.xml,PMC4880116.xml,PMC5487382.xml,PMC5579818.xml,PMC6727998.xml" ;;
	riester)     echo "Full Riester demo corpus (172 papers)|PMC10528651.xml,PMC10667925.xml,PMC10759991.xml,PMC11128938.xml,PMC11167878.xml,PMC11219579.xml,PMC11246848.xml,PMC11402677.xml,PMC11548364.xml,PMC11560769.xml,PMC11685751.xml,PMC11707539.xml,PMC11729065.xml,PMC11742904.xml,PMC11779774.xml,PMC11795198.xml,PMC11882683.xml,PMC11898320.xml,PMC12035109.xml,PMC12055610.xml,PMC12187266.xml,PMC12489329.xml,PMC12597749.xml,PMC12620229.xml,PMC12620694.xml,PMC12718677.xml,PMC12748135.xml,PMC12748354.xml,PMC12748365.xml,PMC12748648.xml,PMC12750049.xml,PMC12750060.xml,PMC12750178.xml,PMC12750853.xml,PMC12751073.xml,PMC12751080.xml,PMC12751341.xml,PMC12752834.xml,PMC12753587.xml,PMC12753621.xml,PMC12753947.xml,PMC12754045.xml,PMC12754092.xml,PMC12754332.xml,PMC12754784.xml,PMC12754786.xml,PMC12755470.xml,PMC12756687.xml,PMC12757100.xml,PMC12757276.xml,PMC12757429.xml,PMC12757604.xml,PMC12757875.xml,PMC12758042.xml,PMC12758147.xml,PMC12758711.xml,PMC12759042.xml,PMC12759264.xml,PMC12764803.xml,PMC12764813.xml,PMC12764932.xml,PMC12765497.xml,PMC12765603.xml,PMC12765812.xml,PMC12765899.xml,PMC12766194.xml,PMC12766254.xml,PMC12766425.xml,PMC12766696.xml,PMC12767394.xml,PMC12767491.xml,PMC12767500.xml,PMC12768212.xml,PMC12768587.xml,PMC12769638.xml,PMC12769942.xml,PMC12770061.xml,PMC12770113.xml,PMC12770896.xml,PMC12771675.xml,PMC12772505.xml,PMC12772612.xml,PMC12772651.xml,PMC12773717.xml,PMC12774523.xml,PMC12774643.xml,PMC12774752.xml,PMC12774997.xml,PMC12775167.xml,PMC12775414.xml,PMC12775561.xml,PMC12775696.xml,PMC12776606.xml,PMC12777409.xml,PMC12778157.xml,PMC12778424.xml,PMC12779147.xml,PMC12779267.xml,PMC12779508.xml,PMC12780001.xml,PMC12780067.xml,PMC12780394.xml,PMC12780724.xml,PMC12780783.xml,PMC12781513.xml,PMC12782189.xml,PMC12783088.xml,PMC12783552.xml,PMC12784210.xml,PMC12784249.xml,PMC12784584.xml,PMC12784769.xml,PMC12784773.xml,PMC12785006.xml,PMC12785053.xml,PMC12785104.xml,PMC12785246.xml,PMC12785482.xml,PMC12785631.xml,PMC12787047.xml,PMC12787284.xml,PMC12788344.xml,PMC12788883.xml,PMC12799651.xml,PMC12841279.xml,PMC1525196.xml,PMC2386281.xml,PMC2528071.xml,PMC2874930.xml,PMC3208630.xml,PMC3328824.xml,PMC3328827.xml,PMC3536960.xml,PMC3574134.xml,PMC3607291.xml,PMC3856379.xml,PMC3860380.xml,PMC4082531.xml,PMC4192497.xml,PMC4374115.xml,PMC4398279.xml,PMC4480270.xml,PMC4880116.xml,PMC5346327.xml,PMC5460696.xml,PMC5487382.xml,PMC5579818.xml,PMC5748537.xml,PMC5844663.xml,PMC5961482.xml,PMC6462820.xml,PMC6727998.xml,PMC6728296.xml,PMC6754629.xml,PMC6878942.xml,PMC6921696.xml,PMC7216215.xml,PMC7279272.xml,PMC7364861.xml,PMC7433060.xml,PMC7663948.xml,PMC7969664.xml,PMC7992001.xml,PMC8449520.xml,PMC8584241.xml,PMC8746969.xml,PMC8842976.xml,PMC9266848.xml,PMC9279083.xml,PMC9569845.xml,PMC9694734.xml,PMC9736994.xml" ;;
        *)           echo "" ;;
    esac
}

_show_help() {
    echo "run-ingest-new.sh — Three-pass medlit ingestion using PostgresIdentityServer (fetch_vocab → extract → ingest → build_bundle)"
    echo ""
    echo "Usage:"
    echo "  ./run-ingest-new.sh --list              Show available paper lists"
    echo "  ./run-ingest-new.sh --list <name>       Ingest using the named list"
    echo ""
    echo "Requires:"
    echo "  DATABASE_URL env var pointing to a running Postgres instance."
    echo "  Quickest way: docker-compose --profile local up -d postgres-local"
    echo "  then: export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/kgserver"
    echo ""
    echo "Options:"
    echo "  --list [name]   List paper sets or select one for ingestion"
    echo "  --help, -h      Show this help"
}

_show_lists() {
    echo "Available paper lists:"
    echo ""
    for name in short medium five ten ten-hpylori twenty cushing cushing-6 adrenal ferroptosis misc smorgasbord riester; do
        raw=$(_get_list "$name")
        [ -z "$raw" ] && continue
        desc="${raw%%|*}"
        rest="${raw#*|}"
        count=$(echo "$rest" | tr ',' '\n' | wc -l)
        echo "  $name  ($count papers)  $desc"
    done
    echo ""
    echo "Usage: ./run-ingest-new.sh --list <name>"
}

# -----------------------------------------------------------------------------
# Parse args
# -----------------------------------------------------------------------------
LIST_NAME=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            _show_help
            exit 0
            ;;
        --list)
            if [[ -n "${2:-}" && "$2" != --* ]]; then
                LIST_NAME="$2"
                shift 2
            else
                _show_lists
                exit 0
            fi
            ;;
        *) shift ;;
    esac
done

if [[ -z "$LIST_NAME" ]]; then
    echo "Error: must specify --list <name> to choose a paper list." >&2
    echo "Run './run-ingest-new.sh --help' or './run-ingest-new.sh --list' for usage." >&2
    exit 1
fi

raw=$(_get_list "$LIST_NAME")
if [[ -z "$raw" ]]; then
    echo "Error: unknown list '$LIST_NAME'." >&2
    echo "Run './run-ingest-new.sh --list' to see available lists." >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# Check prerequisites
# -----------------------------------------------------------------------------
if [[ -z "${DATABASE_URL:-}" ]]; then
    echo "Error: DATABASE_URL is not set." >&2
    echo "" >&2
    echo "Start Postgres and set the variable:" >&2
    echo "  docker-compose --profile local up -d postgres-local" >&2
    echo "  export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/kgserver" >&2
    exit 1
fi

PAPER="${raw#*|}"
echo "Using list: $LIST_NAME (${raw%%|*})"
echo "DATABASE_URL: $DATABASE_URL"

# -----------------------------------------------------------------------------
# Fetch any missing PMC XMLs
# -----------------------------------------------------------------------------
(
    cd examples/medlit/pmc_xmls
    for PMC in $(echo $PAPER | tr ',' '\n' | sed 's/\.xml//'); do
        [ -f "${PMC}.xml" ] && continue
        curl -o "${PMC}.xml" \
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=${PMC}&rettype=xml&retmode=xml"
        sleep 0.5
    done
)

# -----------------------------------------------------------------------------
# Clean output dirs
# -----------------------------------------------------------------------------
git rm -rf bundle/* merged/* extracted/* vocab/* || true
rm -rf bundle/* merged/* extracted/* vocab/* || true
git commit -m 'start fresh' || true

# -----------------------------------------------------------------------------
# fetch_vocab: fast vocabulary extraction across all papers
# -----------------------------------------------------------------------------
uv run python -m examples.medlit.scripts.fetch_vocab \
    --input-dir examples/medlit/pmc_xmls \
    --output-dir vocab \
    --llm-backend anthropic \
    --papers $PAPER

# -----------------------------------------------------------------------------
# extract: full entity and relationship extraction with vocabulary context
# -----------------------------------------------------------------------------
uv run python -m examples.medlit.scripts.extract \
    --input-dir examples/medlit/pmc_xmls \
    --output-dir extracted \
    --llm-backend anthropic \
    --vocab-file vocab/vocab.json \
    --papers $PAPER

# -----------------------------------------------------------------------------
# ingest: identity-server-based deduplication and promotion
# (replaces legacy synonym cache + authority lookup chain)
# -----------------------------------------------------------------------------
uv run python -m examples.medlit.scripts.ingest \
    --bundle-dir extracted \
    --output-dir merged \
    --use-identity-server

# -----------------------------------------------------------------------------
# build_bundle: build kgbundle for loading into kgserver
# -----------------------------------------------------------------------------
uv run python -m examples.medlit.scripts.build_bundle \
    --merged-dir merged \
    --bundles-dir extracted \
    --output-dir bundle \
    --pmc-xmls-dir examples/medlit/pmc_xmls

git add bundle/* merged/* extracted/* vocab/*
git commit -m "Ingestion results (identity server): $PAPER"
