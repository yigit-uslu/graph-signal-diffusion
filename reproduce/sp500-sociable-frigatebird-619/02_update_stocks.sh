#!/bin/bash
# Stage 2 (OPTIONAL) — Refresh S&P 500 sector metadata (stocks.csv) from Wikipedia.
#
#   *** PROVENANCE ONLY — not needed for reproduction. ***
#
# Wikipedia is also a live source. The frozen raw/ already contains the exact
# stocks.csv (sector membership) the paper used, so you should NOT run this for a
# faithful reproduction. It is included only to document how sector metadata is
# refreshed independently of a full re-download.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_config.sh"

echo "This refreshes $RAW_DIR/stocks.csv from live Wikipedia and will change the"
echo "frozen sector metadata. Ctrl-C now unless you deliberately want fresh sectors."
sleep 3

$PYRUN -m graph_signal_diffusion.cli.stock.update_stocks
