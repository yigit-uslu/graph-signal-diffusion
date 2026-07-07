#!/bin/bash
# Stage 1 — Download raw S&P 500 price + fundamental data via yfinance.
#
#   *** PROVENANCE ONLY — this is NOT the reproducible entry point. ***
#
# yfinance is a LIVE data source. Re-running this will NOT byte-reproduce the
# frozen data/sp500/raw/: adjusted-close prices are retroactively restated on
# splits/dividends, index membership drifts, and tickers delist over time. The
# canonical reproducible path therefore starts from the FROZEN raw/ that ships
# with the repo (validate it against checksums/raw.sha256 — see README §2).
#
# This script records EXACTLY how that frozen raw was originally produced, so the
# provenance is auditable even though the pull is not bit-reproducible.
#
# Produces: data/sp500/raw/{values.csv, adj.npy, fundamentals.csv, stocks.csv}
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_config.sh"

# Protect the frozen artifact: refuse to clobber an existing, non-empty raw dir.
if [ -d "$RAW_DIR" ] && [ -n "$(ls -A "$RAW_DIR" 2>/dev/null)" ]; then
  echo "REFUSING to overwrite existing '$RAW_DIR' (the frozen raw artifact)." >&2
  echo "The reproducible pipeline uses this frozen raw as-is; you do not need to" >&2
  echo "re-download. To force a fresh (non-reproducible) pull, move it aside first:" >&2
  echo "    mv '$RAW_DIR' '${RAW_DIR}.$(date +%Y%m%d)' && ./01_download_raw.sh" >&2
  exit 1
fi

$PYRUN -m graph_signal_diffusion.cli.stock.download \
    --years-back 10 \
    --end-date 2026-02-14 \
    --coverage-threshold 0.8 \
    --sector-bonus 0.0 \
    --corr-method spearman

echo ""
echo "Downloaded raw -> $RAW_DIR"
echo "NOTE: a fresh yfinance pull will differ from the paper's frozen raw."
echo "Compare against the shipped manifest (from repo root):"
echo "    sha256sum -c \"$REPRO_DIR/checksums/raw.sha256\""
