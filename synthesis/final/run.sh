#!/usr/bin/env bash
# synthesis/final: one table + figures from committed scores/ (see README).
#   bash synthesis/final/run.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PYB=${PYB:-/home/f74134118/anaconda3/bin/python}
$PYB synthesis/final/collect.py
$PYB synthesis/final/make_plots.py
