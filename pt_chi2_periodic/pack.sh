#!/usr/bin/env bash
# pack.sh -- bundle code + README + results + figures into dist/pt_chi2_periodic.zip
#            and mirror the same tree into a target directory (e.g. a git checkout).
# usage: bash pack.sh [target_dir]
set -euo pipefail                                   # stop on any error / unset variable / failed pipe stage
cd "$(dirname "$0")"                                # run from the project folder
NAME=pt_chi2_periodic
STAGE=dist/$NAME
rm -rf dist && mkdir -p "$STAGE"/{results,figures/png}   # fresh staging tree
cp -v *.py pack.sh README.md REPRODUCE.md "$STAGE"/                    # sources, this script, documentation
cp -v results/*.json results/*.npz "$STAGE"/results/      # computed data (JSON is plain text)
cp -v figures/*.svg "$STAGE"/figures/                     # vector figures
cp -v figures/png/*.png "$STAGE"/figures/png/ 2>/dev/null || true   # raster copies, if Chromium was available
cp -v report.html "$STAGE"/                               # self-contained report page
cp -r manuscript "$STAGE"/                                # LaTeX draft, refs.bib, EPS/PDF figures, tables, PLAN.md
rm -f "$STAGE"/manuscript/figures/*.svg                   # intermediate SVGs: EPS/PDF are the deliverables
cp -r manuscript_perf "$STAGE"/                           # performance paper: LaTeX, numbers.tex, tables, EPS/PDF
rm -f "$STAGE"/manuscript_perf/figures/*.svg
for d in manuscript manuscript_perf; do                  # TeX build leftovers, if any
  rm -f "$STAGE"/$d/main.{aux,log,out,dvi,ps,blg} "$STAGE"/$d/arxiv.tar.gz
done
(cd dist && zip -r -q -9 "$NAME.zip" "$NAME")             # -r recurse, -q quiet, -9 max compression
ls -la dist/"$NAME.zip"
if [ "${1:-}" != "" ]; then                               # optional mirror into a checkout
  mkdir -p "$1/$NAME"
  cp -r "$STAGE"/. "$1/$NAME"/
  echo "mirrored into $1/$NAME"
fi
