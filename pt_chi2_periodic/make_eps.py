"""
make_eps.py -- manuscript figures: vector EPS (+ PDF) in manuscript/figures/ from results/*.json.

Route (no TeX, matplotlib or ghostscript in the build environment):
  make_figures.fig_*()  with svgplot.PAPER = True   -> SVG without title/notes, cropped to the panels
  headless Chromium --print-to-pdf (page = SVG size) -> vector PDF, fonts embedded
  pdftocairo -eps (poppler)                          -> vector EPS (heatmaps stay embedded rasters)
The PDF copies let pdflatex use the same \\includegraphics{name} (no extension) that latex+dvips
resolves to the .eps. A 72-dpi PNG of every PDF goes to manuscript/figures/preview/ for checking.
"""
import glob
import os
import re
import subprocess

import svgplot

OUT = 'manuscript/figures'
os.makedirs(os.path.join(OUT, 'preview'), exist_ok=True)
svgplot.PAPER, svgplot.OUTDIR = True, OUT              # switch the plotting module to manuscript mode

import make_figures as mf                               # noqa: E402  (must import after the switch)

for fn in (mf.fig_bands, mf.fig_semiinf, mf.fig_thirdgap, mf.fig_validation, mf.fig_shooting):
    fn()                                                # writes OUT/<name>.svg
    print('svg:', fn.__name__)

from svg2eps import find_chrome                         # noqa: E402
chrome = find_chrome()
assert chrome, 'headless Chromium not found: set CHROME=/path/to/chrome'

for svg in sorted(glob.glob(os.path.join(OUT, '*.svg'))):
    base = svg[:-4]
    txt = open(svg).read()
    w, h = map(float, re.search(r"width='([\d.]+)' height='([\d.]+)'", txt).groups())
    html = os.path.abspath(base + '.html')
    with open(html, 'w') as f:                          # page size = SVG size (CSS px = 1/96 in)
        f.write(f"<html><head><style>@page {{ size: {w}px {h}px; margin: 0 }} "
                f"html, body {{ margin: 0; padding: 0; overflow: hidden }} svg {{ display: block }}</style></head>"
                f"<body>{txt}</body></html>")
    subprocess.run([chrome, '--headless=new', '--no-sandbox', '--disable-gpu', '--no-first-run',
                    '--disable-background-networking', '--disable-component-update',
                    '--no-pdf-header-footer', '--print-to-pdf-no-header',     # new and old flag names
                    f'--print-to-pdf={os.path.abspath(base + ".pdf")}', 'file://' + html],
                   capture_output=True, timeout=120)
    os.remove(html)
    # -f 1 -l 1: first page only (guards against a spill-over blank page); -eps keeps vectors
    subprocess.run(['pdftocairo', '-eps', '-f', '1', '-l', '1', base + '.pdf', base + '.eps'], check=True)
    # rewrite the PDF to its first page too, and a preview PNG for visual checks
    subprocess.run(['pdftocairo', '-pdf', '-f', '1', '-l', '1', base + '.pdf', base + '.p1.pdf'], check=True)
    os.replace(base + '.p1.pdf', base + '.pdf')
    subprocess.run(['pdftocairo', '-png', '-r', '72', '-singlefile', base + '.pdf',
                    os.path.join(OUT, 'preview', os.path.basename(base))], check=True)
    bbox = re.search(rb'%%BoundingBox: ([\d -]+)', open(base + '.eps', 'rb').read(4096))
    print(f"eps: {os.path.basename(base)}.eps  {os.path.getsize(base + '.eps') / 1024:.0f} kB  "
          f"BoundingBox {bbox.group(1).decode() if bbox else '?'}")
