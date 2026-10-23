"""
svg2eps.py -- one SVG -> vector PDF (headless Chromium, page = SVG size) -> EPS (poppler's
pdftocairo), plus a 72-dpi PNG preview.  Same route as make_eps.py, as a function for reuse.
"""
import glob
import os
import re
import subprocess

CHROME = (glob.glob('/opt/pw-browsers/chromium-*/chrome-linux/chrome') or [None])[0]


def convert(svg, preview_dir=None):
    assert CHROME, 'headless Chromium not found (edit the glob in svg2eps.py)'
    base = svg[:-4]
    txt = open(svg).read()
    w, h = map(float, re.search(r"width='([\d.]+)' height='([\d.]+)'", txt).groups())
    html = os.path.abspath(base + '.html')
    with open(html, 'w') as f:                          # CSS px = 1/96 in; page exactly the SVG size
        f.write(f"<html><head><style>@page {{ size: {w}px {h}px; margin: 0 }} "
                f"html, body {{ margin: 0; padding: 0; overflow: hidden }} svg {{ display: block }}</style></head>"
                f"<body>{txt}</body></html>")
    subprocess.run([CHROME, '--headless=new', '--no-sandbox', '--disable-gpu', '--no-first-run',
                    '--disable-background-networking', '--disable-component-update',
                    '--no-pdf-header-footer', '--print-to-pdf-no-header',
                    f'--print-to-pdf={os.path.abspath(base + ".pdf")}', 'file://' + html],
                   capture_output=True, timeout=120)
    os.remove(html)
    subprocess.run(['pdftocairo', '-eps', '-f', '1', '-l', '1', base + '.pdf', base + '.eps'], check=True)
    subprocess.run(['pdftocairo', '-pdf', '-f', '1', '-l', '1', base + '.pdf', base + '.p1.pdf'], check=True)
    os.replace(base + '.p1.pdf', base + '.pdf')
    if preview_dir:
        os.makedirs(preview_dir, exist_ok=True)
        subprocess.run(['pdftocairo', '-png', '-r', '72', '-singlefile', base + '.pdf',
                        os.path.join(preview_dir, os.path.basename(base))], check=True)
    return base + '.eps'
