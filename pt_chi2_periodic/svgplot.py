"""
svgplot.py -- minimal dependency-free plotting to SVG (matplotlib is not available here).

Figure(w, h) holds panels; each Axes maps data -> pixels linearly and emits SVG
primitives: polylines, markers, shaded spans, text, and heatmaps (embedded as
base64 PNG written with zlib, no imaging library needed).
"""
import base64
import math
import struct
import zlib

import numpy as np

FONT = "font-family='Helvetica, Arial, sans-serif'"
# Manuscript mode (set by make_eps.py): no figure title or notes -- the LaTeX caption carries them --
# and the page is cropped to the panels; save() then writes into OUTDIR under the same file name.
PAPER = False
OUTDIR = None
COLORS = ['#1f5fbf', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#17becf', '#8c564b', '#e377c2']

# viridis, 9 anchor colours (interpolated linearly)
_VIR = np.array([[68, 1, 84], [71, 44, 122], [59, 81, 139], [44, 113, 142], [33, 144, 141],
                 [39, 173, 129], [92, 200, 99], [170, 220, 50], [253, 231, 37]], float)


def colormap(v):
    """v in [0,1] (array) -> uint8 RGB via piecewise-linear viridis."""
    v = np.clip(np.asarray(v, float), 0, 1) * (len(_VIR) - 1)
    i = np.minimum(v.astype(int), len(_VIR) - 2)
    t = (v - i)[..., None]
    return ((1 - t) * _VIR[i] + t * _VIR[i + 1]).astype(np.uint8)


def png_bytes(rgb):
    """Encode an (H, W, 3) uint8 array as PNG (filter 0 on every row)."""
    h, w, _ = rgb.shape
    raw = b''.join(b'\x00' + rgb[r].tobytes() for r in range(h))
    def chunk(tag, data):
        c = struct.pack('>I', len(data)) + tag + data
        return c + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b''))


def nice_ticks(lo, hi, n=5):
    """Round tick positions covering [lo, hi] (1-2-5 steps)."""
    if lo > hi:                       # reversed axis (e.g. matrix rows downward): same ticks
        return nice_ticks(hi, lo, n)
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10)), key=lambda s: abs(s - raw))
    t0 = math.ceil(lo / step - 1e-9) * step
    ticks = []
    t = t0
    while t <= hi + 1e-9 * step:
        ticks.append(round(t, 10))
        t += step
    return ticks


def fmt(v):
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e4 or a < 1e-3:
        return f"{v:.0e}"
    s = f"{v:.4f}".rstrip('0').rstrip('.')
    return s


class Axes:
    def __init__(self, fig, x, y, w, h, xlim, ylim, xlabel='', ylabel='', title='', logy=False, logx=False):
        self.fig, self.x, self.y, self.w, self.h = fig, x, y, w, h
        self.xlim, self.ylim, self.logy, self.logx = xlim, ylim, logy, logx
        self.items = []
        self.xlabel, self.ylabel, self.title = xlabel, ylabel, title
        self.legend_items = []

    def X(self, v):
        v = np.asarray(v, float)
        lo, hi = self.xlim
        if self.logx:
            v, lo, hi = np.log10(np.maximum(v, 1e-300)), math.log10(lo), math.log10(hi)
        return self.x + (v - lo) / (hi - lo) * self.w

    def Y(self, v):
        v = np.asarray(v, float)
        lo, hi = self.ylim
        if self.logy:
            v, lo, hi = np.log10(np.maximum(v, 1e-300)), math.log10(lo), math.log10(hi)
        return self.y + self.h - (v - lo) / (hi - lo) * self.h

    def line(self, xs, ys, color=COLORS[0], width=1.5, dash=None, label=None, opacity=1.0):
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        ok = np.isfinite(xs) & np.isfinite(ys)
        if ok.sum() < 2:
            return
        pts = ' '.join(f"{a:.1f},{b:.1f}" for a, b in zip(self.X(xs[ok]), self.Y(ys[ok])))
        d = f" stroke-dasharray='{dash}'" if dash else ""
        self.items.append(f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='{width}'"
                          f" stroke-linejoin='round' stroke-opacity='{opacity}'{d}/>")
        if label:
            self.legend_items.append((label, color, width, dash, 'line'))

    def scatter(self, xs, ys, color=COLORS[0], r=2.5, label=None, hollow=False):
        for a, b in zip(self.X(xs), self.Y(ys)):
            if np.isfinite(a) and np.isfinite(b):
                f = "none" if hollow else color
                self.items.append(f"<circle cx='{a:.1f}' cy='{b:.1f}' r='{r}' fill='{f}' stroke='{color}' stroke-width='1'/>")
        if label:
            self.legend_items.append((label, color, r, None, 'dot'))

    def vspan(self, x0, x1, color='#cccccc', opacity=0.5):
        a, b = sorted((float(self.X(max(x0, self.xlim[0]))), float(self.X(min(x1, self.xlim[1])))))
        if b > a:
            self.items.append(f"<rect x='{a:.1f}' y='{self.y}' width='{b - a:.1f}' height='{self.h}'"
                              f" fill='{color}' fill-opacity='{opacity}'/>")

    def hspan(self, y0, y1, color='#cccccc', opacity=0.5):
        a, b = sorted((float(self.Y(max(y0, self.ylim[0]))), float(self.Y(min(y1, self.ylim[1])))))
        if b > a:
            self.items.append(f"<rect x='{self.x}' y='{a:.1f}' width='{self.w}' height='{b - a:.1f}'"
                              f" fill='{color}' fill-opacity='{opacity}'/>")

    def text(self, xd, yd, s, size=11, color='#222', anchor='start'):
        self.items.append(f"<text x='{float(self.X(xd)):.1f}' y='{float(self.Y(yd)):.1f}' font-size='{size}' "
                          f"fill='{color}' text-anchor='{anchor}' {FONT}>{s}</text>")

    def image(self, Z, extent, vmin=None, vmax=None):
        """Heatmap of Z[row=y, col=x] over extent (x0, x1, y0, y1); y increases upward."""
        Z = np.asarray(Z, float)
        vmin = np.nanmin(Z) if vmin is None else vmin
        vmax = np.nanmax(Z) if vmax is None else vmax
        rgb = colormap((Z - vmin) / (vmax - vmin + 1e-300))[::-1]       # first row at the top
        b64 = base64.b64encode(png_bytes(np.ascontiguousarray(rgb))).decode()
        x0, x1, y0, y1 = extent
        X0, X1 = float(self.X(x0)), float(self.X(x1))
        Y1, Y0 = float(self.Y(y1)), float(self.Y(y0))
        self.items.append(f"<image x='{X0:.1f}' y='{Y1:.1f}' width='{X1 - X0:.1f}' height='{Y0 - Y1:.1f}' "
                          f"preserveAspectRatio='none' image-rendering='pixelated' "
                          f"href='data:image/png;base64,{b64}'/>")
        self.cbar = (vmin, vmax)

    def legend(self, loc='ur'):
        self.legend_loc = loc

    def render(self):
        o = []
        cid = f"clip{id(self)}"
        o.append(f"<clipPath id='{cid}'><rect x='{self.x}' y='{self.y}' width='{self.w}' height='{self.h}'/></clipPath>")
        o.append(f"<rect x='{self.x}' y='{self.y}' width='{self.w}' height='{self.h}' fill='white' stroke='none'/>")
        o.append(f"<g clip-path='url(#{cid})'>" + ''.join(self.items) + "</g>")
        o.extend(getattr(self, 'outside', []))          # annotations allowed outside the panel (not clipped)
        o.append(f"<rect x='{self.x}' y='{self.y}' width='{self.w}' height='{self.h}' fill='none' stroke='#333' stroke-width='1'/>")
        if self.logx:
            xt = [10 ** e for e in range(math.ceil(math.log10(self.xlim[0])), math.floor(math.log10(self.xlim[1])) + 1)]
        else:
            xt = nice_ticks(*self.xlim)
        for t in xt:
            X = float(self.X(t))
            o.append(f"<line x1='{X:.1f}' y1='{self.y + self.h}' x2='{X:.1f}' y2='{self.y + self.h + 4}' stroke='#333'/>")
            o.append(f"<text x='{X:.1f}' y='{self.y + self.h + 16}' font-size='10' text-anchor='middle' {FONT}>{fmt(t)}</text>")
        if self.logy:
            yt = [10 ** e for e in range(math.ceil(math.log10(self.ylim[0])), math.floor(math.log10(self.ylim[1])) + 1)]
        else:
            yt = nice_ticks(*self.ylim)
        for t in ([] if getattr(self, 'noyticks', False) else yt):     # noyticks: categorical axis
            Y = float(self.Y(t))
            o.append(f"<line x1='{self.x - 4}' y1='{Y:.1f}' x2='{self.x}' y2='{Y:.1f}' stroke='#333'/>")
            o.append(f"<text x='{self.x - 6}' y='{Y + 3.5:.1f}' font-size='10' text-anchor='end' {FONT}>{fmt(t)}</text>")
        if self.xlabel:
            o.append(f"<text x='{self.x + self.w / 2}' y='{self.y + self.h + 32}' font-size='12' text-anchor='middle' {FONT}>{self.xlabel}</text>")
        if self.ylabel:
            cx, cy = self.x - 42, self.y + self.h / 2
            o.append(f"<text x='{cx}' y='{cy}' font-size='12' text-anchor='middle' transform='rotate(-90 {cx} {cy})' {FONT}>{self.ylabel}</text>")
        if self.title:
            o.append(f"<text x='{self.x + self.w / 2}' y='{self.y - 8}' font-size='12' font-weight='bold' text-anchor='middle' {FONT}>{self.title}</text>")
        if self.legend_items:
            loc = getattr(self, 'legend_loc', 'ur')
            lx = self.x + (8 if 'l' in loc else self.w - 150)
            ly = self.y + (12 if 'u' in loc else self.h - 14 * len(self.legend_items) - 4)
            for k, (lab, col, wd, dash, kind) in enumerate(self.legend_items):
                yy = ly + 14 * k
                if kind == 'line':
                    d = f" stroke-dasharray='{dash}'" if dash else ""
                    o.append(f"<line x1='{lx}' y1='{yy}' x2='{lx + 22}' y2='{yy}' stroke='{col}' stroke-width='{wd}'{d}/>")
                else:
                    o.append(f"<circle cx='{lx + 11}' cy='{yy}' r='3' fill='{col}'/>")
                o.append(f"<text x='{lx + 28}' y='{yy + 4}' font-size='10.5' {FONT}>{lab}</text>")
        if hasattr(self, 'cbar'):
            vmin, vmax = self.cbar
            bx, by, bh = self.x + self.w + 8, self.y, self.h
            grad = colormap(np.linspace(1, 0, 64))[:, None, :].repeat(6, axis=1)
            b64 = base64.b64encode(png_bytes(np.ascontiguousarray(grad))).decode()
            o.append(f"<image x='{bx}' y='{by}' width='10' height='{bh}' preserveAspectRatio='none' href='data:image/png;base64,{b64}'/>")
            for v in (vmin, 0.5 * (vmin + vmax), vmax):
                yy = by + bh - (v - vmin) / (vmax - vmin + 1e-300) * bh
                o.append(f"<text x='{bx + 13}' y='{yy + 3:.1f}' font-size='9' {FONT}>{fmt(v)}</text>")
        return '\n'.join(o)


class Figure:
    def __init__(self, w, h, title=None):
        self.w, self.h, self.axes, self.title = w, h, [], title
        self.notes = []

    def add(self, x, y, w, h, xlim, ylim, **kw):
        ax = Axes(self, x, y, w, h, xlim, ylim, **kw)
        self.axes.append(ax)
        return ax

    def note(self, x, y, s, size=11, color='#333', anchor='start', bold=False):
        fw = " font-weight='bold'" if bold else ""
        self.notes.append(f"<text x='{x}' y='{y}' font-size='{size}' fill='{color}' text-anchor='{anchor}'{fw} {FONT}>{s}</text>")

    def svg(self):
        x0, y0, w, h = 0, 0, self.w, self.h
        if PAPER:
            # bounding box of the panels incl. tick labels (10 px), axis labels (y: x-42, x: y+h+32),
            # panel titles (y-8) and colour bars (x+w+8 .. +65)
            render = [ax.render() for ax in self.axes]            # render first: colour bars are set lazily
            x0 = max(0, min(a.x - getattr(a, 'pad_left', 58 if a.ylabel else 40) for a in self.axes))
            y0 = max(0, min(a.y - (22 if a.title else 8) for a in self.axes))
            x1 = min(self.w, max(a.x + a.w + (68 if hasattr(a, 'cbar') else 14) for a in self.axes))
            y1 = min(self.h, max(a.y + a.h + (40 if a.xlabel else 24) for a in self.axes))
            w, h = x1 - x0, y1 - y0
        head = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='{x0} {y0} {w} {h}' "
                f"width='{w}' height='{h}'>")
        body = [f"<rect x='{x0}' y='{y0}' width='{w}' height='{h}' fill='white'/>"]
        if PAPER:
            return head + '\n'.join(body + render) + '</svg>'
        if self.title:
            body.append(f"<text x='{self.w / 2}' y='20' font-size='14' font-weight='bold' text-anchor='middle' {FONT}>{self.title}</text>")
        body += [ax.render() for ax in self.axes] + self.notes
        return head + '\n'.join(body) + '</svg>'

    def save(self, path):
        if PAPER and OUTDIR:
            import os
            path = os.path.join(OUTDIR, os.path.basename(path))
        with open(path, 'w') as f:
            f.write(self.svg())
