"""
make_figures.py -- all figures as standalone SVG (figures/*.svg) from results/*.json.
Thick lines = linearly stable (no localized eigenvalue with Im λ > 1e-5), thin = unstable.
"""
import os
import numpy as np

from io_utils import load
from model import discriminant, Q_ff, Q_sh
from svgplot import Figure, COLORS

os.makedirs('figures', exist_ok=True)
RED, BLUE, GREEN, GREY = '#d62728', '#1f5fbf', '#2ca02c', '#bdbdbd'
GTOL = 5e-5          # growth below this (wide solitons touching the box) is treated as zero
REMAX = 3.0          # |Re lam| above this: high-frequency, box-size-dependent growth (marked separately)
IMRE, IMG = 0.05, 1e-3   # purely imaginary (|Re lam| < IMRE) AND weaker than IMG: near-edge box artefact --
                         # these rates fall 6-18x from L = 40 pi to 56 pi (results/convergence.json)


def cplx(a):
    a = np.asarray(a, float)
    return a[..., 0] + 1j * a[..., 1]


def bands_from_edges(edges, blo):
    """Edges (any order) -> list of band intervals, top band first; the lowest open band runs to blo."""
    e = sorted(edges, reverse=True)
    out = []
    for i in range(0, len(e), 2):
        hi = e[i]
        lo = e[i + 1] if i + 1 < len(e) else blo
        out.append((lo, hi))
    return out


def stab_segments(ax, b, P, g, color, label=None):
    """Draw P(b) with thick (stable) / thin (unstable) segments; g = growth (None = not computed)."""
    st = []
    last = True
    for gi in g:
        if gi is not None:
            last = gi <= GTOL
        st.append(last)
    for i in range(len(b) - 1):
        thick = st[i] and st[i + 1]
        ax.line(b[i:i + 2], P[i:i + 2], color=color, width=3.2 if thick else 1.0)
    if label:
        ax.legend_items.append((label, color, 2.5, None, 'line'))


# ---------------------------------------------------------------- Fig. 1: bands
def fig_bands():
    B = load('results/bands.json')
    sh = B['sh_edges']
    blo, bhi = -1.5, 1.0
    fig = Figure(980, 430, title="Band-gap spectrum (V1 = V2 = 1, q = 0)  -- cf. paper Fig. 1")
    # (a) dispersion b(k) at al = 0.4 from the Floquet discriminant: cos(k pi) = D(b)
    ax = fig.add(70, 50, 380, 320, (-1, 1), (blo, bhi), xlabel='Bloch wavenumber k', ylabel='b',
                 title='(a) α = 0.4: FF (blue) and SH (green dashed)')
    bs = np.linspace(blo, bhi, 700)
    for Qf, col, dash in ((lambda b: Q_ff(1.0, 0.4, b), BLUE, None), (lambda b: Q_sh(1.0, 0.0, b), GREEN, '6,3')):
        D = np.array([discriminant(Qf(b)).real for b in bs])
        k = np.where(np.abs(D) <= 1, np.arccos(np.clip(D, -1, 1)) / np.pi, np.nan)
        # split into continuous pieces (bands) and mirror to negative k
        idx = np.where(np.isfinite(k))[0]
        pieces = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        for p in pieces:
            if len(p) > 1:
                ax.line(k[p], bs[p], color=col, width=1.8, dash=dash)
                ax.line(-k[p], bs[p], color=col, width=1.8, dash=dash)
    i04 = int(np.argmin(np.abs(np.array(B['alphas']) - 0.4)))
    for lo, hi in bands_from_edges(B['ff_edges'][i04], blo - 1):
        ax.hspan(lo, hi, '#9ecae1', 0.35)
    for lo, hi in bands_from_edges(sh, blo - 1):
        ax.hspan(lo, hi, '#a1d99b', 0.35)
    ax.text(0, 0.75, 'semi-infinite gap', anchor='middle')
    # (b) bands vs al
    ax = fig.add(560, 50, 380, 320, (0, 1), (blo, bhi), xlabel='α (gain/loss amplitude)', ylabel='b',
                 title='(b) shaded = band of FF and/or SH; white = total gap')
    al = np.array(B['alphas'])
    ff = B['ff_edges']
    for lo, hi in bands_from_edges(sh, blo - 1):                # SH bands: al-independent
        ax.hspan(lo, hi, GREY, 0.8)
    dal = al[1] - al[0]
    for a, edges in zip(al, ff):                                # FF bands, one vertical slice per al
        for lo, hi in bands_from_edges(edges, blo - 1):
            x0, x1 = max(a - dal / 2, 0), min(a + dal / 2, 1)
            y0, y1 = float(ax.Y(min(hi, bhi))), float(ax.Y(max(lo, blo)))
            ax.items.append(f"<rect x='{ax.X(x0):.1f}' y='{y0:.1f}' width='{ax.X(x1) - ax.X(x0) + 0.5:.1f}' "
                            f"height='{y1 - y0:.1f}' fill='{GREY}' fill-opacity='0.8'/>")
    nb = min(len(e) for e in ff[:-1])
    E = np.array([sorted(e, reverse=True)[:nb] for e in ff[:-1]])
    for j in range(nb):
        ax.line(al[:-1], E[:, j], color='#555', width=1.0)
    for e in sh:
        ax.line([0, 1], [e, e], color='#555', width=1.0, dash='4,3')
    ax.text(0.04, 0.7, 'semi-infinite gap', size=10)
    ax.text(0.04, -1.12, '3rd finite gap', size=10)
    ax.text(0.04, -0.54, '2nd', size=10)
    ax.text(0.6, 0.22, '1st finite gap', size=10)
    fig.note(70, 415, f"Computed: SH top edge b₂₊⁽⁰⁾ = {max(sh):.4f} (paper 0.3784);  "
             f"q₊⁽⁰'⁰⁾(α=0.7) = {B['q00_al07']:.4f} (paper -0.316);  FF and SH edges coalesce at b = -1.2932 for α = 0.7919.",
             size=11)
    fig.save('figures/fig1_bands.svg')


# ---------------------------------------------------------------- branches
def branch_arrays(points):
    """b, P, P1, P2, low-frequency growth (drives thick/thin), high-frequency growth (marked only)."""
    b = np.array([p['b'] for p in points])
    P = np.array([p['P'] for p in points])
    P1 = np.array([p['P1'] for p in points])
    P2 = np.array([p['P2'] for p in points])
    glo, ghi = [], []
    for p in points:
        gm, lu = p.get('gmax'), p.get('lam_unst')
        if gm is None:
            glo.append(None); ghi.append(None); continue
        hi = lu is not None and (abs(lu[0]) > REMAX or (abs(lu[0]) < IMRE and gm < IMG))
        glo.append(0.0 if hi else gm)
        ghi.append(gm if hi else 0.0)
    o = np.argsort(b)
    return b[o], P[o], P1[o], P2[o], [glo[i] for i in o], [ghi[i] for i in o]


def fig_semiinf():
    S = load('results/semiinf.json')
    m = S['meta']
    fig = Figure(1100, 640, title="Semi-infinite gap, α = 0.7: fundamental soliton branches  -- cf. paper Fig. 3")
    specs = (('case1', f"case 1: q = q_+^(0,0) = {m['q00']:.4f}"), ('case2', 'case 2: q = 0'),
             ('case3', 'case 3: q = -0.5134'))
    for k, (name, title) in enumerate(specs):
        C = S['cases'][name]
        b, P, P1, P2, g, gh = branch_arrays(C['branch'])
        ax = fig.add(70 + 350 * k, 50, 280, 300, (0, 1.0), (0, 4), xlabel='b', ylabel='P' if k == 0 else '',
                     title=title)
        edge = m['b2p'] + C['q'] if name == 'case2' else m['b1p']
        ax.vspan(-1, max(edge, m['b2p'] + C['q']), GREY, 0.7)
        stab_segments(ax, b, P, g, RED)
        ax.line(b, P1, color=BLUE, width=1, dash='5,3', label='P₁')
        ax.line(b, P2, color=GREEN, width=1, dash='2,2', label='P₂')
        pr = C['profile']
        ax.scatter([pr['b']], [pr['P']], color='black', r=4)
        ax.legend('ul')
        ax.legend_items.insert(0, ('P stable', RED, 3.2, None, 'line'))
        ax.legend_items.insert(1, ('P unstable', RED, 1.0, None, 'line'))
        ax.text(0.55, 0.25, f"dot: b={pr['b']}, P={pr['P']:.3f}", size=10)
        gr = fig.add(70 + 350 * k, 400, 280, 150, (0, 1.0), (1e-5, 1), xlabel='b', logy=True,
                     ylabel='max Im λ' if k == 0 else '')
        gg = np.array([np.nan if (v is None or v <= GTOL) else v for v in g], float)
        gh_ = np.array([np.nan if (v is None or v <= GTOL) else v for v in gh], float)
        gr.scatter(b, gg, color=RED, r=2.4, label='internal mode (|Re λ| < 3)' if k == 0 else None)
        gr.scatter(b, gh_, color='#7f7f7f', r=2.4, hollow=True, label='high-frequency, box-dependent' if k == 0 else None)
        if k == 0:
            gr.legend('ll')
    fig.note(70, 610, "Grey: band of FF and/or SH.  Thick: no internal-mode instability (|Re λ| < 3, Im λ > 5e-5).  Open grey "
             f"markers: high-frequency growth whose rate changes with box size.  Fourier grid L = 16π, h = {m['h']:.3f}.", size=11)
    fig.save('figures/fig3_semiinf_branches.svg')

    # profiles (paper Fig. 4)
    fig = Figure(900, 760, title="Semi-infinite gap profiles |w_l|^2 and currents j_l = Im(w_l* w_l')  -- cf. paper Fig. 4")
    for k, (name, title) in enumerate(specs):
        pr = S['cases'][name]['profile']
        x = np.array(pr['x']) / np.pi
        ax = fig.add(80, 50 + 235 * k, 330, 175, (-2, 2), (0, 1.1 * max(max(pr['I1']), max(pr['I2']))),
                     xlabel='ξ / π', ylabel='|w₁|², |w₂|²', title=f"{title}, b = {pr['b']}, P = {pr['P']:.3f}")
        ax.line(x, pr['I1'], color=BLUE, width=2, label='|w₁|²')
        ax.line(x, pr['I2'], color=GREEN, width=1.2, label='|w₂|²')
        ax.legend('ur')
        jm = 1.15 * max(np.max(np.abs(pr['j1'])), np.max(np.abs(pr['j2'])))
        ax = fig.add(510, 50 + 235 * k, 330, 175, (-2, 2), (-jm, jm), xlabel='ξ / π', ylabel='j₁, j₂')
        ax.line(x, pr['j1'], color=RED, width=2, label='j₁ (FF)')
        ax.line(x, pr['j2'], color='#17becf', width=1.2, label='j₂ (SH)')
        ax.line([-2, 2], [0, 0], color='#999', width=0.6)
        ax.legend('ur')
    fig.save('figures/fig4_semiinf_profiles.svg')

    # dynamics (paper Figs. 7, 18)
    for tag, names, figname, cite in (('case 1', ('case1_b0.25', 'case1_b0.50'), 'fig7_case1_dynamics', 'Fig. 7'),
                                      ('case 3', ('case3_b0.21', 'case3_b0.25'), 'fig18_case3_dynamics', 'Fig. 18')):
        dynamics_figure(S, names, f"Semi-infinite gap, {tag} (α = 0.7): split-step Fourier with 10% noise and "
                                  f"linear spectrum  -- cf. paper {cite}", f"figures/{figname}.svg", xmax_pi=4)


def dynamics_figure(D, names, title, path, xmax_pi):
    fig = Figure(1000, 850, title=title)
    for k, name in enumerate(names):
        r = D['ssf'][name]
        sp = D['spectra'][name]
        x = np.array(r['x']) / np.pi
        z = np.array(r['z'])
        I = np.array(r['I1'])
        sel = np.abs(x) <= xmax_pi
        x0 = 70 + 480 * k
        ax = fig.add(x0, 50, 360, 280, (-xmax_pi, xmax_pi), (0, z[-1]), xlabel='ξ / π', ylabel='ζ',
                     title=f"|u₁|², b = {r['b']}")
        ax.image(I[:, sel], (x[sel][0], x[sel][-1], 0, z[-1]), vmin=0, vmax=np.percentile(I[:, sel], 99.8))
        ax = fig.add(x0, 400, 360, 170, (0, z[-1]), (0, max(1.5, 1.1 * max(r['peak']) / r['peak0'])),
                     xlabel='ζ', ylabel='max|u₁|² / max|w₁|²')
        ax.line(z, np.array(r['peak']) / r['peak0'], color=RED, width=1.4, label='peak intensity')
        ax.line(z, np.array(r['P']) / r['P'][0], color=BLUE, width=1.4, label='P(ζ)/P(0)')
        ax.legend('ul')
        lam = cplx(sp['lam'])
        gm = float(sp['gmax'])
        lim = max(0.02, 1.4 * gm)
        ax = fig.add(x0, 640, 360, 170, (-4, 4), (-lim, lim), xlabel='Re λ', ylabel='Im λ',
                     title=f"max Im λ = {gm:.2e}  (absorbing layer)")
        ax.line([-4, 4], [0, 0], color='#aaa', width=0.6)
        ax.scatter(lam.real, lam.imag, color='#1f5fbf', r=1.8, hollow=True, label='eigenvalues')
        if sp.get('lam_unst') is not None:
            lu = cplx(sp['lam_unst'])
            ax.scatter([lu.real, -lu.real], [lu.imag, lu.imag], color=RED, r=4, label='most unstable pair')
        ax.legend('ul')
    fig.save(path)


def fig_thirdgap():
    T = load('results/thirdgap.json')
    m = T['meta']
    fig = Figure(1060, 470, title="Third finite gap, q = 0: fundamental soliton branches  -- cf. paper Fig. 5")
    ax = fig.add(80, 50, 520, 340, (-1.3, -1.0), (0, 2.5), xlabel='b', ylabel='P')
    gr = fig.add(690, 50, 330, 340, (-1.3, -1.0), (1e-5, 1), xlabel='b', ylabel='max Im λ', logy=True,
                 title='growth rate (filled: internal mode; open: box-dependent)')
    ax.vspan(-2, m['b_sh'], GREY, 0.8)
    cols = {'0.0': '#7f0000', '0.7': '#d62728', '0.7919': '#ff7f0e', '0.9': '#9467bd', '0.95': '#1f5fbf'}
    for key, B in T['branches'].items():
        b, P, P1, P2, g, gh = branch_arrays(B['points'])
        stab_segments(ax, b, P, g, cols.get(key, '#333'), label=f"α = {key}")
        gg = np.array([np.nan if (v is None or v <= GTOL) else v for v in g], float)
        gh_ = np.array([np.nan if (v is None or v <= GTOL) else v for v in gh], float)
        gr.scatter(b, gg, color=cols.get(key, '#333'), r=2.4)
        gr.scatter(b, gh_, color=cols.get(key, '#333'), r=2.4, hollow=True)
    for name, pr in T['profiles'].items():
        ax.scatter([pr['b']], [pr['P']], color='black', r=4)
    ax.legend('ul')
    fig.note(80, 430, "Soliton centred at the potential minimum (x₀ = π/2).  Grey: SH band.  Thick = linearly stable "
             "(localized modes), thin = unstable.  Dots: examples of Fig. 6.", size=11)
    fig.note(80, 448, f"Computed: α = 0.7919, b = −1.207 gives P = {T['profiles']['case1']['P']:.4f} (paper: P = 0.5).", size=11)
    fig.save('figures/fig5_thirdgap_branches.svg')

    fig = Figure(900, 760, title="Third finite gap profiles and currents  -- cf. paper Fig. 6")
    for k, name in enumerate(('case1', 'case2', 'case3')):
        pr = T['profiles'][name]
        x = np.array(pr['x']) / np.pi
        ax = fig.add(80, 50 + 235 * k, 330, 175, (-4, 4), (0, 1.1 * max(max(pr['I1']), max(pr['I2']))),
                     xlabel='(ξ − π/2) / π', ylabel='|w₁|², |w₂|²',
                     title=f"{name}: α = {pr['al']}, b = {pr['b']}, P = {pr['P']:.3f}")
        ax.line(x, pr['I1'], color=BLUE, width=1.4, label='|w₁|²')
        ax.line(x, pr['I2'], color=GREEN, width=1.2, label='|w₂|²')
        ax.legend('ur')
        jm = 1.15 * max(np.max(np.abs(pr['j1'])), np.max(np.abs(pr['j2'])))
        ax = fig.add(510, 50 + 235 * k, 330, 175, (-4, 4), (-jm, jm), xlabel='(ξ − π/2) / π', ylabel='j₁, j₂')
        ax.line(x, pr['j1'], color=RED, width=1.4, label='j₁ (FF)')
        ax.line(x, pr['j2'], color='#17becf', width=1.2, label='j₂ (SH)')
        ax.line([-4, 4], [0, 0], color='#999', width=0.6)
        ax.legend('lr' if np.mean(pr['j1']) > 0 else 'ur')
    fig.save('figures/fig6_thirdgap_profiles.svg')

    dynamics_figure(T, ('b-1.207', 'b-1.101'), "Third finite gap, α = 0.7919 (case 1): split-step Fourier with 10% "
                    "noise and linear spectrum  -- cf. paper Fig. 8", 'figures/fig8_thirdgap_dynamics.svg', xmax_pi=12)


def fig_shooting():
    S = load('results/semiinf.json')
    T = load('results/thirdgap.json')
    fig = Figure(1000, 600, title="Edge shooting (α = 0): converged profiles on the adaptive RKF45 mesh, and the step sizes")
    for k, (lab, sh) in enumerate((('semi-infinite gap, case 1, b = 0.25', S['cases']['case1']['shoot']),
                                   ('third gap, b = −1.207 (x₀ = π/2)', T['shoot']))):
        xs = np.array(sh['xs'])
        x0 = 70 + 480 * k
        lim = 1.1 * max(np.max(np.abs(sh['w1'])), np.max(np.abs(sh['w2'])))
        ax = fig.add(x0, 50, 380, 240, (0, xs[-1] / np.pi), (-lim, lim), xlabel='ξ / π', ylabel='w (real, α = 0)',
                     title=f"{lab}: {sh['shots']} shots, {sh['mesh']} steps")
        ax.line(xs / np.pi, sh['w1'], color=BLUE, width=1.6, label='w1')
        ax.line(xs / np.pi, sh['w2'], color=GREEN, width=1.4, label='w2')
        ax.line([0, xs[-1] / np.pi], [0, 0], color='#aaa', width=0.6)
        ax.legend('ur')
        hstep = np.diff(xs)
        ax = fig.add(x0, 360, 380, 180, (0, xs[-1] / np.pi), (1e-3, 0.2), xlabel='ξ / π', ylabel='accepted step h',
                     logy=True, title='step refinement where the solution varies fast')
        ax.line(xs[1:] / np.pi, hstep, color=RED, width=1.0)
    fig.note(70, 585, "Integration runs from ξ_f (right) to 0 starting on the decaying Floquet manifold "
             "(+ forced SH tail); rtol = 1e-12.  The thesis flowchart (nested bracket + bisection) sets the two tail amplitudes.", size=11)
    fig.save('figures/fig_shooting.svg')


def fig_validation():
    V = load('results/validation.json')
    fig = Figure(1000, 330, title="Validation: convergence of each building block")
    ax = fig.add(70, 50, 190, 210, (-12.5, -5.5), (1e-14, 1e-4), xlabel='log10 rtol', ylabel='global error', logy=True,
                 title='RKF45, y″ = −y')
    ax.line([np.log10(r['rtol']) for r in V['rk45']], [r['err'] for r in V['rk45']], color=RED, width=1.6)
    ax.scatter([np.log10(r['rtol']) for r in V['rk45']], [r['err'] for r in V['rk45']], color=RED)
    ax = fig.add(320, 50, 190, 210, (1.6, 2.7), (1e-9, 1e-2), xlabel='log10 knots', ylabel='max error', logy=True,
                 title='cubic spline (~n^-4)')
    n = np.array([r['n'] for r in V['spline']]); e = np.array([r['err'] for r in V['spline']])
    ax.line(np.log10(n), e, color=BLUE, width=1.6)
    ax.scatter(np.log10(n), e, color=BLUE)
    ax.line(np.log10(n), e[0] * (n / n[0]) ** -4.0, color='#999', width=1, dash='4,3')
    Pref = V['newton'][-1]['P']
    ax = fig.add(570, 50, 190, 210, (0.0, 0.35), (1e-13, 1e-3), xlabel='h', ylabel='|P(h) − P_ref|', logy=True,
                 title='Newton: fd4 vs Fourier')
    for deriv, col in (('fd4', RED), ('fourier', BLUE)):
        rs = [r for r in V['newton'] if r['deriv'] == deriv]
        hh = np.array([r['h'] for r in rs]); ee = np.array([max(abs(r['P'] - Pref), 1e-13) for r in rs])
        ax.line(hh, ee, color=col, width=1.6, label=deriv)
        ax.scatter(hh, ee, color=col)
    ax.legend('ul')
    ax = fig.add(810, 50, 160, 210, (0, 1), (0, 1), title='growth rate check')
    gr = V['growth']
    ax.text(0.05, 0.8, f"Im λ = {cplx(gr['eig']).imag:.4f}", size=11)
    ax.text(0.05, 0.62, f"SSF growth = {gr['ssf_rate']:.4f}", size=11)
    ax.text(0.05, 0.40, f"SSF order 2: {V['ssf_order']['2']['order']:.2f}", size=11)
    ax.text(0.05, 0.25, f"SSF order 4: {V['ssf_order']['4']['order']:.2f}", size=11)
    ax.text(0.05, 0.08, f"Jacobian err {V['jacobian_rel_err']:.0e}", size=11)
    fig.save('figures/fig_validation.svg')


def svg_to_png(svg_path, png_path):
    """Rasterize with the pre-installed headless Chromium (no imaging library available here)."""
    import glob, re, subprocess
    chrome = (glob.glob('/opt/pw-browsers/chromium-*/chrome-linux/chrome') or [None])[0]
    if chrome is None:
        return False
    txt = open(svg_path).read()
    w, h = map(int, re.search(r"width='(\d+)' height='(\d+)'", txt).groups())
    html = os.path.abspath(svg_path) + '.html'
    with open(html, 'w') as f:
        f.write(f"<html><body style='margin:0;background:white'>{txt}</body></html>")
    subprocess.run([chrome, '--headless=new', '--no-sandbox', '--disable-gpu', '--hide-scrollbars',
                    '--disable-background-networking', '--disable-component-update', '--no-first-run',
                    '--force-device-scale-factor=2', f'--window-size={w},{h + 120}', f'--screenshot={os.path.abspath(png_path)}',
                    'file://' + html], capture_output=True, timeout=120)
    os.remove(html)
    # the headless window height includes ~90 px of browser chrome: render taller, then crop to the SVG size
    subprocess.run(['convert', png_path, '-crop', f'{2 * w}x{2 * h}+0+0', '+repage', png_path], capture_output=True)
    return os.path.exists(png_path)


if __name__ == '__main__':
    for fn, need in ((fig_bands, 'results/bands.json'), (fig_semiinf, 'results/semiinf.json'),
                     (fig_thirdgap, 'results/thirdgap.json'), (fig_validation, 'results/validation.json')):
        if os.path.exists(need):
            fn()
            print("made", fn.__name__)
    if os.path.exists('results/semiinf.json') and os.path.exists('results/thirdgap.json'):
        fig_shooting()
        print("made fig_shooting")
    os.makedirs('figures/png', exist_ok=True)
    for f in sorted(os.listdir('figures')):
        if f.endswith('.svg'):
            svg_to_png(os.path.join('figures', f), os.path.join('figures/png', f[:-4] + '.png'))
    print("png export done")
