"""
build_report.py -- assemble report.html (inline SVG figures + tables) from results/*.json and figures/*.svg.
"""
import html
import os
import re

import numpy as np

from io_utils import load

S = load('results/semiinf.json')
T = load('results/thirdgap.json')
B = load('results/bands.json')
V = load('results/validation.json') if os.path.exists('results/validation.json') else None


def c(a):
    return complex(a[0], a[1]) if isinstance(a, list) else a


def svg(name):
    p = f'figures/{name}.svg'
    if not os.path.exists(p):
        return f'<p class="missing">figure {name} not generated</p>'
    t = open(p).read()
    # responsive: keep the viewBox, let CSS size it
    t = re.sub(r"width='(\d+)' height='(\d+)'", "width='100%' preserveAspectRatio='xMidYMin meet'", t, count=1)
    return t


def fig(name, caption):
    return (f'<figure><div class="plate">{svg(name)}</div>'
            f'<figcaption>{caption}</figcaption></figure>')


cs = S['cases']
sp = S['spectra']
tp = T['profiles']
ts = T['spectra']
m = S['meta']


def branch_growth(points, b0, b1):
    vals = [p.get('gmax') for p in points if b0 <= p['b'] <= b1 and p.get('gmax') is not None]
    return max(vals) if vals else float('nan')


rows = [
    ("SH top band edge b<sub>2,+</sub><sup>(0)</sup> (V<sub>2</sub>=1, q=0)", "0.3784", f"{max(B['sh_edges']):.4f}"),
    ("Mismatch q<sub>+</sub><sup>(0,0)</sup> for α = 0.7", "−0.316", f"{m['q00']:.4f}"),
    ("FF/SH edges coalesce in the 3rd gap", "α ≈ 0.792, b = −1.29",
     f"α = 0.7919: FF {T['meta']['ff_edge']['0.7919']:.5f}, SH {T['meta']['b_sh']:.5f}"),
    ("Semi-infinite, case 1, b = 0.25 (Fig. 4 top): P", "0.5 (caption; dot drawn at b ≈ 0.27, P ≈ 0.47)",
     f"{cs['case1']['profile']['P']:.4f}"),
    ("Semi-infinite, case 2, b = 0.43, q = 0: P", "0.5", f"{cs['case2']['profile']['P']:.4f}"),
    ("Semi-infinite, case 3, b = 0.21, q = −0.5134: P", "0.5", f"{cs['case3']['profile']['P']:.4f}"),
    ("3rd gap, α = 0.7919, b = −1.207 (Fig. 6 top): P", "0.5", f"{tp['case1']['P']:.4f}"),
    ("3rd gap, α = 0.7, b = −1.234: P", "0.5", f"{tp['case2']['P']:.4f}"),
    ("3rd gap, α = 0.9, b = −1.153: P", "0.5", f"{tp['case3']['P']:.4f}"),
    ("Case 1, b = 0.25: stability", "stable", "stable" if sp['case1_b0.25']['gmax'] == 0 else f"growth {sp['case1_b0.25']['gmax']:.1e}"),
    ("Case 1, b = 0.5: stability", "unstable (quartet)",
     f"unstable, λ = ±{abs(c(sp['case1_b0.50']['lam_unst']).real):.3f} + {c(sp['case1_b0.50']['lam_unst']).imag:.3f}i"),
    ("3rd gap α = 0.7919, b = −1.101: stability", "unstable",
     f"unstable, growth {ts['b-1.101']['gmax']:.3f}" if ts['b-1.101']['gmax'] > 0 else "stable"),
]
table = ''.join(f"<tr><td>{a}</td><td>{b}</td><td class='num'>{x}</td></tr>" for a, b, x in rows)

g_c3 = sp['case3_b0.21']['gmax']
g_gap = ts['b-1.207']['gmax']
_a0 = [p for p in T['branches']['0.0']['points'] if p.get('gmax') is not None]   # alpha = 0 points with a spectrum
g_a0, g_a0_min = max(p['gmax'] for p in _a0), min(p['gmax'] for p in _a0)
b_a0 = (min(p['b'] for p in _a0), max(p['b'] for p in _a0))

val_rows = ''
if V:
    nr = V['newton']
    fd = [r for r in nr if r['deriv'] == 'fd4']
    fo = [r for r in nr if r['deriv'] == 'fourier']
    Pref = fo[-1]['P']
    val_rows = ''.join([
        f"<tr><td>RKF45, y″ = −y over 10 periods</td><td class='num'>error {V['rk45'][-1]['err']:.1e} at rtol 1e-12; "
        f"{V['rk45'][-1]['nrej']} rejected steps</td></tr>",
        f"<tr><td>Clamped cubic spline, non-uniform knots</td><td class='num'>error {V['spline'][0]['err']:.1e} → "
        f"{V['spline'][-1]['err']:.1e} for 50 → 400 knots</td></tr>",
        f"<tr><td>Newton, 4th-order central differences</td><td class='num'>|P − P<sub>ref</sub>| = "
        + ', '.join(f"{abs(r['P'] - Pref):.0e}" for r in fd) + f" for h = " + ', '.join(f"{r['h']:.2f}" for r in fd) + "</td></tr>",
        f"<tr><td>Newton, Fourier-spectral</td><td class='num'>P = {Pref:.10f}, quadratic convergence "
        f"({fo[-1]['its']} iterations from the spline)</td></tr>",
        f"<tr><td>Shooting vs Newton profile (ξ &lt; 4π)</td><td class='num'>{fo[-1]['shoot_vs_newton']:.1e}</td></tr>",
        f"<tr><td>Analytic Jacobian vs finite differences</td><td class='num'>{V['jacobian_rel_err']:.0e}</td></tr>",
        f"<tr><td>Newton state under split-step (same grid, ζ = 50)</td><td class='num'>{V['ssf_stationary']:.0e}</td></tr>",
        f"<tr><td>Split-step order (Strang / Yoshida)</td><td class='num'>{V['ssf_order']['2']['order']:.2f} / "
        f"{V['ssf_order']['4']['order']:.2f}</td></tr>",
        f"<tr><td>Growth rate: eigenvalue vs split-step</td><td class='num'>{c(V['growth']['eig']).imag:.4f} vs "
        f"{V['growth']['ssf_rate']:.4f}</td></tr>",
    ])

page = f"""<title>PT χ(2) Gap Solitons</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Public+Sans:wght@400;600&family=JetBrains+Mono:wght@400&display=swap">
<style>
/* Layout: one reading column (~70ch) with full-width figure plates; tables scroll inside their own box. */
:root {{
  --bg: #f4f6f8; --fg: #16202a; --muted: #55616d; --line: #d5dce3; --panel: #ffffff;
  --accent: #1d4e89; --warn: #a3261c; --plate: #ffffff;
  --display: 'Newsreader', 'Iowan Old Style', Georgia, serif;
  --body: 'Public Sans', 'Helvetica Neue', Arial, sans-serif;
  --mono: 'JetBrains Mono', ui-monospace, Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #0f151b; --fg: #e3e9ee; --muted: #9aa7b2; --line: #2a3540; --panel: #151d25;
  --accent: #8db8ef; --warn: #ff8f84; --plate: #ffffff; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #0f151b; --fg: #e3e9ee; --muted: #9aa7b2; --line: #2a3540; --panel: #151d25;
  --accent: #8db8ef; --warn: #ff8f84; --plate: #ffffff; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--fg); font-family: var(--body); font-size: 16px; line-height: 1.6; }}
main {{ max-width: 1080px; margin: 0 auto; padding-block: 28px 64px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: grid; gap: 28px; }}
.prose {{ max-width: 70ch; display: grid; gap: 12px; min-width: 0; }}
h1, h2 {{ font-family: var(--display); font-weight: 600; text-wrap: balance; line-height: 1.15; margin: 0; }}
h1 {{ font-size: clamp(2rem, 5vw, 2.9rem); }}
h2 {{ font-size: 1.55rem; padding-top: 8px; border-top: 1px solid var(--line); }}
p, ol, ul {{ margin: 0; }}
li + li {{ margin-top: 6px; }}
.eyebrow {{ font-family: var(--mono); font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }}
.lede {{ font-size: 1.12rem; color: var(--fg); }}
a {{ color: var(--accent); }}
code, .num {{ font-family: var(--mono); font-size: 0.86em; font-variant-numeric: tabular-nums; }}
.tablebox {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 6px; background: var(--panel); min-width: 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--line); vertical-align: top; }}
th {{ font-size: 12px; letter-spacing: 0.05em; text-transform: uppercase; color: var(--muted); font-weight: 600; }}
tr:last-child td {{ border-bottom: none; }}
.finding {{ border-left: 3px solid var(--warn); padding: 4px 0 4px 14px; }}
.finding strong {{ color: var(--warn); }}
figure {{ margin: 0; display: grid; gap: 8px; min-width: 0; }}
.plate {{ background: var(--plate); border: 1px solid var(--line); border-radius: 6px; padding: 8px; overflow-x: auto; }}
.plate svg {{ display: block; width: 100%; height: auto; min-width: 560px; }}
figcaption {{ font-size: 14px; color: var(--muted); max-width: 80ch; }}
pre {{ background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 14px; overflow-x: auto;
  font-family: var(--mono); font-size: 13px; line-height: 1.5; margin: 0; }}
.missing {{ color: var(--warn); }}
</style>
<main>
<header class="prose">
  <div class="eyebrow">Moreira, Konotop &amp; Malomed · Phys. Rev. A 87, 013832 (2013)</div>
  <h1>Gap solitons in a PT-symmetric χ<sup>(2)</sup> lattice</h1>
  <p class="lede">The thesis algorithms, run end to end. Edge shooting with an adaptive RKF4(5) integrator
  follows the Fig. 3.1 flowchart; a cubic spline carries the result into Newton–Raphson on a symmetric
  central-difference grid; then come continuation in α and b, linear stability, and split-step Fourier
  propagation. Everything is written in plain numpy.</p>
</header>

<section class="prose">
  <h2>Pipeline</h2>
  <ol>
    <li><b>Floquet spectrum.</b> Monodromy over one period gives band edges, gaps and the decaying tail
    (w, w′) = C(1, r) at ξ<sub>f</sub> = nπ.</li>
    <li><b>Edge shooting at α = 0.</b> RKF4(5) integrates inward from ξ<sub>f</sub> to 0. A step is
    rejected and refined when max|y<sub>5</sub> − y<sub>4</sub>| exceeds tolerance. The SH tail adds
    the Floquet particular solution forced by w<sub>1</sub>². The nested flowchart finds C<sub>2</sub>
    with w<sub>2</sub>′(0) = 0, then C<sub>1</sub> with w<sub>1</sub>′(0) = 0, and a 2-D Newton step
    polishes the last digits. Inward integration is stable because the unwanted Floquet solution decays
    on the way in.</li>
    <li><b>Cubic spline.</b> A clamped spline maps the adaptive RK mesh onto the uniform grid, with the
    end slopes taken from the RK state.</li>
    <li><b>Newton–Raphson.</b> The unknowns live on the half grid. Ghost points use the mirror rule
    (Re w even, Im w odd), which is the central-difference form of the PT symmetry and removes the phase
    zero mode. The box is periodic with L a multiple of π/2, because Dirichlet walls break the PT
    symmetry of a finite lattice.</li>
    <li><b>Continuation</b> switches on α, then follows each branch in b with a secant predictor.</li>
    <li><b>Linear stability</b> uses the paper's Eq. (14) on the same grid and the same d² matrix as
    Newton. A complex absorbing layer pushes box modes to Im λ &lt; 0, so the growth rate is
    max Im λ.</li>
    <li><b>Split-step Fourier</b> (Strang or Yoshida-4) has exact dispersion and an RK4 χ<sup>(2)</sup>
    substep. Its initial state is the Newton solution on the same grid, plus 10% noise.</li>
  </ol>
</section>

<section class="prose">
  <h2>Checks against the paper</h2>
  <p>The figures only match the paper's Eqs. (3) with b → −b. With b as printed, the semi-infinite gap
  would lie at b &lt; −0.378. All numbers below use the figure convention.</p>
</section>
<div class="tablebox"><table>
  <thead><tr><th>Quantity</th><th>Paper</th><th>This code</th></tr></thead>
  <tbody>{table}</tbody>
</table></div>

<section class="prose">
  <h2>Where the results differ from the paper</h2>
  <p class="finding"><strong>Third finite gap, α = 0.</strong> The branch the paper draws as stable
  (Fig. 5) has an oscillatory instability at every computed point, b from {b_a0[0]:.3f} to {b_a0[1]:.3f},
  with growth {g_a0_min:.3f} to {g_a0:.3f} (smallest near the band edge).
  An absorbing layer of any strength gives the same rate, and direct propagation confirms it: the
  deviation grows at 0.028 against Im λ = 0.028 at b = −1.197. The unstable eigenvector keeps an
  oscillating tail. That fits an internal mode resonating with the opposite-signature continuum, which
  overlaps in finite gaps.</p>
  <p class="finding"><strong>Weak instabilities below the paper's resolution.</strong> Two points the
  paper calls stable have small growth rates that converge in box size: case 3 at b = 0.21
  (Im λ = {g_c3:.1e}) and the third gap at α = 0.7919, b = −1.207 (Im λ = {g_gap:.1e}). Over ζ = 1000
  that amounts to e<sup>{g_c3 * 1000:.1f}</sup> and e<sup>{g_gap * 1000:.1f}</sup>, which is hard to
  see in noisy propagation or on eigenvalue axes of ±0.1.</p>
  <p class="finding"><strong>Typos in the paper.</strong> Eq. (9) has q with the opposite sign to
  Eq. (10); the numbers follow Eq. (10). The third-gap matching value is printed as 1.29 but is −1.29.
  The "P = 0.5" captions are rounded: the dots sit at P ≈ 0.45–0.47.</p>
  <p class="finding"><strong>Lattice centring.</strong> In the third gap the fundamental soliton sits on
  the potential minimum, ξ<sub>0</sub> = π/2, because the FF band-edge Bloch mode is odd about ξ = 0.
  Shooting about ξ = 0 only finds multi-hump states there.</p>
</section>

{fig('fig1_bands', 'Band-gap spectrum from Floquet discriminants (paper Fig. 1). (a) Dispersion at α = 0.4. (b) Bands against α: the third finite gap closes where the FF edge meets the SH edge at b = −1.293, α = 0.7919.')}
{fig('fig_shooting', 'Converged edge shots (α = 0) on the adaptive RKF4(5) mesh. The lower panels show the accepted step size: the integrator refines where the solution varies quickly and coarsens in the tails.')}
{fig('fig3_semiinf_branches', 'Semi-infinite gap, α = 0.7, cases 1–3 (paper Fig. 3). Thick: linearly stable; thin: unstable. The lower row shows the growth rate max Im λ along each branch.')}
{fig('fig4_semiinf_profiles', 'Profiles and currents j<sub>l</sub> = Im(w<sub>l</sub>* w<sub>l</sub>′) at the paper’s example points (Fig. 4).')}
{fig('fig7_case1_dynamics', 'Case 1: split-step propagation with 10% noise (b = 0.25, stable, ζ up to 1000; b = 0.5, unstable) and the linear spectrum with the absorbing layer (paper Fig. 7).')}
{fig('fig18_case3_dynamics', 'Case 3: b = 0.21 has a weak, box-converged instability (Im λ ≈ 3.6e-3), so it looks stable for a long time; b = 0.25 for comparison (paper Fig. 18).')}
{fig('fig5_thirdgap_branches', 'Third finite gap, q = 0, soliton centred at ξ<sub>0</sub> = π/2 (paper Fig. 5). Right: growth rate along each branch.')}
{fig('fig6_thirdgap_profiles', 'Third-gap profiles and currents at the paper’s examples (Fig. 6).')}
{fig('fig8_thirdgap_dynamics', 'Third gap, α = 0.7919: b = −1.207 and b = −1.101, split-step propagation and spectra (paper Fig. 8).')}

<section class="prose">
  <h2>Validation</h2>
  <p>Each building block was checked on its own (from <code>validate.py</code>):</p>
</section>
<div class="tablebox"><table><thead><tr><th>Check</th><th>Result</th></tr></thead><tbody>{val_rows}</tbody></table></div>
{fig('fig_validation', 'Convergence of the RK integrator, the spline and the Newton discretisations, plus the eigenvalue-versus-propagation growth check.')}

<section class="prose">
  <h2>Reproduce it</h2>
  <p>The zip bundle holds the code, the computed results, these figures and <code>REPRODUCE.md</code>,
  a step-by-step guide with the output each step should print and fixes for the usual failures.
  Only numpy is needed. The results folder is included, so the last two commands alone rebuild
  every figure and this page.</p>
<pre>python3 validate.py        # building-block checks              (~5 min)
python3 run_bands.py       # Floquet band edges                  (~2 min)
python3 run_semiinf.py     # semi-infinite gap, cases 1-3        (~35 min)
python3 run_thirdgap.py    # third finite gap                    (~45 min)
python3 validate_weak.py   # weak instabilities by propagation   (~15 min)
python3 make_figures.py    # SVG figures (+PNG if Chromium exists)
python3 build_report.py    # this page</pre>
</section>
</main>
"""
open('report.html', 'w').write(page)
print("report.html written,", round(len(page) / 1e6, 2), "MB")
