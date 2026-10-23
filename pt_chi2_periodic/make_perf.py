"""
make_perf.py -- figures, tables and number macros of the performance paper (manuscript_perf/), from
results/perf.json (bench_perf.py), results/arnoldi.json (validate_arnoldi.py) and
results/convergence.json (run_convergence.py).  No number in the paper is typed by hand:
  manuscript_perf/numbers.tex   \\newcommand macros for every number quoted in the text
  manuscript_perf/tables/*.tex  kernels, anatomy, throughput, projection, arnoldi
  manuscript_perf/figures/      fig_pipeline, fig_dag, fig_spy, fig_dense, fig_anatomy, fig_interp
Cost model and scheduler (projection table): see model_tasks() and schedule() below.
"""
import base64
import math
import os

import numpy as np

import svgplot
from svgplot import Figure, FONT, png_bytes
from io_utils import load
from svg2eps import convert

OUT = 'manuscript_perf'
FIG, TAB = f'{OUT}/figures', f'{OUT}/tables'
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)
svgplot.PAPER, svgplot.OUTDIR = True, FIG             # figure title/notes off, crop to panels

R = load('results/perf.json')
AR = load('results/arnoldi.json') if os.path.exists('results/arnoldi.json') else None
A = {int(t): {int(n): v for n, v in d.items()} for t, d in R['A'].items()}      # A[threads][n] = times
C, D = R['C'], R['D']
ST = C['stages']
BLUE, ORANGE, GREY, GREEN, RED = '#1f5fbf', '#e07b00', '#888888', '#2ca02c', '#d62728'
macros = {}


def mac(name, value):
    """record a LaTeX macro \\name -> value (string) for numbers.tex"""
    macros[name] = value
    return value


# ---------------------------------------------------------------------------- fits and models
def fit_power(ns, ts):
    """least-squares slope of log t vs log n -> (exponent p, prefactor c) with t = c n^p"""
    p, lc = np.polyfit(np.log(ns), np.log(ts), 1)
    return p, math.exp(lc)


def t_kernel(kind, n, threads):
    """time of a dense kernel at size n: power law fitted to the measured sizes n >= 2000 for this
    thread count, anchored at the largest measured size; eigensolver times calibrated to the real
    stability matrix (k_cal)"""
    nmax = max(A[threads])
    big = [k for k in sorted(A[threads]) if k >= 2000]
    p = np.polyfit(np.log(big), np.log([A[threads][k][kind] for k in big]), 1)[0]
    cal = k_cal if kind == 'eig' else 1.0
    return cal * A[threads][nmax][kind] * (n / nmax) ** p


ns_fit = [n for n in sorted(A[1]) if n >= 2000]          # near-asymptotic range
p_eig, _ = fit_power(ns_fit, [A[1][n]['eig'] for n in ns_fit])
p_inv, _ = fit_power(ns_fit, [A[1][n]['inv'] for n in ns_fit])
p_sol, _ = fit_power(ns_fit, [A[1][n]['solve'] for n in ns_fit])
nP = max(A[1])                                         # production size 4024 = 4 M at L = 40 pi, h = 0.25
gf_solve = {t: (8 / 3) * nP ** 3 / A[t][nP]['solve'] / 1e9 for t in A}      # complex LU: 8/3 n^3 real flops
gf_inv = {t: 8 * nP ** 3 / A[t][nP]['inv'] / 1e9 for t in A}                # LU + inverse: 8 n^3 real flops
sp_eig = {t: A[1][nP]['eig'] / A[t][nP]['eig'] for t in A}
sp_sol = {t: A[1][nP]['solve'] / A[t][nP]['solve'] for t in A}


def amdahl(speedups):
    """parallel fraction p of Amdahl's law s(t) = 1 / ((1-p) + p/t), least squares in 1/s"""
    ts = np.array([t for t in speedups if t > 1], float)
    s = np.array([speedups[int(t)] for t in ts])
    x, y = 1 - 1 / ts, 1 - 1 / s
    return float((x * y).sum() / (x * x).sum())


p_am_eig, p_am_sol = amdahl(sp_eig), amdahl(sp_sol)

# convergence-study rows: measured wall time per row from the cumulative log stamps
rows = R.get('conv_rows', [])
for i, r in enumerate(rows):
    r['t'] = r['t_cum'] - (rows[i - 1]['t_cum'] if i else 0)

# per-step Newton cost of the b-continuation at L = 40 pi (11 steps measured in part C)
t_bstep40 = ST['b_continuation_11'] / 11
t_alpha40 = ST['alpha_continuation']
t_shot = ST['shoot_polish'] + ST['shoot_profile'] + ST['floquet_tails']            # grid independent
t_grid40 = ST['grid_D2'] + ST['spline'] + ST['newton_al0']                          # scales with the grid
# zgeev on the real stability matrix vs on a random matrix of the same size (both 3 threads)
k_cal = ST['eig_g0_1'] / A[3][max(A[3])]['eig']
N40 = C['n_newton']


def grid_scale(Lpi, h):
    """Newton cost ~ N^3 with N ~ L/h (dense LU of the half-grid Jacobian) relative to L=40pi, h=0.25"""
    return ((Lpi / h) / (40 / 0.25)) ** 3


def nstab(Lpi, h):
    return 4 * int(round(2 * Lpi * math.pi / h))


def nbsteps(al, b):
    return int(math.ceil(round(abs(b - (-1.207)) / 0.005, 6)))


TARGETS = [(0.0, -1.197), (0.0, -1.282), (0.7, -1.282), (0.7919, -1.262), (0.7919, -1.207)]
GRIDS = [(40, 0.25), (40, 0.2), (56, 0.25)]


def model_tasks(threads, arnoldi_big=False):
    """task list of the convergence study: (name, duration, dependencies).
    per grid: setup (shot polish + spline + Newton) -> alpha chain;
    per target: b chain (after the alpha chain) -> two spectra (g0 = 1, 0.25)."""
    tasks = []
    for gi, (Lpi, h) in enumerate(GRIDS):
        sc = grid_scale(Lpi, h)
        tasks.append((f'setup{gi}', t_shot + t_grid40 * sc, []))
        tasks.append((f'alpha{gi}', t_alpha40 * sc, [f'setup{gi}']))
        for ti, (al, b) in enumerate(TARGETS):
            tasks.append((f'b{gi}{ti}', t_bstep40 * sc * nbsteps(al, b), [f'alpha{gi}']))
            n = nstab(Lpi, h)
            if arnoldi_big and Lpi == 56:     # refine the eigenvalue found on the 40 pi box: one shift
                te = ST['arnoldi_target'] * t_kernel('inv', n, threads) / A[3][nP]['inv']
            else:
                te = t_kernel('eig', n, threads)
            for k in range(2):
                tasks.append((f'e{gi}{ti}{k}', te, [f'b{gi}{ti}']))
    return tasks


def schedule(tasks, workers):
    """greedy list scheduling: whenever a worker is free, start the longest ready task.
    Returns the makespan (wall time)."""
    done, start_t, finish = set(), {}, {}
    free = [0.0] * workers
    pending = list(tasks)
    while pending:
        w = int(np.argmin(free))
        t_now = free[w]
        ready = [tk for tk in pending if all(d in finish and finish[d] <= t_now for d in tk[2])]
        if not ready:                                   # wait for the earliest dependency to finish
            nxt = min(finish[d] for tk in pending for d in tk[2] if d in finish and finish[d] > t_now)
            free[w] = nxt
            continue
        tk = max(ready, key=lambda z: z[1])
        pending.remove(tk)
        finish[tk[0]] = t_now + tk[1]
        free[w] = finish[tk[0]]
    return max(finish.values())


proj = [
    ('as run: 1 process, 3 BLAS threads, in series', schedule(model_tasks(3), 1)),
    ('1 process, 4 BLAS threads, in series', schedule(model_tasks(4), 1)),
    ('4 processes x 1 thread, task-parallel', schedule(model_tasks(1), 4)),
    ('4 processes x 1 thread + Arnoldi refinement on the 56$\\pi$ box', schedule(model_tasks(1, True), 4)),
]
t_meas = rows[-1]['t_cum'] if rows else float('nan')
mac('modelErr', f"{100 * (proj[0][1] - t_meas) / t_meas:+.0f}")

# ---------------------------------------------------------------------------- macros
mac('nCPU', str(R['machine']['cpus']))
mac('cpuName', R['machine']['cpu'].replace('(R)', '').replace('@', 'at'))
mac('numpyVer', R['machine']['numpy'])
mac('nStab', f"{C['n_stab']}")
mac('nNewton', f"{C['n_newton']}")
mac('pEig', f"{p_eig:.2f}"); mac('pInv', f"{p_inv:.2f}"); mac('pSolve', f"{p_sol:.2f}")
mac('tEigOne', f"{A[1][nP]['eig']:.0f}"); mac('tEigFour', f"{A[4][nP]['eig']:.0f}"); mac('tEigThree', f"{A[3][nP]['eig']:.0f}")
mac('spEigFour', f"{sp_eig[4]:.2f}"); mac('spSolveFour', f"{sp_sol[4]:.2f}")
mac('pAmEig', f"{p_am_eig:.2f}"); mac('pAmSolve', f"{p_am_sol:.2f}")
mac('gfSolveOne', f"{gf_solve[1]:.0f}"); mac('gfSolveFour', f"{gf_solve[4]:.0f}")
mac('ratioEigSolve', f"{A[1][nP]['eig'] / A[1][nP]['solve']:.0f}")
mac('kCal', f"{k_cal:.2f}")
n2 = sorted(A[1])[-2]
mac('pEigLocal', f"{math.log(A[1][nP]['eig'] / A[1][n2]['eig']) / math.log(nP / n2):.2f}")
mac('peakCore', '67')                                   # 2.1 GHz x 32 flop/cycle (AVX-512, 2 FMA units)
mac('gfInvOne', f"{gf_inv[1]:.0f}"); mac('gfInvFour', f"{gf_inv[4]:.0f}")
mac('spInvFour', f"{A[1][nP]['inv'] / A[4][nP]['inv']:.2f}")
mac('spEigThree', f"{sp_eig[3]:.2f}"); mac('spSolveThree', f"{sp_sol[3]:.2f}")
mac('thrVsSerial', f"{4 * A[1][2000]['eig'] / R['B']['4x1']['wall']:.1f}")
mac('tEigTwoK', f"{A[1][2000]['eig']:.1f}")
mac('ratioEigInv', f"{A[1][nP]['eig'] / A[1][nP]['inv']:.1f}")
B = R['B']
mac('thrOneFour', f"{B['1x4']['t']:.1f}"); mac('thrTwoTwo', f"{B['2x2']['wall']:.1f}"); mac('thrFourOne', f"{B['4x1']['wall']:.1f}")
mac('thrGain', f"{B['1x4']['t'] / B['4x1']['wall']:.2f}")
eig_rows = ST['eig_g0_1'] + ST['eig_g0_025']
tot_row = sum(v for k, v in ST.items() if k not in ('jacobian_assembly', 'jacobian_solve', 'arnoldi_target'))
mac('tEigRow', f"{eig_rows:.0f}"); mac('tEigSpec', f"{ST['eig_g0_1']:.0f}"); mac('shEigRow', f"{100 * eig_rows / tot_row:.0f}")
mac('tRowTotal', f"{tot_row:.0f}")
mac('tAlpha', f"{ST['alpha_continuation']:.1f}"); mac('nAlpha', str(C['n_alpha']))
mac('tBstep', f"{t_bstep40:.2f}")
mac('tJacAsm', f"{ST['jacobian_assembly']:.2f}"); mac('tJacSolve', f"{ST['jacobian_solve']:.2f}")
mac('tPolish', f"{ST['shoot_polish']:.1f}"); mac('tArnTarget', f"{ST['arnoldi_target']:.1f}")
mac('spArnTarget', f"{ST['eig_g0_1'] / ST['arnoldi_target']:.1f}")
mac('tConvMeas', f"{t_meas:.0f}"); mac('tConvMeasMin', f"{t_meas / 60:.0f}")
mac('tProjRun', f"{proj[0][1]:.0f}"); mac('tProjFourThr', f"{proj[1][1]:.0f}")
mac('tProjProc', f"{proj[2][1]:.0f}"); mac('tProjArn', f"{proj[3][1]:.0f}")
mac('spProjProc', f"{proj[0][1] / proj[2][1]:.1f}"); mac('spProjArn', f"{proj[0][1] / proj[3][1]:.1f}")
mac('usRKstep', f"{D['rk_shot']['us_per_step']:.0f}")
mac('nRKsteps', f"{D['rk_shot']['nacc'] + D['rk_shot']['nrej']}")
mac('tRKshot', f"{D['rk_shot']['t']:.2f}")
mac('usRHSscalar', f"{D['rhs_scalar_us']:.1f}")
rv = {int(k): v for k, v in D['rhs_vec_us_per_state'].items()}
mac('usRHSvecBig', f"{rv[max(rv)]:.3f}"); mac('spRHS', f"{D['rhs_scalar_us'] / rv[max(rv)]:.0f}")
ss = {int(k): v for k, v in D['ssf'].items()}
mac('usSSFsmall', f"{ss[1024]['us_per_step']:.0f}"); mac('gfSSFsmall', f"{ss[1024]['gflops']:.2f}")
mac('gfSSFbig', f"{ss[max(ss)]['gflops']:.2f}"); mac('MSSFbig', str(max(ss)))
sb = {int(k): v for k, v in D['ssf_batch'].items()}
kb = min(sb, key=sb.get)
mac('spBatch', f"{sb[1] / sb[kb]:.1f}"); mac('KBatch', str(kb)); mac('KBatchMax', str(max(sb)))
mac('spBatchMax', f"{sb[1] / sb[max(sb)]:.1f}")
mac('usRHSvecOne', f"{rv[1]:.1f}")
mpk = max(ss, key=lambda M: ss[M]['gflops'])
mac('gfSSFpeak', f"{ss[mpk]['gflops']:.1f}"); mac('MSSFpeak', str(mpk)); mac('gfSSFsmallest', f"{ss[min(ss)]['gflops']:.1f}")
S1 = load('results/semiinf.json')['cases']['case1']['shoot']
mac('nShotsCaseOne', str(S1['shots'])); mac('tFlowchart', f"{S1['shots'] * D['rk_shot']['t']:.0f}")
if AR:
    ar = AR['rows']
    d1 = max(abs(r['target']['g'] - r['dense']['g']) for r in ar)
    d2 = max(abs(r['scan']['g'] - r['dense']['g']) for r in ar)
    mac('arAgree', 'to the 6 significant digits stored' if d1 == d2 == 0 else
        f'to within {d1:.1e} (one shift) and {d2:.1e} (scan)')
    big = max(ar, key=lambda r: r['n'])
    mac('arN', str(big['n'])); mac('arDense', f"{big['dense']['t']:.1f}")
    mac('arTarget', f"{big['target']['t']:.1f}"); mac('arScan', f"{big['scan']['t']:.0f}")


# ---------------------------------------------------------------------------- tables
with open(f'{TAB}/kernels.tex', 'w') as f:
    f.write("% generated by make_perf.py -- do not edit\n\\begin{tabular}{rrrrrr}\n\\hline\\hline\n"
            "$n$ & eig, 1 thr. & eig, 4 thr. & inv, 1 thr. & solve, 1 thr. & solve GFLOP/s\\\\\n\\hline\n")
    for n in (1000, 2000, 2800, nP):
        f.write(f"{n} & {A[1][n]['eig']:.2f} & {A[4][n]['eig']:.2f} & {A[1][n]['inv']:.2f} & "
                f"{A[1][n]['solve']:.3f} & {(8 / 3) * n ** 3 / A[1][n]['solve'] / 1e9:.1f}\\\\\n")
    f.write(f"\\hline\nfitted exponent & {p_eig:.2f} & & {p_inv:.2f} & {p_sol:.2f} & \\\\\n\\hline\\hline\n\\end{{tabular}}\n")

labels = [('floquet_tails', 'Floquet tails (monodromy, RKF45)'), ('shoot_polish', 'shot: 2-D Newton polish'),
          ('shoot_profile', 'shot: final profile'), ('grid_D2', 'Fourier $D_2$ and folding'),
          ('spline', 'cubic spline'), ('newton_al0', 'Newton at $\\alpha=0$'),
          ('alpha_continuation', f"$\\alpha$ continuation ({C['n_alpha']} steps)"),
          ('b_continuation_11', '$b$ continuation (11 steps)'),
          ('eig_g0_1', 'spectrum, $g_0=1$ (zgeev)'), ('eig_g0_025', 'spectrum, $g_0=0.25$ (zgeev)')]
with open(f'{TAB}/anatomy.tex', 'w') as f:
    f.write("% generated by make_perf.py -- do not edit\n\\begin{tabular}{lrr}\n\\hline\\hline\n"
            "stage & time (s) & share\\\\\n\\hline\n")
    for k, lab in labels:
        f.write(f"{lab} & {ST[k]:.2f} & {100 * ST[k] / tot_row:.1f}\\%\\\\\n")
    f.write(f"\\hline\ntotal & {tot_row:.1f} & 100\\%\\\\\n"
            f"\\hline\none Newton iteration: Jacobian assembly / dense solve & {ST['jacobian_assembly']:.2f} / "
            f"{ST['jacobian_solve']:.2f} & \\\\\nArnoldi, one shift (replaces one spectrum) & {ST['arnoldi_target']:.2f} & \\\\\n"
            "\\hline\\hline\n\\end{tabular}\n")

with open(f'{TAB}/throughput.tex', 'w') as f:
    f.write("% generated by make_perf.py -- do not edit\n\\begin{tabular}{lrr}\n\\hline\\hline\n"
            "configuration (4 eigensolves, $n=2000$) & wall time (s) & relative\\\\\n\\hline\n")
    for key, lab, t in (('1x4', '1 process $\\times$ 4 threads, in series', B['1x4']['t']),
                        ('2x2', '2 processes $\\times$ 2 threads', B['2x2']['wall']),
                        ('4x1', '4 processes $\\times$ 1 thread', B['4x1']['wall'])):
        f.write(f"{lab} & {t:.1f} & {t / B['1x4']['t']:.2f}\\\\\n")
    f.write("\\hline\\hline\n\\end{tabular}\n")

with open(f'{TAB}/projection.tex', 'w') as f:
    f.write("% generated by make_perf.py -- do not edit\n\\begin{tabular}{lrr}\n\\hline\\hline\n"
            "strategy & model wall time (s) & speedup\\\\\n\\hline\n")
    for lab, t in proj:
        f.write(f"{lab} & {t:.0f} & {proj[0][1] / t:.1f}\\\\\n")
    f.write(f"\\hline\nmeasured production run (1 process, 3 threads) & {t_meas:.0f} & \\\\\n\\hline\\hline\n\\end{{tabular}}\n")

if AR:
    with open(f'{TAB}/arnoldi.tex', 'w') as f:
        f.write("% generated by make_perf.py from results/arnoldi.json -- do not edit\n"
                "\\begin{tabular}{lrrrrr}\n\\hline\\hline\n"
                "soliton & $n$ & $\\max\\mathrm{Im}\\,\\lambda$ (dense) & dense (s) & one shift (s) & scan (s)\\\\\n\\hline\n")
        for r in AR['rows']:
            f.write(f"{r['label']} & {r['n']} & {r['dense']['g']:.6f} & {r['dense']['t']:.1f} & "
                    f"{r['target']['t']:.1f} & {r['scan']['t']:.0f}\\\\\n")
        f.write("\\hline\\hline\n\\end{tabular}\n")

# ---------------------------------------------------------------------------- diagrams (hand SVG)
DEFS = ("<defs><marker id='ar' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' "
        "orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' fill='#444'/></marker></defs>")


def svg_doc(w, h, body):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' width='{w}' height='{h}'>"
            f"{DEFS}<rect width='{w}' height='{h}' fill='white'/>{body}</svg>")


def box(x, y, w, h, title, lines, color):
    o = [f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='7' fill='{color}' fill-opacity='0.13' "
         f"stroke='{color}' stroke-width='1.6'/>",
         f"<text x='{x + w / 2}' y='{y + 18}' font-size='12.5' font-weight='bold' text-anchor='middle' {FONT}>{title}</text>"]
    for i, s in enumerate(lines):
        o.append(f"<text x='{x + w / 2}' y='{y + 35 + 14 * i}' font-size='10.5' text-anchor='middle' {FONT}>{s}</text>")
    return ''.join(o)


def arrow(x1, y1, x2, y2, dash=False):
    d = " stroke-dasharray='4,3'" if dash else ""
    return f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='#444' stroke-width='1.4' marker-end='url(#ar)'{d}/>"


def text(x, y, s, size=11, anchor='start', color='#222', bold=False):
    fw = " font-weight='bold'" if bold else ""
    return f"<text x='{x}' y='{y}' font-size='{size}' fill='{color}' text-anchor='{anchor}'{fw} {FONT}>{s}</text>"


# Fig. 1: pipeline, coloured by what bounds each stage
t_eig3 = ST['eig_g0_1']
b = []
W, H, bw, bh = 1000, 330, 178, 96
xs = [15, 213, 411, 609, 807]
b.append(box(xs[0], 30, bw, bh, 'Edge shooting', ['RKF4(5), Python scalars', f"{D['rk_shot']['us_per_step']:.0f} µs per step",
                                                  f"{D['rk_shot']['nacc'] + D['rk_shot']['nrej']} steps, {1e3 * D['rk_shot']['t']:.0f} ms per shot",
                                                  f"{load('results/semiinf.json')['cases']['case1']['shoot']['shots']} shots: "
                                                  f"{load('results/semiinf.json')['cases']['case1']['shoot']['shots'] * D['rk_shot']['t']:.0f} s"], ORANGE))
b.append(box(xs[1], 30, bw, bh, 'Spline + Newton', ['clamped spline O(N)', f"dense Jacobian, N = {C['n_newton']}",
                                                    f"assembly {ST['jacobian_assembly']:.2f} s", f"LU solve {ST['jacobian_solve']:.2f} s"], BLUE))
b.append(box(xs[2], 30, bw, bh, 'Continuation', ['α, then b: secant predictor', 'each step: 2–3 Newton its',
                                                 f"{C['n_alpha']} α steps: {ST['alpha_continuation']:.0f} s", 'SEQUENTIAL chain'], BLUE))
b.append(box(xs[3], 30, bw, bh, 'Stability spectrum', [f"dense zgeev, n = 4M = {C['n_stab']}", 'O(n³), Hessenberg + QR',
                                                       f"{t_eig3:.0f} s per spectrum (3 thr.)", f"threads: ×{sp_eig[4]:.1f} on 4"], RED))
b.append(box(xs[4], 30, bw, bh, 'Split-step Fourier', ['FFT + RK4 local step', f"{ss[1024]['us_per_step']:.0f} µs per step, M = 1024",
                                                       '10⁵ steps per ζ = 1000', 'NumPy call overhead'], ORANGE))
for i in range(4):
    b.append(arrow(xs[i] + bw, 78, xs[i + 1] - 4, 78))
# legend and the parallel structure underneath
b.append(f"<rect x='15' y='160' width='14' height='14' fill='{ORANGE}' fill-opacity='0.3' stroke='{ORANGE}'/>")
b.append(text(36, 172, 'interpreter-bound: time per Python / NumPy call, not flops'))
b.append(f"<rect x='15' y='184' width='14' height='14' fill='{BLUE}' fill-opacity='0.3' stroke='{BLUE}'/>")
b.append(text(36, 196, f"dense LU, moderate size: {gf_solve[1]:.0f} GFLOP/s on one core (BLAS-3)"))
b.append(f"<rect x='15' y='208' width='14' height='14' fill='{RED}' fill-opacity='0.3' stroke='{RED}'/>")
b.append(text(36, 220, f"dense nonsymmetric eigensolver: dominant ({100 * eig_rows / tot_row:.0f}% of a convergence row), "
                       f"threads help little (×{sp_eig[4]:.2f} on 4)"))
b.append(text(15, 258, 'Parallel width:', bold=True))
b.append(text(130, 258, 'shooting: candidate amplitudes independent  |  continuation: none (each step needs the last)  |'))
b.append(text(130, 276, 'spectra: every branch point, grid and absorber independent  |  split-step: noise realizations independent'))
b.append(text(15, 304, 'Used in the production runs: BLAS threads only, inside one Python process.', color=RED))
open(f'{FIG}/fig_pipeline.svg', 'w').write(svg_doc(W, H - 10, ''.join(b)))

# Fig. 2: task graph of the convergence study
b = []
W, H = 1000, 380
lane_y = [48, 158, 268]
for gi, ((Lpi, h), y) in enumerate(zip(GRIDS, lane_y)):
    b.append(text(10, y + 30, f"L = {Lpi}π", bold=True)); b.append(text(10, y + 46, f"h = {h}"))
    b.append(box(70, y, 120, 60, 'setup', ['shot, spline,', 'Newton α = 0'], BLUE))
    b.append(box(220, y, 120, 60, 'α chain', [f"{C['n_alpha']} Newton solves", 'sequential'], BLUE))
    b.append(arrow(190, y + 30, 216, y + 30))
    # bus from the alpha chain over all b chains of the lane (no line crosses an eig box)
    xe = 380 + (len(TARGETS) - 1) * 122 + 30
    b.append(f"<path d='M340,{y + 30} H356 V{y - 9} H{xe}' fill='none' stroke='#444' stroke-width='1.4'/>")
    for ti, (al, bb) in enumerate(TARGETS):
        x = 380 + ti * 122
        b.append(f"<rect x='{x}' y='{y + 2}' width='60' height='26' rx='5' fill='{BLUE}' fill-opacity='0.13' stroke='{BLUE}'/>")
        b.append(text(x + 30, y + 19, f"b chain", 9.5, 'middle'))
        b.append(arrow(x + 30, y - 9, x + 30, y))                  # drop from the bus into this b chain
        for k in range(2):
            ex = x + 66 + 0 * k
            ey = y + 2 + 30 * k
            b.append(f"<rect x='{ex}' y='{ey}' width='46' height='26' rx='5' fill='{RED}' fill-opacity='0.18' stroke='{RED}'/>")
            b.append(text(ex + 23, ey + 17, 'eig', 9.5, 'middle'))
            b.append(arrow(x + 60, y + 15, ex - 1, ey + 13))
        b.append(text(x + 30, y + 72, f"α={al}", 8.5, 'middle')); b.append(text(x + 30, y + 83, f"b={bb}", 8.5, 'middle'))
n_eig_tasks = 2 * len(TARGETS) * len(GRIDS)
b.append(text(70, 366, f"{len(GRIDS)} independent lanes; {n_eig_tasks} spectra (red) are independent once their b chain is done.  "
                       f"Production run: every box executed one after another in one process.", 11))
open(f'{FIG}/fig_dag.svg', 'w').write(svg_doc(W, H + 8, ''.join(b)))

# Fig. 3: sparsity of the stability matrix, Fourier vs 4th-order differences (small box for visibility)
from newton_fd import Grid
from stability import stability_matrix


def spy(ax, Z):
    """black = nonzero, white = zero, as an embedded PNG (pixel per matrix entry)"""
    n = Z.shape[0]
    rgb = np.where(Z[:, :, None], np.array([20, 20, 20], np.uint8), np.array([255, 255, 255], np.uint8))
    b64 = base64.b64encode(png_bytes(np.ascontiguousarray(rgb.astype(np.uint8)))).decode()
    X0, X1, Y0, Y1 = float(ax.X(0)), float(ax.X(n)), float(ax.Y(0)), float(ax.Y(n))
    ax.items.append(f"<image x='{X0:.1f}' y='{min(Y0, Y1):.1f}' width='{X1 - X0:.1f}' height='{abs(Y1 - Y0):.1f}' "
                    f"preserveAspectRatio='none' image-rendering='pixelated' href='data:image/png;base64,{b64}'/>")


fig = Figure(760, 400)
for j, deriv in enumerate(('fourier', 'fd4')):
    g = Grid(L=6 * np.pi, h=0.4, deriv=deriv)          # M = 94 -> n = 376: small enough to see the structure
    x = g.x_full if hasattr(g, 'x_full') else np.arange(g.M)
    w = 0.3 * np.exp(-x ** 2) + 0j
    Am = stability_matrix(x, w, w, dict(V1=1., V2=1., al=0.7, q=0., b=0.5), g.D2)
    nz = np.abs(Am) > 1e-14
    n = Am.shape[0]
    ax = fig.add(70 + 360 * j, 40, 300, 300, (0, n), (n, 0),
                 title=f"{'Fourier' if deriv == 'fourier' else '4th-order differences'}: "
                       f"{100 * nz.mean():.1f}% nonzero")
    spy(ax, nz)
    for k in range(1, 4):                               # block boundaries of the 4 x 4 block operator
        ax.line([k * n / 4] * 2, [0, n], color='#d62728', width=0.8, dash='3,3')
        ax.line([0, n], [k * n / 4] * 2, color='#d62728', width=0.8, dash='3,3')
fig.save('figures/fig_spy.svg')

# ---------------------------------------------------------------------------- plots from the measurements
fig = Figure(940, 360)
ns = sorted(A[1])
ax = fig.add(80, 40, 420, 260, (100, 1e4), (1e-3, 300), xlabel='matrix size n', ylabel='wall time (s)',
             logx=True, logy=True, title='dense LAPACK kernels (complex)')
for t, dash in ((1, None), (4, '6,3')):
    ax.line(ns, [A[t][n]['eig'] for n in ns], color=RED, width=2, dash=dash, label=f"eigvals, {t} thr.")
    ax.line(ns, [A[t][n]['inv'] for n in ns], color=GREEN, width=1.6, dash=dash, label=f"inv, {t} thr.")
    ax.line(ns, [A[t][n]['solve'] for n in ns], color=BLUE, width=1.4, dash=dash, label=f"solve, {t} thr.")
ref = A[1][2000]['eig']
ax.line([250, 6000], [ref * (250 / 2000) ** 3, ref * (6000 / 2000) ** 3], color=GREY, width=1, dash='2,3', label='∝ n³')
ax.scatter([nP, nstab(56, 0.25)], [A[1][nP]['eig'], t_kernel('eig', nstab(56, 0.25), 1)], color='black', r=3.5)
ax.text(nP * 1.05, A[1][nP]['eig'] * 0.55, 'L = 40π', size=9.5)
ax.text(nstab(56, 0.25) * 1.05, t_kernel('eig', nstab(56, 0.25), 1) * 0.45, '56π', size=9.5)
ax.legend('ul')
ts = sorted(A)
ax2 = fig.add(600, 40, 300, 260, (0.8, 4.2), (0.8, 4.2), xlabel='BLAS threads', ylabel=f'speedup at n = {nP}',
              title='thread scaling')
ax2.line([1, 4], [1, 4], color=GREY, width=1, dash='3,3', label='ideal')
ax2.line(ts, [sp_eig[t] for t in ts], color=RED, width=2, label=f"eigvals (Amdahl p = {p_am_eig:.2f})")
ax2.scatter(ts, [sp_eig[t] for t in ts], color=RED, r=3)
ax2.line(ts, [sp_sol[t] for t in ts], color=BLUE, width=2, label=f"solve (p = {p_am_sol:.2f})")
ax2.scatter(ts, [sp_sol[t] for t in ts], color=BLUE, r=3)
tt = np.linspace(1, 4, 50)
ax2.line(tt, 1 / ((1 - p_am_eig) + p_am_eig / tt), color=RED, width=0.8, dash='2,2')
ax2.legend('ul')
fig.save('figures/fig_dense.svg')


def hbar(ax, y, x0, x1, color, hgt=0.6):
    X0, X1 = float(ax.X(x0)), float(ax.X(x1))
    Yt, Yb = float(ax.Y(y + hgt / 2)), float(ax.Y(y - hgt / 2))
    ax.items.append(f"<rect x='{X0:.1f}' y='{min(Yt, Yb):.1f}' width='{max(X1 - X0, 0.5):.1f}' height='{abs(Yb - Yt):.1f}' "
                    f"fill='{color}' fill-opacity='0.75'/>")


fig = Figure(980, 380)
names = [l for _, l in labels]
ax = fig.add(250, 40, 300, 280, (0, max(ST[k] for k, _ in labels) * 1.15), (-0.7, len(labels) - 0.3),
             xlabel='wall time (s)', title=f"one convergence row, L = 40π (total {tot_row:.0f} s)")
ax.pad_left, ax.noyticks = 235, True                  # room for the stage names; no numeric y ticks
for i, (k, lab) in enumerate(labels):
    y = len(labels) - 1 - i
    col = RED if k.startswith('eig') else (ORANGE if k in ('floquet_tails', 'shoot_polish', 'shoot_profile') else BLUE)
    hbar(ax, y, 0, ST[k], col)
    plain = lab.replace('$\\alpha=0$', 'α = 0').replace('$\\alpha$', 'α').replace('$b$', 'b').replace('$D_2$', 'D₂') \
               .replace('$g_0=1$', 'g₀ = 1').replace('$g_0=0.25$', 'g₀ = 0.25')
    ax.outside = getattr(ax, 'outside', []) + [f"<text x='{ax.x - 6}' y='{float(ax.Y(y)) + 3.5:.1f}' font-size='10' "
                                              f"text-anchor='end' {FONT}>{plain}</text>"]
if rows:
    tmax = max(r['t'] for r in rows) * 1.15
    ax2 = fig.add(640, 40, 310, 280, (0.4, len(rows) + 0.6), (0, tmax), xlabel='row of the convergence study',
                  ylabel='wall time (s)', title='production run: measured (bars) vs model (dots)')
    gcol = {(40, 0.25): BLUE, (40, 0.2): GREEN, (56, 0.25): RED}
    for i, r in enumerate(rows):
        X0, X1 = float(ax2.X(i + 1 - 0.35)), float(ax2.X(i + 1 + 0.35))
        Y0, Y1 = float(ax2.Y(0)), float(ax2.Y(r['t']))
        ax2.items.append(f"<rect x='{X0:.1f}' y='{Y1:.1f}' width='{X1 - X0:.1f}' height='{Y0 - Y1:.1f}' "
                         f"fill='{gcol[(r['L_pi'], r['h'])]}' fill-opacity='0.6'/>")
    # model per row with the 3-thread kernel times (as run): b chain + 2 spectra (+ setup and alpha chain on a grid's first row)
    mod = []
    for i, r in enumerate(rows):
        sc = grid_scale(r['L_pi'], r['h'])
        m = t_bstep40 * sc * nbsteps(r['al'], r['b']) + 2 * t_kernel('eig', nstab(r['L_pi'], r['h']), 3)
        if i == 0 or (rows[i - 1]['L_pi'], rows[i - 1]['h']) != (r['L_pi'], r['h']):
            m += t_shot + t_grid40 * sc + t_alpha40 * sc
        mod.append(m)
    ax2.scatter(range(1, len(rows) + 1), mod, color='black', r=3)
    r4 = [r for r in rows if (r['L_pi'], r['h'], r['al'], r['b']) == (40, 0.25, 0.7919, -1.262)][0]
    mac('tSpecProd', f"{(r4['t'] - t_bstep40 * 11) / 2:.1f}")       # same matrix as Table anatomy, production run
    mac('modelRowErr', f"{100 * max(abs(a - r['t']) / r['t'] for a, r in zip(mod, rows)):.0f}")
    for (Lpi, h), col in gcol.items():
        ax2.legend_items.append((f"L = {Lpi}π, h = {h}", col, 6, None, 'line'))
    ax2.legend('ul')
fig.save('figures/fig_anatomy.svg')

fig = Figure(940, 360)
Ms = sorted(ss)
ax = fig.add(80, 40, 380, 260, (150, 1e5), (10, 1e5), xlabel='grid points M', ylabel='µs per Strang step',
             logx=True, logy=True, title='split-step Fourier: cost per step')
ax.line(Ms, [ss[M]['us_per_step'] for M in Ms], color=ORANGE, width=2, label='measured')
ax.scatter(Ms, [ss[M]['us_per_step'] for M in Ms], color=ORANGE, r=3)
Mb = Ms[-1]
ax.line(Ms, [ss[Mb]['us_per_step'] * (M * math.log2(M)) / (Mb * math.log2(Mb)) for M in Ms],
        color=GREY, width=1, dash='3,3', label='∝ M log M (flop count)')
ax.legend('ul')
Ks = sorted(sb)
ax2 = fig.add(560, 40, 340, 260, (0.8, 6000), (1e-3, 1e5), xlabel='batch size K (independent problems)',
              ylabel='µs per problem', logx=True, logy=True, title='batching into one array')
ax2.line(Ks, [sb[K] for K in Ks], color=ORANGE, width=2, label='split-step step, M = 1024')
ax2.scatter(Ks, [sb[K] for K in Ks], color=ORANGE, r=3)
Kr = sorted(rv)
ax2.line(Kr, [rv[K] for K in Kr], color=BLUE, width=2, label='shooting RHS, vectorized')
ax2.scatter(Kr, [rv[K] for K in Kr], color=BLUE, r=3)
ax2.line([0.8, 6000], [D['rhs_scalar_us']] * 2, color=BLUE, width=1, dash='4,3', label='shooting RHS, scalar Python')
ax2.legend('ul')
fig.save('figures/fig_interp.svg')

for name in ('fig_pipeline', 'fig_dag', 'fig_spy', 'fig_dense', 'fig_anatomy', 'fig_interp'):
    convert(f'{FIG}/{name}.svg', preview_dir=f'{FIG}/preview')
    print('figure', name)
# numbers.tex last: some macros (e.g. modelRowErr) are computed while drawing the figures
with open(f'{OUT}/numbers.tex', 'w') as f:
    f.write("% generated by make_perf.py from results/perf.json, arnoldi.json -- do not edit\n")
    for k, v in macros.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print('numbers.tex:', len(macros), 'macros')
