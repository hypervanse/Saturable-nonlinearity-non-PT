"""
validate_arnoldi.py -- shift-invert Arnoldi (stability.growth_arnoldi) against the dense solver
(stability.growth_cap) on the same matrices, plus the symmetry the shift scan relies on.

For each test soliton (rebuilt from the stored edge shots: Newton polish -> spline -> Newton ->
continuation in alpha, then b):
  dense     all eigenvalues, max Im(lam)                        (reference)
  scan      Arnoldi at the default shifts 0..3 (+0.05i)          (blind search, |Re lam| <= 3)
  target    Arnoldi at one shift next to the dense eigenvalue   (refinement mode, e.g. on a bigger box)
  symmetry  max over the dense spectrum of dist(-conj(lam), spectrum)   (should be ~1e-10)
If SciPy is importable it also checks that the sparse assembly equals the dense matrix and runs
the sparse backend with the fd4 derivative.  Output: results/arnoldi.json
"""
import json
import time
import numpy as np

import stability
from stability import growth_cap, growth_arnoldi, stability_matrix
from shooting import tails, newton_polish, profile_from_shot
from newton_fd import Grid, solve, continuation, full_line
from pipeline import to_grid, solution_at
from io_utils import dump

T0 = time.time()
S = json.load(open('results/semiinf.json'))
Tg = json.load(open('results/thirdgap.json'))


def soliton(g, par0, n, c1, c2, al, b):
    """alpha = 0 shot at par0 (amplitudes polished from the stored ones) -> grid -> Newton ->
    continuation to (al, b).  Returns the full-box x, W1, W2 and the parameters."""
    T = tails(par0, n)
    c1, c2 = newton_polish(par0, T, c1, c2, verbose=False)
    xs, w1, p1, w2, p2, info = profile_from_shot(par0, T, c1, c2)
    v1, v2, it, res = solve(g, *to_grid(g, dict(xs=xs, w1=w1, p1=p1, w2=w2, p2=p2)), par0)
    if al > 0:
        als = list(np.round(np.arange(0.05, al, 0.05), 3)) + [al]
        br = continuation(g, v1, v2, par0, 'al', als, verbose=False)
        v1, v2 = br[-1]['w1'], br[-1]['w2']
    par = dict(par0, al=al)
    v1, v2 = solution_at(g, v1, v2, par, b, db=0.005)
    x, W1, W2 = full_line(g, v1, v2)
    return x, W1, W2, dict(par, b=b)


c1s, c3s = S['cases']['case1'], S['cases']['case3']
sg = Tg['shoot']
TESTS = [  # label, grid, alpha=0 shooting parameters, n periods, stored (c1, c2), target (al, b)
    ('semi case 1, b=0.5', Grid(12 * np.pi, 0.2, 'fourier'),
     dict(V1=1., V2=1., al=0., q=c1s['q'], b=c1s['b_shoot'], x0=0.), c1s['n_periods'], c1s['shoot'], (0.7, 0.5)),
    ('semi case 1, b=0.5, fd4', Grid(12 * np.pi, 0.1, 'fd4'),
     dict(V1=1., V2=1., al=0., q=c1s['q'], b=c1s['b_shoot'], x0=0.), c1s['n_periods'], c1s['shoot'], (0.7, 0.5)),
    ('semi case 3, b=0.25', Grid(12 * np.pi, 0.2, 'fourier'),
     dict(V1=1., V2=1., al=0., q=c3s['q'], b=c3s['b_shoot'], x0=0.), c3s['n_periods'], c3s['shoot'], (0.7, 0.25)),
    ('gap3 al=0, b=-1.197', Grid(28 * np.pi, 0.25, 'fourier'),
     dict(V1=1., V2=1., al=0., q=0., b=-1.207, x0=np.pi / 2), 8, sg, (0.0, -1.197)),
    ('gap3 al=0.7919, b=-1.207', Grid(28 * np.pi, 0.25, 'fourier'),
     dict(V1=1., V2=1., al=0., q=0., b=-1.207, x0=np.pi / 2), 8, sg, (0.7919, -1.207)),
    ('gap3 al=0.7919, b=-1.101', Grid(28 * np.pi, 0.25, 'fourier'),
     dict(V1=1., V2=1., al=0., q=0., b=-1.207, x0=np.pi / 2), 8, sg, (0.7919, -1.101)),
]

rows = []
for label, g, par0, n, sh, (al, b) in TESTS:
    x, W1, W2, par = soliton(g, par0, n, sh['c1'], sh['c2'], al, b)
    nA = 4 * len(x)
    t = time.time(); gd, ld, lam_all = growth_cap(x, W1, W2, par, g.D2, g0=1.0); td = time.time() - t
    sym = max(np.min(np.abs(lam_all - (-np.conj(l)))) for l in lam_all)       # lam -> -conj(lam)
    t = time.time(); gs, ls, fs = growth_arnoldi(x, W1, W2, par, g.D2, g0=1.0, backend='dense'); ts = time.time() - t
    # target mode: one shift offset from the reference eigenvalue (as a guess from a coarser grid would be)
    guess = complex(abs(ld.real), ld.imag) + (0.02 + 0.01j) if gd > 0 else 0.05j
    t = time.time(); gt, lt, ft = growth_arnoldi(x, W1, W2, par, g.D2, g0=1.0, shifts=[guess], backend='dense'); tt = time.time() - t
    row = dict(label=label, n=nA, deriv=g.deriv, dense=dict(g=gd, lam=ld, t=td),
               scan=dict(g=gs, lam=ls, t=ts, nfound=int(fs.size)),
               target=dict(g=gt, lam=lt, t=tt), symmetry=sym)
    if stability.HAVE_SCIPY:                    # sparse assembly == dense, and the sparse backend
        A_d = stability_matrix(x, W1, W2, par, g.D2)
        A_s = stability.stability_matrix_sparse(x, W1, W2, par, g.D2).toarray()
        row['sparse_assembly_err'] = float(np.abs(A_d - A_s).max())
        if g.deriv == 'fd4':
            t = time.time(); gp, lp, fp = growth_arnoldi(x, W1, W2, par, g.D2, g0=1.0, backend='sparse')
            row['sparse'] = dict(g=gp, lam=lp, t=time.time() - t)
    rows.append(row)
    fmt = lambda l: f"{l.real:+.5f}{l.imag:+.6f}i" if l is not None else "None"
    print(f"{label:26s} n={nA:5d}  dense {gd:.6e} {fmt(ld)} [{td:5.1f}s] | scan {gs:.6e} {fmt(ls)} "
          f"[{ts:5.1f}s, {fs.size} eigs] | target {gt:.6e} [{tt:4.1f}s] | sym {sym:.1e}"
          + (f" | sparse-assembly err {row['sparse_assembly_err']:.1e}" if 'sparse_assembly_err' in row else "")
          + (f" | sparse backend {row['sparse']['g']:.6e} [{row['sparse']['t']:.2f}s]" if 'sparse' in row else ""),
          flush=True)

dump('results/arnoldi.json', dict(have_scipy=stability.HAVE_SCIPY, rows=rows))
print(f"all done [{time.time() - T0:.0f}s]")
