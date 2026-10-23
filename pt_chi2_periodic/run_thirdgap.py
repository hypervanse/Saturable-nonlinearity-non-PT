"""
run_thirdgap.py -- third finite gap, V1 = V2 = 1, q = 0 (paper Figs. 5, 6, 8).

The fundamental gap soliton here is centred on the potential MINIMUM, x0 = pi/2:
at al = 0 the FF band-edge Bloch mode of this gap is odd about x = 0 (sin x-like)
and even about x = pi/2.  x is measured from x0 throughout.

  edge shooting at al = 0, b = -1.207  ->  spline  ->  Newton
  al = 0 branch;  al-continuation to 0.7 and 0.7919 at b = -1.207;
  al = 0.9, 0.95 reached at b = -1.10 (their branches start at larger b)
  every branch traced in b with the linear spectrum at every 2nd point
  profiles at the paper's examples; spectra + split-step runs at al = 0.7919.
Output: results/thirdgap.json
"""
import os
import time
import numpy as np

from model import band_edges, Q_ff, Q_sh
from newton_fd import Grid, solve, continuation, full_line, power_half
from pipeline import shoot, to_grid, trace_branch, solution_at, stability_of, currents_full, ssf_run, fourier_refine
from io_utils import dump

os.makedirs('results', exist_ok=True)
T0 = time.time()
X0 = np.pi / 2
g = Grid(L=40 * np.pi, h=0.25, deriv='fourier')         # M = 1006 points on [-40pi, 40pi)
b_sh = band_edges(lambda b: Q_sh(1.0, 0.0, b), -1.35, -1.25, nscan=101).max()       # SH edge -1.2932
ff_edge = {}
for al in (0.0, 0.7, 0.7919, 0.9, 0.95):
    e = band_edges(lambda b: Q_ff(1.0, al, b), -1.5, -1.05, nscan=181)
    ff_edge[al] = e.max() if len(e) else None
    print(f"al={al}: FF edge (top of band 1) = {ff_edge[al]}")
print(f"SH edge (top of SH band) = {b_sh:.5f}")

par0 = dict(V1=1.0, V2=1.0, al=0.0, q=0.0, b=-1.207, x0=X0)
sh = shoot(par0, 8, 0.9, 0.1, 0.05, 0.3)
w1, w2 = to_grid(g, sh)
w1, w2, it, nF = solve(g, w1, w2, par0)
print(f"al=0, b=-1.207: P = {sum(power_half(g, w1, w2)):.5f}")

out = dict(meta=dict(V1=1.0, V2=1.0, q=0.0, x0=X0, L=g.L, h=g.h, M=g.M, deriv=g.deriv,
                     b_sh=b_sh, ff_edge={str(k): v for k, v in ff_edge.items()}),
           shoot=dict(c1=sh['c1'], c2=sh['c2'], shots=sh['shots'], mesh=len(sh['xs']),
                      w1_0=sh['w1'][0].real, w2_0=sh['w2'][0].real, mu1=sh['mu1'], mu2=sh['mu2'],
                      xs=sh['xs'], w1=sh['w1'].real, w2=sh['w2'].real),
           branches={}, profiles={}, spectra={}, ssf={})

# starting solutions for each al
starts = {0.0: (-1.207, w1, w2)}
br = continuation(g, w1, w2, par0, 'al', list(np.round(np.arange(0.05, 0.7501, 0.05), 3)) + [0.7919],
                  verbose=False)
for r in br:
    if r['al'] in (0.7, 0.7919):
        starts[r['al']] = (-1.207, r['w1'], r['w2'])
v1, v2 = solution_at(g, w1, w2, par0, -1.10, db=0.01)
br = continuation(g, v1, v2, dict(par0, b=-1.10), 'al', list(np.round(np.arange(0.05, 0.9501, 0.05), 3)),
                  verbose=False)
for r in br:
    if r['al'] in (0.9, 0.95):
        starts[r['al']] = (-1.10, r['w1'], r['w2'])

for al in (0.0, 0.7, 0.7919, 0.9, 0.95):
    b0, s1, s2 = starts[al]
    par = dict(par0, al=al, b=b0)
    print(f"\n=== branch al = {al}  (start b = {b0}) ===")
    lo = ff_edge[al] if (ff_edge[al] is not None and ff_edge[al] > b_sh) else b_sh
    # broad solitons near the FF edge: accept tails up to 5e-3 of the peak amplitude at |x| = L
    up = trace_branch(g, s1, s2, par, np.round(np.arange(b0 + 0.01, -0.995, 0.01), 4), stab_every=2, tail_tol=5e-3)
    dn = trace_branch(g, s1, s2, par, np.round(np.arange(b0 - 0.005, lo, -0.005), 4), stab_every=2, tail_tol=5e-3)
    A0 = stability_of(g, dict(w1=s1, w2=s2), par)
    P1, P2 = power_half(g, s1, s2)
    mid = dict(b=b0, P=P1 + P2, P1=P1, P2=P2, gmax=A0['gmax'], lam_unst=A0['lam_unst'])
    out['branches'][str(al)] = dict(al=al, edge=lo, points=dn[::-1] + [mid] + up)
    print(f"  [{time.time() - T0:.0f}s total]")

# example profiles (paper Fig. 6): case 1 al=0.7919 b=-1.207, case 2 al=0.7 b=-1.234, case 3 al=0.9 b=-1.153
for name, al, bex in (('case1', 0.7919, -1.207), ('case2', 0.7, -1.234), ('case3', 0.9, -1.153)):
    b0, s1, s2 = starts[al]
    e1, e2 = solution_at(g, s1, s2, dict(par0, al=al, b=b0), bex, db=0.005)
    x, W1, W2 = full_line(g, e1, e2)
    j1, j2 = currents_full(g, W1, W2)
    xr, W1r = fourier_refine(x, W1)                      # 4x band-limited refinement for smooth plots
    _, W2r = fourier_refine(x, W2)
    _, j1r = fourier_refine(x, j1)
    _, j2r = fourier_refine(x, j2)
    sel = np.abs(xr) <= 5 * np.pi
    P = sum(power_half(g, e1, e2))
    out['profiles'][name] = dict(al=al, b=bex, P=P, x=xr[sel], I1=(np.abs(W1r) ** 2)[sel],
                                 I2=(np.abs(W2r) ** 2)[sel], j1=j1r[sel].real, j2=j2r[sel].real)
    print(f"profile {name}: al={al} b={bex} P={P:.5f}")
    if name == 'case1':
        c1w, c2w = e1, e2

# spectra + propagation at al = 0.7919 (paper Fig. 8): b = -1.207 stable, b = -1.101 unstable
parS = dict(par0, al=0.7919, b=-1.207)
for name, bS, zmax, dz in (('b-1.207', -1.207, 1000.0, 0.01), ('b-1.101', -1.101, 100.0, 0.005)):
    f1, f2 = solution_at(g, c1w, c2w, parS, bS, db=0.005)
    p = dict(parS, b=bS)
    A = stability_of(g, dict(w1=f1, w2=f2), p)
    keep = (np.abs(A['lam'].real) < 4) & (A['lam'].imag > -0.6)
    out['spectra'][name] = dict(b=bS, lam=A['lam'][keep], gmax=A['gmax'], lam_unst=A['lam_unst'])
    r = ssf_run(g, f1, f2, p, zmax, dz=dz, noise=0.1, seed=1, nsave=250)
    xs = slice(None, None, 3)
    sel = np.abs(r['x'][xs]) <= 16 * np.pi
    out['ssf'][name] = dict(b=bS, zmax=zmax, z=r['z'], x=r['x'][xs][sel], I1=r['I1'][:, xs][:, sel],
                            P=r['P'], peak=np.max(r['I1'], axis=1), peak0=np.max(np.abs(r['W1']) ** 2))
    print(f"{name}: growth={A['gmax']:.3e} at {A['lam_unst']} | ssf z={zmax}: "
          f"P {r['P'][0]:.4f} -> {r['P'][-1]:.4f} [{r['time']:.0f}s]")

dump('results/thirdgap.json', out)
print(f"\nall done [{time.time() - T0:.0f}s]")
