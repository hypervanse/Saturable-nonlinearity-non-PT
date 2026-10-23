"""
run_semiinf.py -- semi-infinite gap, V1 = V2 = 1, al = 0.7 (paper Figs. 3, 4, 7, 18).

For each case (1: q = q_+^(0,0) band-edge matching, 2: q = 0, 3: q = -0.5134):
  edge shooting at al = 0  ->  spline  ->  Newton  ->  al-continuation to 0.7
  ->  b-continuation along the branch with the linear spectrum at every point
  ->  profiles / currents at the paper's example b
Then split-step runs with 10% noise: case 1 b = 0.25 (z = 1000) and b = 0.5 (z = 40),
case 3 b = 0.21 (z = 1000) and b = 0.25 (z = 100).
Output: results/semiinf.json  (+ results/semiinf_fields.npz)
"""
import os
import time
import numpy as np

from model import band_edges, Q_ff, Q_sh
from newton_fd import Grid, solve, continuation, full_line, power_half
from pipeline import shoot, to_grid, trace_branch, solution_at, stability_of, currents_full, ssf_run
from io_utils import dump

os.makedirs('results', exist_ok=True)
T0 = time.time()
AL = 0.7
g = Grid(L=16 * np.pi, h=0.15, deriv='fourier')         # M = 670 points on [-16pi, 16pi)

b1p = band_edges(lambda b: Q_ff(1.0, AL, b), -0.2, 0.3, nscan=201).max()     # FF edge at al = 0.7
b2p = band_edges(lambda b: Q_sh(1.0, 0.0, b), 0.2, 0.5, nscan=121).max()     # SH edge, q = 0
q00 = b1p - b2p
print(f"b_1+^(0)(al=0.7) = {b1p:.5f}, b_2+^(0) = {b2p:.5f}, q_+^(0,0) = {q00:.5f}")

cases = [
    # name, q, b_shoot, n periods, c1_start, dc1, dc2, c2_guess, example b (paper Fig. 4), bifurcation edge
    dict(name='case1', q=q00, b0=0.25, n=6, c1s=0.05, dc1=0.05, dc2=0.1, c2g=0.0, bex=0.25, edge=b1p),
    dict(name='case2', q=0.0, b0=0.43, n=6, c1s=0.02, dc1=0.05, dc2=0.1, c2g=-1.0, bex=0.43, edge=b2p),
    dict(name='case3', q=-0.5134, b0=0.21, n=4, c1s=0.05, dc1=0.05, dc2=0.1, c2g=-1.0, bex=0.21, edge=b1p),
]
out = dict(meta=dict(V1=1.0, V2=1.0, al=AL, L=g.L, h=g.h, M=g.M, deriv=g.deriv, b1p=b1p, b2p=b2p, q00=q00),
           cases={}, ssf={}, spectra={})
fields = {}

for C in cases:
    print(f"\n=== {C['name']}  q = {C['q']:+.5f} ===")
    par0 = dict(V1=1.0, V2=1.0, al=0.0, q=C['q'], b=C['b0'])
    sh = shoot(par0, C['n'], C['c1s'], C['dc1'], C['dc2'], C['c2g'])
    w1, w2 = to_grid(g, sh)
    w1, w2, it, nF = solve(g, w1, w2, par0)
    P_al0 = sum(power_half(g, w1, w2))
    br = continuation(g, w1, w2, par0, 'al', np.round(np.arange(0.05, AL + 1e-9, 0.05), 3), verbose=False)
    par = dict(par0, al=AL)
    wA1, wA2 = br[-1]['w1'], br[-1]['w2']
    print(f"  al=0 P={P_al0:.5f} -> al={AL} P={br[-1]['P']:.5f}")

    # branch: up in b to 1.0, down to the bifurcation edge
    up = trace_branch(g, wA1, wA2, par, np.round(np.arange(C['b0'] + 0.02, 1.0001, 0.02), 4))
    dn = trace_branch(g, wA1, wA2, par, np.round(np.arange(C['b0'] - 0.01, C['edge'], -0.01), 4))
    p0 = dict(par)
    A0 = stability_of(g, dict(w1=wA1, w2=wA2), p0)
    P1, P2 = power_half(g, wA1, wA2)
    mid = dict(b=C['b0'], P=P1 + P2, P1=P1, P2=P2, gmax=A0['gmax'], lam_unst=A0['lam_unst'])
    branch = dn[::-1] + [mid] + up

    # example profile (paper Fig. 4) and currents (Eq. 11)
    e1, e2 = solution_at(g, wA1, wA2, par, C['bex'])
    x, W1, W2 = full_line(g, e1, e2)
    j1, j2 = currents_full(g, W1, W2)
    sel = np.abs(x) <= 4 * np.pi
    Pex = sum(power_half(g, e1, e2))
    fields[C['name']] = (e1, e2)
    out['cases'][C['name']] = dict(
        q=C['q'], b_shoot=C['b0'], n_periods=C['n'],
        shoot=dict(c1=sh['c1'], c2=sh['c2'], shots=sh['shots'], mesh=len(sh['xs']),
                   w1_0=sh['w1'][0].real, w2_0=sh['w2'][0].real, mu1=sh['mu1'], mu2=sh['mu2'],
                   xs=sh['xs'], w1=sh['w1'].real, w2=sh['w2'].real),
        P_al0=P_al0, branch=branch,
        profile=dict(b=C['bex'], P=Pex, x=x[sel], I1=np.abs(W1[sel]) ** 2, I2=np.abs(W2[sel]) ** 2,
                     j1=j1[sel], j2=j2[sel]))
    print(f"  example b={C['bex']}: P={Pex:.5f}  [{time.time() - T0:.0f}s total]")


def spectrum_record(name, w1, w2, par):
    A = stability_of(g, dict(w1=w1, w2=w2), par)
    keep = (np.abs(A['lam'].real) < 6) & (A['lam'].imag > -0.6)
    out['spectra'][name] = dict(b=par['b'], lam=A['lam'][keep], gmax=A['gmax'], lam_unst=A['lam_unst'])
    print(f"  spectrum {name}: growth = {A['gmax']:.3e} at {A['lam_unst']}")


def ssf_record(name, w1, w2, par, zmax, dz):
    r = ssf_run(g, w1, w2, par, zmax, dz=dz, noise=0.1, seed=1, nsave=250)
    xs = slice(None, None, 2)
    sel = np.abs(r['x'][xs]) <= 8 * np.pi
    out['ssf'][name] = dict(b=par['b'], zmax=zmax, z=r['z'], x=r['x'][xs][sel],
                            I1=r['I1'][:, xs][:, sel], P=r['P'],
                            peak=np.max(r['I1'], axis=1), peak0=np.max(np.abs(r['W1']) ** 2))
    print(f"  ssf {name}: z={zmax} P0={r['P'][0]:.4f} Pend={r['P'][-1]:.4f} "
          f"peak/peak0 end={np.max(r['I1'][-1]) / np.max(np.abs(r['W1']) ** 2):.3f} [{r['time']:.0f}s]")


print("\n=== spectra and propagation (paper Figs. 7, 18) ===")
par1 = dict(V1=1.0, V2=1.0, al=AL, q=q00, b=0.25)
a1, a2 = fields['case1']
spectrum_record('case1_b0.25', a1, a2, par1)
ssf_record('case1_b0.25', a1, a2, par1, 1000.0, 0.01)
b1, b2 = solution_at(g, a1, a2, par1, 0.5)
spectrum_record('case1_b0.50', b1, b2, dict(par1, b=0.5))
ssf_record('case1_b0.50', b1, b2, dict(par1, b=0.5), 40.0, 0.005)

par3 = dict(V1=1.0, V2=1.0, al=AL, q=-0.5134, b=0.21)
c1, c2 = fields['case3']
spectrum_record('case3_b0.21', c1, c2, par3)
ssf_record('case3_b0.21', c1, c2, par3, 1000.0, 0.01)
d1, d2 = solution_at(g, c1, c2, par3, 0.25)
spectrum_record('case3_b0.25', d1, d2, dict(par3, b=0.25))
ssf_record('case3_b0.25', d1, d2, dict(par3, b=0.25), 100.0, 0.005)

dump('results/semiinf.json', out)
np.savez('results/semiinf_fields.npz', X=g.X, **{f"{k}_w1": v[0] for k, v in fields.items()},
         **{f"{k}_w2": v[1] for k, v in fields.items()})
print(f"\nall done [{time.time() - T0:.0f}s]")
