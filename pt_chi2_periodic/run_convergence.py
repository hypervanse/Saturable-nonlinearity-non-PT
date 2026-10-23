"""
run_convergence.py -- grid / box / absorber convergence of the third-gap growth rates that the
manuscript reports as new (paper Fig. 5 draws these branches as stable).

For each grid (L, h) the alpha = 0 soliton is rebuilt from the stored edge shot (c1, c2 polished
again by 2-D Newton), splined onto the grid, solved, continued in alpha at b = -1.207, then in b to
each target.  The growth rate max Im(lambda) is computed with absorbing layers of strength
g0 = 1 and 0.25 (a physical rate must not depend on g0, a box artefact does).
Output: results/convergence.json
"""
import json
import time
import numpy as np

from shooting import tails, newton_polish, profile_from_shot
from newton_fd import Grid, solve, continuation, power_half
from pipeline import to_grid, solution_at, stability_of
from io_utils import dump

T0 = time.time()
X0 = np.pi / 2
par0 = dict(V1=1.0, V2=1.0, al=0.0, q=0.0, b=-1.207, x0=X0)
sh0 = json.load(open('results/thirdgap.json'))['shoot']          # converged amplitudes of the production run

# edge shot rebuilt once: Newton polish from the stored (c1, c2) needs only a few shots
T = tails(par0, 8)
c1, c2 = newton_polish(par0, T, sh0['c1'], sh0['c2'], verbose=False)
xs, w1, p1, w2, p2, info = profile_from_shot(par0, T, c1, c2)
sh = dict(xs=xs, w1=w1, p1=p1, w2=w2, p2=p2)
print(f"shot: c1={c1:.10f} c2={c2:+.10f}  (stored {sh0['c1']:.10f} {sh0['c2']:+.10f})")

GRIDS = [(40, 0.25), (40, 0.20), (56, 0.25)]                     # (L / pi, h): production, finer h, larger box
TARGETS = [(0.0, -1.197), (0.0, -1.282), (0.7, -1.282), (0.7919, -1.262), (0.7919, -1.207)]
G0S = (1.0, 0.25)                                                # absorber strengths

rows = []
for Lpi, h in GRIDS:
    g = Grid(L=Lpi * np.pi, h=h, deriv='fourier')
    v1, v2, it, res = solve(g, *to_grid(g, sh), par0)              # alpha = 0, b = -1.207
    start = {0.0: (v1, v2)}
    br = continuation(g, v1, v2, par0, 'al', list(np.round(np.arange(0.05, 0.7501, 0.05), 3)) + [0.7919],
                      verbose=False)
    for r in br:
        if r['al'] in (0.7, 0.7919):
            start[r['al']] = (r['w1'], r['w2'])
    for al, b in TARGETS:
        s1, s2 = start[al]
        par = dict(par0, al=al)
        e1, e2 = solution_at(g, s1, s2, par, b, db=0.005)
        pb = dict(par, b=b)
        P = sum(power_half(g, e1, e2))
        row = dict(L_pi=Lpi, h=h, M=g.M, al=al, b=b, P=P)
        for g0 in G0S:
            A = stability_of(g, dict(w1=e1, w2=e2), pb, g0=g0)
            row[f'g_{g0}'] = A['gmax']
            row[f'lam_{g0}'] = A['lam_unst']
        rows.append(row)
        lam = row['lam_1.0']
        print(f"L={Lpi}pi h={h} al={al:<6} b={b:+.3f} P={P:.5f}  growth g0=1: {row['g_1.0']:.3e}  "
              f"g0=0.25: {row['g_0.25']:.3e}  Re lam={lam.real if lam is not None else float('nan'):+.4f}  "
              f"[{time.time() - T0:.0f}s]", flush=True)

dump('results/convergence.json', dict(grids=GRIDS, targets=TARGETS, g0=G0S, rows=rows))
print(f"all done [{time.time() - T0:.0f}s]")
