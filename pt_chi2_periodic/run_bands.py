"""
run_bands.py -- band-gap spectrum (paper Fig. 1) from Floquet discriminants.

(a) bands of FF and SH vs b at al = 0.4 (V1 = V2 = 1, q = 0)
(b) FF band edges vs al in [0, 1]; SH edges do not depend on al
Also prints the matching mismatches q_+^(0,0) used for semi-infinite case 1.
Output: results/bands.json
"""
import os
import time
import numpy as np

from model import band_edges, Q_ff, Q_sh
from io_utils import dump

os.makedirs('results', exist_ok=True)
t0 = time.time()
V1 = V2 = 1.0
blo, bhi = -1.6, 1.0

# SH edges (al-independent), q = 0
sh = band_edges(lambda b: Q_sh(V2, 0.0, b), blo, bhi, nscan=521)
print("SH edges:", np.round(sh, 5))

# FF edges vs al
alphas = np.round(np.linspace(0.0, 1.0, 26), 4)
ff = []
for al in alphas:
    e = band_edges(lambda b: Q_ff(V1, al, b), blo, bhi, nscan=521)
    ff.append(e)
    print(f"al={al:.2f}  FF edges: {np.round(e, 5)}")

# matching values quoted in the paper
ff07 = band_edges(lambda b: Q_ff(V1, 0.7, b), -0.2, 0.3, nscan=201)
q00 = ff07.max() - sh.max()
print(f"q_+^(0,0)(al=0.7) = b_1+^(0) - b_2+^(0) = {ff07.max():.5f} - {sh.max():.5f} = {q00:.5f}  (paper: -0.316)")
ff_al = band_edges(lambda b: Q_ff(V1, 0.7919, b), -1.35, -1.25, nscan=101)
print(f"FF edge near -1.29 at al=0.7919: {ff_al}  vs SH edge {sh.min():.5f}  (paper: coalesce at al~0.792)")

dump('results/bands.json', dict(V1=V1, V2=V2, b_range=[blo, bhi], sh_edges=sh,
                                alphas=alphas, ff_edges=[list(e) for e in ff], q00_al07=q00,
                                ff_edges_al07=ff07))
print(f"done [{time.time() - t0:.0f}s]")
