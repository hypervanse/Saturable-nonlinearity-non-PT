"""
validate_weak.py -- confirm the weak instabilities (below the paper's resolution) by direct propagation.

For each case: absorbing-layer spectrum -> most unstable eigenvalue -> its eigenvector by one step of
inverse iteration, (A - sigma I) v = r with sigma = lam + 1e-9 -> seed the split-step run with
eps * (a + conj c, d + conj f) (Eq. 12 at z = 0) -> fit the envelope growth of the |u1|^2 deviation.
Output: results/weak.json
"""
import numpy as np

from pipeline import shoot, to_grid
from newton_fd import Grid, solve, continuation, full_line
from spline import spline_resample
from stability import stability_matrix, cap_profile
from ssf import propagate, absorber
from io_utils import dump


def check(label, g, w1, w2, par, zmax, dz, eps=1e-5, g0=1.0):
    x, W1, W2 = full_line(g, w1, w2)
    A = stability_matrix(x, W1, W2, par, g.D2)
    A[np.diag_indices_from(A)] -= 1j * np.tile(cap_profile(x, g.L, g0=g0), 4)
    lam = np.linalg.eigvals(A)
    lu = lam[np.argmax(lam.imag)]
    rng = np.random.default_rng(0)
    v = np.linalg.solve(A - (lu + 1e-9) * np.eye(len(A)), rng.standard_normal(len(A)))   # inverse iteration
    M = len(x)
    d, a, f, c = v[:M], v[M:2 * M], v[2 * M:3 * M], v[3 * M:]
    s = eps / np.max(np.abs(np.concatenate((a, c))))
    k = 2 * np.pi * np.fft.fftfreq(g.M, d=g.h)
    r = propagate(W1 + s * (a + np.conj(c)), W2 + s * (d + np.conj(f)), x, k, par, dz, zmax,
                  nsave=2000, order=2, g=absorber(x, g.L, 0.15, g0), verbose=False)
    dev = np.sqrt(np.sum((r['I1'] - np.abs(W1) ** 2) ** 2, axis=1))
    pk = [j for j in range(1, len(dev) - 1) if dev[j] > dev[j - 1] and dev[j] > dev[j + 1]]
    zp, dp = r['z'][pk], dev[pk]
    mm = (dp < 100 * dp[0]) & (zp > 0.05 * zmax)
    rate = np.polyfit(zp[mm], np.log(dp[mm]), 1)[0]
    print(f"{label}: Im(lam) = {lu.imag:.4e} (Re {lu.real:+.4f});  split-step envelope growth = {rate:.4e}  "
          f"(fit over z in [{zp[mm][0]:.0f}, {zp[mm][-1]:.0f}])", flush=True)
    return dict(lam=lu, ssf_rate=rate, z=r['z'][::4], dev=dev[::4])


out = {}
# case 3, semi-infinite gap, b = 0.21, al = 0.7
g = Grid(L=16 * np.pi, h=0.15, deriv='fourier')
par3 = dict(V1=1.0, V2=1.0, al=0.0, q=-0.5134, b=0.21)
sh = shoot(par3, 4, 0.05, 0.05, 0.1, -1.0, verbose=False)
w1, w2 = to_grid(g, sh)
w1, w2, *_ = solve(g, w1, w2, par3)
s3 = continuation(g, w1, w2, par3, 'al', np.round(np.arange(0.05, 0.7001, 0.05), 3), verbose=False)[-1]
out['case3_b0.21'] = check("case 3, b = 0.21, al = 0.7", g, s3['w1'], s3['w2'], dict(par3, al=0.7), 1500.0, 0.01)

# third gap, al = 0.7919, b = -1.207 (x0 = pi/2)
g = Grid(L=28 * np.pi, h=0.2, deriv='fourier')
par = dict(V1=1.0, V2=1.0, al=0.0, q=0.0, b=-1.207, x0=np.pi / 2)
sh = shoot(par, 8, 0.9, 0.1, 0.05, 0.3, verbose=False)
w1, w2 = to_grid(g, sh)
w1, w2, *_ = solve(g, w1, w2, par)
e = continuation(g, w1, w2, par, 'al', list(np.round(np.arange(0.05, 0.7501, 0.05), 3)) + [0.7919], verbose=False)[-1]
out['gap3_b-1.207'] = check("third gap, b = -1.207, al = 0.7919", g, e['w1'], e['w2'], dict(par, al=0.7919), 4000.0, 0.02)
dump('results/weak.json', out)
