"""
validate.py -- numerical checks of every building block; writes results/validation.json
and prints a plain-text report.  Each test states what is compared and why it matters.
"""
import math
import time
import numpy as np

from rk45 import rkf45
from spline import spline_resample
from model import band_edges, Q_ff, Q_sh
from shooting import tails, nested_flowchart, newton_polish, profile_from_shot
from newton_fd import Grid, solve, continuation, full_line, power_half, residual_jacobian, pack
from stability import analyze
from ssf import propagate, absorber
from pipeline import to_grid
from io_utils import dump

R = {}
line = lambda s: print(s, flush=True)

# 1. RKF45 on y'' = -y over 10 periods: global error vs tolerance, accepted/rejected steps
line("\n[1] RKF45 accuracy and step refinement: y'' = -y, y(0)=1, y'(0)=0, x in [0, 20 pi]")
R['rk45'] = []
for rtol in (1e-6, 1e-8, 1e-10, 1e-12):
    xs, ys, info = rkf45(lambda x, y: (y[1], -y[0]), 0.0, (1.0, 0.0), 20 * math.pi, rtol=rtol, atol=rtol * 1e-3)
    err = abs(ys[-1][0].real - 1.0)
    R['rk45'].append(dict(rtol=rtol, err=err, nacc=info['nacc'], nrej=info['nrej']))
    line(f"    rtol={rtol:.0e}: |y(20pi)-1| = {err:.2e}   accepted {info['nacc']:5d}  rejected {info['nrej']}")

# 2. clamped cubic spline on a non-uniform mesh: error ~ h^4
line("\n[2] clamped cubic spline on smoothly non-uniform knots, f = exp(-x/5) sin(3x)")
f = lambda x: np.exp(-x / 5) * np.sin(3 * x)
df = lambda x: np.exp(-x / 5) * (3 * np.cos(3 * x) - np.sin(3 * x) / 5)
rng = np.random.default_rng(0)
t = np.linspace(0, 10, 4001)
R['spline'] = []
for n in (50, 100, 200, 400):
    tt = np.linspace(0, 1, n)                       # smooth non-uniform knots: x = 10 (t + 0.3 sin(2 pi t)/(2 pi))
    x = 10 * (tt + 0.3 * np.sin(2 * np.pi * tt) / (2 * np.pi))     # spacing varies by a factor 1.3/0.7
    e = np.max(np.abs(spline_resample(x, f(x), t, d0=df(0), dn=df(10)).real - f(t)))
    R['spline'].append(dict(n=n, err=e))
    line(f"    n={n:4d} knots: max error {e:.2e}")

# 3. Floquet band edges vs Mathieu characteristic values and the paper
line("\n[3] Floquet band edges")
sh = band_edges(lambda b: Q_sh(1.0, 0.0, b), 0.3, 0.4, nscan=101)
# SH: w'' + 4(cos2x - b) w = 0  <->  Mathieu y'' + (a - 2Q cos2x) y = 0 with a = -4b, Q = -2;
# a_0(2) = -1.513956885 (DLMF table) -> b = -a_0/4 = 0.378489
R['bands'] = dict(sh_top=sh.max(), mathieu=1.513956885 / 4)
line(f"    SH top edge {sh.max():.6f}   Mathieu -a0(2)/4 = {1.513956885 / 4:.6f}   paper 0.3784")
ff = band_edges(lambda b: Q_ff(1.0, 0.7, b), 0.0, 0.2, nscan=101).max()
line(f"    q_+^(0,0)(al=0.7) = {ff - sh.max():.5f}   paper -0.316")
R['bands']['q00'] = ff - sh.max()

# 4. shooting -> spline -> Newton (fd4 and fourier): convergence in h, Jacobian check, agreement
line("\n[4] Newton on the symmetric half grid (semi-infinite gap, case 1, al = 0, b = 0.25)")
par = dict(V1=1.0, V2=1.0, al=0.0, q=ff - sh.max(), b=0.25)
T = tails(par, 6)
t0 = time.time()
c1, c2, ns = nested_flowchart(par, T, 0.05, 0.05, 0.1, 0.0, eps=1e-9, verbose=False)
c1, c2 = newton_polish(par, T, c1, c2, verbose=False)
xs, w1s, p1s, w2s, p2s, info = profile_from_shot(par, T, c1, c2)
sh_rec = dict(xs=xs, w1=w1s, p1=p1s, w2=w2s, p2=p2s)
line(f"    flowchart: {ns} shots, {time.time() - t0:.0f}s; |D| = {abs(p1s[0]):.1e}, {abs(p2s[0]):.1e}")
R['newton'] = []
for deriv, h in (('fd4', 0.2), ('fd4', 0.1), ('fd4', 0.05), ('fourier', 0.3), ('fourier', 0.2), ('fourier', 0.15)):
    g = Grid(L=8 * np.pi, h=h, deriv=deriv)
    a1, a2 = to_grid(g, sh_rec)
    v1, v2, it, nF = solve(g, a1, a2, par)
    P = sum(power_half(g, v1, v2))
    # shooting vs Newton on [0, 4 pi] (away from the shooting edge at 6 pi)
    m = g.X < 4 * np.pi
    dw = np.max(np.abs(v1[m] - a1[m]))
    R['newton'].append(dict(deriv=deriv, h=g.h, P=P, its=it, res=nF, shoot_vs_newton=dw))
    line(f"    {deriv:7s} h={g.h:.3f}: P = {P:.10f}  Newton its {it}  |F| {nF:.1e}  max|w1_newton - w1_shoot| (x<4pi) = {dw:.1e}")
Pref = R['newton'][-1]['P']
e1, e2, e3 = [abs(r['P'] - Pref) for r in R['newton'][:3]]
line(f"    fd4 observed order from P: {math.log2(e1 / e2):.2f}, {math.log2(e2 / e3):.2f}   (expected 4)")
g = Grid(L=8 * np.pi, h=0.15, deriv='fourier')
xv = pack(*to_grid(g, sh_rec))
xv = xv + 1e-2 * rng.standard_normal(len(xv))
dv = rng.standard_normal(len(xv))
for j in g.sym_rows:
    dv[g.N + j] = dv[3 * g.N + j] = 0
pj = dict(par, al=0.5)
F0, J0 = residual_jacobian(xv, g, pj)
F1, _ = residual_jacobian(xv + 1e-7 * dv, g, pj)
jerr = np.max(np.abs((F1 - F0) / 1e-7 - J0 @ dv)) / np.max(np.abs(J0 @ dv))
R['jacobian_rel_err'] = jerr
line(f"    analytic Jacobian vs finite difference: relative error {jerr:.1e}")

# 5. continuation to al = 0.7, stability, SSF checks
line("\n[5] al-continuation, linear stability, split-step Fourier")
v1, v2, *_ = solve(g, *to_grid(g, sh_rec), par)
s = continuation(g, v1, v2, par, 'al', np.round(np.arange(0.1, 0.7001, 0.1), 3), verbose=False)[-1]
p7 = dict(par, al=0.7)
x, W1, W2 = full_line(g, s['w1'], s['w2'])
k = 2 * np.pi * np.fft.fftfreq(g.M, d=g.h)
# (a) the Newton state is an exact stationary state of the SSF on the same grid
r = propagate(W1.copy(), W2.copy(), x, k, p7, 0.01, 50.0, nsave=10, order=4, verbose=False)
st = np.max(np.abs(r['I1'][-1] - np.abs(W1) ** 2))
R['ssf_stationary'] = st
line(f"    stationary state under SSF (z = 50, same grid): max| |u1|^2 - |w1|^2 | = {st:.1e}")
# (b) order of Strang / Yoshida
u10 = W1 * (1 + 0.05 * np.exp(-x ** 2))
ref = propagate(u10.copy(), W2.copy(), x, k, p7, 0.0025, 2.0, order=4, verbose=False)['u1']
R['ssf_order'] = {}
for order in (2, 4):
    errs = [np.max(np.abs(propagate(u10.copy(), W2.copy(), x, k, p7, dz, 2.0, order=order, verbose=False)['u1'] - ref))
            for dz in (0.04, 0.02)]
    R['ssf_order'][order] = dict(errs=errs, order=math.log2(errs[0] / errs[1]))
    line(f"    SSF order {order}: errors {errs[0]:.2e}, {errs[1]:.2e}  -> observed {math.log2(errs[0] / errs[1]):.2f}")
# (c) growth rate: eigenvalue vs SSF of an eigenvector seed at b = 0.5
s5 = continuation(g, s['w1'], s['w2'], p7, 'b', np.round(np.arange(0.275, 0.5001, 0.025), 4), verbose=False)[-1]
p5 = dict(p7, b=0.5)
x, W1, W2 = full_line(g, s5['w1'], s5['w2'])
A = analyze(x, W1, W2, p5, g.D2)
v = A['vec_unst']
M = len(x)
dd, a, ff_, cc = v[:M], v[M:2 * M], v[2 * M:3 * M], v[3 * M:]
eps = 1e-4 / np.max(np.abs(np.concatenate((a, cc))))
r = propagate(W1 + eps * (a + np.conj(cc)), W2 + eps * (dd + np.conj(ff_)), x, k, p5, 0.005, 30.0,
              nsave=1500, order=2, g=absorber(x, g.L), verbose=False)
dev = np.sqrt(np.sum((r['I1'] - np.abs(W1) ** 2) ** 2, axis=1))
pk = [j for j in range(1, len(dev) - 1) if dev[j] > dev[j - 1] and dev[j] > dev[j + 1]]
zp, dp = r['z'][pk], dev[pk]
mm = dp < 30 * dp[0]
rate = np.polyfit(zp[mm], np.log(dp[mm]), 1)[0]
R['growth'] = dict(eig=A['lam_unst'], ssf_rate=rate)
line(f"    b = 0.5: Im(lam) = {A['lam_unst'].imag:.4f} (Re {A['lam_unst'].real:.4f});  SSF envelope growth {rate:.4f}")

dump('results/validation.json', R, nd=8)
line("\nvalidation done")
