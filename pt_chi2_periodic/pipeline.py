"""
pipeline.py -- the full chain used by the run_*.py drivers:

  1. edge shooting at al = 0 (thesis flowchart, nested; RKF45 with step refinement)
  2. clamped cubic spline: adaptive RK mesh -> uniform half grid
  3. Newton-Raphson on the symmetric half grid (mirrored central differences)
  4. continuation in al (to the target) and then in b (along the branch)
  5. linear stability of every branch point (same grid, same d^2)
  6. split-step Fourier propagation of selected solutions

Everything is plain functions passing plain dicts/arrays.
"""
import time
import numpy as np

from shooting import tails, nested_flowchart, newton_polish, profile_from_shot
from spline import spline_resample
from newton_fd import Grid, solve, continuation, full_line, power_half
from stability import analyze, growth_cap, growth_arnoldi
from ssf import propagate, absorber


def shoot(par, n, c1_start, dc1, dc2, c2_guess, eps=1e-9, verbose=True):
    """Steps 1: returns the RKF45 trajectory (ascending x) of the converged edge shot."""
    t = time.time()
    T = tails(par, n)
    c1, c2, ns = nested_flowchart(par, T, c1_start, dc1, dc2, c2_guess, eps=eps, verbose=False)
    c1, c2 = newton_polish(par, T, c1, c2, verbose=False)
    xs, w1, p1, w2, p2, info = profile_from_shot(par, T, c1, c2)
    if verbose:
        print(f"  [shoot] b={par['b']:+.4f} q={par['q']:+.4f} x0={par.get('x0', 0):.4f}: "
              f"c1={c1:.8f} c2={c2:+.8f} shots={ns} RK mesh={len(xs)} (rejected steps {info['nrej']}) "
              f"w1(0)={w1[0].real:+.5f} w2(0)={w2[0].real:+.5f} D=({abs(p1[0]):.1e},{abs(p2[0]):.1e}) "
              f"[{time.time() - t:.0f}s]")
    return dict(xs=xs, w1=w1, p1=p1, w2=w2, p2=p2, c1=c1, c2=c2, shots=ns, mu1=T['mu1'], mu2=T['mu2'])


def to_grid(g, sh):
    """Step 2: clamped cubic spline of the shot onto the half grid (zero beyond xi_f)."""
    w1 = spline_resample(sh['xs'], sh['w1'], g.X, d0=sh['p1'][0], dn=sh['p1'][-1])
    w2 = spline_resample(sh['xs'], sh['w2'], g.X, d0=sh['p2'][0], dn=sh['p2'][-1])
    return w1, w2


def currents_full(g, W1, W2):
    """j_l = Im(conj(w_l) w_l') with the spectral first derivative on the periodic box."""
    k = 2 * np.pi * np.fft.fftfreq(g.M, d=g.h)
    d = lambda f: np.fft.ifft(1j * k * np.fft.fft(f))
    return np.imag(np.conj(W1) * d(W1)), np.imag(np.conj(W2) * d(W2))


def fourier_refine(x, f, factor=4):
    """Band-limited (exact for the Fourier grid) interpolation of a periodic sample by zero padding."""
    M = len(x)
    F = np.fft.fftshift(np.fft.fft(f))
    pad = np.zeros(factor * M, complex)
    pad[(factor * M - M) // 2:(factor * M + M) // 2] = F
    xf = x[0] + (x[1] - x[0]) / factor * np.arange(factor * M)
    return xf, np.fft.ifft(np.fft.ifftshift(pad)) * factor


def stability_of(g, rec, par, g0=1.0, method='dense', shifts=None):
    """Linear stability with an absorbing layer: returns dict(gmax, lam_unst, lam) -- gmax =
    max Im(lam), 0 if below 1e-6.  method='dense': all eigenvalues (stability.growth_cap);
    method='arnoldi': shift-invert Arnoldi near `shifts` (default scan |Re lam| <= 3, or
    [lam_guess] to refine a known eigenvalue; stability.growth_arnoldi), lam = eigenvalues found."""
    x, W1, W2 = full_line(g, rec['w1'], rec['w2'])
    if method == 'arnoldi':
        gm, lu, lam = growth_arnoldi(x, W1, W2, par, g.D2, g0=g0, shifts=shifts)
    else:
        gm, lu, lam = growth_cap(x, W1, W2, par, g.D2, g0=g0)
    return dict(gmax=gm, lam_unst=(complex(lu) if gm > 0 else None), lam=lam)


def trace_branch(g, w1, w2, par, b_values, stab_every=1, tail_tol=1e-4, verbose=True):
    """
    Steps 4-5 along b: continuation from (w1, w2) at par['b'] through b_values
    (monotone list starting next to par['b']).  Stops when Newton fails or when
    the solution reaches the box edge (|w(L)| > tail_tol * max|w|): the branch
    then approaches a band edge and needs a larger box.
    Returns a list of point summaries (no full fields except for small arrays).
    """
    out = []
    w1g, w2g = w1, w2
    prev = []
    for i, b in enumerate(b_values):
        p = dict(par, b=b)
        if len(prev) == 2:                                   # secant predictor in b
            (b0, a1, a2), (b1, c1, c2) = prev
            t = (b - b1) / (b1 - b0)
            w1g, w2g = c1 + t * (c1 - a1), c2 + t * (c2 - a2)
        try:
            v1, v2, it, nF = solve(g, w1g, w2g, p)
        except (RuntimeError, np.linalg.LinAlgError) as e:
            if verbose:
                print(f"    b={b:+.4f}: Newton failed ({e}); branch stops")
            break
        mx = max(np.abs(v1).max(), np.abs(v2).max())
        if mx < 1e-6:
            if verbose:
                print(f"    b={b:+.4f}: converged to the trivial state; branch stops")
            break
        edge = max(abs(v1[-1]), abs(v2[-1])) / mx
        P1, P2 = power_half(g, v1, v2)
        rec = dict(b=b, P=P1 + P2, P1=P1, P2=P2, w1_0=v1[0], w2_0=v2[0], edge=edge,
                   amp1=float(np.abs(v1).max()), amp2=float(np.abs(v2).max()))
        if i % stab_every == 0:
            t = time.time()
            A = stability_of(g, dict(w1=v1, w2=v2), p)
            rec.update(gmax=A['gmax'], lam_unst=A['lam_unst'])
            ts = time.time() - t
        else:
            ts = 0.0
        out.append(rec)
        if verbose:
            gs = f"gmax={rec['gmax']:.2e}" if 'gmax' in rec else ""
            print(f"    b={b:+.4f}  P={rec['P']:.5f}  its={it}  edge={edge:.1e}  {gs}  [{ts:.0f}s]")
        prev = (prev + [(b, v1, v2)])[-2:]
        w1g, w2g = v1, v2
        if edge > tail_tol:
            if verbose:
                print(f"    b={b:+.4f}: solution reaches the box edge; branch stops")
            break
    return out


def solution_at(g, w1, w2, par, b_target, db=0.01):
    """Continue from par['b'] to b_target in steps ~db; returns (w1, w2) at b_target."""
    b0 = par['b']
    if abs(b_target - b0) < 1e-14:
        return w1, w2
    nst = max(1, int(np.ceil(abs(b_target - b0) / db)))
    bs = np.linspace(b0, b_target, nst + 1)[1:]
    br = continuation(g, w1, w2, par, 'b', bs, verbose=False)
    return br[-1]['w1'], br[-1]['w2']


def ssf_run(g, w1, w2, par, zmax, dz=0.01, noise=0.1, seed=0, nsave=200, order=2, absorb=True):
    """
    Step 6: propagate the solution (unfolded onto the very grid it was computed on,
    so it is an exact discrete stationary state) with multiplicative random noise
    u_l(0) = w_l (1 + noise * r_l(x)), r uniform in [-1, 1].
    """
    x, W1, W2 = full_line(g, w1, w2)
    k = 2 * np.pi * np.fft.fftfreq(g.M, d=g.h)
    rng = np.random.default_rng(seed)
    u1 = W1 * (1 + noise * rng.uniform(-1, 1, g.M))
    u2 = W2 * (1 + noise * rng.uniform(-1, 1, g.M))
    gab = absorber(x, g.L, 0.15, 1.0) if absorb else None
    t = time.time()
    r = propagate(u1, u2, x, k, par, dz, zmax, nsave=nsave, order=order, g=gab, verbose=False)
    r['x'] = x
    r['W1'], r['W2'] = W1, W2
    r['time'] = time.time() - t
    return r
