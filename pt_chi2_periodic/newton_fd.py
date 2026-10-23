"""
newton_fd.py -- Newton-Raphson for the stationary problem (A),(B) on a
symmetric half-domain, with central differences that use mirrored ghosts.

THE BOX
-------
Full periodic box x_j = -L + j h, j = 0..M-1, M = 2n, h = L/n, L a multiple of
pi/2 (the potential has period pi, and PT symmetry about x = 0 plus
periodicity then gives the same mirror symmetry about x = L).
WHY periodic and not Dirichlet walls: a PT lattice cut by Dirichlet walls is
no longer PT-symmetric as a finite operator; its spectrum can turn complex
(max|Im| = 1.3e-2 for the bare FF operator on [-16pi, 16pi]) and those
complex pairs contaminate the stability spectrum.  A periodic box of whole
periods samples the true (real) Bloch bands exactly.

SYMMETRY  ("central difference for symmetry")
---------------------------------------------
PT solutions satisfy {w1, w2}(-x) = conj{w1, w2}(x):  Re w even, Im w odd.
We keep only the half grid x_i = i h, i = 0..n (x_0 = 0, x_n = L) and FOLD the
full-box d^2 matrix onto it: every column that refers to a point x < 0 (or
x > L) is redirected to its mirror image, with sign +1 for Re and -1 for Im.
For the 4th-order central stencil
    w''_i ~ (-w_{i-2} + 16 w_{i-1} - 30 w_i + 16 w_{i+1} - w_{i+2}) / (12 h^2)
this is literally the ghost-point rule  R_{-k} = R_k,  I_{-k} = -I_k
(i.e. w'(0) = 0 for the real part, w(0) = 0 for the imaginary part).
The same folding works for the Fourier-spectral d^2 (dense circulant),
offered as deriv='fourier': then the Newton solution is an exact discrete
stationary state of the split-step propagator on the same grid.
Removing the odd/even freedom also removes the U(1) phase zero mode
(w1, w2) -> (w1 e^{i phi}, w2 e^{2 i phi}), whose generator (i w1, 2 i w2)
has an EVEN imaginary part -- the Jacobian is therefore nonsingular.

UNKNOWNS / EQUATIONS (real arithmetic; conj(w1) is not complex-analytic)
  x = [R1, I1, R2, I2]  (4(n+1) reals),  w_l = R_l + i I_l,  c = cos2x, s = sin2x
  F1R = R1'' + (V1 c - b) R1 - al s I1 + 2 (R1 R2 + I1 I2)
  F1I = I1'' + (V1 c - b) I1 + al s R1 + 2 (R1 I2 - I1 R2)
  F2R = R2''/2 + 2 (V2 c + q - b) R2 + R1^2 - I1^2
  F2I = I2''/2 + 2 (V2 c + q - b) I2 + 2 R1 I1
  rows F1I, F2I at x = 0 and x = L are replaced by Im w1 = Im w2 = 0 there.
The Jacobian is assembled analytically (dense numpy; 4(n+1) <= ~4000).
"""
import numpy as np


# --------------------------------------------------------------------------
# second-derivative matrices on the full periodic box
# --------------------------------------------------------------------------
def d2_fd4_periodic(M, h):
    """4th-order central difference, circulant (periodic wrap)."""
    st = np.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / (12.0 * h * h)
    D = np.zeros((M, M))
    for k, ck in zip(range(-2, 3), st):
        D += ck * np.roll(np.eye(M), k, axis=1)        # column (j + k) mod M
    return D


def d2_fourier(M, h):
    """Fourier pseudo-spectral d^2: apply -k^2 in Fourier space to each unit vector."""
    k = 2 * np.pi * np.fft.fftfreq(M, d=h)
    return np.real(np.fft.ifft(-(k ** 2)[:, None] * np.fft.fft(np.eye(M), axis=0), axis=0))


def fold(D2, n):
    """
    Fold a full-box operator (M = 2n points, x_j = -L + j h) onto the half grid
    x_i = i h, i = 0..n.  Full index of half point i is J(i) = n + i (mod M).
    A full column J is mapped to half index i = dist if dist = (J - n) mod M <= n,
    otherwise to its mirror image i = M - dist with parity sign -1 (odd part).
    """
    M = 2 * n
    De = np.zeros((n + 1, n + 1))
    Do = np.zeros((n + 1, n + 1))
    cols = np.arange(M)
    dist = (cols - n) % M
    mirrored = dist > n
    target = np.where(mirrored, M - dist, dist)
    sign = np.where(mirrored, -1.0, 1.0)
    for i in range(n + 1):
        row = D2[(n + i) % M]
        np.add.at(De[i], target, row)
        np.add.at(Do[i], target, sign * row)
    return De, Do


class Grid:
    """Half grid + folded difference matrices + the full-box operator (a bundle of arrays)."""
    def __init__(self, L, h, deriv='fd4'):
        m = L / (np.pi / 2)
        assert abs(m - round(m)) < 1e-9, "periodic box needs L = m*pi/2"
        self.L = L
        self.n = int(round(L / h))
        self.h = L / self.n
        self.M = 2 * self.n
        self.deriv = deriv
        self.X = self.h * np.arange(self.n + 1)                 # 0 .. L
        self.x_full = -L + self.h * np.arange(self.M)           # -L .. L-h  (box order)
        if deriv == 'fd4':
            self.D2 = d2_fd4_periodic(self.M, self.h)
        elif deriv == 'fourier':
            self.D2 = d2_fourier(self.M, self.h)
        else:
            raise ValueError(deriv)
        self.De, self.Do = fold(self.D2, self.n)
        self.sym_rows = [0, self.n]                             # Im w = 0 at x = 0 and x = L
        self.N = self.n + 1


# --------------------------------------------------------------------------
# residual, Jacobian, Newton
# --------------------------------------------------------------------------
def unpack(xv, N):
    return xv[:N], xv[N:2 * N], xv[2 * N:3 * N], xv[3 * N:]


def pack(w1, w2):
    return np.concatenate((w1.real, w1.imag, w2.real, w2.imag))


def fields(xv, N):
    R1, I1, R2, I2 = unpack(xv, N)
    return R1 + 1j * I1, R2 + 1j * I2


def residual_jacobian(xv, g, par):
    V1, V2, al, q, b = par['V1'], par['V2'], par['al'], par['q'], par['b']
    N, X, De, Do = g.N, g.X, g.De, g.Do
    R1, I1, R2, I2 = unpack(xv, N)
    x0 = par.get('x0', 0.0)                 # lattice offset of the soliton centre
    c = np.cos(2 * (X + x0))
    s = np.sin(2 * (X + x0))
    P1 = V1 * c - b                       # Re(U1) - b              (FF, eq. A)
    P2 = 2.0 * (V2 * c + q - b)           # 2 (V2 cos2x + q - b)    (SH, eq. B)
    F = np.concatenate((
        De @ R1 + P1 * R1 - al * s * I1 + 2 * (R1 * R2 + I1 * I2),
        Do @ I1 + P1 * I1 + al * s * R1 + 2 * (R1 * I2 - I1 * R2),
        0.5 * (De @ R2) + P2 * R2 + R1 ** 2 - I1 ** 2,
        0.5 * (Do @ I2) + P2 * I2 + 2 * R1 * I1))
    D = np.diag
    Z = np.zeros((N, N))
    J = np.block([
        [De + D(P1 + 2 * R2), D(-al * s + 2 * I2), D(2 * R1),         D(2 * I1)],
        [D(al * s + 2 * I2),  Do + D(P1 - 2 * R2), D(-2 * I1),        D(2 * R1)],
        [D(2 * R1),           D(-2 * I1),          0.5 * De + D(P2),  Z],
        [D(2 * I1),           D(2 * R1),           Z,                 0.5 * Do + D(P2)]])
    for j in g.sym_rows:                  # symmetry rows: Im w1 = Im w2 = 0 at x = 0, L
        for r in (N + j, 3 * N + j):
            F[r] = xv[r]
            J[r, :] = 0.0
            J[r, r] = 1.0
    return F, J


def newton(xv, g, par, tol=1e-11, maxit=40, verbose=False):
    """Damped Newton-Raphson: full step unless it fails to reduce ||F||_inf, then halve."""
    F, J = residual_jacobian(xv, g, par)
    nF = np.max(np.abs(F))
    for it in range(maxit):
        if verbose:
            print(f"      newton it={it:2d}  |F|_inf={nF:.3e}")
        if nF < tol:
            return xv, it, nF
        dx = np.linalg.solve(J, -F)
        t = 1.0
        while True:                                   # backtracking line search
            xt = xv + t * dx
            Ft, Jt = residual_jacobian(xt, g, par)
            nFt = np.max(np.abs(Ft))
            if nFt < nF or t < 1e-4:
                break
            t *= 0.5
        if nFt >= nF:
            break
        xv, F, J, nF = xt, Ft, Jt, nFt
    raise RuntimeError(f"newton: no convergence, |F| = {nF:.3e}")


def solve(g, w1, w2, par, **kw):
    """Newton from an initial guess (w1, w2) on the half grid; returns (w1, w2, its, |F|)."""
    xv, it, nF = newton(pack(w1, w2), g, par, **kw)
    w1, w2 = fields(xv, g.N)
    return w1, w2, it, nF


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def full_line(g, w1, w2):
    """Unfold the half-grid solution onto the full box x = -L .. L-h with w(-x) = conj w(x)."""
    idx = (np.arange(g.M) - g.n) % g.M                 # distance from x=0 in grid steps (mod M)
    mir = idx > g.n
    i = np.where(mir, g.M - idx, idx)
    W1 = np.where(mir, np.conj(w1[i]), w1[i])
    W2 = np.where(mir, np.conj(w2[i]), w2[i])
    return g.x_full, W1, W2


def power_half(g, w1, w2):
    """Full-box powers from half-grid data (trapezoid on [0, L], doubled = periodic sum)."""
    P1 = 2.0 * np.trapezoid(np.abs(w1) ** 2, g.X)
    P2 = 2.0 * 2.0 * np.trapezoid(np.abs(w2) ** 2, g.X)   # P2 = 2 int |w2|^2
    return P1, P2


def continuation(g, w1, w2, par, key, values, verbose=True, callback=None):
    """
    Natural-parameter continuation in par[key] over `values`, with a secant
    predictor (linear extrapolation from the last two solutions).
    Returns list of dicts {key value, w1, w2, P1, P2, P}.
    """
    out = []
    for v in values:
        p = dict(par)
        p[key] = v
        if len(out) >= 2 and out[-1][key] != out[-2][key]:
            t = (v - out[-1][key]) / (out[-1][key] - out[-2][key])    # secant predictor
            w1g = out[-1]['w1'] + t * (out[-1]['w1'] - out[-2]['w1'])
            w2g = out[-1]['w2'] + t * (out[-1]['w2'] - out[-2]['w2'])
        elif out:
            w1g, w2g = out[-1]['w1'], out[-1]['w2']
        else:
            w1g, w2g = w1, w2
        w1n, w2n, it, nF = solve(g, w1g, w2g, p)
        P1, P2 = power_half(g, w1n, w2n)
        rec = {key: v, 'w1': w1n, 'w2': w2n, 'P1': P1, 'P2': P2, 'P': P1 + P2,
               'it': it, 'res': nF}
        out.append(rec)
        if verbose:
            print(f"   {key}={v:+.5f}  P={P1 + P2:.6f} (P1={P1:.5f}, P2={P2:.5f})  "
                  f"newton its={it}  |F|={nF:.1e}")
        if callback:
            callback(rec, p)
    return out
