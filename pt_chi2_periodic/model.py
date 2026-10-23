"""
model.py -- the chi(2) system with a PT-symmetric periodic potential on the FF.

Methods: PhD thesis of F. C. Moreira.  Model and figures compared with:
F. C. Moreira, V. V. Konotop, B. A. Malomed, PRA 87, 013832 (2013).

SIGN CONVENTION (read this first)
---------------------------------
Eqs. (1) and (3) as printed in the paper, with the ansatz u_l = w_l exp(i l b z),
place the SH band edge of the Mathieu problem at b = -0.378 and the semi-infinite
gap at b -> -infinity.  Every number quoted in the figures (b_{2,+}^(0) = 0.3784,
q_+^(0,0) = -0.316, third-gap edge b = -1.293, semi-infinite gap at b > 0.38)
is reproduced only after b -> -b.  We therefore use the figure convention:

  stationary (x = xi):
    w1'' + [V1 cos2x + i*al*sin2x - b] w1 + 2 conj(w1) w2 = 0            (A)
    (1/2) w2'' + 2 [V2 cos2x + q - b] w2 + w1^2         = 0              (B)

  evolution (z = zeta), consistent with (A),(B) through u_l = w_l e^{i l b z}:
    i u1_z = -u1_xx     - [V1 cos2x + i*al*sin2x] u1 - 2 conj(u1) u2     (C)
    i u2_z = -(1/2)u2_xx - 2 [V2 cos2x + q] u2       - u1^2              (D)

(C),(D) are the printed Eqs. (1) with z -> -z, i.e. the mirror image x -> -x
of the gain/loss pattern; this does not change existence or stability.

CENTRING: x is measured from the soliton centre x0 (par['x0'], 0 or pi/2;
both are PT-symmetry points of the lattice).  Every potential is evaluated
as cos(2(x + x0)), sin(2(x + x0)).  x0 = pi/2 is needed e.g. in the third
finite gap, where the FF band-edge Bloch mode is odd about x = 0 (sin x-like)
but even about x = pi/2.

Conserved (for al = 0) / balanced (PT, al > 0) power:  P = P1 + P2,
P_l = l * int |w_l|^2 dx   (paper's definition below Eq. (11)).
"""
import math
import numpy as np

from rk45 import rkf45


# --------------------------------------------------------------------------
# Potentials.  U1 acts on the FF (complex, PT-symmetric: U1(-x) = conj U1(x)),
# U2 on the SH (real).  Kept as explicit functions so a different potential
# (e.g. the localized Scarf-type one of thesis ch. 6) is a one-line change.
# --------------------------------------------------------------------------
def U1(x, V1, al):
    """FF potential  V1 cos(2x) + i al sin(2x)   (works on scalars and arrays)."""
    return V1 * np.cos(2 * x) + 1j * al * np.sin(2 * x)


def U2(x, V2, q):
    """SH potential  2 [V2 cos(2x) + q]."""
    return 2.0 * (V2 * np.cos(2 * x) + q)


# --------------------------------------------------------------------------
# First-order form of (A),(B):  y = (w1, w1', w2, w2')  (complex)
#   w1'' = -(V1 c + i al s - b) w1 - 2 conj(w1) w2
#   w2'' = -4 (V2 c + q - b) w2 - 2 w1^2          [(B) multiplied by 2]
# Written with python complex scalars + math.cos: this is the inner loop of
# every shot, and scalar math is ~10x faster than numpy on 4-element arrays.
# --------------------------------------------------------------------------
def make_rhs(V1, V2, al, q, b, x0=0.0):
    def f(x, y):
        w1, p1, w2, p2 = y
        c = math.cos(2.0 * (x + x0))
        s = math.sin(2.0 * (x + x0))
        dp1 = -complex(V1 * c - b, al * s) * w1 - 2.0 * w1.conjugate() * w2
        dp2 = -4.0 * (V2 * c + q - b) * w2 - 2.0 * w1 * w1
        return (p1, dp1, p2, dp2)
    return f


# --------------------------------------------------------------------------
# Floquet theory of the linearized (w -> 0) equations, needed for
#   (i) the band-gap spectrum (Fig. 1 of the paper)
#  (ii) the decaying tail used as the initial condition of edge shooting.
#
# Linear Hill equation  y'' = -Q(x) y  with pi-periodic Q:
#   FF:  Q1 = V1 cos2x + i al sin2x - b
#   SH:  Q2 = 4 (V2 cos2x + q - b)
# Monodromy M = Phi(pi), Phi(0) = I.  det M = 1 (no y' term -> Wronskian const),
# so multipliers are mu, 1/mu.  Discriminant D = tr(M)/2:
#   |D| <= 1 (real D)  -> Bloch band (bounded solutions)
#   |D| >  1           -> gap: one solution decays as x -> +inf (|mu| < 1).
# --------------------------------------------------------------------------
def monodromy(Qfun, rtol=1e-12, atol=1e-14):
    """Monodromy matrix of y'' = -Q(x) y over one period [0, pi]."""
    def f(x, y):
        # y = (y_a, y_a', y_b, y_b'): both fundamental solutions at once
        Q = Qfun(x)
        return (y[1], -Q * y[0], y[3], -Q * y[2])
    y0 = (1.0 + 0j, 0j, 0j, 1.0 + 0j)          # Phi(0) = identity
    xs, ys, info = rkf45(f, 0.0, y0, math.pi, h0=0.01, rtol=rtol, atol=atol, store=False)
    ya, dya, yb, dyb = ys[-1]
    return np.array([[ya, yb], [dya, dyb]])     # columns = the two solutions


def Q_ff(V1, al, b, x0=0.0):
    return lambda x: complex(V1 * math.cos(2 * (x + x0)) - b, al * math.sin(2 * (x + x0)))


def Q_sh(V2, q, b, x0=0.0):
    return lambda x: 4.0 * (V2 * math.cos(2 * (x + x0)) + q - b) + 0j


def discriminant(Qfun):
    """D = tr(M)/2.  Real for real b when the PT phase is unbroken."""
    M = monodromy(Qfun)
    return 0.5 * (M[0, 0] + M[1, 1])


def decaying_tail(Qfun):
    """
    Slope ratio r = y'/y of the Floquet solution that DECAYS as x -> +inf,
    evaluated at x = n*pi (same potential phase as x = 0).
    Edge shooting starts from (w, w') = C (1, r) at xi_f = n*pi.
    Returns (r, mu) with |mu| < 1 the decay factor per period, or (None, mu)
    if b sits inside a band (|mu| = 1: no decaying linear tail exists).
    """
    M = monodromy(Qfun)
    mu, V = np.linalg.eig(M)
    k = int(np.argmin(np.abs(mu)))
    if abs(abs(mu[k]) - 1.0) < 1e-9:
        return None, mu[k]
    v = V[:, k]
    return v[1] / v[0], mu[k]


def band_edges(Qfactory, b_lo, b_hi, nscan=1201, tol=1e-12):
    """
    All b in [b_lo, b_hi] where |Re D(b)| = 1, i.e. Bloch band edges.
    Qfactory(b) -> Q(x).  Scan, then bisect each sign change of |Re D| - 1.
    """
    bs = np.linspace(b_lo, b_hi, nscan)
    g = np.array([abs(discriminant(Qfactory(b)).real) - 1.0 for b in bs])
    edges = []
    for i in range(nscan - 1):
        if g[i] * g[i + 1] < 0:                      # bracketed an edge
            lo, hi, glo = bs[i], bs[i + 1], g[i]
            while hi - lo > tol:                     # plain bisection
                mid = 0.5 * (lo + hi)
                gm = abs(discriminant(Qfactory(mid)).real) - 1.0
                if gm * glo > 0:
                    lo, glo = mid, gm
                else:
                    hi = mid
            edges.append(0.5 * (lo + hi))
    return np.array(edges)


# --------------------------------------------------------------------------
# Observables of a stationary solution on a grid
# --------------------------------------------------------------------------
def power(x, w1, w2):
    """P1 = int|w1|^2,  P2 = 2 int|w2|^2  (trapezoid on the given grid)."""
    P1 = np.trapezoid(np.abs(w1) ** 2, x)
    P2 = 2.0 * np.trapezoid(np.abs(w2) ** 2, x)
    return P1, P2


def currents(x, w):
    """Energy flow j = |w|^2 d(arg w)/dx = Im(conj(w) w')  (paper Eq. (11))."""
    dw = np.gradient(w, x)
    return np.imag(np.conj(w) * dw)
