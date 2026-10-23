"""
shooting.py -- edge shooting for the conservative case (al = 0), thesis Fig. 3.1.

WHY SHOOT FROM THE EDGE (xi_f -> 0) INSTEAD OF FROM THE CENTRE (0 -> xi_f)
-------------------------------------------------------------------------
In a gap every linearized component has one Floquet solution decaying and
one growing as xi -> +inf.  Integrating outward from the centre, any
round-off along the growing solution is amplified by |mu|^{-n} and the
trajectory peels off the soliton after a few periods; the only way to see the
target is the sign of the divergence at xi_f.
Integrating INWARD from xi_f, the unwanted (growing-at-infinity) solution
DECAYS by |mu|^{n} on the way in, so errors in the initial condition are
damped.  At xi_f we sit exactly on the decaying Floquet manifold:
    (w_l, w_l')(xi_f) = C_l (1, r_l),   r_l = slope ratio of the decaying
                                         Floquet solution, xi_f = n*pi.
The unknowns are the tail amplitudes C1, C2.

SYMMETRY CONDITION AT THE CENTRE
--------------------------------
For al = 0 the fundamental soliton is real and even, so the boundary
conditions at the centre are
    D1 = w1'(0) = 0,   D2 = w2'(0) = 0.
(In the finite-difference Newton solver the same symmetry is imposed with
mirrored ghost points: a central difference across x = 0 with w(-h) = w(h)
is what "w'(0) = 0" means on a grid.)

THE FLOWCHART (thesis Fig. 3.1), applied to a generic scalar c and D(c)
-----------------------------------------------------------------------
  Evaluate D0 = D(c0)
  repeat: c1 = c0 + dc ; D1 = D(c1)
          if D0*D1 > 0 : c0 <- c1, D0 <- D1     (no sign change yet: keep marching)
          else         : bracket found
  bisection: c = (c0+c1)/2, err = |c1-c0|
          if err < eps : return c
          D = D(c)
          if D*D1 < 0  : c0 <- c, D0 <- D        (root in [c, c1])
          else         : c1 <- c, D1 <- D        (root in [c0, c])
Inner loop: c = C2 at fixed C1, D = D2.  Outer loop: c = C1, D = D1 evaluated
at the inner-converged C2(C1)  ("substitute C2, D2 by C1, D1").

We march in log-amplitude s = ln C (tail amplitudes span many decades; a
constant ds is a constant RELATIVE step).  The algorithm is unchanged.
"""
import math
import numpy as np

from rk45 import rkf45
from model import make_rhs, decaying_tail, monodromy, Q_ff, Q_sh


def tails(par, n):
    """
    Everything the edge needs, for xi_f = n*pi:
      r1, r2 : slope ratios w'/w of the decaying Floquet solutions (FF, SH)
      mu1,mu2: their multipliers per period (|mu| < 1)
      z      : (psi, psi') at xi_f of the FORCED SH tail, see below
      s1, s2 : scales |mu1|^n, |mu2|^n so that  C_l = c_l * s_l  with c_l = O(1)
    WHY THE FORCED TAIL: w2 is driven by w1^2 ~ C1^2 mu1^{2x/pi}.  When
    |mu1|^2 > |mu2| (SH gap deeper than twice the FF one) this driven part,
    not the homogeneous SH solution, dominates the SH tail.  Leaving it out
    makes the required C2 a huge cancelling number (~1e3-1e4 in units of s2).
    psi solves  psi'' = -Q2 psi - 2 phi1^2  with multiplier mu1^2 per period:
      (psi,psi')(pi) = M2 (psi,psi')(0) + g  and  (psi,psi')(pi) = mu1^2 (psi,psi')(0)
      =>  (psi,psi')(0) = (mu1^2 I - M2)^{-1} g ,  g = response from zero data.
    """
    V1, V2, al, q, b = par['V1'], par['V2'], par['al'], par['q'], par['b']
    x0 = par.get('x0', 0.0)
    r1, mu1 = decaying_tail(Q_ff(V1, al, b, x0))
    r2, mu2 = decaying_tail(Q_sh(V2, q, b, x0))
    if r1 is None:
        raise ValueError(f"b = {b} lies in a FF band: no decaying FF tail")
    embedded = r2 is None
    if embedded:
        r2, mu2 = 0.0, 1.0   # SH inside its band: no decaying SH mode, only the forced tail
    # forced tail: integrate phi1 (from (1, r1)) together with psi (from 0) over one period
    Q1, Q2 = Q_ff(V1, al, b, x0), Q_sh(V2, q, b, x0)
    def f(x, y):
        ph, dph, ps, dps = y
        return (dph, -Q1(x) * ph, dps, -Q2(x) * ps - 2.0 * ph * ph)
    _, ys, _ = rkf45(f, 0.0, (1.0, r1, 0.0, 0.0), math.pi, h0=0.01, rtol=1e-12,
                     atol=1e-14, store=False)
    g = np.array([ys[-1][2], ys[-1][3]])
    M2 = monodromy(Q2)
    z = np.linalg.solve(mu1 ** 2 * np.eye(2) - M2, g)
    return dict(n=n, xf=n * math.pi, r1=r1, r2=r2, mu1=mu1, mu2=mu2, z=z,
                s1=abs(mu1) ** n, s2=abs(mu2) ** n, embedded=embedded)


def edge_shot(c1, c2, par, T, rtol=1e-10, atol=1e-16, ymax=50.0, store=False):
    """
    One shot: start on the decaying manifold at xi_f and integrate (A),(B)
    inward to xi = 0 with RKF45.  Tail amplitudes C_l = c_l * s_l:
        (w1, w1')(xi_f) = C1 (1, r1)
        (w2, w2')(xi_f) = C2 (1, r2) + C1^2 (z0, z1)      [homogeneous + forced]
    Returns (D1, D2, xs, ys, info) with D_l = w_l'(0) (real parts; al = 0).
    If the trajectory runs away (|w| > ymax) the shot stops early; the signs of
    the derivatives at that point still tell on which side of the root we are.
    """
    f = make_rhs(par['V1'], par['V2'], par['al'], par['q'], par['b'], par.get('x0', 0.0))
    C1, C2 = c1 * T['s1'], c2 * T['s2']
    z = T['z']
    y0 = (C1, C1 * T['r1'], C2 + C1 * C1 * z[0], C2 * T['r2'] + C1 * C1 * z[1])
    xs, ys, info = rkf45(f, T['xf'], y0, 0.0, h0=0.05, rtol=rtol, atol=atol,
                         hmax=0.1, ymax=ymax, store=store)
    _, p1, _, p2 = ys[-1]
    return p1.real, p2.real, xs, ys, info


def flowchart_root(Dfun, c0, dc, eps, maxscan=400):
    """Thesis Fig. 3.1: march c by dc until D changes sign, then bisect to |c1-c0| < eps."""
    D0 = Dfun(c0)                               # Evaluate D^(0)
    for _ in range(maxscan):
        c1 = c0 + dc                            # Set C -> C + dC
        D1 = Dfun(c1)                           # Evaluate D^(1)
        if D0 * D1 > 0:                         # D^(0) D^(1) > 0 : still same side
            c0, D0 = c1, D1                     # Set D^(0) -> D^(1)
        else:
            break
    else:
        raise RuntimeError("flowchart_root: no sign change found while marching")
    while True:                                 # bisection on the bracket [c0, c1]
        c = 0.5 * (c0 + c1)
        if abs(c1 - c0) < eps:                  # C_error < eps  -> final result
            return c
        D = Dfun(c)                             # Evaluate D at the midpoint
        if D * D1 < 0:                          # root between c and c1
            c0, D0 = c, D                       # Set C^(0) -> C, D^(0) -> D
        else:                                   # root between c0 and c
            c1, D1 = c, D                       # Set C^(1) -> C, D^(1) -> D


def inner_root(Dfun, c_guess, dc, eps, backoff=8):
    """
    Flowchart with a warm start: begin marching `backoff` steps below the
    previous root.  If no sign change is met (start was above the root),
    double the back-off and try again.  The flowchart itself is untouched.
    """
    for _ in range(6):
        try:
            return flowchart_root(Dfun, c_guess - backoff * dc, dc, eps, maxscan=4 * backoff + 40)
        except RuntimeError:
            backoff *= 2
    raise RuntimeError("inner_root: could not bracket")


def nested_flowchart(par, T, c1_start, dc1, dc2, c2_guess=0.0, eps=1e-10, verbose=True):
    """
    Thesis algorithm: inner flowchart finds c2*(c1) with D2 = 0 at fixed c1,
    outer flowchart finds c1 with D1(c1, c2*(c1)) = 0.
    Returns (c1, c2, nshots).
    """
    count = [0]
    last_c2 = [c2_guess]

    def D2_at(c1, c2):
        count[0] += 1
        return edge_shot(c1, c2, par, T)[1]

    def D1_outer(c1):
        c2 = inner_root(lambda c: D2_at(c1, c), last_c2[0], dc2, eps)   # inner flowchart
        last_c2[0] = c2
        count[0] += 1
        D1 = edge_shot(c1, c2, par, T)[0]
        if verbose:
            print(f"   outer: c1={c1:.12f}  c2*={c2:+.12f}  D1={D1:+.3e}  shots={count[0]}")
        return D1

    c1 = flowchart_root(D1_outer, c1_start, dc1, eps)                   # outer flowchart
    c2 = inner_root(lambda c: D2_at(c1, c), last_c2[0], dc2, eps)       # c2 at the final c1
    return c1, c2, count[0]


def newton_polish(par, T, c1, c2, tol=1e-10, maxit=20, verbose=True):
    """
    Optional last step: 2D Newton on (c1, c2) -> (D1, D2) = 0 with a forward
    finite-difference Jacobian.  Bisection gains one bit per shot; inside the
    basin Newton doubles the number of correct digits per iteration.
    """
    F = lambda a, c: np.array(edge_shot(a, c, par, T, rtol=1e-12)[:2])
    nD_old = np.inf
    for it in range(maxit):
        D = F(c1, c2)
        nD = np.max(np.abs(D))
        if verbose:
            print(f"   shooting-newton {it}: c1={c1:.14f} c2={c2:+.14f} |D|={nD:.2e}")
        if nD < tol or nD > 0.5 * nD_old:      # converged, or stalled at the RK noise floor
            break
        nD_old = nD
        d = 1e-7
        J = np.column_stack(((F(c1 + d, c2) - D) / d, (F(c1, c2 + d) - D) / d))
        dc = np.linalg.solve(J, -D)
        c1, c2 = c1 + dc[0], c2 + dc[1]
    return c1, c2


def profile_from_shot(par, T, c1, c2, rtol=1e-12):
    """Stored RKF45 trajectory of the converged shot, ascending in x:
    (xs, w1, w1', w2, w2', info).  xs is the adaptive (non-uniform) mesh."""
    _, _, xs, ys, info = edge_shot(c1, c2, par, T, rtol=rtol, store=True)
    xs = np.array(xs)[::-1]                 # integration ran xi_f -> 0; flip to ascending
    Y = np.array(ys)[::-1]
    return xs, Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3], info
