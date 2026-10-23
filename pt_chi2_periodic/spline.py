"""
spline.py -- C2 cubic spline on non-uniform knots (complex data allowed).

Used to carry the shooting solution, which lives on the ADAPTIVE (non-uniform)
RKF45 mesh, onto the UNIFORM finite-difference grid of the Newton solver.

Unknowns are the knot second derivatives M_i = S''(x_i).  Continuity of S' at
interior knots gives the tridiagonal system (h_i = x_{i+1} - x_i):
   h_{i-1} M_{i-1} + 2(h_{i-1}+h_i) M_i + h_i M_{i+1}
        = 6 [ (y_{i+1}-y_i)/h_i - (y_i-y_{i-1})/h_{i-1} ]
End conditions:
   clamped  (S'(x_0)=d0 known):  2 h_0 M_0 + h_0 M_1 = 6[(y_1-y_0)/h_0 - d0]
   natural  (no slope given):    M_0 = 0               (same at the right end)
Clamped ends are the right choice here: RKF45 gives w' at every mesh point,
in particular w'(0) = 0 (symmetry) and w'(xi_f) from the Floquet tail.
"""
import numpy as np


def thomas(a, b, c, d):
    """Solve tridiagonal  a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i  (a_0, c_{n-1} unused)."""
    n = len(b)
    cp = np.zeros(n, dtype=complex)
    dp = np.zeros(n, dtype=complex)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):                       # forward elimination
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.zeros(n, dtype=complex)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):              # back substitution
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def spline_fit(x, y, d0=None, dn=None):
    """Knot second derivatives M for data (x, y); d0/dn = end slopes (None -> natural)."""
    x = np.asarray(x, float)
    y = np.asarray(y, complex)
    n = len(x)
    h = np.diff(x)
    a = np.zeros(n, complex); b = np.zeros(n, complex)
    c = np.zeros(n, complex); d = np.zeros(n, complex)
    # interior rows: continuity of the first derivative
    a[1:-1] = h[:-1]
    b[1:-1] = 2.0 * (h[:-1] + h[1:])
    c[1:-1] = h[1:]
    d[1:-1] = 6.0 * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1])
    # left end
    if d0 is None:
        b[0], c[0], d[0] = 1.0, 0.0, 0.0
    else:
        b[0], c[0], d[0] = 2.0 * h[0], h[0], 6.0 * ((y[1] - y[0]) / h[0] - d0)
    # right end
    if dn is None:
        a[-1], b[-1], d[-1] = 0.0, 1.0, 0.0
    else:
        a[-1], b[-1], d[-1] = h[-1], 2.0 * h[-1], 6.0 * (dn - (y[-1] - y[-2]) / h[-1])
    return thomas(a, b, c, d)


def spline_eval(x, y, M, t):
    """Evaluate the spline at points t (inside [x_0, x_{n-1}])."""
    x = np.asarray(x, float)
    y = np.asarray(y, complex)
    t = np.asarray(t, float)
    i = np.clip(np.searchsorted(x, t) - 1, 0, len(x) - 2)   # interval index for each t
    h = x[i + 1] - x[i]
    A = (x[i + 1] - t)                                      # distance to right knot
    B = (t - x[i])                                          # distance to left knot
    return (M[i] * A ** 3 + M[i + 1] * B ** 3) / (6 * h) \
        + (y[i] / h - M[i] * h / 6) * A + (y[i + 1] / h - M[i + 1] * h / 6) * B


def spline_resample(x, y, t, d0=None, dn=None):
    """Convenience: fit + evaluate; points of t outside [x_0, x_{n-1}] are set to 0."""
    M = spline_fit(x, y, d0, dn)
    out = np.zeros(len(t), complex)
    inside = (t >= x[0]) & (t <= x[-1])
    out[inside] = spline_eval(x, y, M, t[inside])
    return out
