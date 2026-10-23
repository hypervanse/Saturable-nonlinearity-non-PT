"""
ssf.py -- split-step Fourier propagation of the evolution equations (C),(D).

   i u1_z = -u1_xx      - U1 u1 - 2 conj(u1) u2        U1 = V1 cos2x + i al sin2x
   i u2_z = -(1/2)u2_xx - U2 u2 - u1^2                 U2 = 2 (V2 cos2x + q)

Split the generator into
   D (dispersion, diagonal in Fourier space, solved EXACTLY):
        u1^ <- exp(-i k^2 dz) u1^ ,   u2^ <- exp(-i k^2/2 dz) u2^
   N (everything local in x: potentials, gain/loss, chi(2) coupling):
        u1_z = i U1 u1 + 2 i conj(u1) u2 - g(x) u1
        u2_z = i U2 u2 + i u1^2          - g(x) u2
     solved pointwise by classical RK4 over the step (the chi(2) step has no
     closed form, unlike the Kerr phase rotation).
Strang:      S(dz) = D(dz/2) N(dz) D(dz/2)                    -> O(dz^2) global
Yoshida-4:   S4(dz) = S(w1 dz) S(w0 dz) S(w1 dz),
             w1 = 1/(2 - 2^{1/3}), w0 = -2^{1/3}/(2 - 2^{1/3})  -> O(dz^4) global
(RK4 inside N is 4th order, so it never limits either scheme.)
Consecutive half-steps D(dz/2) D(dz/2) are merged into D(dz) between samples.

Box: x in [-L, L) with L a multiple of pi/2 so that the potential is periodic
on the FFT grid.  Optional absorber g(x) >= 0 in the outer `absorb` fraction
of the box (smooth ramp ~ s^2): radiation leaving the soliton is damped
instead of wrapping around the periodic box.
"""
import numpy as np


def make_box(L, N):
    x = -L + (2 * L / N) * np.arange(N)            # contains x = 0 at index N/2
    k = 2 * np.pi * np.fft.fftfreq(N, d=2 * L / N)
    return x, k


def absorber(x, L, frac=0.15, g0=1.0):
    """g(x) = g0 * s^2 on the outer `frac` of the box, s in [0,1] the depth into the layer."""
    xa = L * (1 - frac)
    s = np.clip((np.abs(x) - xa) / (L - xa), 0.0, 1.0)
    return g0 * s ** 2


def _N_rhs(u1, u2, U1, U2, g):
    du1 = 1j * U1 * u1 + 2j * np.conj(u1) * u2 - g * u1
    du2 = 1j * U2 * u2 + 1j * u1 * u1 - g * u2
    return du1, du2


def _N_step(u1, u2, dz, U1, U2, g):
    """Classical RK4 for the local (x-pointwise) part over dz."""
    a1, a2 = _N_rhs(u1, u2, U1, U2, g)
    b1, b2 = _N_rhs(u1 + 0.5 * dz * a1, u2 + 0.5 * dz * a2, U1, U2, g)
    c1, c2 = _N_rhs(u1 + 0.5 * dz * b1, u2 + 0.5 * dz * b2, U1, U2, g)
    d1, d2 = _N_rhs(u1 + dz * c1, u2 + dz * c2, U1, U2, g)
    return (u1 + dz / 6 * (a1 + 2 * b1 + 2 * c1 + d1),
            u2 + dz / 6 * (a2 + 2 * b2 + 2 * c2 + d2))


def _D_step(u1, u2, E1, E2):
    """Exact dispersion step with precomputed phase factors E1, E2."""
    return np.fft.ifft(E1 * np.fft.fft(u1)), np.fft.ifft(E2 * np.fft.fft(u2))


def propagate(u1, u2, x, k, par, dz, zmax, nsave=200, order=2, g=None, verbose=True):
    """
    Propagate (u1, u2) from z = 0 to zmax.  Returns dict with
      z (nsave+1,), |u1|^2 and |u2|^2 snapshots, power P(z) = int |u1|^2 + 2|u2|^2,
      and the final fields.
    order = 2 (Strang) or 4 (Yoshida triple jump).
    """
    V1, V2, al, q = par['V1'], par['V2'], par['al'], par['q']
    x0 = par.get('x0', 0.0)                       # x is measured from the soliton centre x0
    U1 = V1 * np.cos(2 * (x + x0)) + 1j * al * np.sin(2 * (x + x0))
    U2 = 2.0 * (V2 * np.cos(2 * (x + x0)) + q)
    if g is None:
        g = np.zeros_like(x)
    dx = x[1] - x[0]
    nsteps = int(round(zmax / dz))
    dz = zmax / nsteps
    every = max(1, nsteps // nsave)

    if order == 2:
        subs = [1.0]
    elif order == 4:
        c = 2.0 ** (1.0 / 3.0)
        w1 = 1.0 / (2.0 - c)
        subs = [w1, -c * w1, w1]                    # Yoshida weights (sum = 1)
    else:
        raise ValueError("order must be 2 or 4")
    # phase factors for half sub-steps  exp(-i k^2 (w dz/2)) and for SH exp(-i k^2/2 (w dz/2))
    Eh = {w: (np.exp(-1j * k ** 2 * w * dz / 2), np.exp(-0.5j * k ** 2 * w * dz / 2)) for w in set(subs)}

    zs, A1, A2, P = [0.0], [np.abs(u1) ** 2], [np.abs(u2) ** 2], []
    P.append(np.sum(np.abs(u1) ** 2 + 2 * np.abs(u2) ** 2) * dx)
    for n in range(1, nsteps + 1):
        for w in subs:                              # one Strang step per sub-weight
            E1, E2 = Eh[w]
            u1, u2 = _D_step(u1, u2, E1, E2)
            u1, u2 = _N_step(u1, u2, w * dz, U1, U2, g)
            u1, u2 = _D_step(u1, u2, E1, E2)
        if n % every == 0 or n == nsteps:
            zs.append(n * dz)
            A1.append(np.abs(u1) ** 2)
            A2.append(np.abs(u2) ** 2)
            P.append(np.sum(np.abs(u1) ** 2 + 2 * np.abs(u2) ** 2) * dx)
            if not np.isfinite(P[-1]):
                raise FloatingPointError(f"ssf: field blew up at z = {n * dz}")
            if verbose and len(zs) % 20 == 0:
                print(f"      z={n * dz:8.2f}  P={P[-1]:.6f}  max|u1|^2={A1[-1].max():.4f}")
    return dict(z=np.array(zs), I1=np.array(A1), I2=np.array(A2), P=np.array(P),
                u1=u1, u2=u2)


def stationary_on_box(xs_half, w1_half, w2_half, x):
    """
    Place a PT-symmetric stationary solution, known on a half grid x >= 0,
    onto the propagation box x via w(-x) = conj w(x) and the cubic spline
    (clamped: Re w'(0) = 0 by symmetry; zero outside the solution's domain).
    """
    from spline import spline_resample
    out = []
    for w in (w1_half, w2_half):
        # even real part and odd imaginary part, interpolated separately in |x|
        re = spline_resample(xs_half, w.real, np.abs(x), d0=0.0).real
        im = spline_resample(xs_half, w.imag, np.abs(x)).real * np.sign(x)
        out.append(re + 1j * im)
    return out[0], out[1]
