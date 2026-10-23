"""
stability.py -- linear stability spectrum, paper Eqs. (12)-(15), in our sign convention.

Ansatz (paper Eq. (12)):
   u1 = (w1 + a e^{-i lam z} + conj(c) e^{i conj(lam) z}) e^{i b z}
   u2 = (w2 + d e^{-i lam z} + conj(f) e^{i conj(lam) z}) e^{2 i b z}
Insert into (C),(D), keep first order, collect e^{-i lam z} and e^{+i conj(lam) z}
(the latter conjugated).  With
   M1 = d^2/dx^2 + U1 - b,          U1 = V1 cos2x + i al sin2x
   M2 = (1/2) d^2/dx^2 + U2 - 2b,   U2 = 2 (V2 cos2x + q)
one gets, for the vector (d, a, f, c) = (delta2+, delta1+, delta2-, delta1-):

   lam | d |   | -M2      -2 w1      0        0     | | d |
       | a | = | -2 w1*   -M1        0       -2 w2  | | a |
       | f |   |  0        0         M2       2 w1* | | f |
       | c |   |  0        2 w2*     2 w1     M1*   | | c |

(identical to the paper's Eq. (14) after b -> -b and an overall lam -> -lam,
i.e. the same spectrum mirrored).  A perturbation grows as |e^{-i lam z}| =
e^{Im(lam) z}:  the soliton is linearly UNSTABLE iff some Im(lam) > 0.
Validated against split-step propagation: an eigenvector seed grows at
0.1791 in SSF vs Im(lam) = 0.1793 (b = 0.5, al = 0.7, q = -0.3156).

Discretization: the full periodic box and the SAME d^2 matrix (g.D2) that the
Newton solver folded onto its half grid, so the unfolded Newton solution is an
exact discrete stationary state of this operator.  Consistency matters: if
operator and solution came from different discretizations, the double zero
eigenvalue (phase mode + generalized partner) splits by ~sqrt(residual) and
fakes a small instability.

Finite-box caveat: continuous spectra of the a/d (and c/f) blocks overlap,
and in a finite box their discrete samples can form tiny complex pairs.  Such
pairs have DELOCALIZED eigenvectors and Im(lam) shrinking with L.  A genuine
(internal-mode) instability has an eigenvector concentrated on the soliton.
We therefore report max Im(lam) over eigenvectors that are small at the box
edge (`edge_ratio` < 0.05).  Example: third gap, b = -1.207, al = 0.7919:
the largest Im(lam) ~ 6.5e-3 belongs to eigenvectors with edge/max = 0.6-0.9
(box modes), while the soliton itself is ~5e-4 at the edge; the soliton is
stable, as found in the paper and confirmed by propagation.
"""
import numpy as np


def stability_matrix(x, w1, w2, par, D2):
    V1, V2, al, q, b = par['V1'], par['V2'], par['al'], par['q'], par['b']
    x0 = par.get('x0', 0.0)
    U1 = V1 * np.cos(2 * (x + x0)) + 1j * al * np.sin(2 * (x + x0))
    U2 = 2.0 * (V2 * np.cos(2 * (x + x0)) + q)
    M1 = D2 + np.diag(U1 - b)
    M1c = D2 + np.diag(np.conj(U1) - b)
    M2 = 0.5 * D2 + np.diag(U2 - 2 * b)
    W1, W2 = np.diag(w1), np.diag(w2)
    W1c, W2c = np.diag(np.conj(w1)), np.diag(np.conj(w2))
    Z = np.zeros_like(D2)
    return np.block([
        [-M2,       -2 * W1,  Z,        Z],
        [-2 * W1c,  -M1,      Z,        -2 * W2],
        [Z,          Z,       M2,       2 * W1c],
        [Z,          2 * W2c, 2 * W1,   M1c]])


def cap_profile(x, L, frac=0.15, g0=0.5):
    """Absorber Gamma(x) = g0 s^2 in the outer `frac` of the box (same shape as the SSF absorber)."""
    xa = L * (1 - frac)
    s = np.clip((np.abs(x) - xa) / (L - xa), 0.0, 1.0)
    return g0 * s ** 2


def growth_cap(x, w1, w2, par, D2, L=None, g0=0.5, tol=1e-6):
    """
    Fast stability test with a complex absorbing layer.  Adding the damping -Gamma(x) u to
    the evolution (exactly what the split-step runs do near the box edge) adds -i Gamma(x)
    to every diagonal block of the lam-operator.  Box-quantized continuum modes live in
    the whole box, so they acquire Im(lam) ~ -Gamma * (their weight in the layer) < 0;
    localized (internal) modes shift only by ~ Gamma * |tail|^2.  Hence
        growth = max Im(lam)  over ALL eigenvalues of  A - i Gamma,
    and no eigenvectors are needed (eigvals is ~3x cheaper than eig with vectors).
    Returns (growth, lam_at_max, all lam).
    """
    if L is None:
        L = -x[0]
    A = stability_matrix(x, w1, w2, par, D2)
    G = np.tile(cap_profile(x, L, g0=g0), 4)
    A[np.diag_indices_from(A)] -= 1j * G
    lam = np.linalg.eigvals(A)
    i = int(np.argmax(lam.imag))
    g = lam.imag[i]
    return (g if g > tol else 0.0), lam[i], lam


def spectrum(x, w1, w2, par, D2, vectors=False):
    """All eigenvalues lam of the discretized operator (dense); optionally eigenvectors."""
    A = stability_matrix(x, w1, w2, par, D2)
    if vectors:
        return np.linalg.eig(A)
    return np.linalg.eigvals(A)


def localization(x, vecs, w1, w2, frac=0.999):
    """
    Fraction of each eigenvector's norm inside the soliton core, where the core
    is the smallest symmetric interval |x| < xc holding `frac` of the soliton power.
    """
    dens = np.abs(w1) ** 2 + 2 * np.abs(w2) ** 2
    order = np.argsort(np.abs(x))
    cum = np.cumsum(dens[order]) / dens.sum()
    xc = np.abs(x[order][min(np.searchsorted(cum, frac), len(x) - 1)])
    inside = np.tile(np.abs(x) <= xc, 4)
    nrm = np.sum(np.abs(vecs) ** 2, axis=0)
    return np.sum(np.abs(vecs[inside]) ** 2, axis=0) / nrm, xc


def edge_ratio(x, vecs, L, frac=0.15):
    """max |v| over the outer `frac` of the box divided by max |v| (all 4 components summed pointwise).
    Internal modes decay exponentially (ratio << 1); box-quantized continuum modes are O(1)."""
    M = len(x)
    amp = sum(np.abs(vecs[k * M:(k + 1) * M]) for k in range(4))
    outer = np.abs(x) > (1 - frac) * L
    return np.max(amp[outer], axis=0) / np.max(amp, axis=0)


def analyze(x, w1, w2, par, D2, L=None, edge_max=0.05, tol=1e-6):
    """
    Spectrum + classification.  Returns dict:
      lam         all eigenvalues
      edge        eigenvector amplitude at the box edge / its maximum
      gmax_all    max Im(lam) over everything (includes finite-box continuum pairs)
      gmax_loc    max Im(lam) over LOCALIZED eigenvectors (edge < edge_max): the physical growth rate
      lam_unst    that localized eigenvalue (or None), vec_unst its eigenvector
    """
    if L is None:
        L = -x[0]
    lam, V = spectrum(x, w1, w2, par, D2, vectors=True)
    edge = edge_ratio(x, V, L)
    sel = edge < edge_max
    gl = lam.imag[sel].max() if sel.any() else 0.0
    i_unst = np.where(sel)[0][np.argmax(lam.imag[sel])] if sel.any() else None
    unstable = i_unst is not None and gl > tol
    return dict(lam=lam, edge=edge,
                gmax_all=max(lam.imag.max(), 0.0),
                gmax_loc=gl if gl > tol else 0.0,
                lam_unst=lam[i_unst] if unstable else None,
                vec_unst=V[:, i_unst] if unstable else None)
