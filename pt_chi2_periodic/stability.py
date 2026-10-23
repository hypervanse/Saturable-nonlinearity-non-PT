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
edge (`edge_ratio` < 0.05), or -- the production choice -- add a complex absorbing
layer and take max Im(lam) over all eigenvalues (`growth_cap`).  Example: third gap,
b = -1.207, al = 0.7919: without either filter the largest Im(lam) ~ 6.5e-3 belongs to
box modes (edge/max = 0.6-0.9); with the absorber the physical rate is 1.0e-3, confirmed
by seeded propagation (1.07e-3).

Two solvers for the same matrix:
  growth_cap      dense: all eigenvalues (LAPACK zgeev, ~25 n^3 flops, n = 4M)
  growth_arnoldi  shift-invert Arnoldi: only eigenvalues near chosen shifts sigma.
                  (A - sigma)^{-1} maps the eigenvalues nearest sigma to the largest-|theta|
                  ones, which Arnoldi finds in a few dozen steps.  Factorization: sparse LU
                  (SciPy, fd4 derivative -> banded blocks, ~O(n)) or, NumPy only, one dense
                  inverse (~2 n^3, no LU-factor routine in NumPy).
"""
import numpy as np

try:                                    # optional: sparse LU for the Arnoldi path
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


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


# ---- shift-invert Arnoldi ---------------------------------------------------------------------

def stability_matrix_sparse(x, w1, w2, par, D2):
    """Same operator as stability_matrix, assembled block by block as a SciPy CSC matrix
    (D2 may be dense or sparse; with the fd4 derivative it is banded + periodic corners)."""
    V1, V2, al, q, b = par['V1'], par['V2'], par['al'], par['q'], par['b']
    x0 = par.get('x0', 0.0)
    U1 = V1 * np.cos(2 * (x + x0)) + 1j * al * np.sin(2 * (x + x0))
    U2 = 2.0 * (V2 * np.cos(2 * (x + x0)) + q)
    D = sp.csc_matrix(D2)
    dg = lambda v: sp.diags(v, 0, format='csc')
    M1, M1c = D + dg(U1 - b), D + dg(np.conj(U1) - b)
    M2 = 0.5 * D + dg(U2 - 2 * b)
    W1, W2, W1c, W2c = dg(w1), dg(w2), dg(np.conj(w1)), dg(np.conj(w2))
    return sp.bmat([[-M2,      -2 * W1,  None,    None],
                    [-2 * W1c, -M1,      None,    -2 * W2],
                    [None,     None,     M2,      2 * W1c],
                    [None,     2 * W2c,  2 * W1,  M1c]], format='csc')


def _shift_invert(A, sigma, sparse):
    """v -> (A - sigma I)^{-1} v.  sparse: one sparse LU, solves O(nnz(L+U)).
    dense: NumPy has no LU-factor routine, so the inverse is formed once (~2 n^3 flops)
    and each Arnoldi step is one O(n^2) product."""
    n = A.shape[0]
    if sparse:
        lu = splu((A - sigma * sp.identity(n, dtype=complex, format='csc')).tocsc())
        return lu.solve
    Minv = np.linalg.inv(A - sigma * np.eye(n))
    return lambda v: Minv @ v


def arnoldi(op, n, m, v0):
    """m Arnoldi steps on `op` with classical Gram-Schmidt applied twice ("twice is enough"
    for orthogonality to working precision).  Returns V (n x m+1), H (m+1 x m): op V_m = V_{m+1} H.
    Stops early at an invariant subspace (H[j+1, j] ~ 0)."""
    V = np.zeros((n, m + 1), complex)
    H = np.zeros((m + 1, m), complex)
    V[:, 0] = v0 / np.linalg.norm(v0)
    for j in range(m):
        w = op(V[:, j])
        for _ in range(2):
            h = V[:, :j + 1].conj().T @ w
            w = w - V[:, :j + 1] @ h
            H[:j + 1, j] += h
        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] < 1e-13 * np.linalg.norm(H[:j + 2, j]):   # invariant subspace: exact Ritz pairs
            return V[:, :j + 1], H[:j + 1, :j + 1]
        V[:, j + 1] = w / H[j + 1, j]
    return V, H


def eig_near(A, sigma, nev=10, m=60, tol=1e-9, maxrestart=6, sparse=False, seed=0):
    """
    The `nev` eigenvalues of A closest to `sigma`, by shift-invert Arnoldi.
    Ritz values theta of (A - sigma)^{-1} give lam = sigma + 1/theta.  Acceptance uses the TRUE
    residual of the original problem, |A x - lam x| / (|A|_1 |x|), not the Arnoldi estimate.
    Not converged after an Arnoldi cycle -> explicit restart from the sum of the wanted Ritz
    vectors (simple; Krylov-Schur thick restart would converge faster but is longer code).
    Returns lam (nev), X (n x nev), res (nev).
    """
    n = A.shape[0]
    op = _shift_invert(A, sigma, sparse)
    anorm = abs(A).sum(axis=0).max()                              # 1-norm (works dense and sparse)
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    for _ in range(maxrestart):
        V, H = arnoldi(op, n, m, v0)
        k = H.shape[1]
        theta, Y = np.linalg.eig(H[:k, :k])
        idx = np.argsort(-np.abs(theta))[:nev]                    # largest |theta| = nearest sigma
        lam = sigma + 1.0 / theta[idx]
        X = V[:, :k] @ Y[:, idx]
        X /= np.linalg.norm(X, axis=0)
        res = np.linalg.norm(A @ X - X * lam, axis=0) / anorm
        if np.all(res < tol) or k < m:
            break
        v0 = X.sum(axis=1)
    return lam, X, res


def default_shifts(remax=3.0, dre=0.25, ims=(0.05, 0.3)):
    """Two rows of shifts, Im(sigma) = 0.05 (weak instabilities, Im lam ~ 1e-4 .. 0.05) and 0.3
    (strong ones, ~0.05 .. 0.4), for 0 <= Re(sigma) <= remax.  One row is not enough: a strongly
    unstable eigenvalue far above a low shift is farther from it than the damped box continuum
    just below the axis.  Only Re >= 0 is needed: the spectrum is symmetric under
    lam -> -conj(lam), also with the absorbing layer (checked in validate_arnoldi.py)."""
    re = np.arange(0.0, remax + dre / 2, dre)
    return np.concatenate([re + 1j * im for im in ims])


def growth_arnoldi(x, w1, w2, par, D2, L=None, g0=0.5, shifts=None, nev=16, m=64, tol=1e-6,
                   restol=1e-9, backend='auto'):
    """
    Drop-in alternative to growth_cap (same operator A - i Gamma, same return values) that
    computes only the eigenvalues near `shifts` (default: scan of default_shifts()).
    Pass shifts=[lam_guess] to refine a known eigenvalue, e.g. on a larger box.
    backend: 'sparse' (SciPy sparse LU; best with the fd4 derivative), 'dense' (NumPy inverse),
             'auto' = sparse if SciPy is importable and D2 has < 5% nonzeros, else dense.
    Eigenvalues with |Re lam| > max Re(shifts) + spacing are NOT searched: high-frequency modes
    (|Re lam| > 3) must be checked with the dense solver.
    Returns (growth, lam_at_max, lam_found) with only residual-converged eigenvalues in lam_found.
    """
    if L is None:
        L = -x[0]
    if backend == 'auto':
        dens = np.count_nonzero(np.asarray(D2.todense() if hasattr(D2, 'todense') else D2)) / D2.shape[0] ** 2
        backend = 'sparse' if (HAVE_SCIPY and dens < 0.05) else 'dense'
    if backend == 'sparse' and not HAVE_SCIPY:
        raise ImportError("backend='sparse' needs SciPy")
    G = np.tile(cap_profile(x, L, g0=g0), 4)
    if backend == 'sparse':
        A = stability_matrix_sparse(x, w1, w2, par, D2) - 1j * sp.diags(G, 0, format='csc')
    else:
        A = stability_matrix(x, w1, w2, par, D2)
        A[np.diag_indices_from(A)] -= 1j * G
    if shifts is None:
        shifts = default_shifts()
    found = []
    for s in np.atleast_1d(shifts):
        lam, X, res = eig_near(A, complex(s), nev=nev, m=m, tol=restol, sparse=(backend == 'sparse'))
        found.extend(lam[res < restol])
    found = np.array(found)
    if found.size == 0:
        return 0.0, None, found
    i = int(np.argmax(found.imag))
    g = found.imag[i]
    return (g if g > tol else 0.0), found[i], found
