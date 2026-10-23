# Gap solitons in a χ⁽²⁾ waveguide with a PT-symmetric periodic potential

Pure-numpy implementation of the numerical methods of the PhD thesis of **F. C. Moreira** (sole
author), applied to the periodic model of the article

> F. C. Moreira, V. V. Konotop, B. A. Malomed, *Solitons in PT-symmetric periodic systems with the
> quadratic nonlinearity*, Phys. Rev. A **87**, 013832 (2013),

whose figures the results are compared with.

namely: **edge shooting** with an adaptive Runge–Kutta–Fehlberg 4(5) integrator (thesis Fig. 3.1
flowchart), **cubic-spline** transfer to a uniform grid, **Newton–Raphson** on a symmetric
half-domain with mirrored central differences, continuation in α and b, **linear stability**
(paper Eqs. 12–15) and **split-step Fourier** propagation.

No scipy / numba / matplotlib was available in the build environment, so every algorithm is written
out explicitly (RK tableau, Thomas solver, Jacobian, Floquet monodromy, SVG writer, …).
Only `numpy` is required. The figures are SVG, written by `svgplot.py`.

---

## 0. Model and sign convention

```
stationary:   w1'' + [V1 cos2ξ + iα sin2ξ − b] w1 + 2 w1* w2 = 0                 (A)
              ½ w2'' + 2[V2 cos2ξ + q − b] w2 + w1²          = 0                 (B)
evolution:    i ∂ζu1 = −∂²ξu1 − [V1 cos2ξ + iα sin2ξ] u1 − 2 u1* u2               (C)
              i ∂ζu2 = −½∂²ξu2 − 2[V2 cos2ξ + q] u2 − u1²                         (D)
power:        P = P1 + P2,   P_l = l ∫|w_l|² dξ
```

**Why not the printed Eqs. (1),(3)?** With u_l = w_l e^{ilbζ} the printed equations put the SH
Mathieu band edge at b = −0.378 and the semi-infinite gap at b → −∞. Every number in the paper's
figures (b₂₊⁽⁰⁾ = 0.3784, q₊⁽⁰'⁰⁾ = −0.316, third-gap edge −1.293, semi-infinite gap at b > 0.38)
is reproduced only after b → −b, which is (A),(B). The evolution equations (C),(D) are the printed
(1) with ζ → −ζ: the mirror image ξ → −ξ of the gain/loss pattern. That changes neither existence
nor stability.

---

## 1. Pipeline (and why each step is done the way it is)

| step | file | what | why |
|---|---|---|---|
| Floquet spectrum | `model.py` | monodromy M of y'' = −Q(ξ)y over one period; band ⇔ \|tr M/2\| ≤ 1 | band edges, gaps, matching mismatch q, and the **decaying tail** for shooting |
| RKF45 | `rk45.py` | Fehlberg 4(5); error = max\|y5 − y4\| scaled; reject + refine if err > 1; h ← 0.9 h err^(−1/5) | exactly the thesis' "order 4 and 5 … error = maximal difference" with interstep refinement |
| edge shooting | `shooting.py` | integrate (A),(B) **inward** from ξ_f = nπ to 0; impose w1'(0) = w2'(0) = 0 | inward, the unwanted (growing-at-∞) Floquet solution *decays*, so the shot is well conditioned |
| tail initial data | `shooting.tails` | (w1,w1') = C1(1, r1); (w2,w2') = C2(1, r2) + C1²(ψ, ψ') | ψ = Floquet particular solution forced by w1² (multiplier μ1²); without it C2 becomes a huge cancelling number whenever \|μ2\| < \|μ1\|² |
| flowchart | `shooting.flowchart_root` | march C by δC until D changes sign, then bisect (Fig. 3.1) | inner loop: C2 at fixed C1 (D2 = 0); outer loop: C1 with D1 = 0 |
| polish | `shooting.newton_polish` | 2-D Newton on (C1, C2) | bisection gains 1 bit/shot; Newton gives the last digits |
| spline | `spline.py` | clamped C² cubic spline (Thomas algorithm), slopes from RK | adaptive RK mesh → uniform Newton grid |
| Newton | `newton_fd.py` | half grid ξ ∈ [0, L], 4th-order central differences with **mirrored ghosts** (Re even, Im odd), or Fourier-spectral d² folded the same way | PT symmetry w(−ξ) = w*(ξ) halves the unknowns and removes the U(1) phase zero mode, so the Jacobian is nonsingular |
| box | `newton_fd.Grid` | periodic box [−L, L), L a multiple of π/2 (mirror symmetry also about ξ = L) | Dirichlet walls break the PT symmetry of the *finite* lattice; its spectrum turns complex (1.3e-2 measured at L = 16π) and pollutes stability |
| continuation | `newton_fd.continuation`, `pipeline.trace_branch` | α from 0 to the target, then b along the branch; secant predictor | the shooting is done for the conservative case only (as in the paper) |
| stability | `stability.py` | dense spectrum of the 4×4 block operator (paper Eq. 14 in our convention) on the **same grid and the same d² matrix** | otherwise the double zero eigenvalue splits by ~√residual and fakes instability |
| fast spectra (optional) | `stability.growth_arnoldi` | shift-invert Arnoldi: only eigenvalues near given shifts; sparse LU if SciPy is present (use `deriv='fd4'`), else one dense inverse | the dense eigensolver costs ~25 n³; a sparse LU is ~O(n), and a single shift next to a known eigenvalue refines it on a larger box in a fraction of the time (`validate_arnoldi.py`) |
| box artefacts | `stability.growth_cap` | complex absorbing layer −iΓ(ξ) at the box edge; growth = max Im λ | box modes acquire Im λ < 0; localized and radiating internal modes keep theirs. Cross-checked with an eigenvector edge-amplitude criterion (`stability.analyze`) |
| propagation | `ssf.py` | Strang (2nd order) or Yoshida (4th order) splitting; exact dispersion in Fourier space; local χ⁽²⁾ + potential step by RK4; optional absorber | linear part exact, nonlinear χ⁽²⁾ step has no closed form (unlike Kerr) |

### Lattice centring
PT-symmetric points of the lattice are ξ = 0 and ξ = π/2. In the **third finite gap** the FF
band-edge Bloch mode (top of band 1) is odd about ξ = 0 (sin ξ-like) and even about ξ = π/2,
so the fundamental soliton is centred on the potential *minimum*, x₀ = π/2. All code takes
`par['x0']`, and ξ is measured from x₀.

---

## 2. How to run

Step-by-step guide with expected outputs and troubleshooting: **REPRODUCE.md**.

```bash
python3 validate.py        # ~5 min: convergence/consistency checks -> results/validation.json
python3 run_bands.py       # ~2 min: Floquet band edges (Fig. 1)
python3 run_semiinf.py     # ~25 min: semi-infinite gap, cases 1-3 (Figs. 3, 4, 7, 18)
python3 run_thirdgap.py    # ~45 min: third finite gap (Figs. 5, 6, 8)
python3 make_figures.py    # SVG figures in figures/ (+ PNG via headless Chromium if present)
```

Minimal interactive use (REPL):

```python
import numpy as np
from pipeline import shoot, to_grid
from newton_fd import Grid, solve, continuation, power_half
par = dict(V1=1., V2=1., al=0., q=-0.3156, b=0.25)          # conservative case for shooting
sh  = shoot(par, n=6, c1_start=0.05, dc1=0.05, dc2=0.1, c2_guess=0.0)   # thesis flowchart
g   = Grid(L=16*np.pi, h=0.15, deriv='fourier')             # or deriv='fd4' (central differences)
w1, w2, its, res = solve(g, *to_grid(g, sh), par)           # spline -> Newton-Raphson
br  = continuation(g, w1, w2, par, 'al', np.arange(0.05, 0.701, 0.05))   # switch on gain/loss
```

---

## 3. Results against the paper

| quantity | paper | this code |
|---|---|---|
| SH top edge b₂₊⁽⁰⁾ (V₂ = 1, q = 0) | 0.3784 | 0.37849 (= −a₀(2)/4 of Mathieu) |
| q₊⁽⁰'⁰⁾ at α = 0.7 | −0.316 | −0.31560 |
| FF/SH edges coalesce in the 3rd gap | α ≈ 0.792, b ≈ −1.29 | α = 0.7919: −1.29322 vs −1.29317 |
| semi-inf. case 1, b = 0.25 | "P = 0.5" (dot drawn at b ≈ 0.27, P ≈ 0.47) | P = 0.388 (0.444 at b = 0.27) |
| semi-inf. case 2, b = 0.43, q = 0 | P = 0.5 | P = 0.4992 |
| semi-inf. case 3, b = 0.21, q = −0.5134 | P = 0.5 | P = 0.4970 |
| 3rd gap, α = 0.7919, b = −1.207 | P = 0.5 | P = 0.50097 |
| 3rd gap, α = 0.7, b = −1.234 / α = 0.9, b = −1.153 | P = 0.5 | 0.502 / 0.523 |
| case 1 stability, b = 0.25 / 0.5 | stable / unstable quartet | stable / λ = ±1.748 + 0.179i |
| case 2 instability band at α = 0.7 (Fig. 10, q = 0) | b ≈ 0.62–0.97, plus a sliver at the SH edge | 0.65–0.97, plus 0.38–0.39 |
| case 3 instability bands at α = 0.7 (Fig. 10, q = −0.5134) | ≈ 0.22–0.58 and 0.76–0.9 | 0.20–0.53 and 0.76–0.87 |
| 3rd gap, α = 0.7919, b = −1.101 | unstable, purely imaginary pair, decays | growth 0.20 with Re λ = 0; P drops 2.10 → 0.54 |

## 4. Where this code disagrees with the paper (and the evidence)

1. **Sign convention.** The figures correspond to Eqs. (3) with b → −b (see §0). Eq. (9) and Eq. (10)
   give q with opposite signs; the numbers follow (10). The third-gap matching value printed as
   1.29 is −1.29. The FF/SH currents come out with the opposite sign to the paper's, as expected for
   the mirrored (ξ → −ξ) convention.
2. **Third-gap centring.** The fundamental third-gap soliton sits at ξ₀ = π/2 (potential minimum).
   Centred at ξ = 0, shooting only finds multi-hump, high-power states.
3. **Third gap, α = 0, is not stable.** Fig. 5 draws this branch thick. Here it has an
   oscillatory instability at every computed point (b = −1.282 … −1.077), growth 0.006–0.087
   along the branch (0.028 at b = −1.197). Evidence:
   the same rate without an absorbing layer (0.0289) and with layers of strength 0.25 and 1
   (0.0277, 0.0281); direct propagation with 1e-4 noise grows at 0.028. The eigenvector has an
   oscillating tail: an internal mode resonating with the opposite-signature continuum, which
   overlaps in finite gaps.
4. **Weak instabilities below the paper's resolution**, converged in box size and confirmed by
   seeded propagation:
   case 3, b = 0.21: Im λ = 3.56e-3 (propagation 3.39e-3);
   3rd gap, α = 0.7919, b = −1.207: Im λ = 1.04e-3 (propagation 1.07e-3).
   Over ζ = 1000 these amount to e^3.6 and e^1.0, hard to see in noisy runs.
5. **Inside the paper's stability windows** (case 1, b ≈ 0.73–1.0; case 3, b ≈ 0.53–0.76) the
   internal-mode instability does switch off, as in the paper. A high-frequency mode
   (Re λ ≈ ±4.7–4.8) with growth 0.02–0.06 remains. Its rate *increases* with the box
   (case 3, b = 0.65: 0.050 at L = 16π, 0.059 at L = 24π; propagation in the 24π box gives 0.055),
   so it is either a radiation (convective-type) amplification by the gain or a finite-box effect.
   It is not counted in the thick/thin lines and is plotted separately (open markers).

## 5. Files

```
model.py        potentials, first-order RHS, Floquet monodromy / band edges / decaying tail
rk45.py         Runge-Kutta-Fehlberg 4(5) with step rejection and refinement
shooting.py     Floquet tails (+ forced SH tail), edge shot, thesis flowchart, nested driver, 2-D Newton polish
spline.py       clamped cubic spline (Thomas algorithm)
newton_fd.py    periodic box, fd4 / Fourier d^2, folding onto the symmetric half grid, Newton, continuation
stability.py    linear-stability operator, absorbing-layer growth rate, eigenvector edge criterion
ssf.py          split-step Fourier (Strang / Yoshida), absorber
pipeline.py     the chain used by the drivers
run_bands.py, run_semiinf.py, run_thirdgap.py     production runs -> results/*.json
validate.py, validate_weak.py                     checks -> results/validation.json, results/weak.json
make_figures.py, svgplot.py                       figures/*.svg (+ figures/png/*.png)
build_report.py                                   report.html
run_convergence.py                                third-gap growth rates vs h, L, absorber -> results/convergence.json
validate_arnoldi.py                               shift-invert Arnoldi vs dense eigensolver -> results/arnoldi.json
make_tables.py, make_eps.py                       manuscript tables (from results/*.json) and EPS/PDF figures
manuscript/                                       LaTeX draft (main.tex, refs.bib, Makefile, check_tex.py, PLAN.md)
bench_perf.py, make_perf.py, svg2eps.py            performance measurements -> results/perf.json -> paper 2
manuscript_perf/                                  paper 2: where the time goes (diagrams, cost model, recommendations)
```

## 6. Manuscript

`manuscript/` holds a draft paper built from these results: REVTeX 4.2 source, bibliography, vector
EPS figures (with PDF copies), and tables written by `make_tables.py` straight from `results/*.json`.
**`manuscript/PLAN.md`** has the claims register (reproduced / correction / new / open, with the
evidence and confidence for each), the computations still to do before submission, the authorship
note, and the arXiv steps. No TeX was available where it was written: `make check` runs static
checks, and `make` builds the PDF on any machine with TeX Live.

`manuscript_perf/` is a second paper on the computation itself: which resource bounds each stage
(dense eigensolver vs interpreter overhead), measured kernel scaling and thread efficiency, the task
graph of the convergence study, a calibrated scheduling model, and ranked recommendations
(process-level parallelism, Arnoldi refinement, sparse factorization, batching). Every number in it
is a macro in `numbers.tex`, written by `make_perf.py` from `results/perf.json` (`bench_perf.py`,
about 10 minutes on an idle machine).
