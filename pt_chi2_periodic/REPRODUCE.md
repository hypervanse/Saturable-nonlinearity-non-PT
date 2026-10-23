# Reproducing the results

## TL;DR

```bash
# 1. only numpy is required (tested: Python 3.13, numpy 2.5)
python3 -m pip install numpy            # skip if numpy is already present
unzip pt_chi2_periodic.zip && cd pt_chi2_periodic

# 2. checks first (~5 min): every building block against a known answer
python3 validate.py                     # prints a report, writes results/validation.json

# 3. the physics (~1.5 h on 4 cores in total; each script is independent)
python3 run_bands.py                    # Floquet band edges          -> results/bands.json      (~2 min)
python3 run_semiinf.py                  # semi-infinite gap, cases 1-3 -> results/semiinf.json   (~35 min)
python3 run_thirdgap.py                 # third finite gap             -> results/thirdgap.json  (~45 min)
python3 validate_weak.py                # weak instabilities by seeded propagation -> results/weak.json (~15 min)

# 4. pictures and the report page
python3 make_figures.py                 # figures/*.svg (+ figures/png/*.png if headless Chromium exists)
python3 build_report.py                 # report.html (open in any browser)
```

The `results/` folder in the bundle already holds the outputs of these runs, so steps 4 alone
regenerate every figure and the report in under a minute.

---

## 1. Requirements and why so few

* Python ≥ 3.10 and **numpy**. Nothing else.
* scipy / numba / matplotlib were unavailable where this was built, so the RK integrator, spline,
  Newton solver, Floquet monodromy, eigen-classification and the SVG/PNG writers are all written
  out in the source. This also means every numerical step is visible in ~1500 lines of Python.
* Optional: a Chromium binary for PNG copies of the figures (`make_figures.svg_to_png` looks in
  `/opt/pw-browsers/chromium-*/chrome-linux/chrome`; edit the glob for your system). SVGs are
  always written and open in any browser.
* Threads: the dense eigenproblems use numpy's BLAS. Two runs in parallel:
  `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 run_semiinf.py &` and the same for
  `run_thirdgap.py`.

## 2. What each step should print (so you know it worked)

| step | look for | expected |
|---|---|---|
| `validate.py` [1] | `rtol=1e-12: \|y(20pi)-1\|` | ≈ 1e-11, with accepted/rejected step counts |
| `validate.py` [3] | `SH top edge` | 0.378489 (Mathieu −a₀(2)/4 = 0.378489) |
| `validate.py` [4] | `fd4 observed order` | ≈ 3.95, 4.00 |
| `validate.py` [4] | `max\|w1_newton - w1_shoot\|` (Fourier) | ≈ 2.5e-7 |
| `validate.py` [5] | `SSF order 2 / 4` | 2.00 / ≈ 3.95 |
| `validate.py` [5] | `Im(lam)` vs `SSF envelope growth` | 0.1793 vs 0.1791 |
| `run_bands.py` | `q_+^(0,0)(al=0.7)` | −0.31560 |
| `run_semiinf.py` | `[shoot] ... case1` | `c1=0.40975…  c2=-1.16861…  shots≈1340` |
| `run_semiinf.py` | `example b=0.43` (case 2) | `P=0.4992` |
| `run_thirdgap.py` | `[shoot] b=-1.2070 ... x0=1.5708` | `c1=1.24373…  c2=+0.69402…  w1(0)=+0.29587` |
| `run_thirdgap.py` | `profile case1` | `P=0.50097` |
| `validate_weak.py` | case 3 / third gap | Im λ 3.56e-3 vs SSF ≈ 3.4e-3; 1.04e-3 vs ≈ 1.07e-3 |

Shot counts can differ by a few between machines (floating-point order in BLAS); the converged
amplitudes agree to the shooting tolerance (1e-9).

## 3. Walk through the pipeline by hand (REPL)

```python
import numpy as np
from shooting import tails, nested_flowchart, newton_polish, profile_from_shot
from newton_fd import Grid, solve, continuation, full_line, power_half
from pipeline import to_grid, stability_of, ssf_run

# --- 1. conservative problem (alpha = 0): semi-infinite gap, case 1 -----------------------
par = dict(V1=1.0, V2=1.0, al=0.0, q=-0.3156, b=0.25)    # figure convention, see README sec. 0
T = tails(par, n=6)                 # Floquet data at xi_f = 6 pi: decay factors mu1, mu2, slopes r1, r2,
print(T['mu1'], T['mu2'])           # forced SH tail z, and the amplitude scales s1, s2

# --- 2. thesis flowchart: inner loop C2 (D2 = w2'(0) = 0), outer loop C1 (D1 = w1'(0) = 0) --
c1, c2, nshots = nested_flowchart(par, T, c1_start=0.05, dc1=0.05, dc2=0.1, c2_guess=0.0)
c1, c2 = newton_polish(par, T, c1, c2)          # 2-D Newton on the shooting residual
xs, w1, p1, w2, p2, info = profile_from_shot(par, T, c1, c2)
print(len(xs), info)                # adaptive RKF45 mesh size, accepted/rejected steps

# --- 3. cubic spline -> 4. Newton on the symmetric half grid ---------------------------------
g = Grid(L=16*np.pi, h=0.15, deriv='fourier')     # deriv='fd4' = 4th-order central differences
sh = dict(xs=xs, w1=w1, p1=p1, w2=w2, p2=p2)
v1, v2, its, res = solve(g, *to_grid(g, sh), par, verbose=True)   # quadratic convergence

# --- 5. continuation: switch on gain/loss, then move along the branch -----------------------
br = continuation(g, v1, v2, par, 'al', np.round(np.arange(0.05, 0.701, 0.05), 3))
p7 = dict(par, al=0.7)
print(br[-1]['P'])                  # 0.3883 at b = 0.25

# --- 6. linear stability (absorbing layer; growth = max Im lambda) --------------------------
A = stability_of(g, br[-1], p7)
print(A['gmax'], A['lam_unst'])     # 0.0, None  -> stable

# --- 7. split-step Fourier with 10 % noise on the same grid ---------------------------------
r = ssf_run(g, br[-1]['w1'], br[-1]['w2'], p7, zmax=200.0, dz=0.01, noise=0.1)
print(r['P'][0], r['P'][-1])        # power stays put for a stable soliton
```

## 4. Changing the problem

* **Other parameters**: everything is a plain dict `par = dict(V1, V2, al, q, b, x0)`.
* **Where the soliton sits**: `x0 = 0` (potential maximum) or `x0 = pi/2` (minimum). In the third
  finite gap use `x0 = pi/2`: the FF band-edge Bloch mode is odd about 0 and even about π/2.
  Check with `model.monodromy(model.Q_ff(V1, al, b_edge))`: the column whose entry in the first row
  stays at ±1 is the Bloch mode (first column = even about 0, second column = odd).
* **Another potential** (e.g. the localized Scarf-type potential of thesis ch. 6): the lattice
  enters through `cos(2(x+x0))`, `sin(2(x+x0))` in `model.make_rhs`, `model.Q_ff/Q_sh`,
  `newton_fd.residual_jacobian`, `stability.stability_matrix` and `ssf.propagate`. A localized
  potential also removes the Floquet structure of the tails: replace `decaying_tail` by the
  exponential tail of the free equation (r = −√(b) for the FF, −2√(b − q) for the SH).

## 5. When something does not converge

| symptom | cause | fix |
|---|---|---|
| `flowchart_root: no sign change` | start of the marching is above the root, or the step jumps over a narrow root pair | lower `c2_guess` / `c1_start`, use a smaller `dc2`; or map all roots first: for a few fixed c1, march c2 over a range with `shooting.edge_shot` and list every sign change of D2 together with D1 there |
| `inner_root: could not bracket` with very large \|c2\| | SH decays faster than w1⁴ (\|μ2\| < \|μ1\|⁴) | shoot from a shorter distance (`n = 4` instead of 6) |
| shooting converges to a multi-hump state, Newton then fails in α | wrong lattice centring | `x0 = pi/2` (finite gaps) |
| branch stops: "solution reaches the box edge" | soliton near a band edge is wider than the box | larger `L` (keep `L` a multiple of π/2), or a larger `tail_tol` in `trace_branch` |
| spurious tiny complex eigenvalues | Dirichlet walls / finite box | already handled: periodic box + absorbing layer (`stability.growth_cap`); confirm a suspicious rate by changing `L` and `g0`, and by `validate_weak.check` |

## 6. Map from the thesis / paper to the code

| thesis / paper | code |
|---|---|
| Fig. 3.1 flowchart (C2 at fixed C1, then C1) | `shooting.flowchart_root`, `shooting.nested_flowchart` |
| RK of order 4 and 5, error = max difference, step refinement | `rk45.rkf45` |
| boundary conditions at the centre (3.2) | D_l = w_l′(0) = 0 in `shooting.edge_shot`; mirrored ghosts in `newton_fd.fold` |
| Newton–Raphson for α > 0 | `newton_fd.residual_jacobian`, `newton_fd.newton`, `newton_fd.continuation` |
| Eqs. (12)–(15), linear stability | `stability.stability_matrix`, `stability.growth_cap`, `stability.analyze` |
| Eq. (11), currents | `pipeline.currents_full` |
| Fig. 1 band-gap spectrum | `model.band_edges`, `run_bands.py` |
| direct simulations with 10 % noise | `ssf.propagate`, `pipeline.ssf_run` |
