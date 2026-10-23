# Manuscript plan: from computed results to an arXiv preprint

Working title: *Gap solitons in a χ⁽²⁾ waveguide with a PT-symmetric lattice revisited: edge shooting,
symmetric Newton continuation, and spectral stability*.

Authors (as requested): **F. C. Moreira** and **Claude Opus 5.5** (Anthropic; work done in Claude Code).
Read §6 before submitting: arXiv does not accept AI systems as authors.

Status: a complete first draft (`main.tex`) with references (`refs.bib`), vector EPS figures (PDF copies
too), and tables generated from the data. **It has not been compiled.** The build container had no TeX
and no network access, by design. `check_tex.py` passes: braces, environments, labels, citations, figure
files. Run `make` on a machine with TeX Live before anything else.

---

## 1. What is in this folder and where each piece comes from

| file | made by | from |
|---|---|---|
| `main.tex` | written by hand | numbers quoted in the text come from `results/*.json` (see §9) |
| `refs.bib` | written by hand | each entry tagged `[MKM-list]` (taken from the 2013 paper's reference list) or `[memory]` (verify) |
| `tables/validation.tex` | `make_tables.py` | `results/validation.json`, `results/weak.json` |
| `tables/thirdgap.tex` | `make_tables.py` | `results/thirdgap.json` |
| `tables/convergence.tex` | `make_tables.py` | `results/convergence.json` (`run_convergence.py`) |
| `figures/*.eps`, `figures/*.pdf` | `make_eps.py` | same plotting code as the report, in paper mode: no title or notes, cropped. SVG → Chromium PDF → `pdftocairo -eps` |
| `figures/preview/*.png` | `make_eps.py` | 72-dpi renderings of the PDFs, for a quick look |
| `check_tex.py` | — | static checker used in place of a compiler |
| `Makefile` | — | `make` (latex+dvips), `make pdf`, `make check`, `make arxiv`, `make figures` |

## 2. Build

```bash
cd manuscript
make check                 # static checks, no TeX needed
make                       # latex -> bibtex -> latex x2 -> dvips -> ps2pdf  => main.pdf
make pdf                   # alternative: pdflatex route with the PDF figures
make arxiv                 # arxiv.tar.gz = main.tex + main.bbl + tables + EPS (arXiv does not run BibTeX)
make figures               # regenerate tables + figures from ../results/*.json
```

Switches at the top of `main.tex`:
* `\aiauthortrue` / `\aiauthorfalse`: authorship (see §6).
* `\showtodostrue` / `\showtodosfalse`: red `[TODO: …]` markers.

## 3. Claims register: what to check, and how sure each claim is

Status values:
* **reproduced**: agrees with the 2013 paper.
* **correction**: the 2013 text and its own figures disagree, and the computation picks one.
* **new**: the 2013 paper says otherwise or does not report it.
* **open**: not settled.

| id | claim | evidence | confidence | status |
|---|---|---|---|---|
| R1 | SH edge b₂₊⁽⁰⁾ = 0.378489, q₊⁽⁰'⁰⁾ = −0.31560, third-gap coalescence at α = 0.7919, b = −1.2932 | Floquet discriminant; Mathieu value −a₀(2)/4 to 6 digits | high | reproduced |
| R2 | example powers 0.4992 (case 2), 0.4970 (case 3), 0.50097 / 0.50242 / 0.52329 (third gap) vs "P = 0.5" | Newton, spectral grid, converged in h | high | reproduced |
| R3 | case-2 and case-3 instability windows (Fig. 10) | spectra along the branches, absorber, filter ‖Re λ‖ < 3 | high (Δb ≈ 0.03–0.05) | reproduced |
| C1 | figures follow the printed stationary equations with **b → −b** | every band edge and gap location matches only with the flipped sign | high | correction |
| C2 | Eqs. (9) and (10) of the paper give q with opposite signs; the numbers follow (10); "1.29" should be "−1.29" | direct comparison with the computed edges | high, **but check against the original derivation** | correction |
| C3 | case 1, b = 0.25: P = 0.388 (the dot in the paper's figure is at b ≈ 0.27, where P = 0.444) | branch data | high | correction (caption rounding) |
| N1 | the fundamental third-gap soliton is centred at the potential **minimum** ξ₀ = π/2 | parity of the band-edge Bloch mode; centred at 0, shooting finds only multi-hump states (P ≈ 15.6) | high | new as a statement; **check whether the 2013 computations already did this without saying so** |
| N2 | the **α = 0 third-gap family is oscillatorily unstable** at every computed point, growth 0.006–0.087 (paper Fig. 5: stable) | absorber-independent at b = −1.197 (0.0282 / 0.0281); same rate at h = 0.25 and 0.20; L = 56π in `tables/convergence.tex`; propagation grows at 0.028 | high: 0.0282 at b = −1.197 on all grids and absorbers; near the edge (b = −1.282) the two absorber values close in from 0.0061 / 0.0103 (L = 40π) to 0.0062 / 0.0070 (L = 56π), so ≈ 0.006 | new |
| N5 | the gain/loss **weakens the oscillatory mode** of the α = 0 family (Re λ ≈ 1.2–1.4) without removing it: at b = −1.207 the rate is 0.026 (α = 0), 4.0e-3 (α = 0.7), 1.0e-3 (α = 0.7919); deeper in the gap a purely imaginary instability of rate 0.1–0.3 takes over (b ≳ −1.12 for α = 0.7, 0.7919) | branch spectra (`results/thirdgap.json`), seeded propagation at α = 0.7919, b = −1.207 | high | new |
| N3 | weak instabilities: case 3, b = 0.21: Im λ = 3.56e-3; third gap α = 0.7919, b = −1.207: 1.0e-3 | seeded propagation 3.39e-3 and 1.07e-3; converged in L | high | new (below the resolution of ζ = 1000 noisy runs) |
| N4 | third gap α = 0.7919, b = −1.101: purely imaginary λ = 0.201i as in the paper, but the soliton **grows** (P 2.09 → 750 by ζ = 100) instead of decaying | propagation with absorber, 10% noise | medium: check what "decays" meant in the paper (amplitude of what, which box, absorber or not) | difference |
| O1 | high-frequency growth, Re λ ≈ ±4.7–4.8, 0.02–0.06, inside the paper's stability windows; rate grows with L | L = 16π, 24π; propagation 0.055 in 24π | low: physical (convective) or finite-box? | open → C4 |
| O2 | near-edge α > 0 purely imaginary rates (3e-5 – 6e-4 at L = 40π) are **box artefacts**: 6.9e-5 → 1.2e-5 (α = 0.7, b = −1.282) and 7.7e-5 → 4.3e-6 (α = 0.7919, b = −1.262) from L = 40π to 56π. Near the edge the α > 0 families are stable within resolution, as in the paper | `tables/convergence.tex`; classification rule ‖Re λ‖ < 0.05 and Im λ < 1e-3 in make_figures / make_tables | high at the two tested points; α = 0.9, 0.95 near-edge points not tested | resolved (agreement) → C2 for α = 0.9, 0.95 |
| X1 | FF/SH currents have the opposite sign to the paper | consequence of the mirrored (ξ → −ξ) convention | high | convention |

## 4. Is it new? Literature checks still needed (deliberately no internet in the build)

The computations were done without literature access. Before claiming novelty, check:

1. **Papers citing PRA 87, 013832.** Search the "cited by" list for later work that revisits the stability of these
   families, especially third-gap and α = 0.
2. **Conservative χ⁽²⁾ gap solitons in periodic potentials (no PT).** N2 is an α = 0 statement, i.e. a purely
   Hamiltonian χ⁽²⁾ lattice problem. Oscillatory instabilities of gap solitons from internal modes colliding with
   the opposite-signature continuum are known in other models (Barashenkov–Pelinovsky–Zemlyanaya 1998;
   Pelinovsky–Sukhorukov–Kivshar 2004). The χ⁽²⁾ lattice case may already be reported. If it is, N2 stays a
   correction of Fig. 5 but is not a new mechanism.
3. **Centring of finite-gap solitons** (N1). This is standard for Kerr lattices: solitons in the first finite gap sit
   at potential minima or maxima depending on the Bloch-mode parity. Present it as a clarification, not a discovery.
4. **Weak instabilities** (N3). These are new only relative to the 2013 paper. They are genuinely small, so
   present them as a resolution issue.

Novelty is a judgment for the author. The realistic framing is a **revisit / erratum-type paper** of the
2013 work: corrections, plus a reproducible reference implementation, plus N2 as the main physics result.

## 5. Computations to finish before submission

| id | what | why | how (script) | cost |
|---|---|---|---|---|
| C1 | h / L / g₀ convergence at five third-gap points | supports N2, N3 and decides O2 | `run_convergence.py` → `make_tables.py` | done (29 min) |
| C2 | near-edge points at L = 56π for α = 0.9, 0.95; α = 0 edge rate at L = 80π | O2 tested at two points only; α = 0 edge rate ≈ 0.006 still drifting | `run_convergence.py` with more targets; `stability_of(..., method='arnoldi', shifts=[λ_guess])` makes large boxes cheap | 1 h |
| C3 | Krein signature of the colliding eigenvalues (α = 0) | supports the mechanism stated for N2 | eigenvectors at b = −1.29 … −1.20; sign of ⟨v, σ₃ v⟩ for the internal mode and the continuum edge; track the collision point in b | 1 h |
| C4 | high-frequency mode | settle O1: convective vs absolute | L = 16, 24, 32, 48π; long-box SSF with a moving window; spatial growth vs temporal growth | 2–4 h |
| C5 | derivative independence | rates must not depend on the d² discretization | repeat C1 with `Grid(..., deriv='fd4')` | 30 min |
| C6 | long seeded runs for N3 | push the agreement below 5% | `validate_weak.check(...)` with ζ = 3000, eps = 1e-6 | 1 h |
| C7 | scope: embedded solitons and the V₂ = 0 case of the 2013 paper | not reproduced here | either add (shooting with SH continuum) or state the scope in the Introduction | — |

Each new result must enter the paper through `results/*.json` → `make_tables.py` / `make_eps.py`, never typed by hand.

## 6. Authorship, AI use, and responsibility

* **Requested:** F. C. Moreira and Claude Opus 5.5. `main.tex` has this author list by default (`\aiauthortrue`).
* **arXiv policy (as known up to the model's training):** generative AI tools do not qualify for authorship. Their use
  should be disclosed, and the human authors take full responsibility for the content. APS journals have the same
  rule. A submission listing an AI as author is likely to be put on hold.
* **One-line fix:** `\aiauthorfalse`. This removes the second author and prints a "Use of AI" statement in the
  acknowledgments. The statement says the code, computations, figures and a first draft were produced by the AI under
  the direction of F. C. Moreira, who reviewed the results and is responsible for the content.
* **Also consider:** the 2013 co-authors (V. V. Konotop, B. A. Malomed). The paper corrects their joint work, and
  informing them or inviting them is usual practice.

## 7. Review checklist (things only the author can verify)

- [ ] Eqs. (1)–(4) and (6) of `main.tex` against the thesis and the 2013 derivation, including the b → −b and ζ → −ζ statements.
- [ ] Claim C2 (Eqs. 9/10 sign; 1.29 → −1.29) against the original derivation notes.
- [ ] N4: what "decays" meant for α = 0.7919, b = −1.101 in the 2013 simulations.
- [ ] Thesis entry in `refs.bib` (`MoreiraThesis`: title, university, year).
- [ ] Every `[memory]` entry in `refs.bib`: volume, page and year; add DOIs.
- [ ] Affiliation, funding, acknowledgments (red TODOs).
- [ ] Figure set: 10 full-width figures is long for PRA. Candidates to move to a supplement: Fig. 2 (shooting), Fig. 3 (validation), Figs. 7–8 (case 1 and 3 dynamics).
- [ ] Decide §6 (author list).
- [ ] Archive the code and data with a DOI (Zenodo) and put it in "Data and code availability".

## 8. arXiv submission

1. `make` (check the log for warnings), read `main.pdf`.
2. Set `\showtodosfalse` and choose `\aiauthorfalse` (§6). Run `make` again.
3. `make arxiv` → upload `arxiv.tar.gz` (latex + dvips route, EPS figures, `.bbl` included).
4. Categories: primary `physics.optics`, cross-list `nlin.PS`.
5. Abstract field: plain text (copy from `main.tex` without LaTeX commands).
6. Comments field: "N pages, 10 figures; code and data at <repository/DOI>".

## 9. Provenance of the numbers in the text

| quantity in `main.tex` | file / key |
|---|---|
| band edges, q₊⁽⁰'⁰⁾ | `results/bands.json`, `results/validation.json:bands` |
| semi-infinite powers, windows, λ | `results/semiinf.json` (cases, spectra, ssf), README §3 |
| high-frequency rates at L = 16π / 24π | logs of the box-size checks (README §4 item 5); **re-run into a JSON file (C4)** |
| third-gap families, growth ranges | `results/thirdgap.json` → `tables/thirdgap.tex` |
| α = 0 rate at b = −1.197, absorber / h / L independence | `results/convergence.json` → `tables/convergence.tex` |
| SSF growth 0.028 from a 1e-4 perturbation (α = 0, b = −1.197) | earlier check (README §4 item 3); **re-run into a JSON file** |
| weak rates and seeded propagation | `results/weak.json` → `tables/validation.tex` |
| convergence | `results/convergence.json` → `tables/convergence.tex` |
