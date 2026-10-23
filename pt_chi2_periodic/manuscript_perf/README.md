# Paper 2: where the time goes

Draft of a computational-methods paper on the solver behind `../manuscript/`: which resource bounds
each stage, how much of the 4-core machine the production runs used, and which changes would pay off
most. Same authorship switch and caveats as paper 1 (see `../manuscript/PLAN.md`, §6). **Not compiled
here** (no TeX in the build container); `make check` passes once `make_perf.py` has run.

## What is in it

| item | content | made by |
|---|---|---|
| `main.tex` | text; every measured number is a macro (`\tEigOne`, `\pAmEig`, …) | written by hand |
| `numbers.tex` | the macros | `make_perf.py` from `results/perf.json` |
| `tables/kernels.tex` | dense LAPACK kernels vs n (1 and 4 threads), fitted exponents, LU GFLOP/s | `make_perf.py` |
| `tables/anatomy.tex` | wall time per stage of one convergence row | `make_perf.py` |
| `tables/throughput.tex` | 4 eigensolves: 1×4 threads vs 2×2 vs 4 processes × 1 | `make_perf.py` |
| `tables/projection.tex` | cost-model projections of the convergence study | `make_perf.py` (list scheduler) |
| `tables/arnoldi.tex` | dense vs shift-invert Arnoldi on 6 solitons | `make_perf.py` from `results/arnoldi.json` |
| `figures/fig_pipeline` | pipeline, coloured by bounding resource | hand-drawn SVG in `make_perf.py` |
| `figures/fig_dag` | task graph of the convergence study | hand-drawn SVG in `make_perf.py` |
| `figures/fig_spy` | nonzero pattern of the stability operator, Fourier vs fd4 | `make_perf.py` |
| `figures/fig_dense`, `fig_anatomy`, `fig_interp` | measurements | `make_perf.py` |

## Reproduce

```bash
python3 bench_perf.py      # ~10 min on an IDLE machine -> results/perf.json (wall-clock times)
python3 make_perf.py       # numbers.tex, tables/, figures/ (EPS + PDF via Chromium + pdftocairo)
cd manuscript_perf && make check && make
```

Timings are machine-specific. On another machine, rerun both scripts and the text updates itself.
Statements that depend on the *ordering* of numbers ("processes beat threads", "the model
reproduces the run") were checked against this machine's `perf.json`; re-read them after a rerun.

## To review

- [ ] Claims about LAPACK internals (Hessenberg reduction, QR sweeps) are textbook-level; check the wording.
- [ ] The cost model's assumptions (§6): N³ scaling of Newton steps, the calibration factor for zgeev on real matrices.
- [ ] The sparse/SciPy path is implemented but **not measured** (no SciPy here): run `validate_arnoldi.py` with SciPy
      installed and add a measured row before claiming a number for it.
- [ ] Target venue: Computer Physics Communications or physics.comp-ph on arXiv; or merge a short version into
      paper 1 as an appendix.
