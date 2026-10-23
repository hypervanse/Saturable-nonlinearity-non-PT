# Session snapshot: run it on your Linux machine

This folder is the complete working directory of the cloud session. It holds the code, results,
logs, figures, report, both manuscripts and the source PDF, and it is a local git repository.
The zip stores it under its absolute path, `/home/user/pt_chi2_solitons`.

## Unpack

```bash
# A: same absolute path as in the session (needs write access to /home/user, hence sudo)
sudo unzip pt_chi2_session.zip -d /                 # -> /home/user/pt_chi2_solitons
sudo chown -R "$USER": /home/user/pt_chi2_solitons   # make it yours

# B: anywhere else -- the code only uses paths relative to this folder
unzip pt_chi2_session.zip -d ~/work                  # -> ~/work/home/user/pt_chi2_solitons
```

Every script is run **from inside the folder** (`cd /home/user/pt_chi2_solitons`), because they read
and write `results/`, `figures/`, `manuscript*/` relative to it.

## Requirements (what the session had)

| need | session version | used by |
|---|---|---|
| Python | 3.13.16 | everything |
| NumPy | 2.5.3 (OpenBLAS) | everything, the only Python dependency |
| Chromium (headless) | Playwright build 1194 | PNG / EPS / PDF figure export only. Set `CHROME=/path/to/chrome` if it is not found |
| poppler `pdftocairo` | 24.02.0 | EPS/PDF figures |
| `zip`, `git` | any | `pack.sh`, history |
| TeX Live (latex, bibtex, dvips, ps2pdf, revtex4-2) | **not available in the session** | building the two papers |
| SciPy | **not available in the session** | optional sparse Arnoldi path |

```bash
python3 -m pip install numpy            # or: python3 -m venv .venv && . .venv/bin/activate && pip install numpy
sudo apt install poppler-utils chromium texlive-latex-extra texlive-publishers ghostscript   # Debian/Ubuntu names
```

## Run (from the folder)

```bash
python3 validate.py                 # checks, ~5 min
python3 run_bands.py && python3 run_semiinf.py && python3 run_thirdgap.py     # physics, ~1.5 h on 4 cores
python3 run_convergence.py          # third-gap convergence study, ~30 min
python3 validate_weak.py && python3 validate_arnoldi.py
python3 make_figures.py && python3 build_report.py                            # figures + report.html
python3 make_tables.py && python3 make_eps.py                                 # paper 1 tables + EPS
python3 bench_perf.py && python3 make_perf.py                                 # paper 2 (benchmark on an IDLE machine)
(cd manuscript && make check && make); (cd manuscript_perf && make check && make)
bash pack.sh                        # the distributable bundle dist/pt_chi2_periodic.zip
```

`results/` already holds the outputs of every long run, so the figure, table, report and paper
steps work without rerunning anything. Timings in paper 2 are machine-specific: rerun
`bench_perf.py` + `make_perf.py` on your machine and the text updates itself.

## What is where

| path | content |
|---|---|
| `*.py` | the solver, drivers, validation, figures, benchmarks (see `README.md`, `REPRODUCE.md`) |
| `results/` | every computed number (JSON) and stored fields (NPZ) |
| `logs/` | console logs of the long runs (the convergence log feeds paper 2) |
| `figures/`, `report.html` | report figures (SVG + PNG) and the self-contained report page |
| `manuscript/` | paper 1 (physics): `main.tex`, `refs.bib`, EPS/PDF figures, tables, **`PLAN.md`** |
| `manuscript_perf/` | paper 2 (performance): `main.tex`, `numbers.tex`, tables, figures, `README.md` |
| `sources/` | the PDF of PRA 87, 013832 supplied to the session (git-ignored: copyrighted) |
| `dist/` | output of `pack.sh` (git-ignored, regenerated) |
| `TOOLS_NEEDED.md` | what to provide in the next session, matched to the session machine |
| `.git/` | local history (`git log`); the GitHub copy is the PR branch `ccr-6948f601-0z0ghj` |
