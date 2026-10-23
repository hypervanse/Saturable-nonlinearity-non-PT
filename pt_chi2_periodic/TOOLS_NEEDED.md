# Tools to provide in the next session

The cloud session could not install anything: pip, apt and CTAN were blocked by the environment's
network policy. Everything below was missing. It is listed in order of importance, with the exact
target, so that what you upload or allow matches the machine.

## The session machine

| item | value |
|---|---|
| OS | Ubuntu 24.04.5 LTS, Linux 6.18 (Firecracker VM) |
| architecture | x86_64, AVX-512, 4 vCPUs, 15 GB RAM |
| C library | glibc 2.39 |
| Python | 3.13.16 at `/usr/bin/python3` (pip 24.0; PEP 668: needs `--break-system-packages` or a venv) |
| present | NumPy 2.5.3, gcc/g++, make, git, zip, poppler 24.02 (`pdftocairo`, `pdftops`), headless Chromium, ImageMagick |
| absent | TeX, ghostscript, SciPy, Numba, gfortran |

## What to provide

### 1. TeX Live, to compile both papers (required)

Needed binaries: `latex`, `pdflatex`, `bibtex`, `dvips`, `ps2pdf` (ghostscript). Needed LaTeX
packages: `revtex4-2`, `natbib`, `amsmath`/`amssymb`, `bm`, `graphicx`, `xcolor`, `hyperref`.

* **Preferred: let the environment install it.** Add this line to the environment's setup script,
  under the cloud environment menu in the session title bar → Edit:
  ```bash
  apt-get update && apt-get install -y --no-install-recommends texlive-latex-base texlive-latex-recommended texlive-latex-extra texlive-publishers texlive-fonts-recommended ghostscript
  ```
  This needs the network policy to allow the Ubuntu archive (`archive.ubuntu.com`, `security.ubuntu.com`).
* **Offline alternative:** upload a TinyTeX tree for **x86_64-linux** with the packages above
  already installed (`tlmgr install revtex natbib bm xcolor hyperref`), as a `.tar.gz`. It will be
  unpacked into the home directory and put on `PATH`. Ghostscript is still needed for `ps2pdf`; with
  TinyTeX only, use `make pdf` (pdflatex with the PDF figures) instead of the dvips route.

### 2. SciPy, for the sparse Arnoldi path (recommended)

This is what turns the O(n³) stability spectra into ~O(n) sparse solves (paper 2, §V-F). The code
switches to it automatically.

* Setup script: `python3 -m pip install --break-system-packages scipy`. This needs `pypi.org` and
  `files.pythonhosted.org` to be allowed.
* Offline: upload a wheel built for **cp313, manylinux, x86_64**, and a version that supports
  NumPy 2.5, e.g. `scipy-<version>-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`.
  It installs with `python3 -m pip install --break-system-packages --no-index ./scipy-*.whl`.

### 3. Numba (optional)

For the compiled shooting integrator and the fused split-step update (paper 2, recommendation 5).
It needs **cp313 x86_64** wheels of `numba` and the matching `llvmlite`, both compatible with
NumPy 2.5.

### 4. Network access for literature checks (optional, your call)

You disabled it on purpose. The bibliography entries tagged `[memory]` and the novelty checks in
`manuscript/PLAN.md` §4 are what it would be used for. Without it, they stay on your review list.

## After the tools are in place, the session will

1. `cd manuscript && make` and `cd manuscript_perf && make`, fix any LaTeX errors, and read both PDFs.
2. With SciPy: run `validate_arnoldi.py` (sparse self-check + timings), then add the measured
   sparse row to paper 2.
3. Run the open computations of `manuscript/PLAN.md` §5 (C2 near-edge points at L = 56π, C3 Krein
   signature, C4 high-frequency mode), using the Arnoldi path for the large boxes.
