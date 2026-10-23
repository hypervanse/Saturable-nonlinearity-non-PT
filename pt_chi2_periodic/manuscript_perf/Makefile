# Makefile -- build the manuscript where TeX Live is installed (it was not in the build container).
#
#   make          latex -> bibtex -> latex x2 -> dvips -> ps2pdf   (EPS figures; the route arXiv uses)
#   make pdf      pdflatex route (uses the PDF copies of the same figures)
#   make check    static checks without TeX (braces, environments, refs, cites, figure files)
#   make arxiv    arxiv.tar.gz: main.tex + main.bbl + tables + EPS only (arXiv does not run BibTeX,
#                 and leaving out the PDF figures keeps arXiv on the latex+dvips route)
#   make figures  regenerate numbers.tex, tables and EPS/PDF figures from ../results/*.json
#   make clean

TEX  = main
FIGS = $(wildcard figures/*.eps)
TABS = $(wildcard tables/*.tex)

all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex refs.bib $(FIGS) $(TABS)
	latex  -interaction=nonstopmode $(TEX)      # 1st pass: writes .aux with \cite and \label
	bibtex $(TEX)                               # .aux + refs.bib -> .bbl
	latex  -interaction=nonstopmode $(TEX)      # 2nd pass: reads .bbl
	latex  -interaction=nonstopmode $(TEX)      # 3rd pass: resolves the cross-references
	dvips  -Ppdf -o $(TEX).ps $(TEX).dvi        # -Ppdf: Type 1 fonts, good for the PDF step
	ps2pdf $(TEX).ps $(TEX).pdf

pdf:
	pdflatex -interaction=nonstopmode $(TEX)
	bibtex   $(TEX)
	pdflatex -interaction=nonstopmode $(TEX)
	pdflatex -interaction=nonstopmode $(TEX)

check:
	python3 check_tex.py $(TEX).tex

$(TEX).bbl: $(TEX).tex refs.bib
	latex  -interaction=nonstopmode $(TEX)
	bibtex $(TEX)

arxiv: $(TEX).bbl
	tar czvf arxiv.tar.gz $(TEX).tex $(TEX).bbl numbers.tex tables/*.tex figures/*.eps

figures:
	cd .. && python3 make_perf.py

clean:
	rm -f $(TEX).aux $(TEX).log $(TEX).out $(TEX).dvi $(TEX).ps $(TEX).bbl $(TEX).blg $(TEX)Notes.bib arxiv.tar.gz

.PHONY: all pdf check arxiv figures clean
