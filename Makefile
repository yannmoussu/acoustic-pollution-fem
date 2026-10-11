PYTHON ?= .venv/bin/python
.PHONY: all calculations check documents archive

all:
	$(MAKE) calculations
	$(MAKE) documents
	$(MAKE) archive

calculations:
	$(PYTHON) mesh_implementation.py
	$(PYTHON) fem_implementation.py
	$(PYTHON) fem_implementation_shifted.py
	$(PYTHON) compare_choices.py

check:
	$(PYTHON) mesh_implementation.py --check-only
	$(PYTHON) fem_implementation.py --check-only
	$(PYTHON) fem_implementation_shifted.py --check-only

documents:
	cd report && latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex
	cd presentation && latexmk -pdf -interaction=nonstopmode -halt-on-error presentation.tex

archive:
	$(PYTHON) -c 'from pathlib import Path; from zipfile import ZipFile, ZIP_DEFLATED; roots=[Path(p) for p in ("compare_choices.py", "mesh_implementation.py", "fem_implementation.py", "fem_implementation_shifted.py", "requirements.txt", "README.md", "Makefile", "report/report.tex", "report/report.pdf", "report/figures", "report/data", "presentation/presentation.tex", "presentation/presentation.pdf")]; assert all(p.exists() for p in roots), "Livrable manquant"; files=sorted({q for p in roots for q in (p.rglob("*") if p.is_dir() else [p]) if q.is_file()}); archive=ZipFile("Yann MOUSSU.zip", "w", ZIP_DEFLATED); [archive.write(p, Path("Yann MOUSSU")/p) for p in files]; archive.close(); print("Archive : Yann MOUSSU.zip (",len(files),"fichiers)")'
