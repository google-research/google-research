all: jpegli.pdf

#
# To build:
#
# env JHEP_REPORTS_CSL=[path to jhep-reports.csl] PANDOC_EQNOS_BINARY=[path to pandoc-eqnos] PANDOC_TEMPLATE=[path to default.latex] make
#
# jhep-reports.csl can be downloaded from
# https://paperpile.com/s/jhep-reports-citation-style/
#
# pandoc-eqnos can be downloaded from https://github.com/tomduck/pandoc-eqnos
#
# default.latex is found at /usr/share/pandoc/data/templates/default.latex in a
# typical Debian installation.
#

jpegli.pdf: jpegli.md meta.yaml bibliography.bib
	pandoc -f markdown -t pdf --citeproc --bibliography bibliography.bib --csl ${JHEP_REPORTS_CSL} --filter ${PANDOC_EQNOS_BINARY} --metadata-file meta.yaml --template ${PANDOC_TEMPLATE} jpegli.md > jpegli.pdf

jpegli.latex: jpegli.md meta.yaml bibliography.bib
	pandoc -f markdown -t latex --citeproc --bibliography bibliography.bib --csl ${JHEP_REPORTS_CSL} --filter ${PANDOC_EQNOS_BINARY} --metadata-file meta.yaml --template ${PANDOC_TEMPLATE} jpegli.md > jpegli.latex

show: jpegli.pdf
	xdg-open jpegli.pdf
