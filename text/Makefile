TEX := pdflatex
BIBER := biber

sources := $(wildcard *.tex)

base_name := $(basename $(sources))


target := $(patsubst %.tex, %.pdf, $(sources))

.PHONY: all clean

all: $(target) Makefile


%.pdf: %.tex
	$(TEX) $^
	$(BIBER) $(base_name)
	$(TEX) $^

clean:
	$(RM) *.toc *.dvi *.log *.aux *.bbl *.bcf *.blg *.xml *blx* *.out
