
all:
	pdflatex paper.tex 
#	open paper.pdf  # Mac specific; no use for Linux machines

clean:
	latexmk -C paper
	rm -f *.fdb_latexmk
	rm -f *.log
	rm -f *.aux
	rm -f *.dvi
	rm -f *.bbl
