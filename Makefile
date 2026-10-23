.PHONY: xmls run stop plots

xmls:
	rm -f data/xmls/*
	bash scripts/generate_xmls.sh

run:
	bash scripts/run_beast_seeds.sh

stop:
	scancel --me

plots:
	cd python && uv run src/benchmarks/convergence_time.py
