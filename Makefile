.PHONY: xmls run stop plots plots_hpc summary

xmls:
	rm -f data/xmls/*
	bash scripts/generate_xmls.sh

run:
	bash scripts/run_beast_seeds.sh

stop:
	scancel --me

plots:
	cd python && uv run src/benchmarks/convergence_time.py

plots_hpc:
	sbatch scripts/run_plots.sh

summary:
	cd python && uv run -m src.data_pipeline.summary
