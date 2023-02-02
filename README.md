# Measuring code complexity via ease of verification
This repo contains code, data, results, etc. of our project.

The key parts are:
* `correlation.py`: used to compute the individual correlations (i.e., RQ1 answers)
* the `data/` folder, which contains the raw output of the verification tools,
* the script `meta-analysis/metafor-based-meta-analysis.r`, which was used to compute the RQ2-4 results,
* the `forest-plot/` folder, which contains forest plots for all meta-analyses (including those mentioned in but
  elided from the paper)
* the `tool-execution` folder, which contains the scripts we used to run the verifiers
* the various gradle subprojects (`simple-datasets`, `dataset6`, `dataset9`), which contain the raw code snippets

More details about how to run the verifiers/reproduce the results are given in the document `how-to-run-the-verifiers.pdf`.

TODO: clean up this repo and remove everything that wasn't actually used in the paper
