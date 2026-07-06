# Transport-Calibrated KME-PMAL experiment suite

This folder turns the paper's theory into the thesis-style benchmark workflow.

## Quick smoke test

```bash
pip install numpy scipy scikit-learn pandas matplotlib
python run_gaussdist_suite.py --repeats 5 --budget 35 --outdir ../results
python run_digitsdist_suite.py --repeats 3 --budget 25 --outdir ../results --pairs "3,8"
python aggregate_and_plot.py --results ../results
```

## Full journal run

Use at least 50 repeats, the same label budget used in the thesis, and all dataset pairs.

```bash
python run_gaussdist_suite.py --repeats 50 --budget 50 --outdir ../results
python run_digitsdist_suite.py --repeats 50 --budget 50 --outdir ../results --pairs "3,8;4,9;1,7;5,6"
python run_20newsgroups_distributional.py --repeats 50 --budget 50 --outdir ../results
python aggregate_and_plot.py --results ../results
```

The code intentionally separates dataset generation, embedding, active selection,
and aggregation so that PMARLDB/PMARLDS implementations from the original thesis
can be inserted as additional methods without changing the rest of the pipeline.

## One-command journal suite

The revised manuscript also includes two utility scripts.

```bash
python run_full_suite.py --repeats 50 --budget 50 --outdir ../results_full --include-20news
```

This runs the GAUSSDIST, digit-distribution, optional 20 Newsgroups protocol,
aggregates the outputs, and writes a LaTeX table through:

```bash
python make_latex_tables.py --results ../results_full --out ../results_full/tables_for_paper.tex
```

Use the generated `tables_for_paper.tex` as a drop-in table after the full run.
