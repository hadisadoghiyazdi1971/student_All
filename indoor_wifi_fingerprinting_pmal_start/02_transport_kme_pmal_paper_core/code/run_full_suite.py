from __future__ import annotations
"""Run the complete journal-style distributional active learning suite.

This wrapper does not change the dataset protocol; it orchestrates the separate
scripts used in the manuscript so that a final paper can reproduce GAUSSDIST,
USPSDIST-style digit distributions, and optionally 20 Newsgroups distributional
experiments with the same budget/repeat settings.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--budget", type=int, default=50)
    p.add_argument("--outdir", type=str, default="../results_full")
    p.add_argument("--digit-pairs", type=str, default="3,8;4,9;1,7;5,6;0,1")
    p.add_argument("--include-20news", action="store_true")
    p.add_argument("--include-hard-gauss", action="store_true")
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    gauss_cmd = [sys.executable, "run_gaussdist_suite.py", "--repeats", str(args.repeats),
                 "--budget", str(args.budget), "--outdir", str(outdir)]
    if args.include_hard_gauss:
        gauss_cmd.append("--include-hard")
    run(gauss_cmd, here)
    run([sys.executable, "run_digitsdist_suite.py", "--repeats", str(args.repeats),
         "--budget", str(args.budget), "--outdir", str(outdir),
         "--pairs", args.digit_pairs], here)
    if args.include_20news:
        run([sys.executable, "run_20newsgroups_distributional.py", "--repeats", str(args.repeats),
             "--budget", str(args.budget), "--outdir", str(outdir)], here)
    run([sys.executable, "aggregate_and_plot.py", "--results", str(outdir)], here)
    run([sys.executable, "make_latex_tables.py", "--results", str(outdir),
         "--out", str(outdir / "tables_for_paper.tex")], here)


if __name__ == "__main__":
    main()
