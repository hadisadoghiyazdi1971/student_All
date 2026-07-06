from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from tc_kme_common import repeated_experiment
from tc_kme_datasets import make_gaussdist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--budget", type=int, default=40)
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--include-hard", action="store_true")
    args = p.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    repeated_experiment(lambda s: make_gaussdist(s, variable_bag=True), outdir, "gaussdist_variable", args.repeats, args.budget, 10, gamma=0.1, n_features=256)
    repeated_experiment(lambda s: make_gaussdist(s, variable_bag=False), outdir, "gaussdist_equal", args.repeats, args.budget, 100, gamma=0.1, n_features=256)
    names = ["gaussdist_variable", "gaussdist_equal"]
    if args.include_hard:
        repeated_experiment(
            lambda s: make_gaussdist(s, variable_bag=True, mean_gap=0.45, covariance_multiplier=1.35),
            outdir, "gaussdist_hard_variable", args.repeats, args.budget, 210,
            gamma=0.1, n_features=256,
        )
        repeated_experiment(
            lambda s: make_gaussdist(s, variable_bag=False, mean_gap=0.45, covariance_multiplier=1.35),
            outdir, "gaussdist_hard_equal", args.repeats, args.budget, 310,
            gamma=0.1, n_features=256,
        )
        names += ["gaussdist_hard_variable", "gaussdist_hard_equal"]

    for name in names:
        df = pd.read_csv(outdir / f"{name}_curves.csv")
        plt.figure(figsize=(6.0, 4.0))
        for method, sub in df.groupby("method"):
            plt.plot(sub["budget"], sub["mean_accuracy"], label=method)
        plt.xlabel("Number of labeled distributions")
        plt.ylabel("Test accuracy")
        plt.title(name.replace("_", " "))
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(outdir / f"fig_{name}.pdf")
        plt.savefig(outdir / f"fig_{name}.png", dpi=200)

if __name__ == "__main__":
    main()
