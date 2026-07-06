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
    args = p.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    repeated_experiment(lambda s: make_gaussdist(s, variable_bag=True), outdir, "gaussdist_variable", args.repeats, args.budget, 10, gamma=0.1, n_features=256)
    repeated_experiment(lambda s: make_gaussdist(s, variable_bag=False), outdir, "gaussdist_equal", args.repeats, args.budget, 100, gamma=0.1, n_features=256)

    for name in ["gaussdist_variable", "gaussdist_equal"]:
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
