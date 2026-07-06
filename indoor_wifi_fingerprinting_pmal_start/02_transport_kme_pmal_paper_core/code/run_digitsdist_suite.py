from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from tc_kme_common import repeated_experiment
from tc_kme_datasets import make_digitsdist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--budget", type=int, default=35)
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--pairs", type=str, default="3,8;4,9")
    args = p.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    for pair_txt in args.pairs.split(";"):
        a, b = [int(x) for x in pair_txt.split(",")]
        name = f"digitdist_{a}v{b}"
        repeated_experiment(lambda s, a=a, b=b: make_digitsdist(s, pair=(a, b)), outdir, name,
                            args.repeats, args.budget, 700 + a * 10 + b, gamma=0.05, n_features=256)
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
