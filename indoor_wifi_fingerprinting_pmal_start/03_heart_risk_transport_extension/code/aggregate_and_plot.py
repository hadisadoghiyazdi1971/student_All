from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    args = p.parse_args()
    results = Path(args.results)
    frames = []
    for f in results.glob("*_summary.csv"):
        frames.append(pd.read_csv(f))
    if not frames:
        raise SystemExit("No *_summary.csv files found.")
    summary = pd.concat(frames, ignore_index=True)
    summary.to_csv(results / "all_summary.csv", index=False)
    pivot = summary.pivot_table(index="dataset", columns="method", values="AULC")
    ranks = pivot.rank(axis=1, ascending=False)
    ranks.mean(axis=0).sort_values().to_csv(results / "average_rank.csv", header=["mean_rank"])
    plt.figure(figsize=(6, 4))
    ranks.mean(axis=0).sort_values().plot(kind="bar")
    plt.ylabel("Mean rank (lower is better)")
    plt.tight_layout()
    plt.savefig(results / "fig_average_rank.pdf")
    plt.savefig(results / "fig_average_rank.png", dpi=200)

if __name__ == "__main__":
    main()
