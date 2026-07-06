from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


TARGET_METHOD = "TC-BKME-PMAL"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    args = p.parse_args()
    results = Path(args.results)
    frames = []
    for f in results.glob("*_summary.csv"):
        if f.name == "all_summary.csv":
            continue
        frames.append(pd.read_csv(f))
    if not frames:
        raise SystemExit("No *_summary.csv files found.")
    summary = pd.concat(frames, ignore_index=True)
    summary.to_csv(results / "all_summary.csv", index=False)
    pivot = summary.pivot_table(index="dataset", columns="method", values="AULC")
    ranks = pivot.rank(axis=1, ascending=False)
    ranks.mean(axis=0).sort_values().to_csv(results / "average_rank.csv", header=["mean_rank"])

    repeat_frames = []
    for f in results.glob("*_repeat_metrics.csv"):
        repeat_frames.append(pd.read_csv(f))
    if repeat_frames:
        repeats = pd.concat(repeat_frames, ignore_index=True)
        stats_rows = []
        for dataset, sub in repeats.groupby("dataset"):
            base = sub[sub["method"] == TARGET_METHOD].sort_values("repeat")
            if base.empty:
                continue
            for method, other in sub.groupby("method"):
                if method == TARGET_METHOD:
                    continue
                other = other.sort_values("repeat")
                merged = base[["repeat", "AULC", "final_accuracy"]].merge(
                    other[["repeat", "AULC", "final_accuracy"]],
                    on="repeat",
                    suffixes=("_tc", "_other"),
                )
                if merged.empty:
                    continue
                diff = merged["AULC_tc"] - merged["AULC_other"]
                if (diff == 0).all():
                    p_value = 1.0
                else:
                    p_value = float(wilcoxon(diff).pvalue)
                stats_rows.append({
                    "dataset": dataset,
                    "comparison": f"{TARGET_METHOD} vs {method}",
                    "mean_aulc_delta": float(diff.mean()),
                    "tc_wins_aulc": int((diff > 0).sum()),
                    "other_wins_aulc": int((diff < 0).sum()),
                    "ties_aulc": int((diff == 0).sum()),
                    "wilcoxon_p_value": p_value,
                    "mean_final_delta": float((merged["final_accuracy_tc"] - merged["final_accuracy_other"]).mean()),
                })
        pd.DataFrame(stats_rows).to_csv(results / "pairwise_stats.csv", index=False)

    plt.figure(figsize=(6, 4))
    ranks.mean(axis=0).sort_values().plot(kind="bar")
    plt.ylabel("Mean rank (lower is better)")
    plt.tight_layout()
    plt.savefig(results / "fig_average_rank.pdf")
    plt.savefig(results / "fig_average_rank.png", dpi=200)

if __name__ == "__main__":
    main()
