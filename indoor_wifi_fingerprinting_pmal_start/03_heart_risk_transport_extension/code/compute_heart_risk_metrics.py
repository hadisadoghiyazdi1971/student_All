from __future__ import annotations
"""Compute Heart-of-Risk summaries from active-learning accuracy curves.

The script reads *_curves.csv files produced by the executable benchmark suite and
writes heart_risk_metrics.csv. Accuracy is converted to loss e_t = 1-accuracy.
The main additional quantity is empirical CVaR_alpha over the worst alpha
fraction of active-learning rounds. This connects the benchmark section to the
Heart-of-Risk view: we report not only average performance but tail behavior over
attainable post-query losses along the learning trajectory.
"""

import argparse
from pathlib import Path
import math
import pandas as pd


def empirical_cvar(losses, alpha: float) -> float:
    vals = sorted([float(x) for x in losses], reverse=True)
    k = max(1, int(math.ceil(alpha * len(vals))))
    return sum(vals[:k]) / k


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="../results")
    p.add_argument("--alpha", type=float, default=0.20)
    p.add_argument("--out", default="../results/heart_risk_metrics.csv")
    args = p.parse_args()
    results = Path(args.results)
    rows = []
    for f in sorted(results.glob("*_curves.csv")):
        df = pd.read_csv(f)
        if not {"dataset", "method", "mean_accuracy"}.issubset(df.columns):
            continue
        for (dataset, method), g in df.groupby(["dataset", "method"]):
            losses = 1.0 - g.sort_values("budget")["mean_accuracy"].astype(float)
            rows.append({
                "dataset": dataset,
                "method": method,
                "mean_future_loss": float(losses.mean()),
                f"CVaR_{args.alpha:.2f}": empirical_cvar(losses, args.alpha),
                "worst_round_loss": float(losses.max()),
                "best_round_loss": float(losses.min()),
                "rounds": int(len(losses)),
            })
    if not rows:
        raise SystemExit(f"No usable *_curves.csv files found in {results}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["dataset", "method"]).to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
