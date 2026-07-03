import argparse
from pathlib import Path
import pandas as pd

from data_utils import (
    load_movielens_100k,
    load_movielens_1m,
    load_csv_ratings,
    make_synthetic_ratings,
    split_ratings,
    user_level_cold_start_split,
)
from experiments import (
    run_baselines_and_blf,
    run_parameter_sensitivity,
    run_fcm_diagnostic,
    save_results,
)


def parse_args():
    p = argparse.ArgumentParser(description="Bayesian Latent Filtering recommender experiments")

    p.add_argument("--dataset", choices=["synthetic", "ml100k", "ml1m", "csv"], default="synthetic")
    p.add_argument("--data-dir", default=None, help="Directory for MovieLens dataset")
    p.add_argument("--ratings-csv", default=None, help="CSV file with columns user_id,item_id,rating")

    p.add_argument("--out", default="results_blf")
    p.add_argument("--components", type=int, default=80)
    p.add_argument("--latent-dim", type=int, default=40)
    p.add_argument("--beta", type=float, default=15.0, help="Smoothing for set-item means")
    p.add_argument("--reg", type=float, default=10.0, help="Bias regularization")
    p.add_argument("--blend-weight", type=float, default=0.75, help="Weight assigned to the BLF mixture prediction")
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--topn", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--split", choices=["random", "cold_start"], default="random")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--cold-start-train-interactions", type=int, default=1)
    p.add_argument("--cold-start-user-frac", type=float, default=0.2)

    return p.parse_args()


def load_dataset(args):
    if args.dataset == "synthetic":
        return make_synthetic_ratings(seed=args.seed)
    if args.dataset == "ml100k":
        if args.data_dir is None:
            raise ValueError("--data-dir is required for ml100k")
        return load_movielens_100k(args.data_dir)
    if args.dataset == "ml1m":
        if args.data_dir is None:
            raise ValueError("--data-dir is required for ml1m")
        return load_movielens_1m(args.data_dir)
    if args.dataset == "csv":
        if args.ratings_csv is None:
            raise ValueError("--ratings-csv is required for csv dataset")
        return load_csv_ratings(args.ratings_csv)
    raise ValueError(args.dataset)


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[load dataset]")
    df, users, items = load_dataset(args)
    n_users = int(df["user"].max() + 1)
    n_items = int(df["item"].max() + 1)

    print(f"ratings={len(df):,}, users={n_users:,}, items={n_items:,}")

    if args.split == "cold_start":
        train_df, test_df = user_level_cold_start_split(
            df,
            min_train_interactions=args.cold_start_train_interactions,
            test_user_frac=args.cold_start_user_frac,
            seed=args.seed,
        )
    else:
        train_df, test_df = split_ratings(df, test_size=args.test_size, seed=args.seed)

    print(f"train={len(train_df):,}, test={len(test_df):,}")

    print("[main comparison]")
    metrics_df, blf_model = run_baselines_and_blf(train_df, test_df, n_users, n_items, args)
    print(metrics_df)

    print("[parameter sensitivity]")
    sens_df = run_parameter_sensitivity(train_df, test_df, n_users, n_items, args)
    print(sens_df)

    print("[FCM diagnostic]")
    fcm_hist = run_fcm_diagnostic(train_df, n_users, n_items, args)

    save_results(metrics_df, sens_df, fcm_hist, out_dir)

    print(f"\nDone. Results saved to: {out_dir.resolve()}")
    print("Key files: metrics.csv, parameter_sensitivity.csv, convergence.csv, convergence.png")


if __name__ == "__main__":
    main()
