from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

from metrics import mae, rmse, precision_recall_f1_at_k
from baselines import ItemMeanBaseline, BiasBaseline, SVDItemMeanBaseline
from blf_recommender import BayesianLatentFilteringRecommender
from fuzzy_cmeans import FuzzyCMeans


def evaluate_model(model, train_df, test_df, n_users, n_items, topn_k=10):
    pred = model.predict_pairs(test_df)
    metrics = {
        "MAE": mae(test_df["rating"].values, pred),
        "RMSE": rmse(test_df["rating"].values, pred),
    }
    users = sorted(test_df["user"].unique().tolist())
    topn = model.recommend_topn(train_df, users, n=topn_k) if hasattr(model, "recommend_topn") else {}
    prf = precision_recall_f1_at_k(test_df, topn, relevance_threshold=4.0)
    metrics.update({
        "Precision": prf["precision"],
        "Recall": prf["recall"],
        "F1": prf["f1"],
    })
    return metrics


def run_baselines_and_blf(train_df, test_df, n_users, n_items, args):
    rows = []

    models = [
        ("ItemMean", ItemMeanBaseline()),
        ("Bias", BiasBaseline(reg=args.reg)),
        ("SVD-lite", SVDItemMeanBaseline(latent_dim=args.latent_dim, random_state=args.seed)),
        ("BayesianLatentFiltering", BayesianLatentFilteringRecommender(
            n_components=args.components,
            latent_dim=args.latent_dim,
            beta=args.beta,
            reg=args.reg,
            blend_weight=args.blend_weight,
            random_state=args.seed,
            max_iter=args.max_iter,
        )),
    ]

    fitted_blf = None
    for name, model in models:
        print(f"[fit] {name}")
        model.fit(train_df, n_users, n_items)
        if name == "BayesianLatentFiltering":
            fitted_blf = model

        print(f"[eval] {name}")
        m = evaluate_model(model, train_df, test_df, n_users, n_items, topn_k=args.topn)
        m["Method"] = name
        rows.append(m)

    return pd.DataFrame(rows), fitted_blf


def run_parameter_sensitivity(train_df, test_df, n_users, n_items, args, components_grid=None):
    if components_grid is None:
        base = args.components
        components_grid = sorted(set([
            max(5, int(base * 0.5)),
            max(5, int(base * 0.75)),
            base,
            int(base * 1.25),
        ]))

    rows = []
    for c in components_grid:
        print(f"[sensitivity] components={c}")
        model = BayesianLatentFilteringRecommender(
            n_components=c,
            latent_dim=args.latent_dim,
            beta=args.beta,
            reg=args.reg,
            blend_weight=args.blend_weight,
            random_state=args.seed,
            max_iter=args.max_iter,
        )
        model.fit(train_df, n_users, n_items)
        m = evaluate_model(model, train_df, test_df, n_users, n_items, topn_k=args.topn)
        m["components"] = c
        rows.append(m)
    return pd.DataFrame(rows)


def run_fcm_diagnostic(train_df, n_users, n_items, args):
    """
    Diagnostic only: compares standard FCM objective against BLF latent objective style.
    This should not be the main claim in the paper.
    """
    rows = train_df["user"].values
    cols = train_df["item"].values
    vals = train_df["rating"].values - train_df["rating"].mean()
    R = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))

    dim = min(args.latent_dim, max(2, min(n_users, n_items) - 1))
    svd = TruncatedSVD(n_components=dim, random_state=args.seed)
    Z = svd.fit_transform(R)

    fcm = FuzzyCMeans(
        n_clusters=min(args.components, max(2, n_users // 3)),
        m=2.0,
        max_iter=args.max_iter,
        random_state=args.seed,
    )
    fcm.fit(Z)

    hist = pd.DataFrame({
        "iteration": np.arange(1, len(fcm.objective_history_) + 1),
        "FCM_objective": fcm.objective_history_,
    })
    return hist


def save_results(metrics_df, sens_df, fcm_hist, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    sens_df.to_csv(out_dir / "parameter_sensitivity.csv", index=False)
    fcm_hist.to_csv(out_dir / "convergence.csv", index=False)

    if len(fcm_hist) > 0:
        plt.figure()
        plt.plot(fcm_hist["iteration"], fcm_hist["FCM_objective"], marker="o")
        plt.xlabel("Iteration")
        plt.ylabel("FCM diagnostic objective")
        plt.title("FCM diagnostic convergence")
        plt.tight_layout()
        plt.savefig(out_dir / "convergence.png", dpi=200)
        plt.close()
