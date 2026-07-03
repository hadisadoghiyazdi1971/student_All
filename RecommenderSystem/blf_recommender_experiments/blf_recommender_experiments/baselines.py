import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


class ItemMeanBaseline:
    def fit(self, train_df, n_users, n_items):
        self.global_mean_ = float(train_df["rating"].mean())
        item_stats = train_df.groupby("item")["rating"].mean()
        self.item_mean_ = np.full(n_items, self.global_mean_, dtype=float)
        self.item_mean_[item_stats.index.values] = item_stats.values
        return self

    def predict_pairs(self, pairs_df):
        return self.item_mean_[pairs_df["item"].values]


class BiasBaseline:
    """
    Baseline: global mean + user bias + item bias with shrinkage.
    """
    def __init__(self, reg=10.0):
        self.reg = reg

    def fit(self, train_df, n_users, n_items):
        self.global_mean_ = float(train_df["rating"].mean())
        self.user_bias_ = np.zeros(n_users, dtype=float)
        self.item_bias_ = np.zeros(n_items, dtype=float)

        user_grp = train_df.groupby("user")
        for u, g in user_grp:
            self.user_bias_[u] = (g["rating"].sum() - len(g) * self.global_mean_) / (self.reg + len(g))

        item_grp = train_df.groupby("item")
        for i, g in item_grp:
            residual = g["rating"].values - self.global_mean_ - self.user_bias_[g["user"].values]
            self.item_bias_[i] = residual.sum() / (self.reg + len(g))
        return self

    def predict_pairs(self, pairs_df):
        u = pairs_df["user"].values
        i = pairs_df["item"].values
        pred = self.global_mean_ + self.user_bias_[u] + self.item_bias_[i]
        return np.clip(pred, 1.0, 5.0)


class SVDItemMeanBaseline:
    """
    Low-rank user embedding + item mean fallback. This is a lightweight baseline,
    not a full matrix factorization optimizer.
    """
    def __init__(self, latent_dim=40, random_state=42):
        self.latent_dim = latent_dim
        self.random_state = random_state

    def fit(self, train_df, n_users, n_items):
        self.global_mean_ = float(train_df["rating"].mean())
        rows = train_df["user"].values
        cols = train_df["item"].values
        vals = train_df["rating"].values - self.global_mean_
        R = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
        self.svd_ = TruncatedSVD(n_components=min(self.latent_dim, min(n_users, n_items) - 1),
                                 random_state=self.random_state)
        self.user_factors_ = self.svd_.fit_transform(R)
        self.item_factors_ = self.svd_.components_.T
        return self

    def predict_pairs(self, pairs_df):
        u = pairs_df["user"].values
        i = pairs_df["item"].values
        pred = self.global_mean_ + np.sum(self.user_factors_[u] * self.item_factors_[i], axis=1)
        return np.clip(pred, 1.0, 5.0)
