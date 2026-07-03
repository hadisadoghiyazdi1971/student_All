import numpy as np


class FuzzyCMeans:
    """
    Minimal FCM implementation for diagnostic comparison.
    This is not the proposed method. It is only a baseline/diagnostic.
    """
    def __init__(self, n_clusters=20, m=2.0, max_iter=100, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.objective_history_ = []

    def fit(self, X):
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape
        U = rng.random((n, self.n_clusters))
        U = U / U.sum(axis=1, keepdims=True)

        prev_obj = np.inf
        for _ in range(self.max_iter):
            Um = U ** self.m
            centers = (Um.T @ X) / (Um.sum(axis=0)[:, None] + 1e-12)

            dist = np.zeros((n, self.n_clusters), dtype=float)
            for k in range(self.n_clusters):
                diff = X - centers[k]
                dist[:, k] = np.sum(diff * diff, axis=1) + 1e-12

            obj = float(np.sum((U ** self.m) * dist))
            self.objective_history_.append(obj)

            if abs(prev_obj - obj) < self.tol:
                break
            prev_obj = obj

            power = 1.0 / (self.m - 1.0)
            ratio = (dist[:, :, None] / dist[:, None, :]) ** power
            U = 1.0 / np.sum(ratio, axis=2)
            U = U / (U.sum(axis=1, keepdims=True) + 1e-12)

        self.membership_ = U
        self.centers_ = centers
        self.labels_ = np.argmax(U, axis=1)
        return self

    def predict_proba(self, X):
        dist = np.zeros((X.shape[0], self.n_clusters), dtype=float)
        for k in range(self.n_clusters):
            diff = X - self.centers_[k]
            dist[:, k] = np.sum(diff * diff, axis=1) + 1e-12

        power = 1.0 / (self.m - 1.0)
        ratio = (dist[:, :, None] / dist[:, None, :]) ** power
        U = 1.0 / np.sum(ratio, axis=2)
        return U / (U.sum(axis=1, keepdims=True) + 1e-12)
