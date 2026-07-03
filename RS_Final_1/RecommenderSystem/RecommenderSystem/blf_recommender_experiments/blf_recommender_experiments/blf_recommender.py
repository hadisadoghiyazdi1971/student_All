import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import GaussianMixture


def _safe_cov(X, weights=None, eps=1e-5):
    if X.shape[0] == 0:
        return np.eye(1)
    if weights is None:
        cov = np.cov(X.T)
    else:
        w = np.asarray(weights, dtype=float)
        w = w / (w.sum() + 1e-12)
        mu = np.sum(X * w[:, None], axis=0)
        Xc = X - mu
        cov = (Xc.T * w) @ Xc
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    cov = np.asarray(cov, dtype=float)
    cov += eps * np.eye(cov.shape[0])
    return cov


def _log_gaussian_diag(X, mean, var):
    """
    log N(X; mean, diag(var)).
    """
    var = np.maximum(var, 1e-6)
    diff = X - mean
    return -0.5 * (np.sum(np.log(2.0 * np.pi * var)) + np.sum(diff * diff / var, axis=1))


def _softmax_log(logits, axis=1):
    m = np.max(logits, axis=axis, keepdims=True)
    e = np.exp(logits - m)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-12)


class BayesianLatentFilteringRecommender:
    """
    Bayesian Latent Filtering Recommender.

    Mathematical chain implemented:
        x_u -> z_u = phi(x_u)
        -> q_i(u)=P(c_i|x_u)
        -> A_ij=f(theta_j|c_i)
        -> R_j(u)=sum_i q_i(u) A_ij
        -> gamma_j(u)
        -> rating prediction.

    In this implementation:
      - x_u is the sparse user rating vector.
      - phi is TruncatedSVD.
      - hidden components c_i are fitted by a Gaussian mixture in latent space.
      - candidate sets S_j are the hard component groups, but prediction uses
        posterior gating, not hard assignment.
    """
    def __init__(
        self,
        n_components=80,
        latent_dim=40,
        beta=15.0,
        reg=10.0,
        blend_weight=0.75,
        covariance_floor=1e-4,
        random_state=42,
        max_iter=100,
    ):
        self.n_components = int(n_components)
        self.latent_dim = int(latent_dim)
        self.beta = float(beta)
        self.reg = float(reg)
        self.blend_weight = float(blend_weight)
        self.covariance_floor = float(covariance_floor)
        self.random_state = random_state
        self.max_iter = int(max_iter)

    def fit(self, train_df, n_users, n_items):
        self.n_users_ = int(n_users)
        self.n_items_ = int(n_items)
        self.global_mean_ = float(train_df["rating"].mean())

        rows = train_df["user"].values
        cols = train_df["item"].values
        vals = train_df["rating"].values
        self.R_csr_ = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))

        # Bias terms.
        self.user_bias_ = np.zeros(n_users, dtype=float)
        self.item_bias_ = np.zeros(n_items, dtype=float)
        for u, g in train_df.groupby("user"):
            self.user_bias_[u] = (g["rating"].sum() - len(g) * self.global_mean_) / (self.reg + len(g))
        for i, g in train_df.groupby("item"):
            residual = g["rating"].values - self.global_mean_ - self.user_bias_[g["user"].values]
            self.item_bias_[i] = residual.sum() / (self.reg + len(g))

        # Encoder phi: low-rank user representation.
        centered_vals = vals - self.global_mean_
        centered_R = csr_matrix((centered_vals, (rows, cols)), shape=(n_users, n_items))
        effective_dim = min(self.latent_dim, max(2, min(n_users, n_items) - 1))
        self.encoder_ = TruncatedSVD(n_components=effective_dim, random_state=self.random_state)
        self.Z_ = self.encoder_.fit_transform(centered_R)

        # Gaussian latent components.
        k = min(self.n_components, max(2, n_users // 3))
        self.n_components_ = k
        self.gmm_ = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            reg_covar=self.covariance_floor,
            max_iter=self.max_iter,
            random_state=self.random_state,
            init_params="kmeans",
        )
        self.gmm_.fit(self.Z_)
        self.q_users_ = self.gmm_.predict_proba(self.Z_)  # P(c_i | x_u)

        self.component_means_ = self.gmm_.means_.copy()              # m_i
        self.component_vars_ = self.gmm_.covariances_.copy()         # diag Lambda_i
        self.component_priors_ = self.gmm_.weights_.copy()           # rho_i

        self._build_set_descriptors(train_df)
        self._build_set_item_models(train_df)
        self._compute_convergence_history(train_df)
        return self

    def _build_set_descriptors(self, train_df):
        """
        Candidate sets S_j are represented by theta_j=(mu_j, Sigma_j).
        Here K=C and set descriptors are derived from posterior-weighted users.
        """
        Z = self.Z_
        Q = self.q_users_
        K = self.n_components_
        d = Z.shape[1]

        self.set_mu_ = np.zeros((K, d), dtype=float)
        self.set_var_ = np.zeros((K, d), dtype=float)
        self.set_prior_ = np.zeros(K, dtype=float)

        for j in range(K):
            w = Q[:, j]
            mass = w.sum() + 1e-12
            mu = (w[:, None] * Z).sum(axis=0) / mass
            diff = Z - mu
            var = (w[:, None] * diff * diff).sum(axis=0) / mass
            var = np.maximum(var, self.covariance_floor)
            self.set_mu_[j] = mu
            self.set_var_[j] = var
            self.set_prior_[j] = mass

        self.set_prior_ = self.set_prior_ / (self.set_prior_.sum() + 1e-12)

        # A_ij = f(theta_j | c_i) proportional to N(m_i; mu_j, Sigma_j) P(theta_j)
        logits = np.zeros((K, K), dtype=float)
        for j in range(K):
            logits[:, j] = _log_gaussian_diag(self.component_means_, self.set_mu_[j], self.set_var_[j])
            logits[:, j] += np.log(self.set_prior_[j] + 1e-12)
        self.component_to_set_ = _softmax_log(logits, axis=1)  # rows i sum over j

        # Confidence omega_j = 1/(1+tr Sigma_j)
        self.set_confidence_ = 1.0 / (1.0 + np.sum(self.set_var_, axis=1))

    def _build_set_item_models(self, train_df):
        """
        Builds local set item means used in local predictions.
        Soft user membership is used, not hard cluster assignment.
        """
        K = self.n_components_
        I = self.n_items_
        Q = self.q_users_

        # Item mean with shrinkage fallback.
        item_count = np.zeros(I, dtype=float)
        item_sum = np.zeros(I, dtype=float)
        for row in train_df.itertuples(index=False):
            item_count[row.item] += 1.0
            item_sum[row.item] += row.rating
        self.item_mean_ = (item_sum + self.beta * self.global_mean_) / (item_count + self.beta)

        numer = np.zeros((K, I), dtype=float)
        denom = np.zeros((K, I), dtype=float)

        for row in train_df.itertuples(index=False):
            u = row.user
            i = row.item
            r = row.rating
            w = Q[u]  # P(c_j | x_u), used as soft set contribution
            numer[:, i] += w * r
            denom[:, i] += w

        self.set_item_mean_ = (numer + self.beta * self.item_mean_[None, :]) / (denom + self.beta)

    def _compute_convergence_history(self, train_df):
        """
        A diagnostic negative log-likelihood curve. GaussianMixture exposes
        lower_bound_ only at the final iteration, so we compute a stable objective
        after fitting as a one-point diagnostic and then a pseudo-history if available.
        """
        log_prob = self.gmm_.score_samples(self.Z_)
        nll = float(-np.mean(log_prob))
        self.convergence_history_ = [nll]

        # sklearn does not store per-iteration history. We keep the final value.
        # experiments.py combines this with FCM objective when needed.

    def posterior_components_for_users(self, user_ids):
        return self.q_users_[np.asarray(user_ids, dtype=int)]

    def relevance_for_users(self, user_ids):
        """
        R_j(x_u)=sum_i P(c_i|x_u) f(theta_j|c_i)
        """
        Q = self.posterior_components_for_users(user_ids)
        return Q @ self.component_to_set_

    def gamma_for_users(self, user_ids):
        R = self.relevance_for_users(user_ids)
        alpha = R / (R.sum(axis=1, keepdims=True) + 1e-12)
        weighted = alpha * self.set_confidence_[None, :]
        gamma = weighted / (weighted.sum(axis=1, keepdims=True) + 1e-12)
        return gamma

    def predict_pairs(self, pairs_df):
        u = pairs_df["user"].values.astype(int)
        i = pairs_df["item"].values.astype(int)

        gamma = self.gamma_for_users(u)
        mixture_pred = np.sum(gamma * self.set_item_mean_[:, i].T, axis=1)

        bias_pred = self.global_mean_ + self.user_bias_[u] + self.item_bias_[i]

        # The Bayesian mixture is blended with a bias model for sparse-rating stability.
        w = np.clip(self.blend_weight, 0.0, 1.0)
        pred = w * mixture_pred + (1.0 - w) * bias_pred
        return np.clip(pred, 1.0, 5.0)

    def recommend_topn(self, train_df, users, n=10, candidate_items=None):
        """
        Returns dict user -> list of top-n item IDs not rated in train.
        This is intentionally simple and readable.
        """
        if candidate_items is None:
            candidate_items = np.arange(self.n_items_, dtype=int)
        else:
            candidate_items = np.asarray(candidate_items, dtype=int)

        rated_by_user = train_df.groupby("user")["item"].apply(set).to_dict()
        out = {}

        for u in users:
            rated = rated_by_user.get(u, set())
            candidates = [int(i) for i in candidate_items if int(i) not in rated]
            if not candidates:
                out[int(u)] = []
                continue
            pairs = pd.DataFrame({"user": [int(u)] * len(candidates), "item": candidates})
            scores = self.predict_pairs(pairs)
            order = np.argsort(-scores)[:n]
            out[int(u)] = [candidates[idx] for idx in order]
        return out
