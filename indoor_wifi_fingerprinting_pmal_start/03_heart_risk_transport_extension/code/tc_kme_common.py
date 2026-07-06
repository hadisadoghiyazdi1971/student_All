"""Common utilities for transport-calibrated KME active learning.

The code is designed to reproduce the experimental style of the Ghafarian thesis
experiments on active learning on distributions: each example is a bag, each bag
is embedded by an empirical kernel mean approximation, and active selection is
performed at the bag/distribution level.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import csv
import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class BagDataset:
    bags_train: List[np.ndarray]
    y_train: np.ndarray
    bags_test: List[np.ndarray]
    y_test: np.ndarray
    name: str

    @property
    def bag_sizes_train(self) -> np.ndarray:
        return np.asarray([len(b) for b in self.bags_train], dtype=float)


class RFFMeanEmbedder:
    """Random Fourier approximation of an RBF kernel mean embedding.

    For each raw sample x in a bag B, z(x) approximates the RKHS feature map.
    The distributional feature is mean_{x in B} z(x). The diagonal covariance of
    z(x) inside the bag approximates posterior/finite-bag uncertainty.
    """

    def __init__(self, gamma: float = 0.1, n_features: int = 512, seed: int = 0):
        self.gamma = gamma
        self.n_features = n_features
        self.seed = seed
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.scaler = StandardScaler()

    def fit(self, bags: List[np.ndarray]) -> "RFFMeanEmbedder":
        X = np.vstack([np.asarray(b) for b in bags])
        self.scaler.fit(X)
        rng = np.random.default_rng(self.seed)
        dim = X.shape[1]
        self.W = rng.normal(scale=np.sqrt(2.0 * self.gamma), size=(dim, self.n_features))
        self.b = rng.uniform(0.0, 2.0 * np.pi, size=self.n_features)
        return self

    def _sample_features(self, bag: np.ndarray) -> np.ndarray:
        if self.W is None or self.b is None:
            raise RuntimeError("RFFMeanEmbedder must be fitted before transform.")
        X = self.scaler.transform(np.asarray(bag))
        return np.sqrt(2.0 / self.n_features) * np.cos(X @ self.W + self.b)

    def transform(self, bags: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        means: List[np.ndarray] = []
        radii: List[float] = []
        for b in bags:
            Z = self._sample_features(b)
            means.append(Z.mean(axis=0))
            # Trace covariance radius for the empirical mean; stable for n=1.
            if len(Z) > 1:
                var = Z.var(axis=0, ddof=1)
            else:
                var = np.ones(Z.shape[1]) * (1.0 / Z.shape[1])
            radii.append(float(np.sqrt(np.sum(var) / max(1, len(Z)))))
        return np.vstack(means), np.asarray(radii)


def init_one_per_class(y: np.ndarray, rng: np.random.Generator) -> List[int]:
    labels = np.unique(y)
    chosen: List[int] = []
    for lab in labels:
        idx = np.flatnonzero(y == lab)
        chosen.append(int(rng.choice(idx)))
    return chosen


def embedding_density(Z: np.ndarray) -> np.ndarray:
    sq = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
    pos = sq[sq > 0]
    sigma2 = float(np.median(pos)) if pos.size else 1.0
    K = np.exp(-sq / (sigma2 + 1e-12))
    return K.mean(axis=1)


def fit_classifier(Z: np.ndarray, y: np.ndarray, labeled: List[int]) -> SVC:
    clf = SVC(kernel="linear", C=1.0)
    clf.fit(Z[labeled], y[labeled])
    return clf


def choose_query(
    method: str,
    clf: SVC,
    Z: np.ndarray,
    unlabeled: np.ndarray,
    radii: np.ndarray,
    density: np.ndarray,
    rng: np.random.Generator,
    alpha_density: float = 0.75,
    eta_radius: float = 3.0,
) -> int:
    if method == "DISTRAND":
        return int(rng.choice(unlabeled))

    margin = np.abs(clf.decision_function(Z[unlabeled]))
    uncertainty = 1.0 / (1.0 + margin)

    if method == "DISTMARGIN":
        return int(unlabeled[int(np.argmax(uncertainty))])

    if method == "HILB-DENSITY":
        return int(unlabeled[int(np.argmax(density[unlabeled]))])

    if method == "PMARLDS-lite":
        reliability = 1.0 / (1.0 + eta_radius / np.sqrt(np.maximum(1.0, radii[unlabeled] ** -2)))
        score = uncertainty * (1.0 + alpha_density * density[unlabeled]) * reliability
        return int(unlabeled[int(np.argmax(score))])

    if method == "TC-BKME-PMAL":
        # Bayesian/finite-bag transport-calibrated reliability.
        reliability = 1.0 / (1.0 + eta_radius * radii[unlabeled])
        score = uncertainty * (1.0 + alpha_density * density[unlabeled]) * reliability
        return int(unlabeled[int(np.argmax(score))])

    raise ValueError(f"Unknown method: {method}")


def run_active_learning(
    data: BagDataset,
    seed: int,
    budget: int,
    gamma: float = 0.1,
    n_features: int = 256,
    methods: Tuple[str, ...] = ("TC-BKME-PMAL", "PMARLDS-lite", "DISTMARGIN", "HILB-DENSITY", "DISTRAND"),
) -> Dict[str, List[float]]:
    rng = np.random.default_rng(seed)
    embedder = RFFMeanEmbedder(gamma=gamma, n_features=n_features, seed=seed)
    embedder.fit(data.bags_train + data.bags_test)
    Z_train, radii_train = embedder.transform(data.bags_train)
    Z_test, _ = embedder.transform(data.bags_test)
    density = embedding_density(Z_train)

    curves: Dict[str, List[float]] = {m: [] for m in methods}
    states: Dict[str, List[int]] = {m: init_one_per_class(data.y_train, rng) for m in methods}

    for _ in range(len(np.unique(data.y_train)), budget + 1):
        for method in methods:
            L = states[method]
            clf = fit_classifier(Z_train, data.y_train, L)
            curves[method].append(float(accuracy_score(data.y_test, clf.predict(Z_test))))
            if len(L) < budget:
                unlabeled = np.asarray(sorted(set(range(len(data.y_train))) - set(L)), dtype=int)
                q = choose_query(method, clf, Z_train, unlabeled, radii_train, density, rng)
                L.append(q)
    return curves


def repeated_experiment(
    data_factory,
    outdir: Path,
    name: str,
    repeats: int,
    budget: int,
    seed: int,
    gamma: float,
    n_features: int,
) -> Dict[str, Dict[str, float]]:
    outdir.mkdir(parents=True, exist_ok=True)
    all_curves: Dict[str, List[List[float]]] = {}
    runtimes: Dict[str, List[float]] = {}
    for r in range(repeats):
        data = data_factory(seed + r)
        t0 = time.perf_counter()
        curves = run_active_learning(data, seed + 10_000 + r, budget, gamma=gamma, n_features=n_features)
        elapsed = time.perf_counter() - t0
        for method, vals in curves.items():
            all_curves.setdefault(method, []).append(vals)
            runtimes.setdefault(method, []).append(elapsed / max(1, len(curves)))

    budgets = np.arange(len(np.unique(data.y_train)), budget + 1)
    summary: Dict[str, Dict[str, float]] = {}
    with open(outdir / f"{name}_curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "budget", "method", "mean_accuracy", "std_accuracy"])
        for method, mat in all_curves.items():
            A = np.asarray(mat, dtype=float)
            mean, std = A.mean(axis=0), A.std(axis=0)
            for b, m, s in zip(budgets, mean, std):
                w.writerow([name, int(b), method, float(m), float(s)])
            summary[method] = {
                "final": float(mean[-1]),
                "aulc": float(mean.mean()),
                "std_final": float(A[:, -1].std()),
                "runtime": float(np.mean(runtimes[method])),
            }

    with open(outdir / f"{name}_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method", "final_accuracy", "std_final", "AULC", "mean_runtime_seconds"])
        for method, vals in summary.items():
            w.writerow([name, method, vals["final"], vals["std_final"], vals["aulc"], vals["runtime"]])
    return summary
