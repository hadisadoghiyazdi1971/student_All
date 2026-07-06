"""Core utilities for Heart-of-Risk transport-calibrated KME active learning.

This module is intentionally lightweight. It exposes the mathematical objects
that appear in the paper:
  - empirical tail-risk summaries for active-learning curves;
  - finite-bag KME radius used as a Hilbert envelope of a transport ball;
  - the HR-TC acquisition score (uncertainty x coverage x reliability).

The full benchmark drivers in this package import or mirror these definitions.
"""
from __future__ import annotations

import numpy as np


def empirical_cvar(losses: np.ndarray, alpha: float = 0.20) -> float:
    """Return empirical CVaR over the worst alpha fraction of losses."""
    losses = np.asarray(losses, dtype=float).ravel()
    if losses.size == 0:
        raise ValueError("losses must be non-empty")
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1]")
    m = int(np.ceil(alpha * losses.size))
    return float(np.sort(losses)[-m:].mean())


def finite_bag_radius(n: int, delta: float, kappa: float = 1.0, c: float = 1.0) -> float:
    """A simple Hilbert/KME radius proportional to sqrt(log(1/delta)/n)."""
    if n <= 0:
        raise ValueError("bag size n must be positive")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")
    return float(c * kappa * np.sqrt(np.log(1.0 / delta) / n))


def transport_radius_from_hilbert_radius(tau: float, L_phi: float) -> float:
    """Conservative implicit W1 radius rho=tau/L_phi."""
    if L_phi <= 0:
        raise ValueError("L_phi must be positive")
    return float(tau / L_phi)


def hr_tc_score(margins: np.ndarray,
                embeddings: np.ndarray,
                radii: np.ndarray,
                alpha: float = 1.0,
                eta: float = 1.0,
                eps_margin: float = 1e-6) -> np.ndarray:
    """Compute HR-TC acquisition scores for unlabeled candidates.

    Parameters
    ----------
    margins:
        Signed classifier margins f_t(z_i). Smaller absolute values imply
        larger label uncertainty.
    embeddings:
        Candidate KME/RFF representations, shape (n_candidates, d).
    radii:
        Hilbert ambiguity radii tau_i. Larger values mean less reliable bags.
    alpha:
        Coverage weight.
    eta:
        Reliability penalty weight.
    eps_margin:
        Stabilizer for zero margins.

    Returns
    -------
    scores:
        HR-TC scores. Query the index with the largest score.
    """
    margins = np.asarray(margins, dtype=float).ravel()
    Z = np.asarray(embeddings, dtype=float)
    radii = np.asarray(radii, dtype=float).ravel()
    if Z.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if margins.shape[0] != Z.shape[0] or radii.shape[0] != Z.shape[0]:
        raise ValueError("margins, embeddings and radii must have matching lengths")

    uncertainty = 1.0 / (eps_margin + np.abs(margins))
    if Z.shape[0] <= 1:
        coverage = np.ones(Z.shape[0])
    else:
        diff = Z[:, None, :] - Z[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        nonzero = dist2[dist2 > 0]
        scale = np.sqrt(np.median(nonzero)) if nonzero.size else 1.0
        K = np.exp(-dist2 / (2.0 * scale * scale + 1e-12))
        np.fill_diagonal(K, 0.0)
        coverage = K.sum(axis=1) / max(1, Z.shape[0] - 1)

    reliability = 1.0 / (1.0 + eta * radii)
    return uncertainty * (1.0 + alpha * coverage) * reliability


def summarize_curve(accuracies: np.ndarray, alpha: float = 0.20) -> dict[str, float]:
    """Return Heart-of-Risk summaries for an active-learning accuracy curve."""
    acc = np.asarray(accuracies, dtype=float).ravel()
    if acc.size == 0:
        raise ValueError("accuracies must be non-empty")
    losses = 1.0 - acc
    return {
        "final_accuracy": float(acc[-1]),
        "AULC": float(acc.mean()),
        "mean_future_loss": float(losses.mean()),
        f"CVaR_{alpha:.2f}": empirical_cvar(losses, alpha=alpha),
        "worst_round_loss": float(losses.max()),
        "best_round_loss": float(losses.min()),
    }
