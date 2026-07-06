"""Dataset generators for thesis-style distributional active-learning benchmarks."""
from __future__ import annotations
from typing import List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from scipy.ndimage import rotate, shift, zoom

from tc_kme_common import BagDataset


def _spd(rng: np.random.Generator, dim: int, scale: float) -> np.ndarray:
    A = rng.normal(size=(dim, dim))
    return scale * (A @ A.T) / dim + 0.05 * np.eye(dim)


def make_gaussdist(seed: int, dim: int = 10, n_per_class: int = 120, test_ratio: float = 0.3,
                   variable_bag: bool = True) -> BagDataset:
    """GAUSSDIST-style synthetic distributions.

    The class means and covariance statistics follow the spirit of the thesis
    protocol: two Gaussian-distribution classes with class-dependent covariance
    generation and finite sample bags of either heterogeneous or equal size.
    """
    rng = np.random.default_rng(seed)
    bags: List[np.ndarray] = []
    y: List[int] = []

    def one_bag(label: int) -> np.ndarray:
        n = int(rng.choice([5, 10, 25, 100])) if variable_bag else 20
        center = np.ones(dim) if label == 1 else 2.0 * np.ones(dim)
        mean = rng.normal(loc=center, scale=np.sqrt(0.5), size=dim)
        cov = _spd(rng, dim, 0.6 if label == 1 else 1.2)
        return rng.multivariate_normal(mean, cov, size=n)

    for label in [1, -1]:
        for _ in range(n_per_class):
            bags.append(one_bag(label)); y.append(label)

    idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=test_ratio,
                                           stratify=np.asarray(y), random_state=seed)
    return BagDataset([bags[i] for i in idx_train], np.asarray(y)[idx_train],
                      [bags[i] for i in idx_test], np.asarray(y)[idx_test],
                      name="GAUSSDIST-var" if variable_bag else "GAUSSDIST-equal")


def _resize_to_8(img: np.ndarray) -> np.ndarray:
    # zoom can change size. Center crop or pad to 8x8.
    h, w = img.shape
    out = np.zeros((8, 8), dtype=float)
    hs = min(h, 8); ws = min(w, 8)
    src_r0 = max(0, (h - hs) // 2); src_c0 = max(0, (w - ws) // 2)
    dst_r0 = max(0, (8 - hs) // 2); dst_c0 = max(0, (8 - ws) // 2)
    out[dst_r0:dst_r0+hs, dst_c0:dst_c0+ws] = img[src_r0:src_r0+hs, src_c0:src_c0+ws]
    return out


def _transform_digit(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    angle = rng.normal(0.0, 10.0)
    sh = rng.normal(0.0, 0.6, size=2)
    scale = float(np.clip(rng.normal(1.0, 0.08), 0.85, 1.15))
    z = zoom(img, scale, order=1, mode="constant", cval=0.0)
    z = _resize_to_8(z)
    z = rotate(z, angle, reshape=False, order=1, mode="constant", cval=0.0)
    z = shift(z, sh, order=1, mode="constant", cval=0.0)
    z += rng.normal(0.0, 0.02, size=z.shape)
    return z.reshape(-1) / 16.0


def make_digitsdist(seed: int, pair: Tuple[int, int] = (3, 8), n_per_class: int = 120,
                    test_ratio: float = 0.3, variable_bag: bool = True) -> BagDataset:
    """USPSDIST-style digit-distribution benchmark using sklearn digits.

    USPS data are not shipped with sklearn, so this script creates the same type
    of distributional object from sklearn digits: each base digit image becomes a
    bag/distribution of geometric transformations (translation, rotation, scale),
    matching the USPSDIST mechanism described in the thesis.
    """
    rng = np.random.default_rng(seed)
    raw = load_digits()
    X = raw.images
    y_raw = raw.target
    selected = np.flatnonzero((y_raw == pair[0]) | (y_raw == pair[1]))
    rng.shuffle(selected)
    bags: List[np.ndarray] = []
    labels: List[int] = []
    for label_digit, lab in [(pair[0], 1), (pair[1], -1)]:
        idx = selected[y_raw[selected] == label_digit][:n_per_class]
        for i in idx:
            n = int(rng.choice([5, 10, 25, 50])) if variable_bag else 20
            bag = np.vstack([_transform_digit(X[i], rng) for _ in range(n)])
            bags.append(bag); labels.append(lab)
    idx_train, idx_test = train_test_split(np.arange(len(labels)), test_size=test_ratio,
                                           stratify=np.asarray(labels), random_state=seed)
    return BagDataset([bags[i] for i in idx_train], np.asarray(labels)[idx_train],
                      [bags[i] for i in idx_test], np.asarray(labels)[idx_test],
                      name=f"DIGITDIST-{pair[0]}v{pair[1]}")
