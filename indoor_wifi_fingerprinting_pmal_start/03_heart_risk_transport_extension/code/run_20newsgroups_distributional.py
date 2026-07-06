"""20 Newsgroups distributional experiment skeleton.

This script is intentionally separate because sklearn may need to download the
20 Newsgroups corpus. It follows the thesis idea: represent each document as a
bag/distribution of word vectors or token features, then run the same active
learning interface used for GAUSSDIST and DIGITDIST.

For an offline replacement, precompute document bags as .npz with keys:
    bags_train, y_train, bags_test, y_test
where bags_* are object arrays of matrices (n_words x d).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from tc_kme_common import BagDataset, repeated_experiment


def make_20news_bags(seed: int, cats=("rec.sport.hockey", "sci.space"), max_docs_per_class: int = 300) -> BagDataset:
    rng = np.random.default_rng(seed)
    corpus = fetch_20newsgroups(subset="all", categories=list(cats), remove=("headers", "footers", "quotes"))
    y = np.where(corpus.target == 0, 1, -1)
    vec = TfidfVectorizer(max_features=5000, stop_words="english", token_pattern=r"(?u)\b\w\w+\b")
    X = vec.fit_transform(corpus.data)
    svd = TruncatedSVD(n_components=50, random_state=seed)
    X_low = svd.fit_transform(X)
    # Each document is approximated as a small bag around its semantic vector.
    # For a stricter word-vector implementation, replace this block by actual
    # word embeddings and one vector per token.
    bags = []
    for row in X_low:
        n = int(rng.choice([10, 25, 50, 75]))
        bags.append(row[None, :] + rng.normal(0, 0.05, size=(n, X_low.shape[1])))
    keep = []
    for lab in [1, -1]:
        idx = np.flatnonzero(y == lab)
        rng.shuffle(idx)
        keep.extend(idx[:max_docs_per_class])
    keep = np.asarray(keep)
    tr, te = train_test_split(np.arange(len(keep)), test_size=0.3, stratify=y[keep], random_state=seed)
    return BagDataset([bags[keep[i]] for i in tr], y[keep][tr], [bags[keep[i]] for i in te], y[keep][te], name="20NEWS-DIST")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="results")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--budget", type=int, default=50)
    args = p.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    repeated_experiment(lambda s: make_20news_bags(s), outdir, "20newsdist", args.repeats, args.budget, 303, gamma=0.05, n_features=256)

if __name__ == "__main__":
    main()
