from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _encode_ids(df, user_col="user_id", item_col="item_id"):
    user_codes, user_uniques = pd.factorize(df[user_col])
    item_codes, item_uniques = pd.factorize(df[item_col])
    out = pd.DataFrame({
        "user": user_codes.astype(int),
        "item": item_codes.astype(int),
        "rating": df["rating"].astype(float).values,
    })
    return out, list(user_uniques), list(item_uniques)


def load_movielens_100k(data_dir):
    """
    Expects MovieLens-100K directory containing u.data.
    u.data columns: user_id, item_id, rating, timestamp separated by tabs.
    """
    data_dir = Path(data_dir)
    path = data_dir / "u.data"
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}. Put MovieLens-100K u.data there.")

    df = pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python"
    )
    df, users, items = _encode_ids(df)
    return df, users, items


def load_movielens_1m(data_dir):
    """
    Expects MovieLens-1M directory containing ratings.dat.
    ratings.dat columns: UserID::MovieID::Rating::Timestamp
    """
    data_dir = Path(data_dir)
    path = data_dir / "ratings.dat"
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}. Put MovieLens-1M ratings.dat there.")

    df = pd.read_csv(
        path,
        sep="::",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python"
    )
    df, users, items = _encode_ids(df)
    return df, users, items


def load_csv_ratings(csv_path):
    """
    Expects CSV columns: user_id,item_id,rating
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    required = {"user_id", "item_id", "rating"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required}. Found: {list(df.columns)}")
    df, users, items = _encode_ids(df)
    return df, users, items


def make_synthetic_ratings(n_users=800, n_items=500, n_groups=12, density=0.06, seed=42):
    """
    A small synthetic recommender dataset with latent user groups and item groups.
    Useful for checking that the whole pipeline runs before using real datasets.
    """
    rng = np.random.default_rng(seed)
    user_group = rng.integers(0, n_groups, size=n_users)
    item_group = rng.integers(0, n_groups, size=n_items)

    group_pref = rng.normal(0, 0.8, size=(n_groups, n_groups))
    user_bias = rng.normal(0, 0.25, size=n_users)
    item_bias = rng.normal(0, 0.25, size=n_items)

    rows = []
    n_obs = int(n_users * n_items * density)
    pairs = set()
    while len(pairs) < n_obs:
        u = int(rng.integers(0, n_users))
        i = int(rng.integers(0, n_items))
        pairs.add((u, i))

    for u, i in pairs:
        score = 3.4 + user_bias[u] + item_bias[i] + group_pref[user_group[u], item_group[i]]
        score += rng.normal(0, 0.55)
        rating = float(np.clip(np.round(score), 1, 5))
        rows.append((u, i, rating))

    df = pd.DataFrame(rows, columns=["user", "item", "rating"])
    users = list(range(n_users))
    items = list(range(n_items))
    return df, users, items


def split_ratings(df, test_size=0.2, seed=42):
    """
    Random split. For a stronger paper, replace this with a temporal or cold-start split.
    """
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=None)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def user_level_cold_start_split(df, min_train_interactions=1, test_user_frac=0.2, seed=42):
    """
    Creates a simple user cold-start split:
    a fraction of users are cold-start users. For each of them, keep only
    min_train_interactions in training and put the rest into test.
    """
    rng = np.random.default_rng(seed)
    users = df["user"].unique()
    cold_users = set(rng.choice(users, size=max(1, int(len(users) * test_user_frac)), replace=False))

    train_rows = []
    test_rows = []

    for u, g in df.groupby("user"):
        g = g.sample(frac=1.0, random_state=seed)
        if u in cold_users and len(g) > min_train_interactions:
            train_rows.append(g.iloc[:min_train_interactions])
            test_rows.append(g.iloc[min_train_interactions:])
        else:
            cut = int(0.8 * len(g))
            train_rows.append(g.iloc[:cut])
            test_rows.append(g.iloc[cut:])

    train = pd.concat(train_rows).reset_index(drop=True)
    test = pd.concat(test_rows).reset_index(drop=True)
    return train, test
