import numpy as np


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def precision_recall_f1_at_k(test_df, topn, relevance_threshold=4.0):
    """
    test_df columns: user, item, rating
    topn: dict[user] -> list[item]
    """
    tp = fp = fn = 0

    grouped = test_df.groupby("user")
    for user, g in grouped:
        relevant = set(g.loc[g["rating"] >= relevance_threshold, "item"].tolist())
        if not relevant:
            continue

        recommended = set(topn.get(user, []))
        tp += len(recommended & relevant)
        fp += len(recommended - relevant)
        fn += len(relevant - recommended)

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def clip_ratings(pred, min_rating=1.0, max_rating=5.0):
    return np.clip(np.asarray(pred, dtype=float), min_rating, max_rating)
