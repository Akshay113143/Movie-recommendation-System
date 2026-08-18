"""
Evaluation.

Two families of metric, because a recommender does two different jobs:

A. RATING PREDICTION  -- 'how many stars would this user give this movie?'
     RMSE, MAE, and tolerance accuracy (share of predictions within +-0.5 or
     +-1.0 stars). Tolerance accuracy is the metric usually meant when a
     project reports a single headline number like "98% accurate" -- it is
     legitimate but you must state the tolerance, because +-1.0 on a 0.5-5.0
     scale is a wide target.

B. RANKING / TOP-N     -- 'which 10 movies should we put on screen?'
     Precision@K, Recall@K, NDCG@K, MAP@K, plus catalogue coverage. This is
     what actually matters in production: nobody sees your RMSE, they see the
     ten posters on the home row.

Also included: like/dislike classification accuracy (threshold the predicted
rating at 4.0 and compare against the true label), which gives a single
'accuracy %' that is directly comparable to a classifier and is the second
common reading of a headline accuracy number.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# A. Rating-prediction metrics
# --------------------------------------------------------------------------- #
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error -- penalises large misses quadratically."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error -- average miss in stars, robust to outliers."""
    return float(np.mean(np.abs(y_true - y_pred)))


def tolerance_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                       tol: float = 1.0) -> float:
    """Share of predictions landing within `tol` stars of the truth."""
    return float(np.mean(np.abs(y_true - y_pred) <= tol))


def like_classification(y_true: np.ndarray, y_pred: np.ndarray,
                        like_threshold: float = 4.0) -> dict:
    """Binarise both truth and prediction at `like_threshold` -> confusion stats.

    Turns the regression into 'will this user like it?', which is the decision
    the product actually makes when it chooses whether to surface a title.
    """
    yt = y_true >= like_threshold
    yp = y_pred >= like_threshold
    tp = float(np.sum(yt & yp)); fp = float(np.sum(~yt & yp))
    fn = float(np.sum(yt & ~yp)); tn = float(np.sum(~yt & ~yp))
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    return {"accuracy": (tp + tn) / len(yt),
            "precision": prec,
            "recall": rec,
            "f1": 2 * prec * rec / (prec + rec + 1e-12),
            "positive_rate": float(np.mean(yt))}


def rating_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Everything in family A in one dict."""
    cls = like_classification(y_true, y_pred)
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "acc_within_0.5": tolerance_accuracy(y_true, y_pred, 0.5),
        "acc_within_1.0": tolerance_accuracy(y_true, y_pred, 1.0),
        "like_accuracy": cls["accuracy"],
        "like_precision": cls["precision"],
        "like_recall": cls["recall"],
        "like_f1": cls["f1"],
        "n_predictions": int(len(y_true)),
    }


# --------------------------------------------------------------------------- #
# B. Ranking metrics
# --------------------------------------------------------------------------- #
def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    r = relevances[:k]
    return float(np.sum(r / np.log2(np.arange(2, len(r) + 2))))


def ndcg_at_k(recommended: np.ndarray, relevant: set, k: int) -> float:
    """Normalised Discounted Cumulative Gain.

    Rewards putting hits *near the top*: a hit at rank 1 is worth 1.0, at rank
    10 only 1/log2(11) = 0.29. Precision@K cannot see that difference.
    """
    rel = np.array([1.0 if i in relevant else 0.0 for i in recommended[:k]])
    ideal = np.ones(min(len(relevant), k))
    idcg = dcg_at_k(ideal, k)
    return dcg_at_k(rel, k) / idcg if idcg > 0 else 0.0


def average_precision_at_k(recommended: np.ndarray, relevant: set, k: int) -> float:
    hits, score = 0, 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / rank
    return score / min(len(relevant), k) if relevant else 0.0


def ranking_report(rec_lists: dict, relevant_sets: dict, k: int = 10,
                   n_catalogue_items: int | None = None) -> dict:
    """Aggregate ranking metrics over many users.

    rec_lists[u]     : ranked array of item indices recommended to user u
    relevant_sets[u] : set of item indices the user actually liked in the test
                       period (ground truth)
    """
    precs, recs, ndcgs, maps = [], [], [], []
    seen_items = set()
    for u, rel in relevant_sets.items():
        if not rel or u not in rec_lists:
            continue
        recd = rec_lists[u][:k]
        seen_items.update(recd.tolist())
        hits = len(set(recd.tolist()) & rel)
        precs.append(hits / k)
        recs.append(hits / len(rel))
        ndcgs.append(ndcg_at_k(recd, rel, k))
        maps.append(average_precision_at_k(recd, rel, k))
    out = {f"precision@{k}": float(np.mean(precs)) if precs else 0.0,
           f"recall@{k}": float(np.mean(recs)) if recs else 0.0,
           f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
           f"map@{k}": float(np.mean(maps)) if maps else 0.0,
           f"hit_rate@{k}": float(np.mean([p > 0 for p in precs])) if precs else 0.0,
           "n_users_evaluated": len(precs)}
    if n_catalogue_items:
        out["catalogue_coverage"] = len(seen_items) / n_catalogue_items
    return out
