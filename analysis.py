"""
Post-training analysis.

    python analysis.py

Answers two questions the headline metrics hide:

1. WHAT DOES 'ACCURACY' ACTUALLY MEAN HERE?
   Reports the share of test predictions inside a whole range of tolerances,
   so any 'X% accurate' claim can be traced to a specific tolerance instead of
   floating free. A single number without its tolerance is not a result.

2. WHO DOES THE MODEL FAIL FOR?
   Slices RMSE and hit-rate by how much history the user has and by how
   popular the item is. Aggregate RMSE hides the fact that a recommender is
   usually excellent for power users and poor for exactly the newcomers whose
   retention you care about most.
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from recsys import data as D
from recsys import evaluate as E
from recsys.hybrid import HybridRatingModel

MODEL = "artifacts/model.pkl"
OUT = "results/analysis.json"


def main():
    print("loading artifacts ...", flush=True)
    with open(MODEL, "rb") as f:
        M = pickle.load(f)
    base, itemcf, mf, cb, hyb = (M["baseline"], M["itemcf"], M["mf"],
                                 M["content"], M["hybrid_rating"])
    mapper, R = M["mapper"], M["R_train"]
    item_support, user_support = M["item_support"], M["user_support"]
    user_means = M["user_means"]

    print("rebuilding the split (same seed -> identical partition) ...", flush=True)
    d = D.prepare("data/ratings.csv", "data/movies.csv", verbose=False)
    test = d["test"].sample(n=200_000, random_state=42)
    tu = mapper.map_users(test["userId"])
    ti = mapper.map_items(test["movieId"])
    ty = test["rating"].to_numpy(dtype=np.float32)

    print("scoring the hybrid on the test sample ...", flush=True)
    b = base.predict(tu, ti)
    cf = itemcf.predict_residual(tu, ti)
    mfr = mf.predict_residual(tu, ti)
    cbs = cb.predict_scores(tu, ti)
    X = hyb.make_features(b, cf, mfr, cbs, item_support[ti], user_support[tu],
                          b - user_means[tu])
    pred = hyb.predict(X)

    out = {}

    # ------------------------------------------------------------------ #
    # 1. tolerance curve
    # ------------------------------------------------------------------ #
    tol_curve = {}
    for tol in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]:
        tol_curve[f"within_{tol}"] = E.tolerance_accuracy(ty, pred, tol)
    out["tolerance_curve"] = tol_curve
    print("\n=== accuracy vs tolerance (hybrid model) ===")
    for k, v in tol_curve.items():
        print(f"  {k:14s} {v:7.2%}")

    # ------------------------------------------------------------------ #
    # 2. error sliced by user history length
    # ------------------------------------------------------------------ #
    hist = user_support[tu]
    bins = [(0, 20), (20, 50), (50, 150), (150, 500), (500, 10 ** 9)]
    by_user = {}
    print("\n=== RMSE by user history length (train ratings) ===")
    for lo, hi in bins:
        m = (hist >= lo) & (hist < hi)
        if m.sum() < 50:
            continue
        by_user[f"{lo}-{hi if hi < 10**9 else 'inf'}"] = {
            "n_test_ratings": int(m.sum()),
            "rmse": E.rmse(ty[m], pred[m]),
            "mae": E.mae(ty[m], pred[m]),
            "acc_within_1.0": E.tolerance_accuracy(ty[m], pred[m], 1.0),
        }
        print(f"  {lo:>4}-{hi if hi < 10**9 else 'inf':<4} "
              f"n={m.sum():>7,}  RMSE={E.rmse(ty[m], pred[m]):.4f}  "
              f"acc±1={E.tolerance_accuracy(ty[m], pred[m], 1.0):.2%}")
    out["by_user_history"] = by_user

    # ------------------------------------------------------------------ #
    # 3. error sliced by item popularity
    # ------------------------------------------------------------------ #
    sup = item_support[ti]
    ibins = [(20, 100), (100, 500), (500, 2000), (2000, 10000), (10000, 10 ** 9)]
    by_item = {}
    print("\n=== RMSE by item popularity (train ratings for that movie) ===")
    for lo, hi in ibins:
        m = (sup >= lo) & (sup < hi)
        if m.sum() < 50:
            continue
        by_item[f"{lo}-{hi if hi < 10**9 else 'inf'}"] = {
            "n_test_ratings": int(m.sum()),
            "rmse": E.rmse(ty[m], pred[m]),
            "acc_within_1.0": E.tolerance_accuracy(ty[m], pred[m], 1.0),
        }
        print(f"  {lo:>6}-{hi if hi < 10**9 else 'inf':<6} "
              f"n={m.sum():>7,}  RMSE={E.rmse(ty[m], pred[m]):.4f}  "
              f"acc±1={E.tolerance_accuracy(ty[m], pred[m], 1.0):.2%}")
    out["by_item_popularity"] = by_item

    # ------------------------------------------------------------------ #
    # 4. like/dislike classification at several thresholds
    # ------------------------------------------------------------------ #
    cls = {}
    print("\n=== 'will the user like it?' classification ===")
    for thr in [3.5, 4.0, 4.5]:
        r = E.like_classification(ty, pred, thr)
        cls[f"threshold_{thr}"] = r
        print(f"  >= {thr}: accuracy={r['accuracy']:.2%} precision={r['precision']:.2%} "
              f"recall={r['recall']:.2%} F1={r['f1']:.2%} "
              f"(base rate {r['positive_rate']:.2%})")
    out["like_classification"] = cls

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
