"""
Movie Recommendation System -- full training + evaluation pipeline.

    python run_pipeline.py --ratings data/ratings.csv --movies data/movies.csv

Stages
    1  data prep       load -> support filter -> chronological split -> CSR
    2  bias baseline   mu + b_u + b_i, and the residual matrix everything else uses
    3  item-item CF    top-k cosine neighbourhood on bias-adjusted item columns
    4  explicit ALS    latent factors for the RATING task (RMSE)
    5  implicit ALS    latent factors for the RANKING task (top-N)
    6  content model   TF-IDF item features + per-user taste profiles
    7  hybrid          ridge stacking (ratings) + weight-tuned fusion (ranking)
    8  evaluation      RMSE/MAE/tolerance-accuracy + P@10/R@10/NDCG@10/coverage
    9  cold start      recommendations with zero interaction history
    10 artifacts       model.pkl, metrics.json, plots
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd

from recsys import data as D
from recsys.collaborative import (BiasBaseline, ItemItemCF,
                                  ALSMatrixFactorization, ImplicitALS)
from recsys.content import ContentBasedRecommender
from recsys.hybrid import (HybridRatingModel, HybridRanker, zscore,
                           quality_adjusted)
from recsys import evaluate as E

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def sample_pairs(df, mapper, n, seed=42):
    if len(df) > n:
        df = df.sample(n=n, random_state=seed)
    return (mapper.map_users(df["userId"]),
            mapper.map_items(df["movieId"]),
            df["rating"].to_numpy(dtype=np.float32))


def build_component_cache(users_raw, split_liked, mapper, R_train, models,
                          base, item_support, tau=50.0):
    """Per-user catalogue-wide scores from every component (for ranking eval)."""
    ials, mf, itemcf, cb = models
    cache, relevant, seen = [], {}, {}
    for raw_u in users_raw:
        u = mapper.user_to_idx[raw_u]
        rel = {mapper.item_to_idx[m] for m in split_liked.loc[raw_u]
               if m in mapper.item_to_idx}
        if not rel:
            continue
        relevant[u] = rel
        seen[u] = R_train.indices[R_train.indptr[u]:R_train.indptr[u + 1]]
        comp = {
            "ials": ials.score_all_items(u),
            "mf": quality_adjusted(mf.score_all_items(u), base.b_i,
                                   item_support, tau),
            "itemcf": quality_adjusted(itemcf.score_all_items(u), base.b_i,
                                       item_support, tau),
            "content": cb.score_all_items(u),
        }
        cache.append((u, comp))
    return cache, relevant, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", default="data/ratings.csv")
    ap.add_argument("--movies", default="data/movies.csv")
    ap.add_argument("--outdir", default="artifacts")
    ap.add_argument("--resultsdir", default="results")
    ap.add_argument("--factors", type=int, default=40)          # explicit ALS
    ap.add_argument("--als-epochs", type=int, default=8)
    ap.add_argument("--implicit-factors", type=int, default=64)
    ap.add_argument("--implicit-epochs", type=int, default=8)
    ap.add_argument("--neighbors", type=int, default=100)
    ap.add_argument("--max-user-history", type=int, default=250)
    ap.add_argument("--eval-pairs", type=int, default=250_000)
    ap.add_argument("--eval-users", type=int, default=2_000)
    ap.add_argument("--tune-users", type=int, default=1_000)
    ap.add_argument("--min-movie-ratings", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.resultsdir, exist_ok=True)
    metrics = {"config": vars(args)}
    K = 10

    # ------------------------------------------------------------ 1. DATA
    log("STAGE 1  data preparation")
    d = D.prepare(args.ratings, args.movies,
                  min_movie_ratings=args.min_movie_ratings)
    train, val, test = d["train"], d["val"], d["test"]
    R_train, mapper, catalogue = d["R_train"], d["mapper"], d["catalogue"]
    n_users, n_items = R_train.shape

    item_support = np.diff(R_train.tocsc().indptr).astype(np.float32)
    user_support = np.diff(R_train.indptr).astype(np.float32)
    user_means = np.zeros(n_users, dtype=np.float32)
    nz = user_support > 0
    user_means[nz] = (np.add.reduceat(R_train.data, R_train.indptr[:-1])[nz]
                      / user_support[nz])

    metrics["dataset"] = {
        "n_ratings_modelled": int(len(train) + len(val) + len(test)),
        "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
        "n_users": int(n_users), "n_items": int(n_items),
        "density_pct": float(100 * R_train.nnz / (n_users * n_items)),
        "global_mean_rating": float(R_train.data.mean()),
        "median_ratings_per_user": float(np.median(user_support)),
        "median_ratings_per_item": float(np.median(item_support)),
    }

    # -------------------------------------------------------- 2. BASELINE
    log("STAGE 2  bias baseline")
    base = BiasBaseline(lam_user=10.0, lam_item=5.0, n_epochs=10).fit(R_train)
    R_res = base.residual_matrix(R_train)
    log(f"         mu={base.mu_:.4f}  mean|b_u|={np.abs(base.b_u).mean():.3f}  "
        f"mean|b_i|={np.abs(base.b_i).mean():.3f}")

    # ----------------------------------------------------- 3. ITEM-ITEM CF
    log("STAGE 3  item-item collaborative filtering")
    itemcf = ItemItemCF(k_neighbors=args.neighbors,
                        max_user_history=args.max_user_history).fit(R_res)
    log(f"         similarity nnz={itemcf.S_.nnz:,} "
        f"({itemcf.S_.nnz / n_items:.0f} neighbours/item)")

    # ------------------------------------------------------ 4. EXPLICIT ALS
    log("STAGE 4  explicit ALS matrix factorisation (rating task)")
    mf = ALSMatrixFactorization(n_factors=args.factors, lam=0.06,
                                n_epochs=args.als_epochs).fit(R_res)

    # ------------------------------------------------------ 5. IMPLICIT ALS
    log("STAGE 5  implicit ALS (ranking task)")
    ials = ImplicitALS(n_factors=args.implicit_factors, lam=8.0, alpha=12.0,
                       n_epochs=args.implicit_epochs).fit(R_train, min_rating=3.5)

    # ---------------------------------------------------------- 6. CONTENT
    log("STAGE 6  content-based TF-IDF model")
    cb = ContentBasedRecommender()
    F = cb.build_item_features(catalogue)
    cb.build_user_profiles(R_train, user_means)
    log(f"         item-feature matrix {F.shape}, nnz={F.nnz:,}")

    # ------------------------------------- 7. RATING-PREDICTION EVALUATION
    log("STAGE 7  rating prediction: components + hybrid stacking")
    tu, ti, ty = sample_pairs(test, mapper, args.eval_pairs)
    vu, vi, vy = sample_pairs(val, mapper, args.eval_pairs, seed=7)

    preds = {"global_mean": np.full(len(ty), float(R_train.data.mean()),
                                    dtype=np.float32)}
    base_test = base.predict(tu, ti)
    preds["bias_baseline"] = base_test

    cf_res_test = itemcf.predict_residual(tu, ti)
    preds["item_item_cf"] = np.clip(base_test + cf_res_test, 0.5, 5.0)
    log("         item-item CF test predictions done")

    mf_res_test = mf.predict_residual(tu, ti)
    preds["matrix_factorization"] = np.clip(base_test + mf_res_test, 0.5, 5.0)

    cb_cos_test = cb.predict_scores(tu, ti)
    preds["content_based"] = np.clip(
        user_means[tu] + 2.0 * (cb_cos_test - cb_cos_test.mean()), 0.5, 5.0)

    base_val = base.predict(vu, vi)
    hyb = HybridRatingModel(alpha=1.0)
    X_val = hyb.make_features(base_val, itemcf.predict_residual(vu, vi),
                              mf.predict_residual(vu, vi), cb.predict_scores(vu, vi),
                              item_support[vi], user_support[vu],
                              base_val - user_means[vu])
    hyb.fit(X_val, vy)
    X_test = hyb.make_features(base_test, cf_res_test, mf_res_test, cb_cos_test,
                               item_support[ti], user_support[tu],
                               base_test - user_means[tu])
    preds["hybrid"] = hyb.predict(X_test)
    log("         stacking weights: "
        + ", ".join(f"{k}={v:+.3f}" for k, v in hyb.coef_.items()))

    metrics["rating_prediction"] = {n: E.rating_report(ty, p)
                                    for n, p in preds.items()}
    for n, m in metrics["rating_prediction"].items():
        log(f"         {n:22s} RMSE={m['rmse']:.4f} MAE={m['mae']:.4f} "
            f"acc+-0.5={m['acc_within_0.5']:.2%} acc+-1.0={m['acc_within_1.0']:.2%} "
            f"like-acc={m['like_accuracy']:.2%}")

    # ------------------------------------------------ 8. RANKING EVALUATION
    log("STAGE 8  top-N ranking: tuning on validation, scoring on test")
    ranker = HybridRanker()
    ranker.set_popularity(item_support, base.b_i)
    models = (ials, mf, itemcf, cb)

    val_liked = val[val["rating"] >= 4.0].groupby("userId")["movieId"].apply(list)
    rng = np.random.default_rng(1)
    vusers = rng.choice(val_liked.index.to_numpy(),
                        size=min(args.tune_users, len(val_liked)), replace=False)
    vcache, vrel, vseen = build_component_cache(vusers, val_liked, mapper,
                                                R_train, models, base, item_support)
    ranker.tune(vcache, vrel, vseen,
                report_fn=lambda r, rel, k: E.ranking_report(r, rel, k, n_items))
    metrics["hybrid_ranking_weights"] = ranker.w

    test_liked = test[test["rating"] >= 4.0].groupby("userId")["movieId"].apply(list)
    rng = np.random.default_rng(0)
    tusers = rng.choice(test_liked.index.to_numpy(),
                        size=min(args.eval_users, len(test_liked)), replace=False)
    tcache, trel, tseen = build_component_cache(tusers, test_liked, mapper,
                                                R_train, models, base, item_support)

    pop_scores = ranker.pop_
    rec = {n: {} for n in ["popularity", "content_based", "item_item_cf",
                           "matrix_factorization", "implicit_als", "hybrid"]}
    for u, comp in tcache:
        s_seen = tseen[u]

        def top(scores):
            s = scores.copy().astype(np.float32)
            s[s_seen] = -np.inf
            t = np.argpartition(-s, K)[:K]
            return t[np.argsort(-s[t])]

        rec["popularity"][u] = top(pop_scores)
        rec["content_based"][u] = top(comp["content"])
        rec["item_item_cf"][u] = top(comp["itemcf"])
        rec["matrix_factorization"][u] = top(comp["mf"])
        rec["implicit_als"][u] = top(comp["ials"])
        rec["hybrid"][u] = ranker.recommend(comp, len(s_seen), s_seen, top_n=K)

    metrics["ranking"] = {n: E.ranking_report(r, trel, K, n_items)
                          for n, r in rec.items()}
    for n, m in metrics["ranking"].items():
        log(f"         {n:22s} P@10={m['precision@10']:.4f} "
            f"R@10={m['recall@10']:.4f} NDCG@10={m['ndcg@10']:.4f} "
            f"MAP@10={m['map@10']:.4f} hit={m['hit_rate@10']:.3f} "
            f"cov={m['catalogue_coverage']:.2%}")

    # ------------------------------------------------- 9. COLD-START CHECK
    log("STAGE 9  cold-start check (content-based, no interaction history)")
    cold = {}
    title_to_idx = {t: i for t, i in zip(catalogue["title"], catalogue["item_idx"])}
    seeds = [t for t in ["Toy Story (1995)", "Matrix, The (1999)",
                         "Godfather, The (1972)", "Inception (2010)"]
             if t in title_to_idx]
    for t in seeds:
        idx = int(title_to_idx[t])
        prof = cb.cold_start_profile([idx])
        s = cb.score_for_profile(prof)
        s[idx] = -np.inf
        topi = np.argsort(-s)[:5]
        cold[t] = catalogue.loc[topi, "title"].tolist()
        log(f"         '{t}' -> {cold[t]}")
    metrics["cold_start_examples"] = cold

    # ---------------------------------------------------------- 10. SAVE
    log("STAGE 10 saving artifacts + plots")
    with open(os.path.join(args.outdir, "model.pkl"), "wb") as f:
        pickle.dump({"baseline": base, "itemcf": itemcf, "mf": mf, "ials": ials,
                     "content": cb, "hybrid_rating": hyb, "ranker": ranker,
                     "mapper": mapper, "catalogue": catalogue,
                     "user_means": user_means, "item_support": item_support,
                     "user_support": user_support, "R_train": R_train},
                    f, protocol=4)
    with open(os.path.join(args.resultsdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    try:
        make_plots(metrics, preds, ty, args.resultsdir, item_support)
    except Exception as exc:                      # plots are cosmetic, never fatal
        log(f"         plotting skipped: {exc}")

    log("PIPELINE COMPLETE")


def make_plots(metrics, preds, y_true, outdir, item_support):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    names = list(metrics["rating_prediction"].keys())
    rmses = [metrics["rating_prediction"][n]["rmse"] for n in names]
    ax[0, 0].barh(names, rmses, color="#4C72B0")
    ax[0, 0].set_xlabel("Test RMSE (lower is better)")
    ax[0, 0].set_title("Rating prediction accuracy")
    for i, v in enumerate(rmses):
        ax[0, 0].text(v, i, f" {v:.4f}", va="center", fontsize=8)

    rnames = list(metrics["ranking"].keys())
    ndcgs = [metrics["ranking"][n]["ndcg@10"] for n in rnames]
    ax[0, 1].barh(rnames, ndcgs, color="#DD8452")
    ax[0, 1].set_xlabel("NDCG@10 (higher is better)")
    ax[0, 1].set_title("Top-10 ranking quality")
    for i, v in enumerate(ndcgs):
        ax[0, 1].text(v, i, f" {v:.4f}", va="center", fontsize=8)

    err = np.abs(y_true - preds["hybrid"])
    ax[1, 0].hist(err, bins=40, color="#55A868")
    ax[1, 0].set_xlabel("|predicted - actual| (stars)")
    ax[1, 0].set_ylabel("test ratings")
    ax[1, 0].set_title("Hybrid absolute error distribution")

    ax[1, 1].loglog(np.sort(item_support)[::-1], color="#C44E52")
    ax[1, 1].set_xlabel("movie rank")
    ax[1, 1].set_ylabel("number of ratings")
    ax[1, 1].set_title("Long tail of the catalogue (why sparsity is the problem)")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "evaluation.png"), dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
