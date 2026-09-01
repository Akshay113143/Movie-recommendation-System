"""
Train the deep autoencoder and measure whether its embeddings actually help.

    python train_autoencoder.py

Three questions, answered with numbers rather than assertions:

  1. Does it reconstruct held-out ratings? (masked RMSE on unseen items)
  2. Are the embeddings semantically meaningful? (nearest neighbours by cosine
     in the 128-d space -- inspected qualitatively)
  3. Does adding it to the hybrid improve ranking? (NDCG@10 with and without
     the autoencoder component, on the same held-out users)

Question 3 is the one that matters. An embedding that looks nice in a
neighbour dump but does not move the metric is not a contribution, and saying
so is more useful than claiming an improvement that is not there.
"""

from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd

from recsys import data as D
from recsys import evaluate as E
from recsys.autoencoder import DeepAutoEncoder, build_anchor_matrix
from recsys.hybrid import zscore, quality_adjusted

MODEL = "artifacts/model.pkl"
OUT_NPZ = "artifacts/autoencoder.npz"
OUT_JSON = "results/autoencoder_metrics.json"
K = 10


def main():
    print("loading trained artifacts ...", flush=True)
    with open(MODEL, "rb") as f:
        M = pickle.load(f)
    R_train = M["R_train"]
    base, mf, ials, itemcf, cb = (M["baseline"], M["mf"], M["ials"],
                                  M["itemcf"], M["content"])
    mapper, catalogue = M["mapper"], M["catalogue"]
    item_support = M["item_support"]
    titles = catalogue["title"].to_numpy()

    # ------------------------------------------------------------ 1. train
    print("\nbuilding anchor matrix ...", flush=True)
    X, anchors, item_means = build_anchor_matrix(R_train, n_anchors=3000)
    print(f"   items x anchors = {X.shape}, nnz = {X.nnz:,} "
          f"({X.nnz / (X.shape[0] * X.shape[1]):.2%} dense)")

    print("\ntraining autoencoder (3000 -> 512 -> 128 -> 512 -> 3000) ...",
          flush=True)
    ae = DeepAutoEncoder(X.shape[1], hidden=512, bottleneck=128,
                         lr=1e-3, noise=0.25, l2=1e-5)
    ae.fit(X, n_epochs=20, batch_size=256, val_frac=0.05)

    emb = ae.encode(X)
    emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    print(f"\nembeddings: {emb.shape}, "
          f"mean L2 norm {np.linalg.norm(emb, axis=1).mean():.3f}")

    metrics = {
        "architecture": "3000 -> 512 -> 128 -> 512 -> 3000, ReLU, masked MSE",
        "n_anchor_users": int(len(anchors)),
        "n_parameters": int(sum(v.size for v in ae.p.values())),
        "train_rmse": float(ae.history_[-1][0]),
        "val_rmse": float(ae.history_[-1][1]),
        "history": [[float(a), float(b)] for a, b in ae.history_],
    }
    print(f"   parameters: {metrics['n_parameters']:,}")

    # -------------------------------------------------- 2. neighbour sanity
    print("\nnearest neighbours in the 128-d embedding space:")
    probes = ["Godfather, The (1972)", "Toy Story (1995)", "Matrix, The (1999)",
              "Pulp Fiction (1994)", "Shining, The (1980)"]
    t2i = {t: i for t, i in zip(catalogue["title"], catalogue["item_idx"])}
    nn_examples = {}
    for t in probes:
        if t not in t2i:
            continue
        i = int(t2i[t])
        sims = emb_n @ emb_n[i]
        sims[i] = -np.inf
        top = np.argsort(-sims)[:5]
        nn_examples[t] = [f"{titles[j]} ({sims[j]:.3f})" for j in top]
        print(f"   {t}")
        for j in top:
            print(f"      {sims[j]:.3f}  {titles[j]}")
    metrics["nearest_neighbours"] = nn_examples

    # ------------------------------------------- 3. does it help the hybrid?
    print("\nre-running ranking evaluation with the autoencoder component ...",
          flush=True)
    d = D.prepare("data/ratings.csv", "data/movies.csv", verbose=False)
    test = d["test"]
    test_liked = test[test["rating"] >= 4.0].groupby("userId")["movieId"].apply(list)
    rng = np.random.default_rng(0)
    users = rng.choice(test_liked.index.to_numpy(),
                       size=min(1500, len(test_liked)), replace=False)

    n_items = R_train.shape[1]
    pop = zscore(np.log1p(item_support) * (1.0 + base.b_i))
    W = M["ranker"].w

    relevant, rec_base, rec_ae, rec_aeonly = {}, {}, {}, {}
    for raw_u in users:
        u = mapper.user_to_idx[raw_u]
        rel = {mapper.item_to_idx[m] for m in test_liked.loc[raw_u]
               if m in mapper.item_to_idx}
        if not rel:
            continue
        relevant[u] = rel
        seen = R_train.indices[R_train.indptr[u]:R_train.indptr[u + 1]]
        vals = R_train.data[R_train.indptr[u]:R_train.indptr[u + 1]]

        s_ials = ials.score_all_items(u)
        s_mf = quality_adjusted(mf.score_all_items(u), base.b_i, item_support)
        s_cf = quality_adjusted(itemcf.score_all_items(u), base.b_i, item_support)
        s_cb = cb.score_all_items(u)

        # autoencoder user profile = rating-weighted mean of item embeddings
        w = np.maximum(vals - vals.mean(), 0.0)
        if w.sum() == 0:
            w = np.ones_like(vals)
        prof = (emb_n[seen] * w[:, None]).sum(0)
        prof /= (np.linalg.norm(prof) + 1e-8)
        s_ae = quality_adjusted(emb_n @ prof, base.b_i, item_support)

        def top(s):
            s = s.copy().astype(np.float32)
            s[seen] = -np.inf
            t = np.argpartition(-s, K)[:K]
            return t[np.argsort(-s[t])]

        blend = (W["ials"] * zscore(s_ials) + W["mf"] * zscore(s_mf)
                 + W["itemcf"] * zscore(s_cf) + W["content"] * zscore(s_cb)
                 + W["pop"] * pop)
        rec_base[u] = top(blend)
        rec_ae[u] = top(blend + 0.20 * zscore(s_ae))
        rec_aeonly[u] = top(s_ae)

    rep = lambda r: E.ranking_report(r, relevant, K, n_items)
    r_base, r_ae, r_only = rep(rec_base), rep(rec_ae), rep(rec_aeonly)
    metrics["ranking"] = {"hybrid_without_ae": r_base,
                          "hybrid_with_ae": r_ae,
                          "autoencoder_alone": r_only}

    print(f"\n   autoencoder alone     NDCG@10={r_only['ndcg@10']:.4f} "
          f"P@10={r_only['precision@10']:.4f}")
    print(f"   hybrid WITHOUT AE     NDCG@10={r_base['ndcg@10']:.4f} "
          f"P@10={r_base['precision@10']:.4f}")
    print(f"   hybrid WITH AE        NDCG@10={r_ae['ndcg@10']:.4f} "
          f"P@10={r_ae['precision@10']:.4f}")
    delta = (r_ae["ndcg@10"] - r_base["ndcg@10"]) / max(r_base["ndcg@10"], 1e-9)
    metrics["ndcg_lift_from_ae_pct"] = float(100 * delta)
    print(f"   -> NDCG@10 change from adding the autoencoder: {100 * delta:+.2f}%")

    # ---------------------------------------------------------- 4. persist
    os.makedirs("results", exist_ok=True)
    np.savez_compressed(OUT_NPZ, embeddings=emb.astype(np.float32),
                        embeddings_norm=emb_n.astype(np.float32),
                        anchors=anchors.astype(np.int32),
                        item_means=item_means.astype(np.float32),
                        **{f"p_{k}": v for k, v in ae.p.items()})
    with open(OUT_JSON, "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"\nwrote {OUT_NPZ} and {OUT_JSON}")


if __name__ == "__main__":
    main()
