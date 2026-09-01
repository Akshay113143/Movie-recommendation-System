"""
Export a deployment-sized bundle from the trained model.

    python export_serving_bundle.py

Why this exists: `artifacts/model.pkl` is ~586 MB, mostly the 20M-non-zero
training matrix and the 162,414 x 64 user-factor matrix. GitHub rejects files
over 100 MB and Streamlit Community Cloud gives you ~1 GB of RAM, so neither
can host it.

The key realisation is that **the user factors do not need to be shipped at
all.** In the deployed app the visitor is not a MovieLens user -- they pick a
few films they like, and we solve for their factor vector on the fly
(`fold-in`, see `app/serving.py`). Everything else needed at serving time is
item-side and small:

    iALS item factors  18,430 x 64  float32   ~4.7 MB
    MF item factors    18,430 x 40  float32   ~2.9 MB
    item-item top-100 similarity (sparse)     ~15  MB
    TF-IDF item features (sparse)             ~2   MB
    item bias, support, catalogue metadata    ~1   MB

Result: ~25 MB, comfortably inside every limit, and the app loads in seconds.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
from scipy import sparse

SRC = "artifacts/model.pkl"
DST_DIR = "app/bundle"


def main():
    os.makedirs(DST_DIR, exist_ok=True)
    print(f"loading {SRC} ...", flush=True)
    with open(SRC, "rb") as f:
        M = pickle.load(f)

    ials, mf, base, itemcf, cb = (M["ials"], M["mf"], M["baseline"],
                                  M["itemcf"], M["content"])
    catalogue = M["catalogue"]

    # ---- dense item-side arrays -------------------------------------- #
    np.savez_compressed(
        os.path.join(DST_DIR, "item_factors.npz"),
        ials_Q=ials.Q.astype(np.float32),
        ials_QtQ=(ials.Q.T @ ials.Q).astype(np.float32),   # precomputed for fold-in
        ials_lam=np.float32(ials.lam),
        ials_alpha=np.float32(ials.alpha),
        mf_Q=mf.Q.astype(np.float32),
        item_bias=base.b_i.astype(np.float32),
        mu=np.float32(base.mu_),
        item_support=M["item_support"].astype(np.float32),
        ranker_weights=np.array([M["ranker"].w[k] for k in
                                 ("ials", "mf", "itemcf", "content", "pop")],
                                dtype=np.float32),
    )

    # ---- autoencoder embeddings (optional) ---------------------------- #
    # float16 halves the file (9.4 MB -> 4.7 MB) and costs nothing: these are
    # only used for cosine similarity, where 3 decimal places is plenty.
    if os.path.exists("artifacts/autoencoder.npz"):
        A = np.load("artifacts/autoencoder.npz")
        np.savez_compressed(os.path.join(DST_DIR, "ae_embeddings.npz"),
                            emb=A["embeddings_norm"].astype(np.float16))
        print("   included autoencoder embeddings")

    # ---- sparse matrices --------------------------------------------- #
    sparse.save_npz(os.path.join(DST_DIR, "item_similarity.npz"),
                    itemcf.S_.astype(np.float32))
    sparse.save_npz(os.path.join(DST_DIR, "content_features.npz"),
                    cb.F_.astype(np.float32))

    # ---- catalogue metadata ------------------------------------------ #
    cols = ["movieId", "title", "clean_title", "year", "genres", "item_idx"]
    # CSV, not parquet: avoids a pyarrow dependency (~90 MB wheel) in the
    # deployed app for a file that is only 1 MB either way.
    catalogue[cols].to_csv(os.path.join(DST_DIR, "catalogue.csv"), index=False)

    total = sum(os.path.getsize(os.path.join(DST_DIR, f))
                for f in os.listdir(DST_DIR))
    print(f"\nwrote {DST_DIR}/  ({total / 1e6:.1f} MB total)")
    for f in sorted(os.listdir(DST_DIR)):
        print(f"   {f:28s} {os.path.getsize(os.path.join(DST_DIR, f)) / 1e6:6.2f} MB")
    print(f"\nsource model.pkl was {os.path.getsize(SRC) / 1e6:.0f} MB "
          f"-- {os.path.getsize(SRC) / total:.0f}x reduction")


if __name__ == "__main__":
    main()
