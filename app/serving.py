"""
Serving logic for the deployed app -- no training code, no training data.

The interesting piece here is `fold_in`. A visitor to the deployed app has no
row in the training matrix, so there is no p_u to look up. Rather than
retraining, we solve the implicit-ALS normal equations for that one user while
holding the item factors Q fixed:

    A = QtQ + Qu^T (Cu - I) Qu + lam * I
    b = Qu^T cu
    p_u = A^-1 b

That is one 64x64 linear solve -- microseconds -- and it is mathematically the
same update the model would apply to this user during a training epoch. It is
how a production system serves someone who signed up an hour ago.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize


def zscore(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / (s + 1e-8)


def quality_adjusted(resid, item_bias, item_support, tau=50.0):
    """Add back item quality and shrink the personalised part by evidence."""
    return (item_bias + resid * (item_support / (item_support + tau))
            ).astype(np.float32)


class ServingModel:
    """Loads the exported bundle and answers recommendation queries."""

    def __init__(self, bundle_dir: str = "app/bundle"):
        d = np.load(f"{bundle_dir}/item_factors.npz")
        self.Q = d["ials_Q"]
        self.QtQ = d["ials_QtQ"]
        self.lam = float(d["ials_lam"])
        self.alpha = float(d["ials_alpha"])
        self.mf_Q = d["mf_Q"]
        self.item_bias = d["item_bias"]
        self.mu = float(d["mu"])
        self.item_support = d["item_support"]
        w = d["ranker_weights"]
        self.weights = dict(zip(("ials", "mf", "itemcf", "content", "pop"), w))

        self.S = sparse.load_npz(f"{bundle_dir}/item_similarity.npz").tocsr()
        self.F = sparse.load_npz(f"{bundle_dir}/content_features.npz").tocsr()
        self.catalogue = pd.read_csv(f"{bundle_dir}/catalogue.csv")

        # Autoencoder embeddings are optional: the app degrades to the other
        # three similarity signals if the file is absent.
        try:
            self.ae_emb = np.load(f"{bundle_dir}/ae_embeddings.npz")["emb"].astype(np.float32)
        except Exception:
            self.ae_emb = None

        self.titles = self.catalogue["title"].to_numpy()
        self.genres = self.catalogue["genres"].to_numpy()
        self.years = self.catalogue["year"].to_numpy()
        self.clean_titles = self.catalogue["clean_title"].to_numpy()
        self.pop = zscore(np.log1p(self.item_support) * (1.0 + self.item_bias))
        self.n_items = len(self.titles)

    # ------------------------------------------------------------------ #
    def search(self, query: str, limit: int = 12) -> list[int]:
        """Substring title search, most-rated matches first."""
        q = query.strip().lower()
        if not q:
            return []
        mask = pd.Series(self.titles).str.lower().str.contains(q, regex=False)
        idx = np.flatnonzero(mask.to_numpy())
        return idx[np.argsort(-self.item_support[idx])][:limit].tolist()

    # ------------------------------------------------------------------ #
    def fold_in(self, item_idx: np.ndarray, ratings: np.ndarray) -> np.ndarray:
        """Solve for one new user's implicit-ALS factor vector."""
        item_idx = np.asarray(item_idx, dtype=np.int64)
        r = np.asarray(ratings, dtype=np.float32) / 5.0
        Qu = self.Q[item_idx]
        c = 1.0 + self.alpha * r
        A = (self.QtQ + (Qu * (c - 1.0)[:, None]).T @ Qu
             + self.lam * np.eye(self.Q.shape[1], dtype=np.float32))
        b = Qu.T @ c
        return np.linalg.solve(A, b).astype(np.float32)

    # ------------------------------------------------------------------ #
    def recommend(self, item_idx, ratings, top_n: int = 12,
                  diversity: float = 0.0, min_year: int | None = None,
                  genre_filter: str | None = None) -> pd.DataFrame:
        """Hybrid top-N for a visitor described only by a few rated films."""
        item_idx = np.asarray(item_idx, dtype=np.int64)
        ratings = np.asarray(ratings, dtype=np.float32)

        # --- component 1: implicit ALS via fold-in ---------------------
        p_u = self.fold_in(item_idx, ratings)
        s_ials = self.Q @ p_u

        # --- component 2: explicit MF, quality-adjusted ----------------
        # least-squares fit of the user's residual taste on MF item factors
        resid = ratings - (self.mu + self.item_bias[item_idx])
        Qm = self.mf_Q[item_idx]
        A = Qm.T @ Qm + 0.06 * len(item_idx) * np.eye(Qm.shape[1], dtype=np.float32)
        p_mf = np.linalg.solve(A, Qm.T @ resid).astype(np.float32)
        s_mf = quality_adjusted(self.mf_Q @ p_mf, self.item_bias,
                                self.item_support)

        # --- component 3: item-item neighbourhood ----------------------
        w = np.maximum(resid, 0.0)
        s_cf_raw = np.asarray((self.S[:, item_idx] @ w)).ravel()
        denom = np.asarray(np.abs(self.S[:, item_idx]).sum(axis=1)).ravel() + 1e-8
        s_itemcf = quality_adjusted((s_cf_raw / denom).astype(np.float32),
                                    self.item_bias, self.item_support)

        # --- component 4: content profile ------------------------------
        liked = np.maximum(ratings - ratings.mean() + 0.5, 0.05)
        prof = normalize(sparse.csr_matrix(liked) @ self.F[item_idx])
        s_content = np.asarray((self.F @ prof.T).todense()).ravel()

        # --- fuse ------------------------------------------------------
        W = self.weights
        score = (W["ials"] * zscore(s_ials)
                 + W["mf"] * zscore(s_mf)
                 + W["itemcf"] * zscore(s_itemcf)
                 + W["content"] * zscore(s_content)
                 + W["pop"] * self.pop)

        # diversity slider: subtract popularity to push into the long tail
        if diversity > 0:
            score = score - diversity * self.pop

        score[item_idx] = -np.inf                      # never re-recommend inputs
        if min_year is not None:
            score[np.nan_to_num(self.years, nan=0) < min_year] = -np.inf
        if genre_filter and genre_filter != "Any":
            mask = ~pd.Series(self.genres).str.contains(
                genre_filter, regex=False, na=False).to_numpy()
            score[mask] = -np.inf

        n = min(top_n, int(np.isfinite(score).sum()))
        if n == 0:
            return pd.DataFrame()
        top = np.argpartition(-score, n - 1)[:n]
        top = top[np.argsort(-score[top])]

        return pd.DataFrame({
            "item_idx": top,
            "title": self.titles[top],
            "clean_title": self.clean_titles[top],
            "year": self.years[top],
            "genres": self.genres[top],
            "n_ratings": self.item_support[top].astype(int),
            "score": score[top],
            "why": [self._why(i, item_idx, s_itemcf, s_content) for i in top],
        })

    # ------------------------------------------------------------------ #
    def _why(self, i: int, seed_idx: np.ndarray, s_itemcf, s_content) -> str:
        """'Because you liked X' -- the strongest seed by item-item similarity."""
        sims = np.asarray(self.S[i, seed_idx].todense()).ravel()
        if sims.max() > 0:
            return f"because you liked {self.clean_titles[seed_idx[int(sims.argmax())]]}"
        g = set(str(self.genres[i]).split("|"))
        seed_g = [x for j in seed_idx for x in str(self.genres[j]).split("|")]
        shared = [x for x in pd.Series(seed_g).value_counts().index[:4] if x in g]
        if shared:
            return "matches your taste for " + "/".join(shared[:2])
        return "highly rated by users with similar taste"

    # ------------------------------------------------------------------ #
    def similar(self, item_idx: int, top_n: int = 12,
                mode: str = "hybrid") -> pd.DataFrame:
        """More-like-this. mode = content | collaborative | neural | hybrid."""
        content = np.asarray((self.F @ self.F[item_idx].T).todense()).ravel()
        collab = np.asarray(self.S[item_idx].todense()).ravel()
        neural = (self.ae_emb @ self.ae_emb[item_idx]
                  if self.ae_emb is not None else np.zeros_like(content))
        if mode == "content":
            score = content
        elif mode == "collaborative":
            score = collab
        elif mode == "neural":
            score = neural
        else:
            score = 0.5 * zscore(content) + 0.5 * zscore(collab)
        score[item_idx] = -np.inf
        top = np.argpartition(-score, top_n)[:top_n]
        top = top[np.argsort(-score[top])]
        return pd.DataFrame({
            "item_idx": top,
            "title": self.titles[top],
            "clean_title": self.clean_titles[top],
            "year": self.years[top],
            "genres": self.genres[top],
            "n_ratings": self.item_support[top].astype(int),
            "content_sim": content[top],
            "collab_sim": collab[top],
            "neural_sim": neural[top],
        })

    def popular(self, n: int = 12) -> list[int]:
        return np.argsort(-self.pop)[:n].tolist()
