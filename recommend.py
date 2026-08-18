"""
Serving layer -- load the trained artifacts and actually recommend something.

    python recommend.py --user 12345                    # existing user, top 10
    python recommend.py --similar "Toy Story (1995)"    # item-to-item
    python recommend.py --new-user "Inception (2010)" "The Dark Knight (2008)"
    python recommend.py --explain --user 12345          # with reasons
    python recommend.py --demo                          # runs all of the above

This is deliberately separate from `run_pipeline.py`: training is a batch job
that runs nightly, serving is a low-latency lookup. Everything expensive
(factors, similarity lists, TF-IDF profiles) was precomputed at training time,
so a request here is a handful of sparse products -- single-digit milliseconds.
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from recsys.hybrid import zscore, quality_adjusted


class RecommenderService:
    def __init__(self, model_path: str = "artifacts/model.pkl"):
        with open(model_path, "rb") as f:
            M = pickle.load(f)
        self.base = M["baseline"]
        self.itemcf = M["itemcf"]
        self.mf = M["mf"]
        self.ials = M["ials"]
        self.cb = M["content"]
        self.hyb = M["hybrid_rating"]
        self.ranker = M["ranker"]
        self.mapper = M["mapper"]
        self.cat = M["catalogue"]
        self.user_means = M["user_means"]
        self.item_support = M["item_support"]
        self.R = M["R_train"]
        self.titles = self.cat["title"].to_numpy()
        self._title_lookup = {t.lower(): i for i, t in enumerate(self.titles)}

    # ------------------------------------------------------------------ #
    def find_title(self, query: str) -> int:
        """Exact match first, then case-insensitive substring."""
        q = query.strip().lower()
        if q in self._title_lookup:
            return self._title_lookup[q]
        hits = [i for t, i in self._title_lookup.items() if q in t]
        if not hits:
            raise KeyError(f"No movie matching '{query}'")
        # prefer the most-rated match ('Batman' -> the 1989 film, not an obscure one)
        return max(hits, key=lambda i: self.item_support[i])

    def _components(self, u: int) -> dict:
        return {
            "ials": self.ials.score_all_items(u),
            "mf": quality_adjusted(self.mf.score_all_items(u), self.base.b_i,
                                   self.item_support, 50.0),
            "itemcf": quality_adjusted(self.itemcf.score_all_items(u),
                                       self.base.b_i, self.item_support, 50.0),
            "content": self.cb.score_all_items(u),
        }

    # ------------------------------------------------------------------ #
    def recommend_for_user(self, raw_user_id: int, top_n: int = 10,
                           explain: bool = False) -> pd.DataFrame:
        """Hybrid top-N for a user who exists in the training matrix."""
        if raw_user_id not in self.mapper.user_to_idx:
            raise KeyError(f"user {raw_user_id} not in the trained model")
        u = self.mapper.user_to_idx[raw_user_id]
        seen = self.R.indices[self.R.indptr[u]:self.R.indptr[u + 1]]
        comp = self._components(u)
        top = self.ranker.recommend(comp, len(seen), seen, top_n=top_n)

        pred = np.clip(self.base.mu_ + self.base.b_u[u] + self.base.b_i[top]
                       + self.mf.predict_residual(np.full(len(top), u), top),
                       0.5, 5.0)
        out = pd.DataFrame({
            "rank": np.arange(1, len(top) + 1),
            "title": self.titles[top],
            "genres": self.cat.loc[top, "genres"].to_numpy(),
            "pred_rating": np.round(pred, 2),
            "n_ratings": self.item_support[top].astype(int),
        })
        if explain:
            out["why"] = [self._explain(u, i, seen) for i in top]
        return out

    def _explain(self, u: int, i: int, seen: np.ndarray) -> str:
        """Cheap post-hoc explanation: strongest neighbour + shared genres.

        Honest framing: the hybrid score is not literally computed from this
        sentence -- latent factors are not human-readable. This finds the
        watched movie most similar to the recommendation, which is the
        standard 'because you watched X' pattern, and it is faithful in the
        sense that removing X from the history would measurably lower the
        item-CF component.
        """
        S = self.itemcf.S_csc_[:, seen]
        row = S[i].toarray().ravel() if S[i].nnz else np.zeros(len(seen))
        parts = []
        if row.max() > 0:
            parts.append(f"because you watched '{self.titles[seen[int(row.argmax())]]}'")
        g_rec = set(str(self.cat.loc[i, "genres"]).split("|"))
        top_seen = seen[np.argsort(-self.R.data[self.R.indptr[u]:self.R.indptr[u + 1]])[:20]]
        g_user = pd.Series([g for j in top_seen
                            for g in str(self.cat.loc[j, "genres"]).split("|")])
        shared = [g for g in g_user.value_counts().index[:5] if g in g_rec]
        if shared:
            parts.append("matches your taste for " + "/".join(shared[:2]))
        return "; ".join(parts) if parts else "popular among users like you"

    # ------------------------------------------------------------------ #
    def similar_movies(self, query: str, top_n: int = 10,
                       mode: str = "hybrid") -> pd.DataFrame:
        """'More like this'. mode = content | collaborative | hybrid.

        Content similarity generalises to movies nobody has rated yet;
        collaborative similarity captures things no metadata field encodes
        (that fans of 'Alien' also love 'The Thing' despite different genre
        tags). The hybrid takes both.
        """
        i = self.find_title(query)
        c_idx, c_sim = self.cb.similar_items(i, top_n=200)
        content = np.zeros(len(self.titles), dtype=np.float32)
        content[c_idx] = c_sim

        collab = self.itemcf.S_[i].toarray().ravel()

        if mode == "content":
            score = content
        elif mode == "collaborative":
            score = collab
        else:
            score = 0.5 * zscore(content) + 0.5 * zscore(collab)
        score[i] = -np.inf
        top = np.argsort(-score)[:top_n]
        return pd.DataFrame({
            "matched": self.titles[i],
            "rank": np.arange(1, top_n + 1),
            "title": self.titles[top],
            "genres": self.cat.loc[top, "genres"].to_numpy(),
            "content_sim": np.round(content[top], 3),
            "collab_sim": np.round(collab[top], 3),
        })

    # ------------------------------------------------------------------ #
    def recommend_for_new_user(self, liked_titles: list[str],
                               top_n: int = 10) -> pd.DataFrame:
        """Cold start: a user with no row in the matrix, only a few likes.

        Pure collaborative filtering cannot answer this at all -- there is no
        p_u to look up. The content model can, because a taste profile is just
        a weighted sum of item feature vectors. A small popularity prior is
        mixed in to stop the list filling with obscure genre-matches.
        """
        idxs = [self.find_title(t) for t in liked_titles]
        prof = self.cb.cold_start_profile(idxs)
        score = zscore(self.cb.score_for_profile(prof)) + 0.35 * self.ranker.pop_
        score[idxs] = -np.inf
        top = np.argsort(-score)[:top_n]
        return pd.DataFrame({
            "rank": np.arange(1, top_n + 1),
            "title": self.titles[top],
            "genres": self.cat.loc[top, "genres"].to_numpy(),
            "n_ratings": self.item_support[top].astype(int),
        })

    # ------------------------------------------------------------------ #
    def user_profile_summary(self, raw_user_id: int, top_n: int = 8) -> pd.DataFrame:
        """What the system thinks this user likes (for sanity-checking)."""
        u = self.mapper.user_to_idx[raw_user_id]
        s, e = self.R.indptr[u], self.R.indptr[u + 1]
        items, ratings = self.R.indices[s:e], self.R.data[s:e]
        order = np.argsort(-ratings)[:top_n]
        return pd.DataFrame({
            "title": self.titles[items[order]],
            "genres": self.cat.loc[items[order], "genres"].to_numpy(),
            "user_rating": ratings[order],
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/model.pkl")
    ap.add_argument("--user", type=int)
    ap.add_argument("--similar")
    ap.add_argument("--new-user", nargs="+")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    svc = RecommenderService(args.model)
    pd.set_option("display.width", 200, "display.max_colwidth", 55)

    if args.user:
        print(f"\n=== profile of user {args.user} (their top-rated films) ===")
        print(svc.user_profile_summary(args.user).to_string(index=False))
        print(f"\n=== hybrid recommendations for user {args.user} ===")
        print(svc.recommend_for_user(args.user, args.top_n,
                                     explain=args.explain).to_string(index=False))
    if args.similar:
        print(f"\n=== movies similar to '{args.similar}' ===")
        print(svc.similar_movies(args.similar, args.top_n).to_string(index=False))
    if args.new_user:
        print(f"\n=== cold-start recommendations from {args.new_user} ===")
        print(svc.recommend_for_new_user(args.new_user, args.top_n).to_string(index=False))

    if args.demo:
        uid = int(svc.mapper.user_ids[np.argsort(-np.diff(svc.R.indptr))[500]])
        print(f"\n=== DEMO: existing user {uid} ===")
        print(svc.user_profile_summary(uid).to_string(index=False))
        print(svc.recommend_for_user(uid, 10, explain=True).to_string(index=False))
        for t in ["Toy Story (1995)", "Matrix, The (1999)", "Godfather, The (1972)"]:
            print(f"\n=== similar to {t} ===")
            print(svc.similar_movies(t, 8).to_string(index=False))
        print("\n=== cold start: user who only told us 2 films they love ===")
        print(svc.recommend_for_new_user(["Inception (2010)", "Interstellar (2014)"],
                                         10).to_string(index=False))


if __name__ == "__main__":
    main()
