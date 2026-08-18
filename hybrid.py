"""
The augmented / hybrid layer.

Collaborative filtering and content-based filtering fail in *opposite*
situations, which is exactly why combining them works:

    Collaborative filtering        Content-based filtering
    ---------------------------    -------------------------------------
    needs many ratings per item    works on a movie released this morning
    cannot explain 'why'           trivially explainable ('same director')
    finds surprising cross-genre   stuck inside the user's known genres
      taste ('serendipity')          ('filter bubble' / over-specialisation)
    cold start = fatal             cold start = fine

Two fusion strategies are implemented, because the two tasks need different
machinery:

1. `HybridRatingModel`  -- STACKING (a.k.a. blending / feature-weighted
   linear stacking). The component predictions become *features* of a small
   ridge regression that is fitted on the validation split. The meta-model
   learns the weights from data instead of us guessing them, and it can learn
   an intercept correction too.

2. `HybridRanker`       -- WEIGHTED SCORE FUSION with per-user *switching*.
   Component scores live on incomparable scales (a latent-factor dot product
   vs a cosine in [0,1]), so each is z-scored per user before mixing. Weights
   are chosen on validation NDCG@10. When a user has fewer than
   `cold_start_threshold` ratings, weight shifts to the content component --
   this is the 'switching hybrid' pattern from Burke's taxonomy.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge


# --------------------------------------------------------------------------- #
def zscore(x: np.ndarray) -> np.ndarray:
    """Standardise a score vector so different models can be added together."""
    s = x.std()
    return (x - x.mean()) / (s + 1e-8)


def quality_adjusted(resid_scores: np.ndarray, item_bias: np.ndarray,
                     item_support: np.ndarray, tau: float = 50.0,
                     bias_weight: float = 1.0) -> np.ndarray:
    """Turn a *residual* score into a rankable score.

    Two corrections, both of which matter enormously for top-N quality and
    neither of which shows up in RMSE:

    1. ADD BACK THE ITEM BIAS. RMSE is measured only on movies the user chose
       to watch, so the model is never asked 'is this movie any good?'. Ranking
       asks exactly that over the whole catalogue. A residual-only ranking
       happily puts a mediocre film at rank 1 because the personalised part is
       +0.4, ignoring that its item bias is -0.9.

    2. SHRINK THE PERSONALISED PART BY SUPPORT: resid * n_i / (n_i + tau).
       A movie with 21 ratings has a latent-factor vector fitted from almost no
       evidence, so its dot product is high-variance noise. Uncorrected, top-N
       lists fill up with obscure items that happened to get a lucky factor --
       high novelty, terrible precision. This is an empirical-Bayes style
       shrink toward the population mean.
    """
    conf = item_support / (item_support + tau)
    return (bias_weight * item_bias + resid_scores * conf).astype(np.float32)


class HybridRatingModel:
    """Ridge stacking over baseline + item-item CF + MF + content signals."""

    FEATURES = ["baseline", "itemcf_resid", "mf_resid", "content_cos",
                "log_item_support", "log_user_support", "user_mean_dev"]

    def __init__(self, alpha: float = 1.0, clip=(0.5, 5.0)):
        self.alpha = alpha
        self.clip = clip

    def make_features(self, baseline_pred, itemcf_resid, mf_resid,
                      content_cos, item_support, user_support,
                      user_mean_dev) -> np.ndarray:
        """Assemble the meta-feature matrix (one row per (u,i) pair).

        Support counts enter in log space because their effect is
        multiplicative, not additive: going from 20 to 200 ratings changes
        confidence far more than 20,000 to 20,180.
        """
        return np.column_stack([
            baseline_pred,
            itemcf_resid,
            mf_resid,
            content_cos,
            np.log1p(item_support),
            np.log1p(user_support),
            user_mean_dev,
        ]).astype(np.float32)

    def fit(self, X_val: np.ndarray, y_val: np.ndarray) -> "HybridRatingModel":
        self.model_ = Ridge(alpha=self.alpha, fit_intercept=True)
        self.model_.fit(X_val, y_val)
        self.coef_ = dict(zip(self.FEATURES, self.model_.coef_))
        self.intercept_ = float(self.model_.intercept_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.clip(self.model_.predict(X), *self.clip).astype(np.float32)


# --------------------------------------------------------------------------- #
class HybridRanker:
    """Weighted + switching fusion of four rankers, tuned on validation NDCG.

    Components
        ials     -- implicit ALS               (main personalisation engine)
        mf       -- explicit ALS, quality-adjusted (rating-quality signal)
        itemcf   -- item-item neighbourhood     (local co-visitation signal)
        content  -- TF-IDF profile cosine       (cold start + explainability)
        pop      -- log-popularity x item bias  (prior / tie-break)

    Every component is z-scored **per user over the catalogue** before mixing.
    Without that step the mix is meaningless: the iALS score is roughly in
    [0, 1.2], the quality-adjusted MF score in [-1.5, 1.0] and the content
    cosine in [0, 0.9], so a naive sum is just 'whichever component happens to
    have the largest numeric range wins'.
    """

    COMPONENTS = ("ials", "mf", "itemcf", "content", "pop")

    def __init__(self, w_ials: float = 0.65, w_mf: float = 0.10,
                 w_itemcf: float = 0.15, w_content: float = 0.10,
                 w_pop: float = 0.15, cold_start_threshold: int = 10):
        self.w = {"ials": w_ials, "mf": w_mf, "itemcf": w_itemcf,
                  "content": w_content, "pop": w_pop}
        self.cold_start_threshold = cold_start_threshold

    def set_popularity(self, item_support: np.ndarray, item_bias: np.ndarray):
        """Popularity prior = log(support) x (1 + learned item bias).

        Widely-seen AND well-liked. Log because popularity is Zipfian: the gap
        between 100 and 1,000 ratings means far more than 30,000 to 31,000.
        Kept at a modest weight on purpose -- push it up and the system
        collapses into a global top-100 chart that is the same for everyone.
        """
        self.pop_ = zscore(np.log1p(item_support) * (1.0 + item_bias))

    # ------------------------------------------------------------------ #
    def weights_for_user(self, n_ratings: int) -> dict:
        """Switching rule: with a thin history, lean on content + popularity.

        A user with 3 ratings has latent factors fitted from 3 observations --
        essentially noise. Their TF-IDF taste profile from those same 3 movies
        is far more reliable, because it inherits structure (genre, era) that
        was estimated from the whole catalogue rather than from their history.
        """
        if n_ratings >= self.cold_start_threshold:
            return dict(self.w)
        t = n_ratings / max(self.cold_start_threshold, 1)
        w = dict(self.w)
        w["content"] = self.w["content"] + (1 - t) * (0.45 - self.w["content"])
        w["pop"] = self.w["pop"] + (1 - t) * (0.30 - self.w["pop"])
        rest = 1.0 - w["content"] - w["pop"]
        base_rest = self.w["ials"] + self.w["mf"] + self.w["itemcf"]
        scale = rest / max(base_rest, 1e-8)
        for k in ("ials", "mf", "itemcf"):
            w[k] = self.w[k] * scale
        return w

    def score(self, comp: dict, n_ratings: int,
              seen_items: np.ndarray | None = None) -> np.ndarray:
        w = self.weights_for_user(n_ratings)
        s = (w["ials"] * zscore(comp["ials"])
             + w["mf"] * zscore(comp["mf"])
             + w["itemcf"] * zscore(comp["itemcf"])
             + w["content"] * zscore(comp["content"])
             + w["pop"] * self.pop_)
        if seen_items is not None and len(seen_items):
            s[seen_items] = -np.inf          # never re-recommend a watched film
        return s

    def recommend(self, comp: dict, n_ratings: int, seen_items=None,
                  top_n: int = 10) -> np.ndarray:
        s = self.score(comp, n_ratings, seen_items)
        top = np.argpartition(-s, top_n)[:top_n]
        return top[np.argsort(-s[top])]

    # ------------------------------------------------------------------ #
    def tune(self, comp_cache: list, relevant: dict, seen: dict,
             grid: list | None = None, k: int = 10, metric: str = "ndcg@10",
             report_fn=None, verbose: bool = True) -> "HybridRanker":
        """Grid-search the fusion weights on the *validation* split.

        Tuning on validation and reporting on test is not a formality here: the
        weights are 5 free parameters chosen by looking at ranking quality, so
        picking them on test would inflate the headline numbers.
        """
        if grid is None:
            grid = [(1.0, 0, 0, 0, 0), (0, 0, 0, 0, 1.0),
                    (0.70, 0.15, 0.10, 0.05, 0.00),
                    (0.65, 0.10, 0.15, 0.10, 0.15),
                    (0.60, 0.15, 0.15, 0.10, 0.10),
                    (0.55, 0.20, 0.15, 0.10, 0.20),
                    (0.50, 0.20, 0.20, 0.10, 0.30),
                    (0.45, 0.15, 0.20, 0.20, 0.20),
                    (0.80, 0.05, 0.10, 0.05, 0.10)]
        best, best_score = None, -np.inf
        for g in grid:
            self.w = dict(zip(self.COMPONENTS, g))
            recs = {}
            for u, comp in comp_cache:
                recs[u] = self.recommend(comp, len(seen[u]), seen[u], top_n=k)
            score = report_fn(recs, relevant, k)[metric]
            if verbose:
                print(f"   [tune] w={g} -> {metric}={score:.4f}", flush=True)
            if score > best_score:
                best, best_score = g, score
        self.w = dict(zip(self.COMPONENTS, best))
        self.tuned_score_ = best_score
        if verbose:
            print(f"   [tune] best weights {self.w} ({metric}={best_score:.4f})")
        return self
