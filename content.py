"""
Content-based filtering: learn from *item attributes*, never from other users.

The archive supplied contains only `movies.csv` (movieId, title, genres) and
`ratings.csv`, so the available content signal is:

    * genres      -- 20 tags, multi-label ('Action|Sci-Fi|Thriller')
    * release year / decade -- extracted from the title string
    * title tokens -- weak but real ('Star Wars: Episode V', 'Harry Potter and
      the ...' share franchise tokens with their sequels)

If you later add MovieLens `tags.csv` or a TMDB join (cast, director, plot
overview, keywords), the *only* thing that changes is `build_item_features` --
everything downstream keeps working, because every model here consumes an
item x feature sparse matrix and nothing else. That is the point of keeping
the feature builder isolated.

Why TF-IDF and not raw multi-hot?
    IDF = log(N / df) down-weights features that are everywhere. 'Drama' tags
    25,606 of 62,423 movies -- two films sharing 'Drama' tells you almost
    nothing. 'Film-Noir' tags 353 -- two films sharing it tells you a lot.
    Raw multi-hot treats both as 1.0; TF-IDF gives Film-Noir ~5x the weight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class ContentBasedRecommender:
    """TF-IDF content model with two query modes.

    Mode A -- item-to-item : 'more movies like Toy Story'  (works for a brand
              new movie with zero ratings -> solves item cold start)
    Mode B -- user-to-items: build a taste profile as the rating-weighted mean
              of the feature vectors of the movies the user liked, then rank
              the catalogue by cosine to that profile (works for a user with
              a single rating -> mitigates user cold start)
    """

    def __init__(self, genre_weight: float = 1.0, title_weight: float = 0.35,
                 decade_weight: float = 0.25, min_df: int = 2,
                 like_threshold: float = 0.0):
        self.genre_weight = genre_weight
        self.title_weight = title_weight
        self.decade_weight = decade_weight
        self.min_df = min_df
        self.like_threshold = like_threshold

    # ------------------------------------------------------------------ #
    def build_item_features(self, catalogue: pd.DataFrame) -> sparse.csr_matrix:
        """Build the item x feature TF-IDF matrix, aligned to matrix columns.

        Three blocks are built separately, each L2-normalised and scaled by its
        own weight, then horizontally stacked. Separate normalisation is what
        stops the 3,000-token title vocabulary from drowning out the 20-token
        genre vocabulary purely by having more columns.
        """
        genres_txt = (catalogue["genres"].fillna("(no genres listed)")
                      .str.replace("(no genres listed)", "nogenre", regex=False)
                      .str.replace("|", " ", regex=False)
                      .str.replace("-", "", regex=False)
                      .str.lower())
        self.genre_vec_ = TfidfVectorizer(token_pattern=r"[^\s]+")
        G = self.genre_vec_.fit_transform(genres_txt)

        titles = catalogue["clean_title"].fillna("").str.lower()
        self.title_vec_ = TfidfVectorizer(min_df=self.min_df,
                                          stop_words="english",
                                          token_pattern=r"[a-z][a-z0-9']+",
                                          sublinear_tf=True)
        T = self.title_vec_.fit_transform(titles)

        decade = (catalogue["year"].fillna(catalogue["year"].median())
                  // 10 * 10).astype(int).astype(str) + "s"
        self.decade_vec_ = TfidfVectorizer(token_pattern=r"[^\s]+")
        D = self.decade_vec_.fit_transform(decade)

        F = sparse.hstack([
            normalize(G) * self.genre_weight,
            normalize(T) * self.title_weight,
            normalize(D) * self.decade_weight,
        ]).tocsr().astype(np.float32)

        self.F_ = normalize(F).astype(np.float32)   # rows are unit vectors
        self.catalogue_ = catalogue
        self.feature_names_ = (list(self.genre_vec_.get_feature_names_out())
                               + list(self.title_vec_.get_feature_names_out())
                               + list(self.decade_vec_.get_feature_names_out()))
        return self.F_

    # ------------------------------------------------------------------ #
    def similar_items(self, item_idx: int, top_n: int = 10,
                      exclude_self: bool = True):
        """Cosine similarity of one item against the catalogue.

        Computed on demand as one sparse matrix-vector product instead of
        materialising the full 18k x 18k similarity matrix (1.4 GB dense).
        Cost is O(nnz) ~ a few milliseconds.
        """
        sims = (self.F_ @ self.F_[item_idx].T).toarray().ravel()
        if exclude_self:
            sims[item_idx] = -np.inf
        top = np.argpartition(-sims, top_n)[:top_n]
        top = top[np.argsort(-sims[top])]
        return top, sims[top]

    # ------------------------------------------------------------------ #
    def build_user_profiles(self, R_train: sparse.csr_matrix,
                            user_means: np.ndarray) -> sparse.csr_matrix:
        """One profile vector per user: rating-weighted mean of liked items.

            profile_u = sum_i w_ui * f_i,   w_ui = max(0, r_ui - mean_u)

        Centring on the user's own mean is essential: a user whose average is
        4.6 has not endorsed a movie by giving it 3.5, and a user who averages
        2.8 has. Negative weights are clipped to 0 -- we build a profile of
        what the user *likes*; dislike information is left to the CF models,
        where it is far more reliable.
        """
        R = R_train.tocsr()
        rows = np.repeat(np.arange(R.shape[0], dtype=np.int32), np.diff(R.indptr))
        w = R.data - user_means[rows]
        w = np.maximum(w, self.like_threshold)
        W = sparse.csr_matrix((w, R.indices, R.indptr), shape=R.shape)
        profiles = W @ self.F_                      # (n_users x n_features)
        self.profiles_ = normalize(profiles).astype(np.float32)
        return self.profiles_

    def score_all_items(self, user_idx: int) -> np.ndarray:
        """Cosine score of every catalogue item against the user's profile."""
        return (self.F_ @ self.profiles_[user_idx].T).toarray().ravel()

    def predict_scores(self, user_idx: np.ndarray,
                       item_idx: np.ndarray) -> np.ndarray:
        """Vectorised cosine(profile_u, f_i) for arbitrary (u, i) pairs."""
        return np.asarray(
            self.profiles_[user_idx].multiply(self.F_[item_idx]).sum(axis=1)
        ).ravel().astype(np.float32)

    # ------------------------------------------------------------------ #
    def cold_start_profile(self, liked_item_idx, ratings=None):
        """Profile for a brand-new user from a handful of liked movies.

        No rating history in the matrix is needed -- this is what lets the
        system respond on the very first interaction, which pure collaborative
        filtering structurally cannot do.
        """
        liked_item_idx = np.asarray(liked_item_idx, dtype=np.int32)
        if ratings is None:
            w = np.ones(len(liked_item_idx), dtype=np.float32)
        else:
            w = np.asarray(ratings, dtype=np.float32)
            w = np.maximum(w - w.mean() + 0.5, 0.05)
        prof = sparse.csr_matrix(w) @ self.F_[liked_item_idx]
        return normalize(prof).astype(np.float32)

    def score_for_profile(self, profile: sparse.csr_matrix) -> np.ndarray:
        return (self.F_ @ profile.T).toarray().ravel()
