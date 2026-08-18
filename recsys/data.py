"""
Data layer for the Movie Recommendation System.

Responsibilities
----------------
1. Load the raw MovieLens files (ratings.csv, movies.csv) memory-efficiently.
2. Apply support filtering (drop very-rarely-rated movies) so that the
   collaborative signal is statistically meaningful.
3. Re-index userId / movieId (which are sparse, arbitrary integers) into dense
   0..n-1 row/column positions, which is what every matrix operation needs.
4. Produce a *chronological* per-user train / validation / test split, which is
   the honest way to evaluate a recommender (you predict the future, not a
   random hole in the past).
5. Build the sparse user x item rating matrix used by the CF models.
"""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd
from scipy import sparse


# --------------------------------------------------------------------------- #
# 1. Loading
# --------------------------------------------------------------------------- #
def load_ratings(path: str) -> pd.DataFrame:
    """Load ratings.csv with explicit narrow dtypes.

    25M rows x 4 columns as int64/float64 would be ~800 MB. Declaring
    int32/float32 halves that. `timestamp` stays int64 only for readability of
    date conversion; it is dropped after the split.
    """
    return pd.read_csv(
        path,
        dtype={"userId": "int32", "movieId": "int32",
               "rating": "float32", "timestamp": "int64"},
    )


def load_movies(path: str) -> pd.DataFrame:
    """Load movies.csv and derive two extra content features from the title.

    MovieLens encodes the release year inside the title string, e.g.
    'Toy Story (1995)'. We pull it out because release decade is a genuine
    content signal (a 1940s noir and a 2015 noir are not interchangeable),
    and we strip it from the title so title tokens stay clean.
    """
    movies = pd.read_csv(path)
    year = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0]
    movies["year"] = pd.to_numeric(year, errors="coerce")
    movies["clean_title"] = (movies["title"]
                             .str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
                             .str.strip())
    movies["genres"] = movies["genres"].fillna("(no genres listed)")
    return movies


# --------------------------------------------------------------------------- #
# 2. Filtering + re-indexing
# --------------------------------------------------------------------------- #
def filter_by_support(ratings: pd.DataFrame,
                      min_movie_ratings: int = 20,
                      min_user_ratings: int = 20) -> pd.DataFrame:
    """Keep only movies/users with enough interactions.

    Why: a movie with 2 ratings gives a co-occurrence similarity that is pure
    noise, and it inflates the item space (59k -> 18k columns here) for no
    predictive gain. MovieLens 25M already guarantees >= 20 ratings per user,
    so the user filter is a no-op safeguard for other datasets.
    """
    movie_counts = ratings.groupby("movieId")["rating"].transform("size")
    ratings = ratings[movie_counts >= min_movie_ratings]
    user_counts = ratings.groupby("userId")["rating"].transform("size")
    ratings = ratings[user_counts >= min_user_ratings]
    return ratings.reset_index(drop=True)


class IndexMapper:
    """Bidirectional map between raw MovieLens IDs and dense matrix positions.

    A CSR matrix needs columns 0..n_items-1. MovieLens movieIds go up to
    209,171 with huge gaps, so using them directly would create a matrix with
    ~190k mostly-empty columns. This class stores the mapping both ways so we
    can go back to human-readable IDs when we print recommendations.
    """

    def __init__(self, user_ids: np.ndarray, item_ids: np.ndarray):
        self.user_ids = np.sort(np.unique(user_ids))
        self.item_ids = np.sort(np.unique(item_ids))
        self.user_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.item_to_idx = {m: i for i, m in enumerate(self.item_ids)}

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def map_users(self, ids: pd.Series) -> np.ndarray:
        return ids.map(self.user_to_idx).to_numpy(dtype=np.int32)

    def map_items(self, ids: pd.Series) -> np.ndarray:
        return ids.map(self.item_to_idx).to_numpy(dtype=np.int32)


# --------------------------------------------------------------------------- #
# 3. Chronological split
# --------------------------------------------------------------------------- #
def chronological_split(ratings: pd.DataFrame,
                        test_frac: float = 0.10,
                        val_frac: float = 0.10,
                        seed: int = 42):
    """Per-user time-ordered split: oldest -> train, newest -> test.

    For each user we sort their ratings by timestamp and cut the last
    `test_frac` into test, the chunk before it into validation, the rest into
    train. Every user therefore appears in train (no cold-start users leaking
    into test), and no future information is used to predict the past.

    A random split would leak: if a user rated 'The Two Towers' in 2003 and
    'Return of the King' in 2004, a random split can put the 2004 rating in
    train and the 2003 one in test, letting the model "predict" a rating it
    already saw the sequel of. Chronological splitting removes that.
    """
    ratings = ratings.sort_values(["userId", "timestamp"], kind="mergesort",
                                  ignore_index=True)
    grp = ratings.groupby("userId", sort=False)["rating"]
    n_per_user = grp.transform("size").to_numpy()
    rank = grp.cumcount().to_numpy()              # 0-based position in time

    n_test = np.maximum(1, np.floor(n_per_user * test_frac)).astype(np.int64)
    n_val = np.maximum(1, np.floor(n_per_user * val_frac)).astype(np.int64)

    is_test = rank >= (n_per_user - n_test)
    is_val = (~is_test) & (rank >= (n_per_user - n_test - n_val))
    is_train = ~(is_test | is_val)

    rng = np.random.default_rng(seed)             # only used for tie-breaking
    _ = rng.random(1)

    # timestamp has done its job; dropping it here saves ~200 MB on 25M rows
    cols = ["userId", "movieId", "rating"]
    return (ratings.loc[is_train, cols].reset_index(drop=True),
            ratings.loc[is_val, cols].reset_index(drop=True),
            ratings.loc[is_test, cols].reset_index(drop=True))


# --------------------------------------------------------------------------- #
# 4. Sparse matrix construction
# --------------------------------------------------------------------------- #
def build_sparse_matrix(df: pd.DataFrame, mapper: IndexMapper) -> sparse.csr_matrix:
    """Build the user x item rating matrix R in CSR format.

    Density check for MovieLens 25M: 25M observed cells out of
    162,541 x 18,430 = 3.0 billion possible cells -> ~0.8% dense. Storing that
    densely as float32 would need 12 GB; CSR needs ~300 MB.
    """
    rows = mapper.map_users(df["userId"])
    cols = mapper.map_items(df["movieId"])
    vals = df["rating"].to_numpy(dtype=np.float32)
    R = sparse.csr_matrix((vals, (rows, cols)),
                          shape=(mapper.n_users, mapper.n_items),
                          dtype=np.float32)
    R.sum_duplicates()
    return R


def prepare(ratings_path: str,
            movies_path: str,
            min_movie_ratings: int = 20,
            test_frac: float = 0.10,
            val_frac: float = 0.10,
            verbose: bool = True):
    """End-to-end data preparation. Returns everything downstream code needs."""
    ratings = load_ratings(ratings_path)
    movies = load_movies(movies_path)
    if verbose:
        print(f"[data] raw ratings           : {len(ratings):,}")

    ratings = filter_by_support(ratings, min_movie_ratings=min_movie_ratings)
    if verbose:
        print(f"[data] after support filter  : {len(ratings):,} "
              f"({ratings.movieId.nunique():,} movies, "
              f"{ratings.userId.nunique():,} users)")

    mapper = IndexMapper(ratings["userId"].to_numpy(),
                         ratings["movieId"].to_numpy())

    train, val, test = chronological_split(ratings, test_frac, val_frac)
    del ratings
    gc.collect()
    if verbose:
        print(f"[data] train/val/test        : {len(train):,} / "
              f"{len(val):,} / {len(test):,}")

    R_train = build_sparse_matrix(train, mapper)
    density = R_train.nnz / (R_train.shape[0] * R_train.shape[1])
    if verbose:
        print(f"[data] R_train shape         : {R_train.shape}, "
              f"density {density:.4%}")

    # movies restricted to the modelled catalogue, aligned to matrix columns
    catalogue = (movies.set_index("movieId")
                       .reindex(mapper.item_ids)
                       .reset_index()
                       .rename(columns={"index": "movieId"}))
    catalogue["item_idx"] = np.arange(len(catalogue), dtype=np.int32)

    return {"train": train, "val": val, "test": test,
            "R_train": R_train, "mapper": mapper,
            "catalogue": catalogue, "movies": movies}
