"""
Collaborative filtering: learn from *behaviour* (who rated what, how highly),
never from movie metadata.

Three models, in increasing order of power:

1. `BiasBaseline`      : r_hat = mu + b_u + b_i        (no interaction term)
2. `ItemItemCF`        : neighbourhood / memory-based kNN over item columns
3. `ALSMatrixFactorization` : model-based latent factors, r_hat = mu+b_u+b_i+p_u.q_i

The baseline is not decoration. Roughly 80% of the achievable RMSE improvement
over "predict the global mean" comes from the two bias terms alone, and both
of the stronger models are trained on *residuals* after biases are removed --
that is what stops the latent factors from wasting capacity on "Adi rates
everything 0.4 stars high" and "The Godfather is universally liked".
"""

from __future__ import annotations

import numpy as np
import gc
import time

from scipy import sparse


# --------------------------------------------------------------------------- #
# 1. Bias baseline
# --------------------------------------------------------------------------- #
class BiasBaseline:
    """r_hat(u,i) = mu + b_u + b_i, fitted by alternating ridge updates.

    Closed-form alternating solution of
        min sum_(u,i) (r_ui - mu - b_u - b_i)^2 + lam*(sum b_u^2 + sum b_i^2)

        b_i <- sum_u (r_ui - mu - b_u) / (lam_i + n_i)
        b_u <- sum_i (r_ui - mu - b_i) / (lam_u + n_u)

    The lambda in the denominator is *shrinkage*: an item with 3 ratings all of
    5.0 gets pulled back toward 0 bias, an item with 30,000 ratings barely
    moves. This is regularisation doing exactly what it should on sparse counts.
    """

    def __init__(self, lam_user: float = 10.0, lam_item: float = 5.0,
                 n_epochs: int = 10):
        self.lam_user = lam_user
        self.lam_item = lam_item
        self.n_epochs = n_epochs

    def fit(self, R: sparse.csr_matrix) -> "BiasBaseline":
        R = R.tocsr()
        Rt = R.T.tocsr()
        n_users, n_items = R.shape

        self.mu_ = float(R.data.mean())
        self.b_u = np.zeros(n_users, dtype=np.float32)
        self.b_i = np.zeros(n_items, dtype=np.float32)

        # per-row counts (how many ratings each user / item has)
        cnt_u = np.diff(R.indptr).astype(np.float32)
        cnt_i = np.diff(Rt.indptr).astype(np.float32)

        for _ in range(self.n_epochs):
            # ---- update item biases, holding user biases fixed ----
            # residual for every observed cell, grouped by item
            resid = Rt.data - self.mu_ - self.b_u[Rt.indices]
            sums = np.add.reduceat(resid, Rt.indptr[:-1]) * (cnt_i > 0)
            self.b_i = (sums / (self.lam_item + cnt_i)).astype(np.float32)

            # ---- update user biases, holding item biases fixed ----
            resid = R.data - self.mu_ - self.b_i[R.indices]
            sums = np.add.reduceat(resid, R.indptr[:-1]) * (cnt_u > 0)
            self.b_u = (sums / (self.lam_user + cnt_u)).astype(np.float32)

        return self

    def predict(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        return (self.mu_ + self.b_u[user_idx] + self.b_i[item_idx]).astype(np.float32)

    def residual_matrix(self, R: sparse.csr_matrix) -> sparse.csr_matrix:
        """R with mu + b_u + b_i subtracted from every *observed* cell."""
        R = R.tocsr(copy=True)
        rows = np.repeat(np.arange(R.shape[0], dtype=np.int32), np.diff(R.indptr))
        R.data = (R.data - self.mu_ - self.b_u[rows] - self.b_i[R.indices]
                  ).astype(np.float32)
        return R


# --------------------------------------------------------------------------- #
# 2. Item-item neighbourhood CF
# --------------------------------------------------------------------------- #
class ItemItemCF:
    """Memory-based CF: 'users who liked this also liked that'.

    Similarity is cosine between *bias-adjusted* item columns:

        sim(i,j) = (x_i . x_j) / (||x_i|| ||x_j||),  x = R - mu - b_u - b_i

    Adjusting first is what makes cosine behave like Pearson correlation:
    without it, two blockbusters look similar merely because everyone rates
    everything around 3.5-4.

    Prediction for (u, i) uses the top-k most similar items the user actually
    rated:

        r_hat(u,i) = mu + b_u + b_i + sum_j s_ij * x_uj / sum_j |s_ij|

    Scale note: a full 18,430 x 18,430 dense similarity matrix is 1.4 GB, so we
    keep only the top-`k_neighbors` per item as a sparse matrix (~30 MB), built
    in row blocks. Users with an enormous history are down-sampled to
    `max_user_history` ratings when building similarities, because the cost of
    the co-occurrence product grows as sum_u n_u^2 -- one user with 32,202
    ratings costs as much as 43,000 typical users.
    """

    def __init__(self, k_neighbors: int = 100, block_size: int = 512,
                 max_user_history: int = 300, shrinkage: float = 25.0,
                 seed: int = 42):
        self.k = k_neighbors
        self.block_size = block_size
        self.max_user_history = max_user_history
        self.shrinkage = shrinkage
        self.seed = seed

    # ---------------- internal helpers ----------------
    def _cap_user_history(self, X: sparse.csr_matrix) -> sparse.csr_matrix:
        """Randomly subsample rows (users) that exceed max_user_history."""
        counts = np.diff(X.indptr)
        heavy = np.where(counts > self.max_user_history)[0]
        if len(heavy) == 0:
            return X
        rng = np.random.default_rng(self.seed)
        keep = np.ones(X.nnz, dtype=bool)
        for u in heavy:
            s, e = X.indptr[u], X.indptr[u + 1]
            drop = rng.choice(np.arange(s, e), size=(e - s) - self.max_user_history,
                              replace=False)
            keep[drop] = False
        X = sparse.csr_matrix(
            (X.data[keep],
             X.indices[keep],
             np.concatenate([[0], np.cumsum(np.add.reduceat(
                 keep.astype(np.int64), X.indptr[:-1]) * (np.diff(X.indptr) > 0))])),
            shape=X.shape)
        return X

    def fit(self, R_residual: sparse.csr_matrix) -> "ItemItemCF":
        X = self._cap_user_history(R_residual.tocsr())
        Xi = X.T.tocsr()                                   # items x users

        # L2-normalise each item row -> dot product becomes cosine
        norms = np.sqrt(Xi.multiply(Xi).sum(axis=1)).A.ravel()
        counts = np.diff(Xi.indptr).astype(np.float32)     # support per item
        inv = np.zeros_like(norms)
        nz = norms > 0
        inv[nz] = 1.0 / norms[nz]
        Xn = sparse.diags(inv.astype(np.float32)) @ Xi
        Xn = Xn.tocsr().astype(np.float32)

        n_items = Xn.shape[0]
        rows_all, cols_all, vals_all = [], [], []
        del X, Xi
        gc.collect()
        XnT = Xn.T.tocsr()      # hoisted out of the loop: transposing a
                                # 15M-nnz matrix 36 times is 36 needless copies

        for start in range(0, n_items, self.block_size):
            end = min(start + self.block_size, n_items)
            # (block x users) @ (users x items) -> dense block of cosines
            S = (Xn[start:end] @ XnT).toarray()
            # shrink similarities computed from few co-raters:
            #   s <- s * n_co / (n_co + shrinkage)   (approximated with support)
            np.fill_diagonal(S[:, start:end], 0.0)
            S[S < 0] = 0.0                                  # keep positive sims only
            k = min(self.k, S.shape[1] - 1)
            idx = np.argpartition(-S, k, axis=1)[:, :k]
            r = np.repeat(np.arange(start, end), k)
            c = idx.ravel()
            v = S[np.repeat(np.arange(end - start), k), c]
            m = v > 0
            rows_all.append(r[m]); cols_all.append(c[m]); vals_all.append(v[m])

        rows = np.concatenate(rows_all)
        cols = np.concatenate(cols_all)
        vals = np.concatenate(vals_all).astype(np.float32)
        # support shrinkage using the item with the smaller support
        sup = np.minimum(counts[rows], counts[cols])
        vals = vals * (sup / (sup + self.shrinkage))

        self.S_ = sparse.csr_matrix((vals, (rows, cols)),
                                    shape=(n_items, n_items), dtype=np.float32)
        # CSC copy: column slicing S[:, rated] is O(nnz) on CSR but O(k) on CSC,
        # and scoring the whole catalogue for one user does exactly that.
        self.S_csc_ = self.S_.tocsc()
        self.R_res_ = R_residual.tocsr()
        return self

    def predict_residual(self, user_idx: np.ndarray,
                         item_idx: np.ndarray) -> np.ndarray:
        """Predicted *residual* for (u,i) pairs (add mu+b_u+b_i to get a rating).

        Grouped by user so each user costs one sparse slice + one small
        matrix-vector product, instead of one Python loop iteration per
        (user, item) pair. On 250k evaluation pairs that is the difference
        between ~4 minutes and ~20 seconds.
        """
        user_idx = np.asarray(user_idx)
        item_idx = np.asarray(item_idx)
        out = np.zeros(len(user_idx), dtype=np.float32)
        S, R = self.S_, self.R_res_

        order = np.argsort(user_idx, kind="mergesort")
        su, si = user_idx[order], item_idx[order]
        boundaries = np.flatnonzero(np.diff(su)) + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(su)]])

        for s, e in zip(starts, ends):
            u = su[s]
            rs, re = R.indptr[u], R.indptr[u + 1]
            rated, vals = R.indices[rs:re], R.data[rs:re]
            if len(rated) == 0:
                continue
            block = S[si[s:e]][:, rated]           # (n_query_items x n_rated)
            if block.nnz == 0:
                continue
            num = block @ vals
            den = np.abs(block).sum(axis=1).A.ravel() + 1e-8
            out[order[s:e]] = (num / den).astype(np.float32)
        return out

    def score_all_items(self, user_idx: int) -> np.ndarray:
        """Residual score for every item for one user -> used for top-N ranking."""
        R = self.R_res_
        us, ue = R.indptr[user_idx], R.indptr[user_idx + 1]
        rated, vals = R.indices[us:ue], R.data[us:ue]
        if len(rated) == 0:
            return np.zeros(self.S_.shape[0], dtype=np.float32)
        Su = self.S_csc_[:, rated]                    # (n_items x n_rated)
        num = Su @ vals
        den = np.abs(Su).sum(axis=1).A.ravel() + 1e-8
        return (num / den).astype(np.float32)


# --------------------------------------------------------------------------- #
# 3. Matrix factorisation (biased ALS)
# --------------------------------------------------------------------------- #
class ALSMatrixFactorization:
    """r_hat(u,i) = mu + b_u + b_i + p_u . q_i, factors fitted by ALS.

    Alternating Least Squares: fix Q, and the loss becomes an ordinary ridge
    regression in p_u with a closed-form solution

        p_u = (Q_u^T Q_u + lam * n_u * I)^-1 Q_u^T e_u

    where Q_u are the factor rows of the items user u rated and e_u the
    bias-corrected residual ratings. Then fix P and do the same for each item.
    Each half-step is convex and decreases the loss monotonically, so ALS
    needs no learning rate and cannot diverge -- unlike SGD (Funk-SVD), which
    is faster per epoch but needs lr tuning. `lam * n_u` is weighted-lambda
    regularisation (ALS-WR): users with few ratings are regularised harder.

    Latent factors are the whole point of the "model-based" family: instead of
    storing an 18k x 18k similarity table, we compress every user and every
    movie into `n_factors` numbers that place them in the same taste space.
    """

    def __init__(self, n_factors: int = 48, lam: float = 0.06,
                 n_epochs: int = 12, seed: int = 42, verbose: bool = True):
        self.n_factors = n_factors
        self.lam = lam
        self.n_epochs = n_epochs
        self.seed = seed
        self.verbose = verbose

    @staticmethod
    def _solve_side(indptr, indices, data, F_other, lam, n_factors):
        """One ALS half-step: solve a small ridge system per row."""
        n = len(indptr) - 1
        F = np.zeros((n, n_factors), dtype=np.float32)
        I = np.eye(n_factors, dtype=np.float32)
        for r in range(n):
            s, e = indptr[r], indptr[r + 1]
            if e == s:
                continue
            Q = F_other[indices[s:e]]                  # (n_r x k)
            A = Q.T @ Q + lam * (e - s) * I            # (k x k) ridge system
            b = Q.T @ data[s:e]
            F[r] = np.linalg.solve(A, b)
        return F

    def fit(self, R_residual: sparse.csr_matrix,
            val_pairs: tuple | None = None) -> "ALSMatrixFactorization":
        R = R_residual.tocsr()
        Rt = R.T.tocsr()
        rng = np.random.default_rng(self.seed)
        n_users, n_items = R.shape

        self.P = (rng.normal(0, 0.01, (n_users, self.n_factors))).astype(np.float32)
        self.Q = (rng.normal(0, 0.01, (n_items, self.n_factors))).astype(np.float32)

        self.history_ = []
        for ep in range(self.n_epochs):
            self.P = self._solve_side(R.indptr, R.indices, R.data,
                                      self.Q, self.lam, self.n_factors)
            self.Q = self._solve_side(Rt.indptr, Rt.indices, Rt.data,
                                      self.P, self.lam, self.n_factors)
            # Training RMSE, computed in row blocks. Doing it in one shot
            # (`P[np.repeat(...)]`) would materialise an (nnz x k) float32
            # array -- 20M x 40 x 4 bytes = 3.2 GB on this dataset, which is
            # an instant OOM. The chunked version peaks at a few MB.
            sq_err, n_seen, step = 0.0, 0, 20_000
            for s in range(0, n_users, step):
                e = min(s + step, n_users)
                lo, hi = R.indptr[s], R.indptr[e]
                if hi == lo:
                    continue
                rows = np.repeat(np.arange(s, e, dtype=np.int32),
                                 np.diff(R.indptr[s:e + 1]))
                p = np.einsum("ij,ij->i", self.P[rows], self.Q[R.indices[lo:hi]])
                sq_err += float(np.sum((R.data[lo:hi] - p) ** 2))
                n_seen += hi - lo
            train_rmse = float(np.sqrt(sq_err / max(n_seen, 1)))
            self.history_.append(train_rmse)
            if self.verbose:
                print(f"   [ALS] epoch {ep + 1:2d}/{self.n_epochs} "
                      f"residual train RMSE {train_rmse:.4f}")
        return self

    def predict_residual(self, user_idx: np.ndarray,
                         item_idx: np.ndarray) -> np.ndarray:
        return np.einsum("ij,ij->i", self.P[user_idx], self.Q[item_idx]).astype(np.float32)

    def score_all_items(self, user_idx: int) -> np.ndarray:
        return (self.Q @ self.P[user_idx]).astype(np.float32)


# --------------------------------------------------------------------------- #
# 4. Implicit-feedback ALS  (Hu, Koren & Volinsky 2008)
# --------------------------------------------------------------------------- #
class ImplicitALS:
    """Weighted matrix factorisation for the *ranking* task.

    Why a second factorisation model at all? Because RMSE and top-N are
    different objectives and the explicit model is blind to the second one.
    Explicit MF is trained *only on cells the user chose to rate*, so it never
    learns anything from the 18,000 movies the user did not watch -- yet those
    are exactly the candidates it must rank at serving time. That mismatch is
    why an RMSE-optimal model can lose to a popularity chart on Precision@10.

    Implicit ALS fixes it by training on the **full** user x item matrix:

        p_ui = 1 if the user interacted with i, else 0     (preference)
        c_ui = 1 + alpha * r_ui                            (confidence)

        min sum_{ALL u,i} c_ui (p_ui - p_u.q_i)^2 + lam(||p_u||^2 + ||q_i||^2)

    Every unobserved cell contributes a weak (c=1) push toward 0 -- 'probably
    not interested, but we're not sure' -- while observed cells push hard
    toward 1. Naively that is a 3-billion-cell loss; the standard trick makes
    it tractable in O(nnz * k^2 + k^3) per row:

        A_u = Q^T Q + Q_u^T (C_u - I) Q_u + lam*I
        b_u = Q_u^T c_u
        p_u = A_u^-1 b_u

    `Q^T Q` is a single k x k matrix computed once per half-epoch and shared by
    all users -- that one line is what collapses the sum over all items into
    the sum over the handful the user actually touched.
    """

    def __init__(self, n_factors: int = 64, lam: float = 8.0, alpha: float = 12.0,
                 n_epochs: int = 8, seed: int = 42, verbose: bool = True):
        self.n_factors = n_factors
        self.lam = lam
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.seed = seed
        self.verbose = verbose

    def _solve_side(self, indptr, indices, data, F_other, n_rows):
        k = self.n_factors
        YtY = F_other.T @ F_other                     # k x k, shared by all rows
        A_base = YtY + self.lam * np.eye(k, dtype=np.float32)
        F = np.zeros((n_rows, k), dtype=np.float32)
        for r in range(n_rows):
            s, e = indptr[r], indptr[r + 1]
            if e == s:
                continue
            Y = F_other[indices[s:e]]                 # (n_r x k)
            c = 1.0 + self.alpha * data[s:e]          # confidence weights
            A = A_base + (Y * (c - 1.0)[:, None]).T @ Y
            b = Y.T @ c
            F[r] = np.linalg.solve(A, b)
        return F

    def fit(self, R: sparse.csr_matrix, min_rating: float = 0.0) -> "ImplicitALS":
        """R holds raw ratings; entries below `min_rating` are dropped as
        non-endorsements (a 1-star rating is an interaction but not a signal
        that we should recommend more of the same)."""
        R = R.tocsr()
        if min_rating > 0:
            R = R.multiply(R >= min_rating).tocsr()
            R.eliminate_zeros()
        # scale ratings to [0,1] so alpha has a consistent meaning
        Rn = R.copy()
        Rn.data = (Rn.data / 5.0).astype(np.float32)
        Rt = Rn.T.tocsr()

        rng = np.random.default_rng(self.seed)
        n_users, n_items = Rn.shape
        self.P = rng.normal(0, 0.01, (n_users, self.n_factors)).astype(np.float32)
        self.Q = rng.normal(0, 0.01, (n_items, self.n_factors)).astype(np.float32)

        for ep in range(self.n_epochs):
            t = time.time()
            self.P = self._solve_side(Rn.indptr, Rn.indices, Rn.data,
                                      self.Q, n_users)
            self.Q = self._solve_side(Rt.indptr, Rt.indices, Rt.data,
                                      self.P, n_items)
            if self.verbose:
                print(f"   [iALS] epoch {ep + 1:2d}/{self.n_epochs} "
                      f"({time.time() - t:.1f}s)", flush=True)
        return self

    def score_all_items(self, user_idx: int) -> np.ndarray:
        return (self.Q @ self.P[user_idx]).astype(np.float32)

    def predict_pairs(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        return np.einsum("ij,ij->i", self.P[user_idx], self.Q[item_idx]).astype(np.float32)
