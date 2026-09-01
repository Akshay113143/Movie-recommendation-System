"""
Deep autoencoder for movie embeddings (I-AutoRec style, Sedhain et al. 2015),
implemented from scratch in NumPy -- forward pass, backprop and Adam, no
autodiff framework.

WHAT IT LEARNS
    Input  : one movie's rating vector over a fixed set of `anchor users`,
             mean-centred, with unrated entries as 0.
    Output : a reconstruction of that same vector.
    Middle : a 128-dimensional bottleneck -- the movie embedding.

The network is forced to squeeze a 3,000-dimensional rating profile through
128 units and back out again. The only way to do that with low error is to
discover the latent structure that actually generates ratings ("gritty crime
drama that cinephiles love", "family animation that parents rate highly"), so
the bottleneck activations become a compact semantic representation of the film.

WHY THIS IS NOT JUST MATRIX FACTORISATION AGAIN
    ALS learns q_i as a *free parameter per item* -- 18,430 independent vectors
    with no function connecting them. The autoencoder learns an *encoder
    function* f(rating profile) -> embedding. Consequences:

      1. Non-linearity. ALS can only express bilinear interactions. The encoder
         stacks ReLU layers, so it can represent "liked by critics AND
         mainstream" as a distinct region of space rather than a linear
         combination.
      2. Inductive, not transductive. A brand-new film with 30 ratings gets an
         embedding by a single forward pass -- no refitting.
      3. Parameter sharing regularises: 18,430 items share one set of weights
         rather than each owning 64 free numbers.

THREE DETAILS THAT MATTER (and are the usual interview questions)

    MASKED LOSS. The loss is computed only over *observed* ratings:
        L = sum_{observed} (r - r_hat)^2
    Without the mask the network's optimal strategy is to output 0 everywhere,
    because ~98% of each input vector is unobserved. This single line is the
    difference between a working AutoRec and a model that predicts nothing.

    DENOISING. Inputs are randomly zeroed with probability `noise` during
    training (scaled by 1/(1-p) to keep expectations right). The network must
    reconstruct ratings it was not shown, which is exactly the task at serving
    time. Without it the model learns an identity map and generalises poorly.

    MEAN-CENTRING. Each item's observed ratings are centred on that item's mean
    before encoding, so the embedding encodes *who* deviates from the norm on
    this film, not how popular or well-liked it is in aggregate. Popularity is
    already handled elsewhere in the hybrid.
"""

from __future__ import annotations

import time

import numpy as np
from scipy import sparse


def he_init(fan_in: int, fan_out: int, rng) -> np.ndarray:
    """He (Kaiming) initialisation: N(0, sqrt(2/fan_in)).

    Scaled to the *input* dimension so activation variance stays ~constant
    through depth. With plain N(0, 0.01) a 4-layer net's signal decays toward
    zero and the early layers barely train; that is the vanishing-signal
    problem He init was designed to fix for ReLU networks.
    """
    return (rng.normal(0, np.sqrt(2.0 / fan_in), (fan_in, fan_out))
            ).astype(np.float32)


class Adam:
    """Adam optimiser.

    Keeps per-parameter running averages of the gradient (m, momentum) and its
    square (v, adaptive scaling), with bias correction because both start at
    zero and would otherwise be biased toward zero early in training. The
    adaptive term is what lets one learning rate work for both the dense hidden
    weights and the sparse output layer, whose gradient magnitudes differ by
    orders of magnitude here.
    """

    def __init__(self, params: dict, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict, grads: dict):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class DeepAutoEncoder:
    """4-layer masked denoising autoencoder: D -> 512 -> 128 -> 512 -> D."""

    def __init__(self, input_dim: int, hidden: int = 512, bottleneck: int = 128,
                 lr: float = 1e-3, noise: float = 0.25, l2: float = 1e-5,
                 seed: int = 42):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.noise = noise
        self.l2 = l2
        self.p = {
            "W1": he_init(input_dim, hidden, rng),
            "b1": np.zeros(hidden, dtype=np.float32),
            "W2": he_init(hidden, bottleneck, rng),
            "b2": np.zeros(bottleneck, dtype=np.float32),
            "W3": he_init(bottleneck, hidden, rng),
            "b3": np.zeros(hidden, dtype=np.float32),
            "W4": he_init(hidden, input_dim, rng),
            "b4": np.zeros(input_dim, dtype=np.float32),
        }
        self.opt = Adam(self.p, lr=lr)
        self.history_ = []

    # ------------------------------------------------------------------ #
    @staticmethod
    def _relu(x):
        return np.maximum(x, 0.0)

    def forward(self, X: np.ndarray, train: bool = True):
        """Returns (reconstruction, cache). Cache holds activations for backprop."""
        p = self.p
        if train and self.noise > 0:
            # inverted dropout on the INPUT: this is the 'denoising' part
            keep = (self.rng.random(X.shape) > self.noise).astype(np.float32)
            Xin = X * keep / (1.0 - self.noise)
        else:
            Xin = X
        z1 = Xin @ p["W1"] + p["b1"]
        a1 = self._relu(z1)
        z2 = a1 @ p["W2"] + p["b2"]          # bottleneck: linear = the embedding
        a2 = z2
        z3 = a2 @ p["W3"] + p["b3"]
        a3 = self._relu(z3)
        out = a3 @ p["W4"] + p["b4"]
        return out, (Xin, z1, a1, a2, z3, a3)

    def backward(self, X, M, out, cache):
        """Backprop through the masked squared-error loss.

        M is the observation mask. `dout = 2*M*(out - X)` is where masking
        enters: unobserved cells get exactly zero gradient, so the network is
        never rewarded for guessing them.
        """
        Xin, z1, a1, a2, z3, a3 = cache
        p = self.p
        n = max(M.sum(), 1)

        dout = (2.0 / n) * M * (out - X)
        g = {}
        g["W4"] = a3.T @ dout + self.l2 * p["W4"]
        g["b4"] = dout.sum(0)

        da3 = dout @ p["W4"].T
        dz3 = da3 * (z3 > 0)
        g["W3"] = a2.T @ dz3 + self.l2 * p["W3"]
        g["b3"] = dz3.sum(0)

        da2 = dz3 @ p["W3"].T
        g["W2"] = a1.T @ da2 + self.l2 * p["W2"]
        g["b2"] = da2.sum(0)

        da1 = da2 @ p["W2"].T
        dz1 = da1 * (z1 > 0)
        g["W1"] = Xin.T @ dz1 + self.l2 * p["W1"]
        g["b1"] = dz1.sum(0)
        return g

    # ------------------------------------------------------------------ #
    def fit(self, X_sparse: sparse.csr_matrix, n_epochs: int = 12,
            batch_size: int = 256, val_frac: float = 0.05, verbose: bool = True):
        """Train on a sparse (items x anchor_users) matrix.

        Rows are densified one mini-batch at a time: the full dense matrix would
        be 18,430 x 3,000 x 4 bytes = 221 MB, while a 256-row batch is 3 MB.
        """
        n = X_sparse.shape[0]
        idx = self.rng.permutation(n)
        n_val = int(n * val_frac)
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        for ep in range(n_epochs):
            t0 = time.time()
            order = self.rng.permutation(train_idx)
            tot, cnt = 0.0, 0
            for s in range(0, len(order), batch_size):
                rows = order[s:s + batch_size]
                Xb = X_sparse[rows].toarray()
                Mb = (Xb != 0).astype(np.float32)
                out, cache = self.forward(Xb, train=True)
                grads = self.backward(Xb, Mb, out, cache)
                self.opt.step(self.p, grads)
                tot += float(np.sum(Mb * (out - Xb) ** 2))
                cnt += int(Mb.sum())
            train_rmse = np.sqrt(tot / max(cnt, 1))
            val_rmse = self.evaluate(X_sparse[val_idx]) if n_val else float("nan")
            self.history_.append((train_rmse, val_rmse))
            if verbose:
                print(f"   [AE] epoch {ep + 1:2d}/{n_epochs} "
                      f"train RMSE {train_rmse:.4f}  val RMSE {val_rmse:.4f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)
        return self

    def evaluate(self, X_sparse: sparse.csr_matrix, batch_size: int = 512) -> float:
        tot, cnt = 0.0, 0
        for s in range(0, X_sparse.shape[0], batch_size):
            Xb = X_sparse[s:s + batch_size].toarray()
            Mb = (Xb != 0).astype(np.float32)
            out, _ = self.forward(Xb, train=False)
            tot += float(np.sum(Mb * (out - Xb) ** 2))
            cnt += int(Mb.sum())
        return float(np.sqrt(tot / max(cnt, 1)))

    # ------------------------------------------------------------------ #
    def encode(self, X_sparse: sparse.csr_matrix, batch_size: int = 512) -> np.ndarray:
        """Bottleneck activations = the movie embeddings (n_items x 128)."""
        outs = []
        for s in range(0, X_sparse.shape[0], batch_size):
            Xb = X_sparse[s:s + batch_size].toarray()
            z1 = Xb @ self.p["W1"] + self.p["b1"]
            a1 = self._relu(z1)
            outs.append((a1 @ self.p["W2"] + self.p["b2"]).astype(np.float32))
        return np.vstack(outs)


# --------------------------------------------------------------------------- #
def build_anchor_matrix(R_train: sparse.csr_matrix, n_anchors: int = 3000,
                        seed: int = 42):
    """Build the (items x anchor_users) mean-centred input matrix.

    Why anchors rather than all 162,414 users? The input layer would need
    162,414 x 512 = 83M weights -- 333 MB of parameters and hours per epoch on
    one core, for almost no gain: the most active users already span the taste
    space, and rarely-active columns contribute mostly noise. Sampling the most
    active 3,000 users keeps 27% of all ratings in 2% of the columns.

    Centring is per item, over observed entries only, so the embedding captures
    *deviation patterns* rather than average popularity.
    """
    rng = np.random.default_rng(seed)
    user_counts = np.diff(R_train.indptr)
    anchors = np.argsort(-user_counts)[:n_anchors]
    anchors = np.sort(anchors)

    X = R_train[anchors].T.tocsr()                     # items x anchors
    counts = np.diff(X.indptr)
    # X.sum rather than np.add.reduceat: reduceat raises on trailing empty rows
    # (their indptr entry equals nnz, which is out of bounds for the data array)
    sums = np.asarray(X.sum(axis=1)).ravel()
    means = np.zeros(X.shape[0], dtype=np.float32)
    nz = counts > 0
    means[nz] = sums[nz] / counts[nz]
    rows = np.repeat(np.arange(X.shape[0], dtype=np.int32), counts)
    X.data = (X.data - means[rows]).astype(np.float32)
    # a centred value of exactly 0 would be read as 'unobserved' by the mask,
    # so nudge those to a tiny non-zero value
    X.data[X.data == 0] = 1e-4
    return X, anchors, means
