# Movie Recommendation System — Collaborative + Content-Based + Hybrid

An information-filtering system that learns each user's taste from 25 million
ratings and predicts which movies they are most likely to enjoy. Built from
scratch in NumPy / SciPy / scikit-learn — no `surprise`, no `implicit`, no
`lightfm`. Every model is inspectable line by line.

Dataset: **MovieLens 25M** (`ratings.csv`, `movies.csv`).

---

## 1. Results at a glance

Trained on 19,984,287 ratings, evaluated on a held-out **chronological** test
split (the newest 10% of every user's history). Full run: **6 min 40 s** on
1 CPU core, peak RSS 1.3 GB.

### Rating prediction — "how many stars would this user give?"

| Model | RMSE ↓ | MAE ↓ | acc ±0.5 | acc ±1.0 | like-acc (≥4★) |
|---|---|---|---|---|---|
| Global mean (no model) | 1.0569 | 0.8295 | 39.90% | 68.83% | 52.72% |
| Bias baseline (μ+bᵤ+bᵢ) | 0.8757 | 0.6627 | 48.58% | 78.16% | 64.93% |
| Item-item CF | 0.8727 | 0.6445 | 51.27% | 79.71% | 69.61% |
| Content-based (TF-IDF) | 1.0147 | 0.7836 | 41.13% | 70.73% | 60.67% |
| Matrix factorisation (ALS) | 0.8175 | 0.6097 | 52.81% | 81.68% | 69.28% |
| **Hybrid (ridge stacking)** | **0.8064** | **0.6052** | 52.73% | **81.93%** | 68.20% |

The hybrid cuts RMSE **23.7%** below the no-model baseline and **7.9%** below
the bias baseline. For reference, the winning Netflix Prize ensemble improved
on Netflix's production system by 10% RMSE.

### Top-10 ranking — "which 10 movies do we actually put on screen?"

Evaluated on 1,500 sampled users; a movie counts as relevant if the user rated
it ≥ 4.0 in the test period.

| Model | P@10 | Recall@10 | NDCG@10 | MAP@10 | Hit-rate | Coverage |
|---|---|---|---|---|---|---|
| Popularity chart | 0.0237 | 0.0468 | 0.0392 | 0.0194 | 17.2% | 0.4% |
| Content-based | 0.0035 | 0.0076 | 0.0065 | 0.0030 | 3.5% | 12.2% |
| Item-item CF | 0.0032 | 0.0089 | 0.0058 | 0.0027 | 3.0% | 17.8% |
| Explicit MF | 0.0070 | 0.0169 | 0.0138 | 0.0077 | 6.0% | 6.7% |
| Implicit ALS | 0.0435 | 0.1026 | 0.0776 | 0.0415 | 30.5% | 6.7% |
| **Hybrid (tuned fusion)** | **0.0471** | **0.1100** | **0.0844** | **0.0449** | **32.7%** | 5.9% |

The hybrid is **2.15× the popularity baseline on NDCG@10** and beats the best
single model by 8.8%. Hit-rate 32.7% means: for roughly one user in three, at
least one of the ten slots is a film they went on to rate 4★ or higher —
chosen from 18,430 candidates.

### About the "98% accuracy" figure

The 98% number in the source material is not a standard recommender metric, and
it is worth being precise about what it can mean, because an unqualified
accuracy claim is the fastest way to lose credibility in a viva or interview.

Measured on this system, accuracy depends entirely on the tolerance you allow:

| Tolerance | Predictions within it |
|---|---|
| ±0.25★ | 28.59% |
| ±0.50★ | 52.74% |
| ±0.75★ | 70.24% |
| ±1.00★ | 81.92% |
| ±1.50★ | 93.44% |
| ±2.00★ | **97.44%** |
| ±2.50★ | **98.98%** |

So a "98% accurate" claim on a 0.5–5.0 star scale corresponds to roughly a
**±2 to ±2.5 star tolerance** — a band nearly half the width of the entire
scale, which is why it sounds impressive and means little. The defensible
headline numbers for this project are **RMSE 0.806**, **81.9% within one
star**, and **NDCG@10 0.0844 at 2.15× the popularity baseline**. If you want a
single accuracy-style number, quote *81.9% of predictions within ±1 star* and
always state the tolerance.

---

## 2. Architecture

```
ratings.csv (25M) ──┐
                    ├─► data prep ──► chronological split ──► sparse R (162,414 × 18,430, 0.67% dense)
movies.csv (62k) ───┘                                              │
                                                                    ▼
                                        ┌──── bias baseline: μ + bᵤ + bᵢ ────┐
                                        │        (residual matrix)           │
                    ┌───────────────────┼───────────────────┬────────────────┤
                    ▼                   ▼                   ▼                ▼
             Item-item CF        Explicit ALS         Implicit ALS     Content TF-IDF
          (co-visitation)      (rating task)        (ranking task)    (genres/year/title)
                    │                   │                   │                │
                    └───────────────────┴─────────┬─────────┴────────────────┘
                                                  ▼
                            ┌─────────────────────────────────────────┐
                            │ HYBRID / AUGMENTED LAYER                │
                            │  • ridge stacking      → star ratings   │
                            │  • weighted fusion     → top-N lists    │
                            │  • switching rule      → cold start     │
                            └─────────────────────────────────────────┘
```

### Live app

A Streamlit front end (`streamlit_app.py`) serves the model with TMDB artwork.
It has three screens: personalised recommendations from a handful of films you
pick, an item-to-item explorer that exposes where content and collaborative
similarity disagree, and a methods page. Deployment instructions are in
[DEPLOYMENT.md](DEPLOYMENT.md).

The deployed app never stores user vectors. A visitor picks a few films and
their factor vector is solved on the spot by **fold-in** — one 64x64 linear
system, ~10 ms — which is why the serving bundle is 19.5 MB instead of 586 MB.

### Repository layout

```
recsys/
  data.py            loading, support filtering, index mapping, chronological split
  collaborative.py   BiasBaseline, ItemItemCF, ALSMatrixFactorization, ImplicitALS
  content.py         ContentBasedRecommender (TF-IDF item + user profiles)
  autoencoder.py     DeepAutoEncoder (NumPy backprop + Adam) + anchor matrix builder
  hybrid.py          HybridRatingModel (stacking), HybridRanker (fusion + switching)
  evaluate.py        RMSE/MAE/tolerance/like-accuracy + P@K/R@K/NDCG@K/MAP@K/coverage
run_pipeline.py      trains everything, evaluates, writes artifacts + plots
train_autoencoder.py      trains the autoencoder, runs the ablation
export_serving_bundle.py  586 MB model.pkl -> 23.8 MB deployable bundle
streamlit_app.py     web UI (3 tabs: recommend / similar / how it works)
app/serving.py       fold-in + hybrid scoring at serving time
app/tmdb.py          TMDB client (posters, cast) with caching + fallback
app/bundle/          exported item factors, similarity, TF-IDF, catalogue
recommend.py         serving layer: user recs, similar movies, cold start, explanations
analysis.py          tolerance curve, error sliced by user history and item popularity
results/             metrics.json, analysis.json, evaluation.png, demo_output.txt
```

### How to run

```bash
pip install -r requirements.txt

python run_pipeline.py --ratings data/ratings.csv --movies data/movies.csv
python analysis.py
python recommend.py --demo
python recommend.py --user 12345 --explain
python recommend.py --similar "Godfather, The (1972)"
python recommend.py --new-user "Inception (2010)" "Interstellar (2014)"
```

---

## 3. The models, and the maths behind them

### 3.0 Data preparation

**Support filtering.** Movies with fewer than 20 ratings are dropped: 59,047 →
18,430 movies, while keeping 99.2% of all ratings. A film with 2 ratings
produces similarity scores that are pure noise and inflates the item space
three-fold for no predictive gain.

**Re-indexing.** MovieLens `movieId` runs to 209,171 with large gaps. Using it
directly as a matrix column index would create ~190k mostly-empty columns, so
`IndexMapper` maps raw IDs to dense 0..n−1 positions and back.

**Chronological split (10% val / 10% test, per user).** Each user's ratings are
sorted by timestamp; the oldest 80% train, then validation, then the newest 10%
test. A random split leaks the future into the past: if a user rated *The Two
Towers* in 2003 and *Return of the King* in 2004, a random split can train on
the 2004 rating and test on the 2003 one. Every user appears in train, so no
unseen users leak into test.

**Sparsity.** 20M observed cells out of 162,414 × 18,430 ≈ 3.0 billion → 0.67%
dense. Dense float32 storage would need 12 GB; CSR needs ~240 MB. Every design
decision downstream follows from this number.

### 3.1 Bias baseline

$$\hat r_{ui} = \mu + b_u + b_i$$

Fitted by alternating ridge updates:

$$b_i \leftarrow \frac{\sum_u (r_{ui} - \mu - b_u)}{\lambda_i + n_i}, \qquad
  b_u \leftarrow \frac{\sum_i (r_{ui} - \mu - b_i)}{\lambda_u + n_u}$$

The λ in the *denominator* is shrinkage: a movie with 3 five-star ratings gets
pulled back toward zero bias; a movie with 30,000 ratings barely moves. On this
data μ = 3.554, mean |bᵤ| = 0.286, mean |bᵢ| = 0.374.

This is not decoration — it takes RMSE from 1.0569 to 0.8757, which is **57% of
the total improvement the full hybrid achieves**. Every other model is trained
on the residuals `r − μ − bᵤ − bᵢ`, so latent factors never waste capacity
learning "this user rates everything 0.4 stars high".

### 3.2 Item-item collaborative filtering (memory-based)

"Users who liked this also liked that." Similarity is cosine between
**bias-adjusted** item columns:

$$sim(i,j) = \frac{x_i \cdot x_j}{\lVert x_i \rVert \lVert x_j \rVert},
\qquad x = r - \mu - b_u - b_i$$

Adjusting first makes cosine behave like Pearson correlation. Without it, two
blockbusters look similar merely because everyone rates everything near 3.5–4.

Prediction uses the top-k similar items the user actually rated:

$$\hat r_{ui} = \mu + b_u + b_i + \frac{\sum_{j \in N_k(i)} s_{ij}\, x_{uj}}{\sum_{j \in N_k(i)} |s_{ij}|}$$

**Two scale tricks that matter:**

- A full 18,430² dense similarity matrix is 1.4 GB. Only the top-100
  neighbours per item are kept, built in row blocks of 512 → 1.84M non-zeros,
  ~22 MB, computed in 21 s.
- Cost of the co-occurrence product grows as Σᵤ nᵤ². One user in this dataset
  has 32,202 ratings and costs as much as 43,000 typical users; Σ nᵤ² = 1.55e10
  uncapped. Capping user history at 250 for the similarity build drops it to
  ~3e9 — a 5× speedup for a negligible accuracy cost.
- Similarities from few co-raters are shrunk by `n/(n+25)`.

### 3.3 Explicit ALS matrix factorisation (model-based, rating task)

$$\hat r_{ui} = \mu + b_u + b_i + p_u^\top q_i$$

Fix Q and the loss becomes an ordinary ridge regression in pᵤ with a
closed-form solution:

$$p_u = (Q_u^\top Q_u + \lambda n_u I)^{-1} Q_u^\top e_u$$

then swap and do the same for each item. Each half-step is convex and decreases
the loss monotonically, so **ALS needs no learning rate and cannot diverge** —
unlike SGD (Funk-SVD), which is faster per epoch but needs lr tuning and
scheduling. The `λ·nᵤ` term is weighted-lambda regularisation (ALS-WR): users
with few ratings are regularised harder.

40 factors, 8 epochs. Instead of an 18k×18k similarity table, each user and
movie is compressed to 40 numbers in a shared taste space.

### 3.4 Implicit ALS — the ranking model (Hu, Koren & Volinsky 2008)

**Why a second factorisation model?** Because RMSE and top-N are different
objectives and the explicit model is structurally blind to the second one.
Explicit MF trains *only on cells the user chose to rate*, so it learns nothing
from the 18,000 movies they didn't watch — yet those are exactly the candidates
it must rank at serving time. That mismatch is why the RMSE-optimal model
(NDCG@10 = 0.0138) loses to a plain popularity chart (0.0392).

Implicit ALS trains on the **full** matrix:

$$p_{ui} = \mathbb{1}[\text{interacted}], \qquad c_{ui} = 1 + \alpha r_{ui}$$
$$\min \sum_{\textbf{all } u,i} c_{ui}\,(p_{ui} - p_u^\top q_i)^2 + \lambda(\lVert p_u \rVert^2 + \lVert q_i \rVert^2)$$

Every *unobserved* cell contributes a weak (c = 1) push toward 0 — "probably
not interested, but we're not sure" — while observed cells push hard toward 1.
Naively that is a 3-billion-cell loss. The standard trick makes it
O(nnz·k² + k³) per row:

$$A_u = Q^\top Q + Q_u^\top (C_u - I) Q_u + \lambda I, \qquad
  b_u = Q_u^\top c_u, \qquad p_u = A_u^{-1} b_u$$

`QᵀQ` is one k×k matrix computed once per half-epoch and shared by all users —
that single line collapses the sum over all 18,430 items into a sum over the
handful each user actually touched. 64 factors, α = 12, λ = 8, 8 epochs, ~16 s
per epoch on 20M interactions. Only ratings ≥ 3.5 count as positives: a 1-star
rating is an interaction, but not an endorsement.

Result: **NDCG@10 = 0.0776, a 5.6× improvement over explicit MF on the same
task.**

### 3.5 Content-based filtering

Available content in this archive: **genres, release year, title tokens**.
There are no cast/director/plot columns — `build_item_features` is isolated
precisely so a TMDB or `tags.csv` join can be dropped in without touching
anything downstream.

**Why TF-IDF and not raw multi-hot?** IDF = log(N/df) down-weights ubiquitous
features. "Drama" tags 25,606 of 62,423 movies — two films sharing it tells you
almost nothing. "Film-Noir" tags 353 — sharing it tells you a lot. Multi-hot
scores both 1.0; TF-IDF gives Film-Noir roughly 5× the weight.

Three blocks (genres, titles, decade) are each L2-normalised, weighted
(1.0 / 0.35 / 0.25) and stacked. Separate normalisation stops the ~3,000-token
title vocabulary from drowning the 20-token genre vocabulary purely by having
more columns.

**User taste profile:**

$$\text{profile}_u = \sum_i \max(0,\; r_{ui} - \bar r_u)\; f_i \quad\text{(then L2-normalised)}$$

Centring on the user's own mean is essential: a user who averages 4.6 has not
endorsed a film by giving it 3.5, and a user who averages 2.8 has. Negative
weights are clipped — this builds a profile of what the user *likes*; dislike
signal is left to CF, where it is far more reliable.

Item-to-item queries are one sparse mat-vec on demand (milliseconds) rather
than a materialised 18k×18k matrix.

### 3.6 Deep autoencoder for movie embeddings

A 4-layer masked denoising autoencoder (I-AutoRec style, Sedhain et al. 2015),
with forward pass, backprop and Adam written from scratch in NumPy — no
autodiff framework.

```
input  3,000 anchor-user ratings (mean-centred, sparse)
        │
      512  ReLU
        │
      128  ← the movie embedding (bottleneck)
        │
      512  ReLU
        │
output 3,000  reconstruction, masked MSE loss
```

3,207,224 parameters. 20 epochs, ~6 s/epoch. **Masked validation RMSE 0.666**
(train 0.556).

**Three details that make or break it:**

1. **Masked loss.** `L = Σ_observed (r − r̂)²`. About 98% of each input vector is
   unobserved, so without the mask the network's optimal strategy is to output
   zero everywhere. This one line is the difference between AutoRec and a model
   that predicts nothing.
2. **Denoising.** Inputs are randomly zeroed with p = 0.25 (inverted dropout),
   forcing the network to reconstruct ratings it was not shown — exactly the
   serving-time task. Without it, it learns an identity map.
3. **Per-item mean-centring**, so the embedding encodes *who deviates* on this
   film rather than how popular it is. Popularity is handled elsewhere.

**Why this is not just matrix factorisation again.** ALS learns `q_i` as a free
parameter per item — 18,430 independent vectors with no function relating them.
The autoencoder learns an *encoder function* `f(rating profile) → embedding`.
That makes it non-linear (stacked ReLUs, not a bilinear form), **inductive**
rather than transductive (a new film gets an embedding from one forward pass,
no refitting), and parameter-shared, which regularises.

**The embeddings are clearly meaningful.** Nearest neighbours by cosine in the
128-d space:

| Query | Top neighbours (cosine) |
|---|---|
| Pulp Fiction | Reservoir Dogs (0.957), Goodfellas (0.864), Inglourious Basterds (0.842), Kill Bill Vol. 2 (0.840) |
| Toy Story | Monsters Inc. (0.943), A Bug's Life (0.897), The Incredibles (0.880) |
| The Shining | Full Metal Jacket (0.842), The Exorcist (0.828), Apocalypse Now (0.803) |

It recovered the Tarantino cluster with **no director, cast or plot data
anywhere in the input** — purely from who rates what.

### 3.7 Ablation: does the autoencoder improve recommendations?

No, and this is worth stating plainly rather than burying.

| Configuration | NDCG@10 | P@10 |
|---|---|---|
| Autoencoder alone | 0.0454 | 0.0258 |
| Hybrid **without** autoencoder | **0.0844** | **0.0471** |
| Hybrid **with** autoencoder (w = 0.20) | 0.0842 | 0.0468 |

A validation sweep of the blend weight put the optimum at exactly **0.0**:

| w_ae | 0.00 | 0.05 | 0.10 | 0.20 | 0.30 | 0.50 |
|---|---|---|---|---|---|---|
| val NDCG@10 | **0.1023** | 0.1018 | 0.1006 | 0.0998 | 0.0994 | 0.0978 |

The autoencoder is largely **redundant with implicit ALS** — both extract
collaborative latent structure from the same interaction signal, and iALS does
it more directly for the ranking objective. On item-to-item similarity quality
the two are comparable (genre Jaccard@10: 0.427 autoencoder vs 0.471 iALS;
franchise retrieval 0.157 vs 0.140).

So the production blend **excludes** it, and it powers the "neural" similarity
mode in the app instead, where the embeddings are genuinely useful. Shipping a
component that measurably does nothing, and calling it an improvement, would be
the alternative.

### 3.8 The hybrid / augmented layer

CF and content-based filtering fail in *opposite* situations, which is exactly
why fusing them works:

| Collaborative filtering | Content-based filtering |
|---|---|
| Needs many ratings per item | Works on a film released this morning |
| Cannot explain "why" | Trivially explainable ("same director") |
| Finds cross-genre surprises (serendipity) | Trapped in known genres (filter bubble) |
| Cold start is fatal | Cold start is fine |

**(a) Rating task — ridge stacking.** Component predictions become *features*
of a small ridge regression fitted on the validation split:

```
features = [baseline, itemcf_resid, mf_resid, content_cos,
            log(item_support), log(user_support), user_mean_dev]
```

Learned weights from the full run: baseline **+0.968**, mf_resid **+0.652**,
itemcf_resid **+0.317**, content_cos **+0.294**. The meta-model learns the
weights from data instead of us guessing them. Support counts enter in log
space because their effect is multiplicative: 20 → 200 ratings changes
confidence far more than 20,000 → 20,180.

**(b) Ranking task — weighted fusion, tuned on validation NDCG@10.** Each
component is **z-scored per user across the catalogue** before mixing, because
raw scales are incomparable (iALS ≈ [0, 1.2], quality-adjusted MF ≈ [−1.5, 1.0],
content cosine ≈ [0, 0.9]) — a naive sum just means "whichever component has the
biggest numeric range wins".

Grid search selected:

```
0.45·iALS + 0.15·MF + 0.20·item-CF + 0.20·content + 0.20·popularity
```

**(c) Two corrections that don't show up in RMSE but dominate top-N quality:**

```python
score = item_bias + residual_score · n_i/(n_i + τ)      # τ = 50
```

1. *Add back the item bias.* RMSE is only measured on films users chose to
   watch, so the model is never asked "is this any good?" Ranking asks exactly
   that over the whole catalogue. Ranking on residuals alone scored
   P@10 = 0.0008 — worse than chance — because it happily promotes a mediocre
   film whose personalised term is +0.4 while its item bias is −0.9.
2. *Shrink the personalised term by item support.* A movie with 21 ratings has
   a latent vector fitted from almost no evidence, so its dot product is
   high-variance noise. Uncorrected, top-N lists fill with obscure items that
   got a lucky factor: high novelty, terrible precision.

Together these lifted explicit-MF P@10 roughly 6×.

**(d) Switching rule for cold start.** Below 10 ratings, weight shifts toward
content (up to 0.45) and popularity (up to 0.30). A user with 3 ratings has
latent factors fitted from 3 observations — essentially noise — while their
TF-IDF profile inherits structure estimated from the whole catalogue.

---

## 4. Qualitative output

**Similar to *The Godfather* (hybrid item-item):** Godfather Part II
(content 0.985 / collab 0.414), Goodfellas, One Flew Over the Cuckoo's Nest,
Shawshank Redemption, Apocalypse Now, Bonnie and Clyde.

Note the split: *Godfather Part II* is found by **both** signals. *One Flew Over
the Cuckoo's Nest* and *Apocalypse Now* have content similarity 0.000 — no
shared genre tags — and are found **only** by collaborative filtering. That is
the serendipity CF buys you, and a pure content system can never produce it.

**Cold start from two films (*Inception*, *Interstellar*), zero rating
history:** Edge of Tomorrow, Gravity, Elysium, Cloud Atlas, V for Vendetta,
Transcendence. Pure CF cannot answer this at all — there is no pᵤ to look up.

**Explained recommendations** (`--explain`) for a classic-cinema user:

```
1  Léon: The Professional (1994)   because you watched 'True Romance (1993)'
4  All About Eve (1950)            because you watched 'A Streetcar Named Desire (1951)'
8  Unforgiven (1992)               because you watched 'The Unforgiven (1960)'
```

---

## 5. Where the model is weak (error analysis)

| User history (train ratings) | RMSE | acc ±1★ |
|---|---|---|
| < 20 | 0.9615 | 73.38% |
| 20–50 | 0.8971 | 77.96% |
| 50–150 | 0.8368 | 80.95% |
| 150–500 | 0.7957 | 82.39% |
| 500+ | 0.7599 | 83.55% |

| Item popularity | RMSE | acc ±1★ |
|---|---|---|
| 20–100 ratings | 0.8553 | 79.94% |
| 100–500 | 0.8275 | 81.05% |
| 2,000–10,000 | 0.7967 | 82.14% |
| 10,000+ | 0.7970 | 82.71% |

Aggregate RMSE hides this: the system is **26% worse for newcomers than for
power users** — precisely the users whose retention matters most. That is the
argument for the switching hybrid, and the argument for onboarding flows that
collect a handful of ratings up front.

Other honest limitations:

- **Catalogue coverage is 5.9%** for the hybrid vs 17.8% for item-CF. Fusion
  buys accuracy by concentrating on well-understood films. Raising the content
  weight trades precision for catalogue exposure — a business decision, not a
  modelling one.
- **No cast/director/plot features** in this archive; content is genre + year +
  title only. Adding TMDB metadata or `tags.csv` is the single highest-value
  extension.
- **Ranking is evaluated against ratings, not views.** A user can only "hit" a
  film they went on to rate, so all top-N numbers here are lower bounds.
- **Popularity bias** is baked into the training signal and only partly
  controlled by the popularity weight.
- **No temporal dynamics.** Koren's timeSVD++ shows user tastes and item
  perceptions drift over years; our factors are static, and the test split is
  the *future*, so this costs real accuracy.

---
## 6. References

- Koren, Bell & Volinsky (2009), *Matrix Factorization Techniques for
  Recommender Systems*, IEEE Computer — biases, ALS/SGD, the Netflix Prize.
- Hu, Koren & Volinsky (2008), *Collaborative Filtering for Implicit Feedback
  Datasets*, ICDM — the confidence-weighted formulation used in §3.4.
- Sarwar et al. (2001), *Item-Based Collaborative Filtering Recommendation
  Algorithms*, WWW.
- Linden, Smith & York (2003), *Amazon.com Recommendations*, IEEE Internet
  Computing.
- Burke (2002), *Hybrid Recommender Systems: Survey and Experiments* — the
  weighted / switching / feature-combination taxonomy used in §3.8.
- Sedhain, Menon, Sanner & Xie (2015), *AutoRec: Autoencoders Meet
  Collaborative Filtering*, WWW — the item-autoencoder formulation in §3.6.
- Liang et al. (2018), *Variational Autoencoders for Collaborative Filtering*,
  WWW — the denoising/multinomial successor to AutoRec.
- Harper & Konstan (2015), *The MovieLens Datasets: History and Context*, ACM
  TiiS — cite this if you publish anything using the data.
