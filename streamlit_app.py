"""
Movie Recommendation System -- Streamlit front end.

    streamlit run streamlit_app.py

Three modes:
    1. "Recommend for me"  -- pick films you love, get a personalised top-N.
                              Uses implicit-ALS fold-in, so it works for a
                              visitor who has never used the system before.
    2. "More like this"     -- item-to-item, with a toggle exposing the
                              difference between content and collaborative
                              similarity (this is the most interesting screen
                              to demo: they disagree, and the disagreement is
                              the whole argument for a hybrid).
    3. "How it works"       -- architecture and metrics, so a visitor who is
                              not you can tell what was actually built.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.serving import ServingModel
from app import tmdb

st.set_page_config(page_title="Movie Recommender | Hybrid CF + Content",
                   page_icon="🎬", layout="wide")

CSS = """
<style>
  .block-container {padding-top: 2rem; max-width: 1200px;}
  .movie-card {border:1px solid rgba(140,140,160,.25); border-radius:10px;
               padding:.6rem; height:100%; background:rgba(140,140,160,.05);}
  .movie-title {font-weight:600; font-size:.92rem; line-height:1.25; margin:.4rem 0 .15rem 0;}
  .movie-meta {font-size:.76rem; opacity:.65; line-height:1.3;}
  .why-chip {font-size:.74rem; background:rgba(90,140,255,.14);
             border-radius:6px; padding:.2rem .45rem; display:inline-block; margin-top:.35rem;}
  .metric-big {font-size:1.6rem; font-weight:700;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Model loading. @st.cache_resource keeps ONE copy in memory across all user
# sessions and reruns -- without it, Streamlit would reload ~25 MB of factors
# on every widget click, since it re-executes the whole script each time.
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading recommendation models…")
def load_model() -> ServingModel:
    return ServingModel("app/bundle")


try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the model bundle: {exc}\n\n"
             "Run `python export_serving_bundle.py` after training to create "
             "`app/bundle/`.")
    st.stop()

HAS_TMDB = tmdb.get_api_key() is not None


# --------------------------------------------------------------------------- #
def movie_card(row, show_why: bool = True, show_sims: bool = False):
    """One poster card. Falls back to a text card when TMDB is unavailable."""
    poster = tmdb.poster_for(row["clean_title"], row.get("year")) if HAS_TMDB else None
    if poster:
        st.image(poster, use_container_width=True)
    else:
        st.markdown(
            f"<div style='height:170px;display:flex;align-items:center;"
            f"justify-content:center;background:rgba(140,140,160,.13);"
            f"border-radius:6px;font-size:.8rem;opacity:.5;padding:.5rem;"
            f"text-align:center'>{row['clean_title']}</div>",
            unsafe_allow_html=True)

    year = "" if pd.isna(row.get("year")) else f" ({int(row['year'])})"
    st.markdown(f"<div class='movie-title'>{row['clean_title']}{year}</div>",
                unsafe_allow_html=True)
    st.markdown(
        f"<div class='movie-meta'>{str(row['genres']).replace('|', ' · ')}<br>"
        f"{int(row['n_ratings']):,} ratings</div>", unsafe_allow_html=True)
    if show_sims:
        extra = (f" · neural {row['neural_sim']:.2f}"
                 if "neural_sim" in row and row["neural_sim"] else "")
        st.markdown(
            f"<div class='movie-meta'>content {row['content_sim']:.2f} · "
            f"collab {row['collab_sim']:.2f}{extra}</div>",
            unsafe_allow_html=True)
    if show_why and row.get("why"):
        st.markdown(f"<div class='why-chip'>{row['why']}</div>",
                    unsafe_allow_html=True)


def grid(df: pd.DataFrame, cols: int = 4, **kw):
    for start in range(0, len(df), cols):
        chunk = df.iloc[start:start + cols]
        columns = st.columns(cols)
        for c, (_, row) in zip(columns, chunk.iterrows()):
            with c.container(border=True):
                movie_card(row, **kw)


# --------------------------------------------------------------------------- #
st.title("🎬 Movie Recommendation System")
st.caption("Hybrid recommender — collaborative filtering + content-based "
           "filtering — trained on 25 million MovieLens ratings")

if not HAS_TMDB:
    st.info("Posters are off: no TMDB API key configured. The recommender "
            "works fully without it — add `TMDB_API_KEY` in app settings → "
            "Secrets to turn on artwork.", icon="🖼️")

tab_rec, tab_sim, tab_about = st.tabs(
    ["✨ Recommend for me", "🔍 More like this", "⚙️ How it works"])


# ------------------------------------------------------------- TAB 1 ------ #
with tab_rec:
    st.subheader("Tell it what you love")
    st.write("Pick a few films and rate them. The system solves for your taste "
             "vector on the spot — no account, no history needed.")

    if "picks" not in st.session_state:
        st.session_state.picks = {}

    c1, c2 = st.columns([3, 2])
    with c1:
        query = st.text_input("Search for a film",
                              placeholder="e.g. Inception, Godfather, Spirited Away")
        if query:
            hits = model.search(query, limit=8)
            if not hits:
                st.warning("No match in the catalogue.")
            for i in hits:
                cc1, cc2 = st.columns([5, 1])
                cc1.write(f"**{model.titles[i]}**  \n"
                          f"<span style='font-size:.78rem;opacity:.6'>"
                          f"{str(model.genres[i]).replace('|', ' · ')}</span>",
                          unsafe_allow_html=True)
                if cc2.button("Add", key=f"add_{i}"):
                    st.session_state.picks[int(i)] = 5.0
                    st.rerun()

    with c2:
        st.markdown("**Your picks**")
        if not st.session_state.picks:
            st.caption("Nothing yet — search and add 3–5 films.")
            if st.button("🎲 Surprise me (use popular films)"):
                for i in model.popular(4):
                    st.session_state.picks[int(i)] = 5.0
                st.rerun()
        for i in list(st.session_state.picks):
            r1, r2 = st.columns([4, 1])
            st.session_state.picks[i] = r1.slider(
                model.clean_titles[i], 0.5, 5.0,
                float(st.session_state.picks[i]), 0.5, key=f"r_{i}")
            if r2.button("✕", key=f"rm_{i}"):
                del st.session_state.picks[i]
                st.rerun()
        if st.session_state.picks and st.button("Clear all"):
            st.session_state.picks = {}
            st.rerun()

    st.divider()
    o1, o2, o3, o4 = st.columns(4)
    n_recs = o1.slider("How many", 4, 24, 12, 4)
    diversity = o2.slider("Adventurousness", 0.0, 1.0, 0.0, 0.1,
                          help="Higher values push past blockbusters into the "
                               "long tail — more novel, less reliable.")
    genre_opts = ["Any", "Action", "Comedy", "Drama", "Horror", "Sci-Fi",
                  "Romance", "Thriller", "Animation", "Documentary", "Crime"]
    genre = o3.selectbox("Genre", genre_opts)
    min_year = o4.number_input("Released after", 1900, 2020, 1900, 10)

    if st.session_state.picks:
        idx = np.array(list(st.session_state.picks.keys()))
        rts = np.array(list(st.session_state.picks.values()), dtype=np.float32)
        with st.spinner("Scoring 18,430 films…"):
            recs = model.recommend(idx, rts, top_n=n_recs, diversity=diversity,
                                   min_year=int(min_year) if min_year > 1900 else None,
                                   genre_filter=genre)
        if recs.empty:
            st.warning("No films match those filters. Try loosening them.")
        else:
            st.subheader(f"Your top {len(recs)}")
            grid(recs, cols=4)
    else:
        st.info("Add at least one film above to get recommendations.", icon="👆")


# ------------------------------------------------------------- TAB 2 ------ #
with tab_sim:
    st.subheader("Find films similar to one you know")
    q = st.text_input("Film title", placeholder="e.g. The Godfather",
                      key="simq")
    modes = ["hybrid", "collaborative", "content"]
    if model.ae_emb is not None:
        modes.append("neural")
    mode = st.radio("Similarity signal", modes, horizontal=True,
                    format_func=lambda m: {
                        "hybrid": "Hybrid (both)",
                        "collaborative": "Collaborative (who watches what)",
                        "content": "Content (genre / era / title)",
                        "neural": "Neural (autoencoder embedding)"}[m])

    if q:
        hits = model.search(q, limit=1)
        if not hits:
            st.warning("Not in the catalogue.")
        else:
            i = hits[0]
            st.markdown(f"### Similar to *{model.titles[i]}*")
            sims = model.similar(i, top_n=12, mode=mode)
            grid(sims, cols=4, show_why=False, show_sims=True)
            st.caption(
                "Look for rows where **content similarity is low but "
                "collaborative similarity is high** — little shared metadata, "
                "yet the same people love both. A content-only system can never "
                "surface those, and that is precisely why this system is hybrid.")


# ------------------------------------------------------------- TAB 3 ------ #
with tab_about:
    st.subheader("What is running under this page")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ratings trained on", "24.8M")
    m2.metric("Test RMSE", "0.806", "-23.7% vs baseline")
    m3.metric("Within ±1 star", "81.9%")
    m4.metric("NDCG@10", "0.0844", "2.15× popularity")

    st.markdown("""
**Five models, fused.** Each is written from scratch in NumPy/SciPy.

| Component | What it contributes | Weight |
|---|---|---|
| Implicit ALS (Hu–Koren–Volinsky) | Main ranking engine; trains on the *full* matrix including non-interactions | 0.45 |
| Item–item CF | "Users who liked this also liked that", top-100 cosine neighbours | 0.20 |
| Content TF-IDF | Genre / decade / title similarity; makes cold start possible | 0.20 |
| Explicit ALS | Rating-quality signal (RMSE-optimal factors) | 0.15 |
| Popularity prior | log(support) × learned item bias, as a tie-break | 0.20 |
| Deep autoencoder | 128-d movie embeddings; powers the "neural" similarity mode | 0 (see below) |

**How it recommends for you specifically.** You have no rating history in the
training data, so there is no factor vector to look up. Instead the app solves
the implicit-ALS normal equations for your picks while holding the item factors
fixed — a single 64×64 linear solve, microseconds:

`p_u = (QᵀQ + Quᵀ(Cu − I)Qu + λI)⁻¹ Quᵀcu`

This is called **fold-in**, and it is how a production system serves a user who
signed up an hour ago without retraining.

**On accuracy claims.** 81.9% of predictions land within ±1 star; 97.4% within
±2 stars. Any headline "accuracy" number for a recommender is meaningless
without its tolerance, so both are stated here. The ranking metric matters more
in practice: NDCG@10 of 0.0844 is 2.15× what a plain popularity chart achieves
on the same held-out, chronologically-split data.

**The autoencoder, and an honest ablation.** A 4-layer denoising autoencoder
(3000 → 512 → **128** → 512 → 3000, 3.2M parameters, backprop and Adam written
from scratch in NumPy) compresses each film's rating profile into a
128-dimensional embedding, reaching 0.666 masked validation RMSE. The
embeddings are clearly meaningful — *Pulp Fiction*'s nearest neighbours are
Reservoir Dogs (0.96), Goodfellas, Inglourious Basterds and both Kill Bills,
i.e. it recovered Tarantino with no director metadata anywhere in the input.

But adding them to the personalised blend **did not help**: a validation sweep
of the blend weight found the optimum at exactly 0.0, and forcing weight 0.2
cost −0.3% NDCG@10. They are largely redundant with implicit ALS, which learns
comparable structure from the same signal more directly. So the production
blend excludes the autoencoder, and it powers the "neural" similarity mode
instead, where its embeddings are genuinely useful. Reporting this rather than
quietly shipping a component that does nothing is the point of running an
ablation.

**Known limits.** Content features are genre + year + title only (the dataset
ships no cast or plot data). Catalogue coverage is 5.9% — fusion buys accuracy
by concentrating on well-understood films. The model is 26% less accurate for
users with under 20 ratings than for power users.
    """)
    st.caption("Data: MovieLens 25M (Harper & Konstan, 2015). "
               "Metadata and artwork: TMDB. This product uses the TMDB API but "
               "is not endorsed or certified by TMDB.")
