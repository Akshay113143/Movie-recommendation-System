# Deploying the app

The trained model is 586 MB — too big for GitHub (100 MB/file limit) and for
Streamlit Community Cloud's memory budget. `export_serving_bundle.py` solves
this by shipping **only the item-side arrays**, 19.5 MB in total, a 30×
reduction.

The trick that makes it possible: the deployed app never looks up a stored user
vector. A visitor picks a few films, and the app solves the implicit-ALS normal
equations for that person on the spot (**fold-in**, one 64×64 linear solve,
~10 ms end to end). So the 162,414 × 64 user-factor matrix — the single largest
object in the model — never needs to leave your laptop.

---

## 1. Get a TMDB API key (2 minutes, free)

1. Create an account at <https://www.themoviedb.org/signup>
2. Go to **Settings → API → Request an API Key → Developer**
3. Fill the form (personal/educational use is fine; for "Application URL" you
   can put your GitHub repo URL)
4. Copy the **API Key (v3 auth)** — a 32-character hex string

The app works without a key — it just renders text cards instead of posters —
so you can deploy first and add artwork later.

## 2. Run locally

```bash
pip install -r requirements.txt

# one-off: train, then export the deployment bundle
python run_pipeline.py --ratings data/ratings.csv --movies data/movies.csv
python export_serving_bundle.py

# local secrets (this file is gitignored — never commit a key)
mkdir -p .streamlit
echo 'TMDB_API_KEY = "your_key_here"' > .streamlit/secrets.toml

streamlit run streamlit_app.py
```

Opens on <http://localhost:8501>.

## 3. Push to GitHub

`app/bundle/` **must be committed** — it is what the deployed app loads. It is
19.5 MB, which is fine. `data/` and `artifacts/` stay ignored.

```bash
git add app streamlit_app.py export_serving_bundle.py requirements.txt \
        .streamlit/config.toml .streamlit/secrets.toml.example DEPLOYMENT.md
git commit -m "Add Streamlit app with TMDB integration and serving bundle"
git push
```

Sanity check before pushing — this should print nothing:

```bash
git status --porcelain | grep -E "secrets.toml$|model.pkl|ratings.csv"
```

## 4. Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in with GitHub
2. **Create app → Deploy a public app from GitHub**
3. Fill in:
   - Repository: `Akshay113143/Movie-recommendation-System`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
4. Before clicking Deploy, open **Advanced settings → Secrets** and paste:

   ```toml
   TMDB_API_KEY = "your_32_char_key"
   ```

5. Deploy. First build takes 2–4 minutes (installing scipy and scikit-learn).

Your app lands at
`https://<something>.streamlit.app` — you can rename it under
**Settings → General → App URL**. Put that link on your resume next to the
project.

## 5. If something breaks

**"Could not load the model bundle"** — `app/bundle/` wasn't committed. Check
with `git ls-files app/bundle` (should list 4 files).

**App is slow on first load** — expected. Community Cloud sleeps inactive apps;
the first request after sleeping reloads 19.5 MB of factors. Subsequent
requests hit `@st.cache_resource` and are instant.

**Posters missing in production but fine locally** — the key is in your local
`secrets.toml` but not in Cloud secrets. Add it under
**App settings → Secrets**, then **Reboot app**.

**"installer returned a non-zero exit code"** — pin versions in
`requirements.txt` if a fresh release breaks the build (`numpy==2.1.0` etc.).

**Rate limited by TMDB** — `@st.cache_data(ttl=24h)` should prevent this. If it
persists, lower the grid size from 12 to 8 cards.

## 6. Optional upgrade: exact TMDB matching

This archive ships only `movies.csv` and `ratings.csv`, so the app matches films
to TMDB by title + year, which resolves most but not all of the catalogue. The
full MovieLens 25M download includes `links.csv` with a real `tmdbId` per movie.
If you add it:

```python
links = pd.read_csv("data/links.csv")          # movieId, imdbId, tmdbId
catalogue = catalogue.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
```

then replace `tmdb.search_movie(title, year)` with a direct
`/movie/{tmdbId}` lookup — one request instead of a search, and 100% match
accuracy.
