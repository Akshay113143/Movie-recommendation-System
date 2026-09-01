"""
TMDB API client -- posters, overviews, cast, ratings.

Design notes that matter for a deployed app:

1. GRACEFUL DEGRADATION. If no API key is configured or TMDB is unreachable,
   every function returns None and the UI falls back to text-only cards. The
   recommender is the product; posters are decoration. An app that crashes
   because an external API is down is a badly built app.

2. CACHING. `@st.cache_data` memoises by argument, so scrolling a results grid
   re-renders without re-hitting the API. TMDB's free tier is rate-limited
   (~50 req/s), and without caching a 12-card grid would burn a request per
   card per rerun -- Streamlit reruns the whole script on every widget
   interaction.

3. TITLE-BASED LOOKUP. The MovieLens archive used here ships only movies.csv
   and ratings.csv, with no `links.csv`, so there is no tmdbId to join on. We
   search by cleaned title + release year, which resolves the large majority of
   the catalogue. If you download the full MovieLens 25M zip, `links.csv` gives
   you an exact tmdbId per movie -- swap `search_movie` for a direct
   /movie/{tmdb_id} lookup and accuracy goes to 100%.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

BASE = "https://api.themoviedb.org/3"
IMG = "https://image.tmdb.org/t/p/w342"
TIMEOUT = 6


def get_api_key() -> str | None:
    """Read the key from Streamlit secrets, then the environment.

    Never hard-code a key in source. On Streamlit Community Cloud you set it
    under App settings -> Secrets; locally you put it in
    .streamlit/secrets.toml, which is gitignored.
    """
    try:
        if "TMDB_API_KEY" in st.secrets:
            return st.secrets["TMDB_API_KEY"]
    except Exception:
        pass
    return os.environ.get("TMDB_API_KEY")


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def search_movie(title: str, year: float | None = None) -> dict | None:
    """Look up one film by title (+year) and return the fields the UI needs."""
    key = get_api_key()
    if not key:
        return None
    params = {"api_key": key, "query": title, "include_adult": "false"}
    if year and year == year:                     # not NaN
        params["year"] = int(year)
    try:
        r = requests.get(f"{BASE}/search/movie", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results and "year" in params:      # retry without the year filter
            params.pop("year")
            r = requests.get(f"{BASE}/search/movie", params=params, timeout=TIMEOUT)
            r.raise_for_status()
            results = r.json().get("results", [])
        if not results:
            return None
        m = results[0]
        return {
            "tmdb_id": m.get("id"),
            "poster": IMG + m["poster_path"] if m.get("poster_path") else None,
            "overview": m.get("overview") or "",
            "tmdb_rating": m.get("vote_average"),
            "tmdb_votes": m.get("vote_count"),
            "release_date": m.get("release_date", ""),
        }
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def movie_details(tmdb_id: int) -> dict | None:
    """Director + top cast, for the detail panel."""
    key = get_api_key()
    if not key or not tmdb_id:
        return None
    try:
        r = requests.get(f"{BASE}/movie/{tmdb_id}",
                         params={"api_key": key, "append_to_response": "credits"},
                         timeout=TIMEOUT)
        r.raise_for_status()
        j = r.json()
        credits = j.get("credits", {})
        directors = [c["name"] for c in credits.get("crew", [])
                     if c.get("job") == "Director"][:2]
        cast = [c["name"] for c in credits.get("cast", [])[:5]]
        return {
            "runtime": j.get("runtime"),
            "tagline": j.get("tagline") or "",
            "directors": directors,
            "cast": cast,
            "homepage": j.get("homepage") or "",
        }
    except Exception:
        return None


def poster_for(title: str, year: float | None = None) -> str | None:
    info = search_movie(title, year)
    return info["poster"] if info else None
