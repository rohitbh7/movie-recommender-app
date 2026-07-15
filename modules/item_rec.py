"""
item_recommender.py

Item-based content recommender using MovieLens genome tag similarity.

Public API
----------
load_artifacts(model_dir)   -> call once at startup
recommend_similar(movie_id, n)      -> returns top-n similar movies
get_movie_title(movie_id)           -> returns title string
"""

import ast
import pickle
import numpy as np
import pandas as pd
import scipy.sparse
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


# ---------- module-level state ----------
_tag_matrix = None   # ndarray (n_movies, n_tags)
_tag_index  = None   # list of movieIds matching rows
_movies     = None   # DataFrame with title/streaming/poster_url, indexed by movieId


def _parse_tags(val):
    if pd.isna(val):
        return {}
    try:
        return ast.literal_eval(val)
    except Exception:
        return {}


def load_artifacts(model_dir: str = "models") -> None:
    global _tag_matrix, _tag_index, _movies

    cache_matrix = Path(model_dir) / "item_tag_matrix.npz"
    cache_meta   = Path(model_dir) / "item_movies.pkl"

    if cache_matrix.exists() and cache_meta.exists():
        _tag_matrix = scipy.sparse.load_npz(str(cache_matrix)).toarray()
        with open(cache_meta, "rb") as f:
            _tag_index, _movies = pickle.load(f)
        print(f"Tag feature matrix: {_tag_matrix.shape} (loaded from cache)")
        return

    _movies = pickle.load(open(Path(model_dir) / "movie_lookup.pkl", "rb"))

    tag_dicts = _movies["genome_tags"].apply(_parse_tags)
    tag_df    = pd.DataFrame(list(tag_dicts), index=_movies.index).fillna(0)

    print(f"Tag feature matrix: {tag_df.shape}")

    _tag_index  = tag_df.index.tolist()
    _tag_matrix = tag_df.values

    # cache for future runs
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    scipy.sparse.save_npz(str(cache_matrix), scipy.sparse.csr_matrix(_tag_matrix))
    with open(cache_meta, "wb") as f:
        pickle.dump((_tag_index, _movies[["title", "streaming", "poster_url"]]), f)
    print("Tag feature matrix cached to models/.")


def recommend_similar(movie_id: int, n: int = 10) -> pd.DataFrame:
    if _tag_matrix is None:
        raise RuntimeError("Call load_artifacts() before recommend_similar().")
    if movie_id not in _tag_index:
        raise ValueError(f"movieId {movie_id} not found.")

    idx    = _tag_index.index(movie_id)
    query  = _tag_matrix[idx : idx + 1]          # (1, n_tags)
    scores = cosine_similarity(query, _tag_matrix).flatten()

    results = pd.Series(scores, index=_tag_index)
    results = results.drop(movie_id).sort_values(ascending=False).head(n)

    return pd.DataFrame({
        "title":      _movies.loc[results.index, "title"],
        "streaming":  _movies.loc[results.index, "streaming"],
        "poster_url": _movies.loc[results.index, "poster_url"],
        "similarity": results,
    })


def get_movie_title(movie_id: int) -> str:
    if _movies is None:
        raise RuntimeError("Call load_artifacts() first.")
    if movie_id not in _movies.index:
        raise ValueError(f"movieId {movie_id} not found.")
    return _movies.loc[movie_id, "title"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Item-based movie recommender")
    parser.add_argument("--movie_id",  type=int, required=True)
    parser.add_argument("--n",         type=int, default=10)
    parser.add_argument("--model_dir", type=str, default="models")
    args = parser.parse_args()

    load_artifacts(args.model_dir)

    title = get_movie_title(args.movie_id)
    print(f"\nMovies similar to: {title}\n")
    print(recommend_similar(args.movie_id, n=args.n).to_string())
