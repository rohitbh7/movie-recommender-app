"""
content_recommender.py

Content-based scoring using sentence-transformer embeddings.

Public API
----------
load_artifacts(model_dir)        -> call once at startup
score_content(user_df, recs_df)  -> adds content_score column to recs_df
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path


# ---------- module-level state ----------
_movie_embeddings = None
_movie_to_idx     = None


def load_artifacts(model_dir: str = "models") -> None:
    """Load embedding matrix and movie->index mapping from model_dir."""
    global _movie_embeddings, _movie_to_idx

    d = Path(model_dir)
    _movie_embeddings = np.load(str(d / "movie_embeddings.npy"))
    _movie_to_idx     = pickle.load(open(d / "movie_to_idx.pkl", "rb"))


def _build_user_vector(user_df: pd.DataFrame) -> np.ndarray:
    """
    Build a weighted average embedding for the user.

    Weights by rating so a 5-star film pulls harder than a 4-star film.
    Only uses movies the user rated >= 4 that exist in the embedding index.

    Parameters
    ----------
    user_df : DataFrame with columns [movieId, rating]

    Returns
    -------
    1-D numpy array of shape (embedding_dim,), or None if no overlap.
    """
    liked = user_df[user_df["rating"] >= 4].copy()
    liked = liked[liked["movieId"].isin(_movie_to_idx)]

    if liked.empty:
        return None

    idxs    = [_movie_to_idx[m] for m in liked["movieId"]]
    weights = liked["rating"].values

    return np.average(_movie_embeddings[idxs], weights=weights, axis=0)


def score_content(user_df: pd.DataFrame, recs: pd.DataFrame) -> pd.DataFrame:
    """
    Add a content_score column to recs by computing cosine-style dot product
    between each candidate's embedding and the user's taste vector.

    Parameters
    ----------
    user_df : DataFrame with columns [movieId, rating]
    recs    : DataFrame with at least a movieId column

    Returns
    -------
    recs with a new content_score column (0.0 if no content signal).
    """
    if _movie_embeddings is None:
        raise RuntimeError("Call load_artifacts() before score_content().")

    user_vector = _build_user_vector(user_df)

    if user_vector is None:
        recs = recs.copy()
        recs["content_score"] = 0.0
        return recs

    # normalise user vector once so dot product ≈ cosine similarity
    norm = np.linalg.norm(user_vector)
    if norm > 0:
        user_vector = user_vector / norm

    def _score(movie_id):
        if movie_id not in _movie_to_idx:
            return 0.0
        idx = _movie_to_idx[movie_id]
        vec = _movie_embeddings[idx]
        vec_norm = np.linalg.norm(vec)
        if vec_norm == 0:
            return 0.0
        return float((vec / vec_norm) @ user_vector)

    recs = recs.copy()
    recs["content_score"] = recs["movieId"].apply(_score)
    return recs