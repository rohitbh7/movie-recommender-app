"""
als_recommender.py

ALS-based candidate generation.

Public API
----------
train(df_path, model_dir)   -> train on full df and save artifacts
load_artifacts(model_dir)   -> call once at startup
recommend_als(user_df, n)   -> takes a user ratings DataFrame, returns candidates
"""

import pickle
import numpy as np
import pandas as pd
import scipy.sparse
from pathlib import Path
from implicit import als


# ---------- module-level state ----------
_model        = None
_user_cat     = None
_movie_cat    = None
_movie_pop    = None
_movie_meta   = None
_movie_lookup = None
_train_matrix = None
_vote_median  = None


def train(model_dir: str = "models") -> None:
    """
    Refit the ALS model using the existing train_matrix.npz (users x movies).
    Overwrites als_model.pkl only — all other artifacts stay the same.
    """
    d = Path(model_dir)

    print("Loading train_matrix.npz...")
    train_matrix = scipy.sparse.load_npz(str(d / "train_matrix.npz"))
    print(f"Train matrix: {train_matrix.shape}  (users x movies)")

    print("Training ALS model...")
    model = als.AlternatingLeastSquares(
        factors=50, regularization=1, iterations=20, use_gpu=False
    )
    model.fit(train_matrix)

    with open(d / "als_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("als_model.pkl saved.")


def load_artifacts(model_dir: str = "models") -> None:
    global _model, _user_cat, _movie_cat, _movie_pop
    global _movie_meta, _movie_lookup, _train_matrix, _vote_median

    d = Path(model_dir)

    with open(d / "als_model.pkl", "rb") as f:
        _model = pickle.load(f)

    _user_cat     = pickle.load(open(d / "user_cat.pkl",     "rb"))
    _movie_cat    = pickle.load(open(d / "movie_cat.pkl",    "rb"))
    _movie_pop    = pickle.load(open(d / "movie_pop.pkl",    "rb"))
    _movie_meta   = pickle.load(open(d / "movie_meta.pkl",   "rb"))
    _movie_lookup = pickle.load(open(d / "movie_lookup.pkl", "rb"))
    _train_matrix = scipy.sparse.load_npz(str(d / "train_matrix.npz"))
    _vote_median  = _movie_meta["vote_average"].median()


def _build_user_vector(user_df: pd.DataFrame) -> scipy.sparse.csr_matrix:
    mean_rating = user_df["rating"].mean()
    user_df = user_df.copy()
    user_df["rating_adj"] = (user_df["rating"] - mean_rating).clip(lower=0)

    known = user_df[user_df["movieId"].isin(_movie_cat.cat.categories)]

    if known.empty:
        n_items = len(_movie_cat.cat.categories)
        return scipy.sparse.csr_matrix((1, n_items))

    alpha      = 20
    movie_idxs = [_movie_cat.cat.categories.get_loc(m) for m in known["movieId"]]
    confidences = (1 + alpha * known["rating_adj"].values).tolist()
    n_items    = len(_movie_cat.cat.categories)

    return scipy.sparse.csr_matrix(
        (confidences, ([0] * len(movie_idxs), movie_idxs)),
        shape=(1, n_items),
    )


def recommend_als(user_df: pd.DataFrame, n: int = 300) -> pd.DataFrame:
    if _model is None:
        raise RuntimeError("Call load_artifacts() before recommend_als().")

    user_vector = _build_user_vector(user_df)
    liked_ids   = set(user_df["movieId"].tolist())

    ids, als_scores = _model.recommend(
        userid=0,
        user_items=user_vector,
        N=n + len(liked_ids),
        filter_already_liked_items=False,
    )

    movie_ids = [_movie_cat.cat.categories[i] for i in ids]
    recs = pd.DataFrame({"movieId": movie_ids, "als_score": als_scores})

    recs = recs[~recs["movieId"].isin(liked_ids)].head(n).reset_index(drop=True)

    recs["num_ratings"] = recs["movieId"].map(_movie_pop).fillna(1)
    recs["pop_score"]   = np.log1p(recs["num_ratings"])
    recs = recs[recs["num_ratings"] >= 50].reset_index(drop=True)

    recs = recs.merge(_movie_meta, on="movieId", how="left")
    recs["vote_average"] = recs["vote_average"].fillna(_vote_median)
    recs["popularity"]   = recs["popularity"].fillna(0)

    recs["title"]      = recs["movieId"].map(_movie_lookup["title"])
    recs["streaming"]  = recs["movieId"].map(_movie_lookup["streaming"])  if "streaming"  in _movie_lookup.columns else None
    recs["poster_url"] = recs["movieId"].map(_movie_lookup["poster_url"]) if "poster_url" in _movie_lookup.columns else None

    return recs[["movieId", "title", "streaming", "poster_url",
                 "als_score", "num_ratings", "pop_score", "vote_average", "popularity"]]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="models")
    args = parser.parse_args()

    train(args.model_dir)
