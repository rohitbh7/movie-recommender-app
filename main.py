"""
main.py

Hybrid movie recommender — merges ALS collaborative filtering with
content-based scoring.

Usage
-----
# Recommend for a known user in the training data:
    python main.py --user_id 15

# Recommend for an arbitrary user from a CSV of their ratings:
    python main.py --ratings_csv my_ratings.csv

# Recommend from an IMDb or Letterboxd ratings export:
    python main.py --imdb my_imdb_ratings.csv
    python main.py --letterboxd my_letterboxd_ratings.csv

Mode flags (one or more, default: all):
    --items     "Because you liked X" rows for top-3 rated movies (item_rec)
    --users     "Users with similar taste like" list (als_rec)
    --overall   "Recommended for you" hybrid merged list

CSV format:
    movieId,rating
    318,5.0
    356,4.5

Optional flags:
    --n          Number of recommendations per list (default: 10)
    --model_dir  Directory containing saved artifacts (default: models)
    --df_path    Path to df.csv required for item_rec (default: df.csv)
"""

import argparse
import pickle
import numpy as np
import pandas as pd
import scipy.sparse
from pathlib import Path

from modules import als_rec
from modules import content_rec
from modules import item_rec
from modules import profile_import


# ---------- weights ----------
W_ALS     = 0.40
W_CONTENT = 0.40
W_POP     = 0.10
W_VOTE    = 0.07
W_TMDB    = 0.03


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _user_df_from_train(user_id: int, model_dir: str) -> pd.DataFrame:
    d = Path(model_dir)
    user_cat     = pickle.load(open(d / "user_cat.pkl",     "rb"))
    movie_cat    = pickle.load(open(d / "movie_cat.pkl",    "rb"))
    movie_lookup = pickle.load(open(d / "movie_lookup.pkl", "rb"))
    train_matrix = scipy.sparse.load_npz(str(d / "train_matrix.npz"))

    if user_id not in user_cat.cat.categories:
        raise ValueError(
            f"user_id {user_id} not found in training data. "
            "Supply a --ratings_csv instead."
        )

    u_idx = user_cat.cat.categories.get_loc(user_id)
    row   = train_matrix[u_idx].tocoo()

    movie_ids   = [movie_cat.cat.categories[i] for i in row.col]
    confidences = row.data.tolist()

    user_df = pd.DataFrame({"movieId": movie_ids, "rating": confidences})
    alpha = 20
    user_df["rating"] = ((user_df["rating"] - 1) / alpha + 3.0).clip(1, 5)
    user_df["title"]  = user_df["movieId"].map(movie_lookup["title"])
    return user_df


def _normalise(series: pd.Series) -> pd.Series:
    mx = series.max()
    if mx <= 0:
        return pd.Series(0.0, index=series.index)
    return series / mx


def _top_rated_movies(user_df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Return the k highest-rated movies for the user, with titles."""
    cols = ["movieId", "rating"] + (["title"] if "title" in user_df.columns else [])
    top = (
        user_df[cols]
        .sort_values("rating", ascending=False)
        .drop_duplicates("movieId")
        .head(k)
        .reset_index(drop=True)
    )
    return top


# ------------------------------------------------------------------ #
#  Recommendation modes                                                #
# ------------------------------------------------------------------ #

def run_items(user_df: pd.DataFrame, n: int) -> None:
    """Print 3 'Because you liked X' lists using item_rec."""
    top3 = _top_rated_movies(user_df, k=3)

    for _, row in top3.iterrows():
        movie_id = int(row["movieId"])
        try:
            title = item_rec.get_movie_title(movie_id)
        except ValueError:
            title = row.get("title", f"movieId {movie_id}")

        print(f"\nBecause you liked: {title}")
        print("-" * 50)
        try:
            results = item_rec.recommend_similar(movie_id, n=n)
            print(results[["title", "streaming", "poster_url", "similarity"]].to_string(index=False))
        except ValueError as e:
            print(f"  (skipped: {e})")


def run_users(user_df: pd.DataFrame, n: int) -> None:
    """Print 'Users with similar taste like' list using als_rec."""
    print("\nUsers with similar taste like:")
    print("-" * 50)
    recs = als_rec.recommend_als(user_df, n=300)
    if recs.empty:
        print("  No results.")
        return
    recs = recs.head(n)
    print(recs[["title", "streaming", "poster_url", "num_ratings", "vote_average"]].to_string(index=False))


def run_overall(user_df: pd.DataFrame, n: int) -> None:
    """Print 'Recommended for you' hybrid list."""
    print("\nRecommended for you:")
    print("-" * 50)

    recs = als_rec.recommend_als(user_df, n=300)
    if recs.empty:
        print("  No results.")
        return

    recs = content_rec.score_content(user_df, recs)

    recs["als_norm"]      = _normalise(recs["als_score"]).clip(upper=0.85)
    recs["content_norm"]  = _normalise(recs["content_score"])
    recs["pop_norm"]      = _normalise(recs["pop_score"])
    recs["vote_norm"]     = _normalise(recs["vote_average"])
    recs["tmdb_pop_norm"] = _normalise(recs["popularity"])

    recs["final_score"] = (
        W_ALS     * recs["als_norm"]
        + W_CONTENT * recs["content_norm"]
        + W_POP     * recs["pop_norm"]
        + W_VOTE    * recs["vote_norm"]
        + W_TMDB    * recs["tmdb_pop_norm"]
    )

    recs = (
        recs
        .sort_values("final_score", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )

    print(
        recs[["title", "streaming", "poster_url", "final_score", "num_ratings", "vote_average"]]
        .to_string(index=False)
    )


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Hybrid movie recommender")

    # user source — one required
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--user_id",     type=int, help="Known user_id from training data")
    source.add_argument("--ratings_csv", type=str, help="CSV with columns [movieId, rating]")
    source.add_argument("--imdb",        type=str, metavar="CSV", help="Path to an IMDb ratings export CSV")
    source.add_argument("--letterboxd",  type=str, metavar="CSV", help="Path to a Letterboxd ratings export CSV")

    # mode flags
    parser.add_argument("--items",   action="store_true",
                        help="'Because you liked X' lists (item-based)")
    parser.add_argument("--users",   action="store_true",
                        help="'Users with similar taste like' list (ALS)")
    parser.add_argument("--overall", action="store_true",
                        help="'Recommended for you' hybrid list")

    # options
    parser.add_argument("--n",          type=int, default=10,
                        help="Recommendations per list (default: 10)")
    parser.add_argument("--model_dir",  type=str, default="models",
                        help="Directory with saved model artifacts (default: models/)")
    parser.add_argument("--df_path",    type=str, default="df.csv",
                        help="Path to df.csv for item_rec (default: df.csv)")

    args = parser.parse_args()

    # if no mode flag given, run all three
    run_all = not (args.items or args.users or args.overall)

    # --- load artifacts ---
    print("Loading artifacts...")

    need_items   = args.items   or run_all
    need_users   = args.users   or run_all
    need_overall = args.overall or run_all

    if need_users or need_overall:
        als_rec.load_artifacts(args.model_dir)

    if need_overall:
        content_rec.load_artifacts(args.model_dir)

    if need_items:
        item_rec.load_artifacts(model_dir=args.model_dir)

    # --- build user_df ---
    if args.user_id is not None:
        print(f"Looking up ratings for user_id={args.user_id}...")
        user_df = _user_df_from_train(args.user_id, args.model_dir)
        print(f"Found {len(user_df)} rated movies.")
    elif args.ratings_csv is not None:
        user_df = pd.read_csv(args.ratings_csv)
        if not {"movieId", "rating"}.issubset(user_df.columns):
            raise ValueError("CSV must contain 'movieId' and 'rating' columns.")
        print(f"Loaded {len(user_df)} ratings from {args.ratings_csv}.")
    else:
        lookup_path = str(Path(args.model_dir) / "movie_lookup.pkl")
        if args.imdb is not None:
            print(f"Converting IMDb export {args.imdb}...")
            user_df = profile_import.build_profile("imdb", args.imdb, lookup_path)
        else:
            print(f"Converting Letterboxd export {args.letterboxd}...")
            user_df = profile_import.build_profile("letterboxd", args.letterboxd, lookup_path)
        print(f"Matched {len(user_df)} ratings to MovieLens movieIds.")

    # --- run requested modes ---
    if need_items:
        run_items(user_df, n=args.n)

    if need_users:
        run_users(user_df, n=args.n)

    if need_overall:
        run_overall(user_df, n=args.n)


if __name__ == "__main__":
    main()