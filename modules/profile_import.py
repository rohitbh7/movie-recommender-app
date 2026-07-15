"""
profile_import.py

Builds a MovieLens-compatible ratings dataframe (columns: [movieId, rating])
from an external ratings export (IMDb or Letterboxd).

Public API
----------
build_profile(source, csv_path, lookup_path)  -> dispatches to the right pipeline
build_imdb_profile(csv_path, lookup_path)     -> IMDb export -> ratings dataframe
build_letterboxd_profile(csv_path, lookup_path) -> Letterboxd export -> ratings dataframe
"""

import re
import pandas as pd
from difflib import SequenceMatcher


# ------------------------------------------------------------------ #
#  IMDb                                                                #
# ------------------------------------------------------------------ #

IMDB_REQUIRED_COLUMNS = {"Const", "Your Rating", "Title", "Year"}


def _load_imdb_export(csv_path: str) -> pd.DataFrame:
    """
    Load a user's IMDb ratings export.
    Export from: https://www.imdb.com/list/ratings -> "..." menu -> Export
    """
    df = pd.read_csv(csv_path)

    missing = IMDB_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected IMDb export format, missing columns: {missing}")

    df = df.rename(columns={
        "Const": "imdb_id",       # e.g. tt0111161
        "Your Rating": "rating",  # 1-10 scale
        "Title": "title",
        "Year": "year",
    })

    df["rating"] = df["rating"].astype(float)
    return df[["imdb_id", "title", "year", "rating"]]


def _map_imdb_to_movielens(user_ratings: pd.DataFrame, lookup_path: str) -> pd.DataFrame:
    """
    Map IMDb IDs to MovieLens movieIds using movie_lookup.pkl, which is
    indexed by movieId and has imdbId stored as a bare int (e.g. 111161
    for tt0111161 -- no 'tt' prefix, no zero-padding).
    """
    lookup = pd.read_pickle(lookup_path)

    # movieId currently lives in the index -- pull it out as a column,
    # and dedupe since the table appears to have one row per (movieId, userId)
    lookup_movies = (
        lookup.reset_index()[["movieId", "imdbId"]]
        .drop_duplicates(subset="movieId")
    )

    # user_ratings["imdb_id"] is a string like "tt0111161" -> strip to bare int
    user_ratings = user_ratings.copy()
    user_ratings["imdbId"] = (
        user_ratings["imdb_id"].str.replace("tt", "", regex=False).astype(int)
    )

    merged = user_ratings.merge(lookup_movies, on="imdbId", how="left")

    matched = merged.dropna(subset=["movieId"])
    unmatched = merged[merged["movieId"].isna()]

    if len(unmatched) > 0:
        print(f"Warning: {len(unmatched)}/{len(merged)} titles had no MovieLens match "
              f"(likely too new/obscure for the ML-20M dataset).")
        print(unmatched[["title", "year", "imdb_id"]].to_string(index=False))

    matched = matched.copy()
    matched["movieId"] = matched["movieId"].astype(int)
    return matched


def _convert_10_to_5_scale(df: pd.DataFrame) -> pd.DataFrame:
    """IMDb ratings are 1-10, MovieLens ratings are 0.5-5 in 0.5 steps."""
    df = df.copy()
    df["rating"] = (df["rating"] / 2).round(1)
    return df


def build_imdb_profile(csv_path: str, lookup_path: str = "models/movie_lookup.pkl") -> pd.DataFrame:
    """Full pipeline: IMDb export -> MovieLens-compatible ratings dataframe."""
    raw = _load_imdb_export(csv_path)
    mapped = _map_imdb_to_movielens(raw, lookup_path)
    scaled = _convert_10_to_5_scale(mapped)
    return scaled[["movieId", "rating"]]


# ------------------------------------------------------------------ #
#  Letterboxd                                                          #
# ------------------------------------------------------------------ #

LETTERBOXD_REQUIRED_COLUMNS = {"Date", "Name", "Year", "Letterboxd URI", "Rating"}


def _load_letterboxd_export(csv_path: str) -> pd.DataFrame:
    """
    Load a user's Letterboxd ratings export.
    Export from: Letterboxd Settings -> Import & Export -> Export Your Data
    (ratings.csv inside the downloaded zip)
    """
    df = pd.read_csv(csv_path)

    missing = LETTERBOXD_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected Letterboxd export format, missing columns: {missing}")

    df = df.rename(columns={
        "Name": "title",
        "Year": "year",
        "Letterboxd URI": "letterboxd_url",
        "Rating": "rating",
        "Date": "date_rated",
    })

    df["rating"] = df["rating"].astype(float)
    df["year"] = df["year"].astype("Int64")  # nullable int, Letterboxd sometimes leaves this blank
    return df[["title", "year", "rating", "letterboxd_url", "date_rated"]]


def _normalize_title(title: str) -> str:
    """Lowercase, strip punctuation/articles for fuzzier matching."""
    title = title.lower().strip()
    title = re.sub(r"^(the|a|an)\s+", "", title)
    title = re.sub(r"[^\w\s]", "", title)
    return title.strip()


def _title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize_title(a), _normalize_title(b)).ratio()


def _map_letterboxd_to_movielens(
    user_ratings: pd.DataFrame,
    lookup_path: str,
    similarity_threshold: float = 0.90,
) -> pd.DataFrame:
    """
    Match Letterboxd (title, year) pairs to MovieLens movieIds using
    movie_lookup.pkl (indexed by movieId, has a 'title' column).

    Strategy:
      1. Exact match on (normalized title, year) -- fast, catches ~90%+
      2. Fuzzy fallback on title within same year for anything unmatched
    """
    lookup = pd.read_pickle(lookup_path)

    lookup_movies = (
        lookup.reset_index()[["movieId", "title"]]
        .drop_duplicates(subset="movieId")
        .copy()
    )

    # MovieLens titles are formatted like "Toy Story (1995)" -- pull year out
    year_extract = lookup_movies["title"].str.extract(r"\((\d{4})\)$")
    lookup_movies["year"] = pd.to_numeric(year_extract[0], errors="coerce").astype("Int64")
    lookup_movies["title_clean"] = (
        lookup_movies["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True)
    )
    lookup_movies["title_norm"] = lookup_movies["title_clean"].apply(_normalize_title)

    user_ratings = user_ratings.copy()
    user_ratings["title_norm"] = user_ratings["title"].apply(_normalize_title)

    # Pass 1: exact match on (title_norm, year)
    exact = user_ratings.merge(
        lookup_movies[["movieId", "title_norm", "year"]],
        on=["title_norm", "year"],
        how="left",
    )

    matched_mask = exact["movieId"].notna()
    matched = exact[matched_mask].copy()
    unmatched = exact[~matched_mask].drop(columns=["movieId"])

    # Pass 2: fuzzy match within same year for whatever pass 1 missed
    fuzzy_matches = []
    if len(unmatched) > 0:
        for _, row in unmatched.iterrows():
            candidates = lookup_movies[lookup_movies["year"] == row["year"]]
            if candidates.empty:
                continue

            scores = candidates["title_clean"].apply(lambda t: _title_similarity(t, row["title"]))
            best_idx = scores.idxmax()
            best_score = scores.loc[best_idx]

            if best_score >= similarity_threshold:
                match_row = row.to_dict()
                match_row["movieId"] = candidates.loc[best_idx, "movieId"]
                fuzzy_matches.append(match_row)

    if fuzzy_matches:
        fuzzy_df = pd.DataFrame(fuzzy_matches)
        matched = pd.concat([matched, fuzzy_df], ignore_index=True)
        still_unmatched = unmatched[~unmatched["title"].isin(fuzzy_df["title"])]
    else:
        still_unmatched = unmatched

    if len(still_unmatched) > 0:
        print(f"Warning: {len(still_unmatched)}/{len(exact)} titles had no MovieLens match.")
        print(still_unmatched[["title", "year"]].to_string(index=False))

    matched["movieId"] = matched["movieId"].astype(int)
    return matched


def build_letterboxd_profile(csv_path: str, lookup_path: str = "models/movie_lookup.pkl") -> pd.DataFrame:
    """Full pipeline: Letterboxd export -> MovieLens-compatible ratings dataframe."""
    raw = _load_letterboxd_export(csv_path)
    mapped = _map_letterboxd_to_movielens(raw, lookup_path)
    # No rating rescale needed -- Letterboxd is already 0.5-5.0
    return mapped[["movieId", "rating"]]


# ------------------------------------------------------------------ #
#  Dispatch                                                            #
# ------------------------------------------------------------------ #

BUILDERS = {
    "imdb": build_imdb_profile,
    "letterboxd": build_letterboxd_profile,
}


def build_profile(source: str, csv_path: str, lookup_path: str = "models/movie_lookup.pkl") -> pd.DataFrame:
    """Dispatch to the right pipeline based on source ('imdb' or 'letterboxd')."""
    try:
        builder = BUILDERS[source]
    except KeyError:
        raise ValueError(f"Unknown profile source: {source!r} (expected one of {list(BUILDERS)})")
    return builder(csv_path, lookup_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert an external ratings export to MovieLens format")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--imdb", type=str, metavar="CSV", help="Path to an IMDb ratings export CSV")
    source.add_argument("--letterboxd", type=str, metavar="CSV", help="Path to a Letterboxd ratings export CSV")
    parser.add_argument("--lookup_path", type=str, default="models/movie_lookup.pkl")
    args = parser.parse_args()

    if args.imdb:
        result = build_profile("imdb", args.imdb, args.lookup_path)
    else:
        result = build_profile("letterboxd", args.letterboxd, args.lookup_path)

    print(result.to_string(index=False))
