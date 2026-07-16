"""
app.py

Streamlit front end for the hybrid movie recommender in main.py.

Upload an IMDb or Letterboxd ratings export and it shows all 5
recommendation rows at once:
    - 3x "Because you liked X" (item-based, for your top-3 rated movies)
    - "Users with similar taste like" (ALS collaborative filtering)
    - "Recommended for you" (hybrid overall ranking)

Run with:
    streamlit run app.py
"""

import contextlib
import csv
import io
import tempfile
from pathlib import Path

import streamlit as st

import main as core
from modules import als_rec, content_rec, item_rec, profile_import

MODEL_DIR = "models"
POSTER_W, POSTER_H = 150, 225
STREAMING_PLATFORMS_CSV = "streaming_platforms.csv"


# ------------------------------------------------------------------ #
#  Artifact loading (cached so it only happens once per server)       #
# ------------------------------------------------------------------ #

@st.cache_resource(show_spinner="Loading model artifacts...")
def load_all_artifacts(model_dir: str) -> bool:
    als_rec.load_artifacts(model_dir)
    content_rec.load_artifacts(model_dir)
    item_rec.load_artifacts(model_dir=model_dir)
    return True


# ------------------------------------------------------------------ #
#  Recommendation builders (mirror main.py's run_items/run_users/run_overall) #
# ------------------------------------------------------------------ #

def get_items_recs(user_df, n, selected_platforms, platforms):
    top3 = core._top_rated_movies(user_df, k=3)
    lists = []
    pool_n = max(n * 5, 100)
    for _, row in top3.iterrows():
        movie_id = int(row["movieId"])
        try:
            title = item_rec.get_movie_title(movie_id)
        except ValueError:
            title = row.get("title", f"movieId {movie_id}")
        try:
            results = item_rec.recommend_similar(movie_id, n=pool_n)
            results = filter_by_platforms(results, selected_platforms, platforms).head(n)
        except ValueError:
            results = None
        lists.append((title, results))
    return lists


def get_users_recs(user_df, n, selected_platforms, platforms):
    recs = als_rec.recommend_als(user_df, n=300)
    recs = filter_by_platforms(recs, selected_platforms, platforms)
    return recs.head(n) if not recs.empty else recs


def get_overall_recs(user_df, n, selected_platforms, platforms):
    recs = als_rec.recommend_als(user_df, n=300)
    if recs.empty:
        return recs

    recs = filter_by_platforms(recs, selected_platforms, platforms)
    if recs.empty:
        return recs

    recs = content_rec.score_content(user_df, recs)

    recs["als_norm"] = core._normalise(recs["als_score"]).clip(upper=0.85)
    recs["content_norm"] = core._normalise(recs["content_score"])
    recs["pop_norm"] = core._normalise(recs["pop_score"])
    recs["vote_norm"] = core._normalise(recs["vote_average"])
    recs["tmdb_pop_norm"] = core._normalise(recs["popularity"])

    recs["final_score"] = (
        core.W_ALS * recs["als_norm"]
        + core.W_CONTENT * recs["content_norm"]
        + core.W_POP * recs["pop_norm"]
        + core.W_VOTE * recs["vote_norm"]
        + core.W_TMDB * recs["tmdb_pop_norm"]
    )

    return recs.sort_values("final_score", ascending=False).head(n).reset_index(drop=True)


# ------------------------------------------------------------------ #
#  Rendering                                                           #
# ------------------------------------------------------------------ #

def _valid_poster(url) -> bool:
    if url is None:
        return False
    s = str(url).strip()
    return s != "" and s.lower() not in {"nan", "none", "n/a"}


def _parse_streaming(val):
    if val is None:
        return []
    if isinstance(val, list):
        return val
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none", "n/a", "[]"}:
        return []
    try:
        import ast
        parsed = ast.literal_eval(s)
        return parsed if isinstance(parsed, list) else [s]
    except (ValueError, SyntaxError):
        return [s]


@st.cache_data
def load_streaming_platforms(path: str = STREAMING_PLATFORMS_CSV):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _match_platform(raw_name, platforms):
    n = raw_name.lower()
    for p in platforms:
        if p["keywords"] and p["keywords"] in n:
            if p["exclude_keywords"] and p["exclude_keywords"] in n:
                continue
            return p
    return None


def _resolve_services(raw_list, platforms):
    """Split raw streaming names into de-duped known platforms and leftover text."""
    matched = {}
    unmatched = []
    for name in raw_list:
        p = _match_platform(name, platforms)
        if p:
            matched[p["platform"]] = p
        elif name not in unmatched:
            unmatched.append(name)
    return list(matched.values()), unmatched


OTHER_KEY = "__other__"
OTHER_LABEL = "Other Streaming Platforms"
NONE_KEY = "__none__"
NONE_LABEL = "None"


def _movie_matches_selection(streaming_val, platforms, selected_platforms):
    raw_list = _parse_streaming(streaming_val)
    if not raw_list:
        return NONE_KEY in selected_platforms

    matched, unmatched = _resolve_services(raw_list, platforms)
    if matched:
        matched_keys = {p["platform"] for p in matched}
        return bool(matched_keys & selected_platforms)

    # only unrecognized/obscure services -- not on any streaming_platforms.csv entry
    return bool(unmatched) and OTHER_KEY in selected_platforms


def filter_by_platforms(df, selected_platforms, platforms):
    """Keep only rows matching selected_platforms. No-op if every option (including Other/None) is selected."""
    if df is None or df.empty or len(selected_platforms) >= len(platforms) + 2:
        return df
    mask = df["streaming"].apply(
        lambda s: _movie_matches_selection(s, platforms, selected_platforms)
    )
    return df[mask].reset_index(drop=True)


def render_movie_card(title, poster_url, streaming):
    if _valid_poster(poster_url):
        st.markdown(
            f'<img src="{poster_url}" '
            f'style="width:{POSTER_W}px;height:{POSTER_H}px;object-fit:cover;border-radius:6px;">',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="width:{POSTER_W}px;height:{POSTER_H}px;'
            f'background:#e0e0e0;border-radius:6px;"></div>',
            unsafe_allow_html=True,
        )
    st.markdown(f"**{title}**")
    services = _parse_streaming(streaming)
    if not services:
        st.caption("Not currently streaming")
        return

    platforms, unmatched = _resolve_services(services, load_streaming_platforms())
    html = '<div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center;">'
    for p in platforms:
        html += (
            f'<a href="{p["website_url"]}" target="_blank" title="{p["label"]}">'
            f'<img src="{p["logo_url"]}" style="width:22px;height:22px;border-radius:4px;">'
            f"</a>"
        )
    for name in unmatched:
        html += f'<span style="font-size:0.75em;color:gray;">{name}</span>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_row(row_title, df, subtitle=None):
    st.subheader(row_title)
    if subtitle:
        st.caption(subtitle)
    if df is None or df.empty:
        st.info("No recommendations available.")
        return
    cols = st.columns(len(df))
    for col, (_, row) in zip(cols, df.iterrows()):
        with col:
            render_movie_card(row["title"], row.get("poster_url"), row.get("streaming"))


# ------------------------------------------------------------------ #
#  App                                                                 #
# ------------------------------------------------------------------ #

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("Movie Recommender")

st.sidebar.header("Your Ratings")
source_label = st.sidebar.radio("Export source", ["IMDb", "Letterboxd"])
uploaded = st.sidebar.file_uploader(
    "Upload ratings export (.csv)",
    type="csv",
    help="IMDb: Your Ratings page -> ... menu -> Export.\n"
         "Letterboxd: Settings -> Import & Export -> Export Your Data (use ratings.csv from the zip).",
)
n = st.sidebar.slider("Recommendations per list", min_value=3, max_value=20, value=10)

all_platforms = load_streaming_platforms()
platform_labels = [p["label"] for p in all_platforms] + [OTHER_LABEL, NONE_LABEL]
label_to_key = {p["label"]: p["platform"] for p in all_platforms}
label_to_key[OTHER_LABEL] = OTHER_KEY
label_to_key[NONE_LABEL] = NONE_KEY
selected_labels = st.sidebar.multiselect(
    "Streaming services",
    options=platform_labels,
    default=platform_labels,
    help="Only recommend movies available on the services you select. "
         "'Other Streaming Platforms' covers services not listed here; "
         "'None' covers movies with no known streaming availability.",
)
selected_platforms = {label_to_key[l] for l in selected_labels}

run_clicked = st.sidebar.button("Get Recommendations", type="primary")

if run_clicked:
    if uploaded is None:
        st.sidebar.error("Please upload a ratings CSV first.")
    else:
        load_all_artifacts(MODEL_DIR)

        suffix = Path(uploaded.name).suffix or ".csv"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        lookup_path = str(Path(MODEL_DIR) / "movie_lookup.pkl")
        source = "imdb" if source_label == "IMDb" else "letterboxd"

        log = io.StringIO()
        user_df = None
        try:
            with st.spinner(f"Matching {source_label} ratings to the movie catalog..."):
                with contextlib.redirect_stdout(log):
                    user_df = profile_import.build_profile(source, tmp_path, lookup_path)
        except ValueError as e:
            st.error(f"Could not parse {source_label} export: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        if user_df is not None:
            if user_df.empty:
                st.error("None of the rated movies could be matched to the MovieLens catalog.")
            else:
                with st.spinner("Computing recommendations..."):
                    st.session_state["results"] = {
                        "items": get_items_recs(user_df, n, selected_platforms, all_platforms),
                        "users": get_users_recs(user_df, n, selected_platforms, all_platforms),
                        "overall": get_overall_recs(user_df, n, selected_platforms, all_platforms),
                        "import_log": log.getvalue(),
                        "num_matched": len(user_df),
                    }

if "results" in st.session_state:
    results = st.session_state["results"]
    st.success(f"Matched {results['num_matched']} rated movies to the MovieLens catalog.")
    if results["import_log"].strip():
        with st.expander("Import details / unmatched titles"):
            st.text(results["import_log"])

    render_row("Recommended for you", results["overall"])
    render_row("Users with similar taste like", results["users"])

    for title, df in results["items"]:
        render_row(f"Because you liked: {title}", df)
else:
    st.info("Upload an IMDb or Letterboxd ratings export in the sidebar, then click **Get Recommendations**.")
