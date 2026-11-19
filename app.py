import io
import zipfile

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")


@st.cache_resource(show_spinner="Loading data and training model (pehli baar thoda time lagega)...")
def load_everything():
    z = zipfile.ZipFile(io.BytesIO(requests.get(URL, timeout=60).content))
    movies = pd.read_csv(z.open("ml-latest-small/movies.csv"))
    ratings = pd.read_csv(z.open("ml-latest-small/ratings.csv"))
    tags = pd.read_csv(z.open("ml-latest-small/tags.csv"))

    # ---------- Content-based ----------
    movies["genres_clean"] = (
        movies["genres"]
        .str.replace("|", " ", regex=False)
        .str.replace("(no genres listed)", "", regex=False)
    )
    tags_agg = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x)).reset_index()
    movies = movies.merge(tags_agg, on="movieId", how="left")
    movies["tag"] = movies["tag"].fillna("")
    movies["content"] = movies["genres_clean"] + " " + movies["tag"]

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["content"])

    # ---------- Collaborative (SVD) ----------
    user_ids = sorted(ratings["userId"].unique())
    movie_ids = sorted(ratings["movieId"].unique())
    user_map = {u: i for i, u in enumerate(user_ids)}
    movie_map = {m: i for i, m in enumerate(movie_ids)}

    R = np.zeros((len(user_ids), len(movie_ids)))
    uidx = ratings["userId"].map(user_map).values
    midx = ratings["movieId"].map(movie_map).values
    R[uidx, midx] = ratings["rating"].values

    mask = R > 0
    user_mean = R.sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
    R_centered = np.where(mask, R - user_mean[:, None], 0)
    U, sigma, Vt = svds(R_centered, k=50)

    pred = U @ np.diag(sigma) @ Vt
    pred = pred + user_mean[:, None]
    pred = np.clip(pred, 0.5, 5)

    movieid_to_row = {m: i for i, m in enumerate(movies["movieId"])}
    movie_rows = np.array([movieid_to_row[m] for m in movie_ids])

    return dict(
        movies=movies, ratings=ratings, tfidf_matrix=tfidf_matrix,
        user_ids=user_ids, movie_ids=movie_ids, user_map=user_map,
        mask=mask, pred=pred, movieid_to_row=movieid_to_row,
        movie_rows=movie_rows, counts=mask.sum(axis=0),
    )


D = load_everything()
movies, ratings = D["movies"], D["ratings"]


def minmax(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-9)


def similar_movies(idx, n):
    sim = cosine_similarity(D["tfidf_matrix"][idx], D["tfidf_matrix"]).flatten()
    top = sim.argsort()[::-1][1:n + 1]
    res = movies.iloc[top][["title", "genres"]].copy()
    res["similarity"] = sim[top].round(3)
    return res


def content_scores_for_user(user_id):
    liked = ratings[(ratings["userId"] == user_id) & (ratings["rating"] >= 4)]["movieId"]
    rows = [D["movieid_to_row"][m] for m in liked if m in D["movieid_to_row"]]
    if not rows:
        return np.zeros(len(D["movie_ids"]))
    profile = np.asarray(D["tfidf_matrix"][rows].mean(axis=0))
    sims = cosine_similarity(profile, D["tfidf_matrix"]).flatten()
    return sims[D["movie_rows"]]


def recommend_for_user(user_id, n, alpha):
    u = D["user_map"][user_id]
    cf = minmax(D["pred"][u])
    ct = minmax(content_scores_for_user(user_id))
    scores = alpha * cf + (1 - alpha) * ct
    scores = scores.copy()
    scores[D["mask"][u]] = -1
    scores[D["counts"] < 20] = -1
    top = np.argsort(scores)[::-1][:n]
    ids = [D["movie_ids"][i] for i in top]
    res = movies.set_index("movieId").loc[ids, ["title", "genres"]]
    res = res.reset_index(drop=True)
    res["score"] = scores[top].round(3)
    return res


# ---------------- UI ----------------
st.title("🎬 Movie Recommendation System")
st.caption("Content-Based + Collaborative (SVD) + Hybrid | MovieLens dataset")

tab1, tab2 = st.tabs(["Similar movies", "For a user"])

with tab1:
    titles = sorted(movies["title"].tolist())
    choice = st.selectbox("Koi movie chuno", titles)
    n1 = st.slider("Kitni recommendations?", 5, 20, 10, key="n1")
    if st.button("Recommend karo", key="b1"):
        idx = movies.index[movies["title"] == choice][0]
        st.dataframe(similar_movies(idx, n1), hide_index=True)

with tab2:
    uid = st.number_input(
        "User ID",
        min_value=int(min(D["user_ids"])),
        max_value=int(max(D["user_ids"])),
        value=1,
        step=1,
    )
    alpha = st.slider("Alpha (1 = sirf CF, 0 = sirf content)", 0.0, 1.0, 0.7, 0.1)
    n2 = st.slider("Kitni recommendations?", 5, 20, 10, key="n2")
    if st.button("Recommend karo", key="b2"):
        st.dataframe(recommend_for_user(int(uid), n2, alpha), hide_index=True)
