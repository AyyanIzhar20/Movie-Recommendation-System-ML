import streamlit as st
import pandas as pd
import numpy as np
import joblib
from surprise import SVD, Dataset, Reader
from sklearn.metrics.pairwise import cosine_similarity
import os

# -------------------------------
# Load dataset
# -------------------------------
DATA_DIR = "ml-100k"
MOVIE_PATH = os.path.join(DATA_DIR, "u.item")
RATINGS_PATH = os.path.join(DATA_DIR, "u.data")

# Movie metadata
movies = pd.read_csv(
    MOVIE_PATH, sep="|", encoding="latin-1",
    header=None, usecols=[0, 1], names=["movieId", "title"]
)

# Ratings
ratings = pd.read_csv(
    RATINGS_PATH, sep="\t", names=["userId", "movieId", "rating", "timestamp"]
)

# User-Item matrix
user_item_matrix = ratings.pivot_table(
    index="userId", columns="movieId", values="rating"
)

# -------------------------------
# Load models
# -------------------------------
MODELS_DIR = "models"

# Optional pre-computed SVD predictions DataFrame
pred_df = None
if os.path.exists(os.path.join(MODELS_DIR, "pred_df.pkl")):
    pred_df = joblib.load(os.path.join(MODELS_DIR, "pred_df.pkl"))

# SVD model
svd_model = None
if os.path.exists(os.path.join(MODELS_DIR, "svd_model.pkl")):
    svd_model = joblib.load(os.path.join(MODELS_DIR, "svd_model.pkl"))

# Item similarity
item_similarity = None
if os.path.exists(os.path.join(MODELS_DIR, "item_similarity.pkl")):
    item_similarity = joblib.load(os.path.join(MODELS_DIR, "item_similarity.pkl"))
else:
    # Compute from scratch if not saved
    item_similarity = cosine_similarity(
        np.nan_to_num(user_item_matrix.T.values)
    )

# -------------------------------
# Helper functions
# -------------------------------
def recommend_svd_df(user_id, top_n=10):
    """Recommend using precomputed prediction dataframe (pred_df)."""
    if pred_df is None:
        return pd.DataFrame(columns=["movieId", "title", "score"])

    user_preds = pred_df.loc[user_id]
    seen_movies = ratings[ratings.userId == user_id]["movieId"].values
    user_preds = user_preds.drop(seen_movies, errors="ignore")

    top_movies = user_preds.sort_values(ascending=False).head(top_n)
    return movies[movies.movieId.isin(top_movies.index)].assign(score=top_movies.values)

def recommend_svd_model(user_id, top_n=10):
    """Recommend using saved SVD model."""
    if svd_model is None:
        return pd.DataFrame(columns=["movieId", "title", "score"])

    all_movies = movies["movieId"].unique()
    seen_movies = ratings[ratings.userId == user_id]["movieId"].values
    unseen_movies = [m for m in all_movies if m not in seen_movies]

    preds = [(m, svd_model.predict(user_id, m).est) for m in unseen_movies]
    preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]

    result = pd.DataFrame(preds, columns=["movieId", "score"])
    return result.merge(movies, on="movieId")

def recommend_item_based(user_id, top_n=10):
    """Recommend using item-based CF similarity."""
    user_ratings = user_item_matrix.loc[user_id].dropna()
    scores = np.zeros(item_similarity.shape[0])

    for movie_id, rating in user_ratings.items():
        scores += rating * item_similarity[movie_id - 1]

    seen_movies = user_ratings.index
    scores = [(i+1, s) for i, s in enumerate(scores) if (i+1) not in seen_movies]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]

    result = pd.DataFrame(scores, columns=["movieId", "score"])
    return result.merge(movies, on="movieId")

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🎬 Movie Recommender System")

user_id = st.number_input("Enter User ID:", min_value=1, max_value=int(ratings.userId.max()), value=50)
top_n = st.slider("Number of recommendations:", 5, 20, 10)
model_choice = st.selectbox("Choose model:", ["SVD (pred_df)", "SVD (model)", "Item-based CF"])

if st.button("Recommend"):
    if model_choice == "SVD (pred_df)":
        recs = recommend_svd_df(user_id, top_n)
    elif model_choice == "SVD (model)":
        recs = recommend_svd_model(user_id, top_n)
    else:
        recs = recommend_item_based(user_id, top_n)

    st.subheader(f"Top {top_n} Recommendations for User {user_id}")
    st.table(recs[["title", "score"]])
