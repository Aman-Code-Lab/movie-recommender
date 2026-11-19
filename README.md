# 🎬 Movie Recommendation System

A hybrid movie recommender that combines **Content-Based Filtering** and **Collaborative Filtering (SVD)**, with a simple Streamlit web app.

🔗 **Live demo:** https://movie-recommender-dcmd7ccr2nhn9pnms669qh.streamlit.app

## Features
- **Content-Based Filtering:** recommends movies similar to a selected movie using TF-IDF on genres and tags, with cosine similarity.
- **Collaborative Filtering:** predicts user ratings with SVD (matrix factorization).
- **Hybrid model:** `score = alpha × CF + (1 − alpha) × Content`, with an adjustable alpha.
- **Streamlit UI:** pick a movie or enter a user ID to get recommendations.

## Dataset
[MovieLens Latest Small](https://grouplens.org/datasets/movielens/latest/): about 100k ratings, 600 users and 9,000 movies.

## How it works
1. **Data preparation:** load movies, ratings and tags, then clean the genres.
2. **Content-based:** TF-IDF vectors for each movie, then cosine similarity.
3. **Collaborative:** build the user-item matrix, mean-center it and apply SVD (k = 50).
4. **Hybrid:** combine the normalized CF and content scores.
5. **Evaluation:** 80/20 train-test split.

## Results

| Method | Precision@10 | Recall@10 |
|---|---|---|
| Popularity baseline | 0.2010 | 0.0911 |
| Content-based | 0.0327 | 0.0132 |
| Collaborative (SVD) | 0.1859 | 0.1052 |
| Hybrid (alpha = 0.7) | 0.0960 | 0.0555 |

**Rating prediction:** Baseline RMSE = 1.0488, SVD RMSE = 0.9322
## Tech stack
Python, pandas, NumPy, SciPy, scikit-learn, Streamlit

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Limitations and future work
- Cold-start problem for new users and new movies
- Content features use only genres and tags
- Could add movie posters (TMDB API), deep learning models and user login

## Author
(Apna naam), B.Tech AIML, (College ka naam)
