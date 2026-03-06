from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

app = Flask(__name__)
CORS(app)

TMDB_KEY = "527b096928be9f627a2f7ce5be8eae31"
TMDB_BASE = "https://api.themoviedb.org/3"
KNN_THRESHOLD = 5  # switch to hybrid at this many watched movies

# ── TMDB Helpers ──────────────────────────────────────────────────────────────
def fetch_genre_map():
    res = requests.get(f"{TMDB_BASE}/genre/movie/list", params={"api_key": TMDB_KEY})
    genres = res.json().get("genres", [])
    return {g["id"]: g["name"] for g in genres}

def fetch_candidates():
    """Pull popular + top_rated + now_playing as recommendation pool."""
    movies = []
    endpoints = ["popular", "top_rated", "now_playing"]
    pages = [1, 2, 3, 4, 5]
    seen = set()
    for endpoint in endpoints:
        for page in pages:
            res = requests.get(
                f"{TMDB_BASE}/movie/{endpoint}",
                params={"api_key": TMDB_KEY, "page": page}
            )
            for m in res.json().get("results", []):
                if m["id"] not in seen:
                    seen.add(m["id"])
                    movies.append(m)
    return movies

def movie_to_text(movie, genre_map):
    """Convert a movie's genres into a text string for TF-IDF."""
    genre_names = [genre_map.get(gid, "") for gid in movie.get("genre_ids", [])]
    # Repeat genres to give them weight, append overview keywords
    genre_text = " ".join(genre_names * 3)
    overview   = movie.get("overview", "")
    return f"{genre_text} {overview}".strip()

# ── ML Engine ─────────────────────────────────────────────────────────────────
def build_model(watched_movies, ratings, candidates, genre_map):
    watched_ids = {m["id"] for m in watched_movies}
    unwatched   = [m for m in candidates if m["id"] not in watched_ids]

    if not unwatched:
        return [], "no_candidates"

    all_movies = watched_movies + unwatched
    use_hybrid = len(watched_movies) >= KNN_THRESHOLD

    # Step 1: Build TF-IDF matrix over all movies
    corpus     = [movie_to_text(m, genre_map) for m in all_movies]
    vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)  # shape: (n_movies, n_features)
    tfidf_norm   = normalize(tfidf_matrix)            # L2 normalize for cosine similarity

    n_watched  = len(watched_movies)
    watched_vecs   = tfidf_norm[:n_watched]           # rows 0..n_watched-1
    unwatched_vecs = tfidf_norm[n_watched:]           # rows n_watched..end

    # Step 2: Build weighted user profile (cosine mode)
    rating_weights = np.array([ratings.get(str(m["id"]), 3) / 5.0 for m in watched_movies])
    user_profile   = np.average(watched_vecs.toarray(), axis=0, weights=rating_weights)
    user_profile   = user_profile / (np.linalg.norm(user_profile) + 1e-9)

    # Cosine similarity: dot product of normalized vectors
    profile_scores = unwatched_vecs.toarray().dot(user_profile)

    if use_hybrid:
        # Step 3: kNN — fit on watched movies, predict for unwatched
        k = min(5, n_watched)
        knn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
        knn.fit(watched_vecs.toarray())

        distances, indices = knn.kneighbors(unwatched_vecs.toarray())
        similarities = 1 - distances  # convert distance → similarity

        knn_scores = []
        for i in range(len(unwatched)):
            neighb_sims    = similarities[i]
            neighb_ratings = np.array([
                ratings.get(str(watched_movies[idx]["id"]), 3)
                for idx in indices[i]
            ]) / 5.0
            total_sim = neighb_sims.sum()
            if total_sim > 0:
                knn_score = np.dot(neighb_sims, neighb_ratings) / total_sim
            else:
                knn_score = 0.0
            knn_scores.append(knn_score)

        knn_scores    = np.array(knn_scores)
        final_scores  = 0.5 * profile_scores + 0.5 * knn_scores
        mode          = "hybrid"
    else:
        final_scores = profile_scores
        mode         = "cosine"

    # Step 4: Rank and return top 10
    top_indices = np.argsort(final_scores)[::-1][:10]
    results = []
    for idx in top_indices:
        movie = unwatched[idx]
        results.append({
            **movie,
            "score": round(float(final_scores[idx]), 4),
        })

    return results, mode

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/api/search")
def search():
    q = request.args.get("q", "")
    if not q:
        return jsonify([])
    res = requests.get(
        f"{TMDB_BASE}/search/movie",
        params={"api_key": TMDB_KEY, "query": q, "include_adult": False}
    )
    return jsonify(res.json().get("results", [])[:7])

@app.route("/api/genres")
def genres():
    return jsonify(fetch_genre_map())

@app.route("/api/recommend", methods=["POST"])
def recommend():
    data          = request.json
    watched       = data.get("watched", [])       # list of movie objects
    ratings       = data.get("ratings", {})       # { "movie_id": 1-5 }

    if not watched:
        return jsonify({"error": "No watched movies provided"}), 400

    genre_map  = fetch_genre_map()
    candidates = fetch_candidates()
    recs, mode = build_model(watched, ratings, candidates, genre_map)

    return jsonify({"recommendations": recs, "mode": mode})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
