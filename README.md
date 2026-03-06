# 🎬 CineMatch Python — ML Movie Recommender

A full-stack movie recommendation system with a **Python/Flask ML backend** and **React frontend**, powered by the TMDB API.

## Architecture

```
frontend/   → React + Vite (runs on port 3000)
backend/    → Flask + scikit-learn (runs on port 5000)
```

## ML Pipeline

The backend uses an adaptive scikit-learn pipeline:

- **Under 5 movies** → TF-IDF cosine similarity (stable with little data)
- **5+ movies** → Hybrid TF-IDF + k-Nearest Neighbours (full ML mode)

### How it works
1. Movie genres and overviews are converted into text and vectorized using **scikit-learn's TfidfVectorizer**
2. A **weighted user profile** is built from your watched movies × your star ratings
3. In hybrid mode, **NearestNeighbors** finds the K most similar watched movies to each candidate and predicts a score from their ratings
4. Final score = `0.5 × cosine profile score + 0.5 × kNN predicted score`

## Tech Stack
- **Backend:** Python, Flask, scikit-learn, NumPy
- **Frontend:** React, Vite
- **API:** TMDB (The Movie Database)

## Getting Started

### 1. Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### 2. Frontend (new terminal)
```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000**

> API data provided by [The Movie Database (TMDB)](https://www.themoviedb.org/)
