🎬 CineMatch — Python ML Movie Recommender
A full-stack movie recommendation web app with a Python/Flask ML backend and React frontend, powered by the TMDB API. Search any movie ever made, rate what you've watched, and let the scikit-learn model learn your taste.
Architecture
cinematch/
├── backend/      → Flask REST API + scikit-learn ML engine (port 5000)
├── frontend/     → React + Vite user interface (port 3000)
└── README.md
How the ML Works
CineMatch uses an adaptive pipeline that automatically upgrades as you add more movies:
Under 5 movies → Cosine Similarity mode
Movie genres and plot overviews are vectorized using scikit-learn's TfidfVectorizer. A weighted user profile is built from your watched list and compared against candidate movies using cosine similarity.
5+ movies → Hybrid TF-IDF + kNN mode
Upgrades to a hybrid model combining cosine profile similarity with scikit-learn's NearestNeighbors. The kNN component finds your K most similar watched movies for each candidate and predicts a score using a weighted average of their ratings — meaning your ratings directly train the model.
Final Score = 0.5 × cosine profile similarity + 0.5 × kNN predicted score
ML Techniques

TF-IDF (Term Frequency–Inverse Document Frequency) — rare genres and plot keywords score higher than common ones, making recommendations more personal
k-Nearest Neighbours (kNN) — learns from your rating history to predict how much you'd enjoy unseen films
Cosine Similarity — measures directional alignment between feature vectors
Weighted User Profiling — higher-rated movies influence your taste profile more than lower-rated ones
Adaptive switching — automatically selects the best algorithm based on available data, mirroring how production recommenders handle the cold start problem

Tech Stack
LayerTechnologiesBackendPython, Flask, scikit-learn, NumPyFrontendReact, ViteDataTMDB API (1M+ movies)
Getting Started
Prerequisites

Python 3.8+
Node.js 16+

1. Start the backend
bashcd backend
pip install -r requirements.txt
python app.py
2. Start the frontend (new terminal)
bashcd frontend
npm install
npm run dev
Open http://localhost:3000 and start adding movies.


API data provided by The Movie Database (TMDB)
