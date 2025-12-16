import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "Data" / "ml-latest-small"
MOVIES_PATH = DATA_DIR / "movies.csv"
RATINGS_PATH = DATA_DIR / "ratings.csv"

# Model paths
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "knn_model.pkl"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)

# Model hyperparameters
MODEL_CONFIG = {
    "name": "KNNWithMeans",  # KNNWithMeans handles user bias better
    "k": 40,  # Number of neighbors
    "sim_options": {
        "name": "cosine",  # Similarity metric: cosine, msd, pearson
        "user_based": True,  # User-based collaborative filtering
    },
    "min_k": 1,  # Minimum number of neighbors
}

RATING_SCALE = (1, 5)

MIN_RATINGS_FOR_RECOMMENDATIONS = 5

# API Configuration
API_TITLE = "Movie Recommendation System"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
Production-ready Movie Recommendation System using KNN algorithm.

## Features:
* Train KNN model on MovieLens dataset
* Get personalized movie recommendations
* Persistent model storage for fast predictions
"""