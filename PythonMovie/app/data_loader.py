import pandas as pd
from surprise import Dataset, Reader
from typing import Tuple, List, Dict, Optional
from app.config import MOVIES_PATH, RATINGS_PATH, RATING_SCALE, MIN_RATINGS_FOR_RECOMMENDATIONS


class DataLoader:
    
    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        self.firebase_ratings = {}
        self._load_data()
    
    def _load_data(self):
        try:
            self.movies_df = pd.read_csv(MOVIES_PATH)
            self.ratings_df = pd.read_csv(RATINGS_PATH)
            
            self.ratings_df = self.ratings_df[self.ratings_df['rating'] > 0]
            self.ratings_df['rating'] = self.ratings_df['rating'].round().astype(int)
            self.ratings_df = self.ratings_df[
                (self.ratings_df['rating'] >= 1) & 
                (self.ratings_df['rating'] <= 5)
            ]
            
            print(f"Loaded {len(self.movies_df)} movies")
            print(f"Loaded {len(self.ratings_df)} ratings")
            print(f"Number of unique users: {self.ratings_df['userId'].nunique()}")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Data files not found. Please ensure movies.csv and ratings.csv "
                f"are in the Data/ml-latest-small/ directory. Error: {e}"
            )
    
    def add_firebase_user_rating(self, firebase_uid: str, movie_id: int, rating: int):
        if firebase_uid not in self.firebase_ratings:
            self.firebase_ratings[firebase_uid] = []
        
        existing = [r for r in self.firebase_ratings[firebase_uid] if r['movieId'] == movie_id]
        if existing:
            existing[0]['rating'] = rating
        else:
            self.firebase_ratings[firebase_uid].append({
                'movieId': movie_id,
                'rating': rating
            })
    
    def get_firebase_user_ratings(self, firebase_uid: str) -> List[Dict]:
        return self.firebase_ratings.get(firebase_uid, [])
    
    def get_firebase_user_rating_count(self, firebase_uid: str) -> int:
        return len(self.firebase_ratings.get(firebase_uid, []))
    
    def can_get_recommendations(self, firebase_uid: str) -> Tuple[bool, int, int]:
        count = self.get_firebase_user_rating_count(firebase_uid)
        return (
            count >= MIN_RATINGS_FOR_RECOMMENDATIONS,
            count,
            MIN_RATINGS_FOR_RECOMMENDATIONS
        )
    
    def get_surprise_dataset(self) -> Dataset:
        reader = Reader(rating_scale=RATING_SCALE)
        data = Dataset.load_from_df(
            self.ratings_df[['userId', 'movieId', 'rating']], 
            reader
        )
        return data
    
    def get_surprise_dataset_with_firebase(self) -> Dataset:
        combined_ratings = self.ratings_df.copy()
        
        if self.firebase_ratings:
            max_user_id = self.ratings_df['userId'].max()
            firebase_user_mapping = {}
            next_id = max_user_id + 1
            
            firebase_rows = []
            for firebase_uid, ratings in self.firebase_ratings.items():
                if firebase_uid not in firebase_user_mapping:
                    firebase_user_mapping[firebase_uid] = next_id
                    next_id += 1
                
                numeric_user_id = firebase_user_mapping[firebase_uid]
                for rating_data in ratings:
                    firebase_rows.append({
                        'userId': numeric_user_id,
                        'movieId': rating_data['movieId'],
                        'rating': rating_data['rating']
                    })
            
            if firebase_rows:
                firebase_df = pd.DataFrame(firebase_rows)
                combined_ratings = pd.concat([combined_ratings, firebase_df], ignore_index=True)
        
        reader = Reader(rating_scale=RATING_SCALE)
        data = Dataset.load_from_df(
            combined_ratings[['userId', 'movieId', 'rating']], 
            reader
        )
        return data
    
    def get_movie_info(self, movie_id: int) -> dict:
        movie = self.movies_df[self.movies_df['movieId'] == movie_id]
        
        if movie.empty:
            return {
                "movieId": movie_id,
                "title": "Unknown",
                "genres": "Unknown"
            }
        
        return {
            "movieId": int(movie_id),
            "title": movie.iloc[0]['title'],
            "genres": movie.iloc[0]['genres']
        }
    
    def get_user_rated_movies(self, user_id: int) -> set:
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        return set(user_ratings['movieId'].values)
    
    def get_firebase_user_rated_movies(self, firebase_uid: str) -> set:
        ratings = self.get_firebase_user_ratings(firebase_uid)
        return set([r['movieId'] for r in ratings])
    
    def get_all_movie_ids(self) -> set:
        return set(self.movies_df['movieId'].values)
    
    def get_random_movies_for_rating(self, count: int = 20, exclude_ids: set = None) -> List[Dict]:
        if exclude_ids is None:
            exclude_ids = set()
        
        # top 200 popular movies
        movie_rating_counts = self.ratings_df['movieId'].value_counts()
        popular_movies = movie_rating_counts.head(200).index.tolist()
        
        available_movies = [mid for mid in popular_movies if mid not in exclude_ids]
        
        import random
        selected = random.sample(available_movies, min(count, len(available_movies)))
        
        return [self.get_movie_info(mid) for mid in selected]
    
    def user_exists(self, user_id: int) -> bool:
        return user_id in self.ratings_df['userId'].values
    
    def firebase_user_exists(self, firebase_uid: str) -> bool:
        return firebase_uid in self.firebase_ratings
    
    def get_dataset_stats(self) -> dict:
        return {
            "total_movies": len(self.movies_df),
            "total_ratings": len(self.ratings_df),
            "total_users": self.ratings_df['userId'].nunique(),
            "firebase_users": len(self.firebase_ratings),
            "avg_ratings_per_user": self.ratings_df.groupby('userId').size().mean(),
            "avg_ratings_per_movie": self.ratings_df.groupby('movieId').size().mean(),
            "rating_scale": RATING_SCALE,
            "min_ratings_required": MIN_RATINGS_FOR_RECOMMENDATIONS
        }