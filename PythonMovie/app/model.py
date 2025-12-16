import pickle
from typing import List, Optional, Tuple, Dict
from surprise import KNNWithMeans, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy
from app.config import MODEL_CONFIG, MODEL_PATH, MIN_RATINGS_FOR_RECOMMENDATIONS
from app.data_loader import DataLoader


class MovieRecommender:
    
    def __init__(self):
        self.model = None
        self.data_loader = DataLoader()
        self.trainset = None
        self.firebase_user_id_mapping = {}
        self.next_user_id = None
        
    def train_model(self, test_size: float = 0.2, include_firebase: bool = False) -> dict:
        print("Starting model training...")
        
        if include_firebase:
            data = self.data_loader.get_surprise_dataset_with_firebase()
            print("Training with Firebase user data included")
        else:
            data = self.data_loader.get_surprise_dataset()
        
        trainset, testset = train_test_split(data, test_size=test_size)
        self.trainset = trainset
        
        # for firebase user id mapping
        self.next_user_id = max([uid for (uid, _, _) in trainset.all_ratings()]) + 1
        
        if MODEL_CONFIG["name"] == "KNNWithMeans":
            self.model = KNNWithMeans(
                k=MODEL_CONFIG["k"],
                sim_options=MODEL_CONFIG["sim_options"],
                min_k=MODEL_CONFIG["min_k"]
            )
        else:
            self.model = KNNBasic(
                k=MODEL_CONFIG["k"],
                sim_options=MODEL_CONFIG["sim_options"],
                min_k=MODEL_CONFIG["min_k"]
            )
        
        print(f"Training {MODEL_CONFIG['name']} with k={MODEL_CONFIG['k']}...")
        self.model.fit(trainset)
        
        predictions = self.model.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        
        print(f"Training complete! RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        self.save_model()
        
        return {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "model_type": MODEL_CONFIG["name"],
            "k": MODEL_CONFIG["k"],
            "similarity": MODEL_CONFIG["sim_options"]["name"],
            "firebase_included": include_firebase
        }
    
    def save_model(self):
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_data = {
            "model": self.model,
            "trainset": self.trainset,
            "config": MODEL_CONFIG,
            "firebase_user_mapping": self.firebase_user_id_mapping,
            "next_user_id": self.next_user_id
        }
        
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {MODEL_PATH}")
    
    def load_model(self) -> bool:
        try:
            with open(MODEL_PATH, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data["model"]
            self.trainset = model_data["trainset"]
            self.firebase_user_id_mapping = model_data.get("firebase_user_mapping", {})
            self.next_user_id = model_data.get("next_user_id", 1000)
            
            print(f"Model loaded successfully from {MODEL_PATH}")
            return True
            
        except FileNotFoundError:
            print(f"No saved model found at {MODEL_PATH}")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def _calculate_genre_similarity(self, genres1: str, genres2: str) -> float:
        """Calculate Jaccard similarity between two genre strings"""
        if genres1 == "Unknown" or genres2 == "Unknown":
            return 0.0
        
        set1 = set(genres1.split('|'))
        set2 = set(genres2.split('|'))
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_dual_user_recommendations(
        self,
        firebase_uid1: str,
        firebase_uid2: str,
        k: int = 10,
        min_rating: int = 3
    ) -> Tuple[List[dict], Dict[str, int]]:
        """Get recommendations for two users together"""
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        # Verify both users have enough ratings
        can_recommend1, count1, required = self.data_loader.can_get_recommendations(firebase_uid1)
        can_recommend2, count2, _ = self.data_loader.can_get_recommendations(firebase_uid2)
        
        if not can_recommend1:
            raise ValueError(
                f"User 1 must rate at least {required} movies. Current: {count1}"
            )
        if not can_recommend2:
            raise ValueError(
                f"User 2 must rate at least {required} movies. Current: {count2}"
            )
        
        # Get individual recommendations for both users
        user1_recs = self._get_user_predictions(firebase_uid1, min_rating, k=50)
        user2_recs = self._get_user_predictions(firebase_uid2, min_rating, k=50)
        
        print(f"User 1 has {len(user1_recs)} potential movies")
        print(f"User 2 has {len(user2_recs)} potential movies")
        
        # Find overlapping movies
        user1_movies = {r['movie_id']: r for r in user1_recs}
        user2_movies = {r['movie_id']: r for r in user2_recs}
        
        overlap_movie_ids = set(user1_movies.keys()).intersection(set(user2_movies.keys()))
        
        recommendations = []
        stats = {
            "overlap_count": 0,
            "knn_similar_count": 0,
            "genre_similar_count": 0
        }
        
        # Add overlapping movies first
        overlap_recs = []
        for movie_id in overlap_movie_ids:
            user1_rating = user1_movies[movie_id]['predicted_rating']
            user2_rating = user2_movies[movie_id]['predicted_rating']
            combined_score = (user1_rating + user2_rating) / 2.0
            
            movie_info = self.data_loader.get_movie_info(movie_id)
            overlap_recs.append({
                "movieId": movie_info["movieId"],
                "title": movie_info["title"],
                "genres": movie_info["genres"],
                "user1_predicted_rating": user1_rating,
                "user2_predicted_rating": user2_rating,
                "combined_score": combined_score,
                "recommendation_type": "overlap"
            })
        
        # Sort by combined score
        overlap_recs.sort(key=lambda x: x["combined_score"], reverse=True)
        recommendations.extend(overlap_recs[:k])
        stats["overlap_count"] = len(overlap_recs[:k])
        
        print(f"Found {len(overlap_recs)} overlapping movies")
        
        # If we need more recommendations
        if len(recommendations) < k:
            needed = k - len(recommendations)
            print(f"Need {needed} more recommendations")
            
            # Get movies rated by both users
            rated1 = self.data_loader.get_firebase_user_rated_movies(firebase_uid1)
            rated2 = self.data_loader.get_firebase_user_rated_movies(firebase_uid2)
            already_recommended = {r["movieId"] for r in recommendations}
            excluded_movies = rated1.union(rated2).union(already_recommended)
            
            # Try to fill with KNN-similar movies
            knn_candidates = self._get_knn_similar_movies(
                user1_recs, user2_recs, excluded_movies, needed
            )
            
            for candidate in knn_candidates:
                if len(recommendations) >= k:
                    break
                
                movie_id = candidate['movie_id']
                numeric_uid1 = self._get_numeric_user_id(firebase_uid1)
                numeric_uid2 = self._get_numeric_user_id(firebase_uid2)
                
                try:
                    pred1 = self.model.predict(numeric_uid1, movie_id)
                    pred2 = self.model.predict(numeric_uid2, movie_id)
                    
                    user1_rating = max(1, min(5, round(pred1.est)))
                    user2_rating = max(1, min(5, round(pred2.est)))
                    
                    if user1_rating >= min_rating and user2_rating >= min_rating:
                        combined_score = (user1_rating + user2_rating) / 2.0
                        movie_info = self.data_loader.get_movie_info(movie_id)
                        
                        recommendations.append({
                            "movieId": movie_info["movieId"],
                            "title": movie_info["title"],
                            "genres": movie_info["genres"],
                            "user1_predicted_rating": user1_rating,
                            "user2_predicted_rating": user2_rating,
                            "combined_score": combined_score,
                            "recommendation_type": "knn_similar"
                        })
                        stats["knn_similar_count"] += 1
                except Exception:
                    continue
            
            print(f"Added {stats['knn_similar_count']} KNN-similar movies")
            
            # If still need more, use genre similarity
            if len(recommendations) < k:
                needed = k - len(recommendations)
                print(f"Still need {needed} more, using genre similarity")
                
                genre_candidates = self._get_genre_similar_movies(
                    firebase_uid1, firebase_uid2, excluded_movies, needed
                )
                
                for candidate in genre_candidates:
                    if len(recommendations) >= k:
                        break
                    
                    recommendations.append(candidate)
                    stats["genre_similar_count"] += 1
                
                print(f"Added {stats['genre_similar_count']} genre-similar movies")
        
        # Final sort by combined score
        recommendations.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return recommendations[:k], stats
    
    def _get_numeric_user_id(self, firebase_uid: str) -> int:
        """Get or create numeric user ID for Firebase user"""
        if firebase_uid not in self.firebase_user_id_mapping:
            self.firebase_user_id_mapping[firebase_uid] = self.next_user_id
            self.next_user_id += 1
        return self.firebase_user_id_mapping[firebase_uid]
    
    def _get_user_predictions(
        self, firebase_uid: str, min_rating: int, k: int = 50
    ) -> List[dict]:
        """Get predictions for a single user"""
        numeric_user_id = self._get_numeric_user_id(firebase_uid)
        
        all_movies = self.data_loader.get_all_movie_ids()
        rated_movies = self.data_loader.get_firebase_user_rated_movies(firebase_uid)
        unrated_movies = all_movies - rated_movies
        
        predictions = []
        for movie_id in unrated_movies:
            try:
                pred = self.model.predict(numeric_user_id, movie_id)
                predicted_rating = max(1, min(5, round(pred.est)))
                if predicted_rating >= min_rating:
                    predictions.append({
                        "movie_id": int(movie_id),
                        "predicted_rating": predicted_rating
                    })
            except Exception:
                continue
        
        predictions.sort(key=lambda x: x["predicted_rating"], reverse=True)
        return predictions[:k]
    
    def _get_knn_similar_movies(
        self, user1_recs: List[dict], user2_recs: List[dict], 
        excluded: set, needed: int
    ) -> List[dict]:
        """Find movies similar to both users' top recommendations using KNN"""
        candidates = []
        
        # Get top movies from each user
        top_user1 = [r['movie_id'] for r in user1_recs[:10]]
        top_user2 = [r['movie_id'] for r in user2_recs[:10]]
        
        all_movies = self.data_loader.get_all_movie_ids()
        candidate_movies = all_movies - excluded
        
        for movie_id in candidate_movies:
            if len(candidates) >= needed * 3:  # Get more candidates to filter
                break
            
            movie_info = self.data_loader.get_movie_info(movie_id)
            if movie_info["title"] == "Unknown":
                continue
            
            # Calculate similarity to top movies of both users
            similarity_score = 0
            for top_movie in top_user1[:5] + top_user2[:5]:
                movie1_genres = self.data_loader.get_movie_info(top_movie)["genres"]
                similarity_score += self._calculate_genre_similarity(
                    movie1_genres, movie_info["genres"]
                )
            
            if similarity_score > 0:
                candidates.append({
                    "movie_id": movie_id,
                    "similarity_score": similarity_score
                })
        
        candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
        return candidates[:needed * 2]
    
    def _get_genre_similar_movies(
        self, firebase_uid1: str, firebase_uid2: str, 
        excluded: set, needed: int
    ) -> List[dict]:
        """Find movies with genres similar to what both users like"""
        # Get genres both users rated highly
        user1_ratings = self.data_loader.get_firebase_user_ratings(firebase_uid1)
        user2_ratings = self.data_loader.get_firebase_user_ratings(firebase_uid2)
        
        user1_genres = set()
        user2_genres = set()
        
        for rating in user1_ratings:
            if rating['rating'] >= 4:
                genres = self.data_loader.get_movie_info(rating['movieId'])["genres"]
                user1_genres.update(genres.split('|'))
        
        for rating in user2_ratings:
            if rating['rating'] >= 4:
                genres = self.data_loader.get_movie_info(rating['movieId'])["genres"]
                user2_genres.update(genres.split('|'))
        
        common_genres = user1_genres.intersection(user2_genres)
        print(f"Common liked genres: {common_genres}")
        
        # Find movies with these genres
        all_movies = self.data_loader.get_all_movie_ids()
        candidate_movies = all_movies - excluded
        
        candidates = []
        numeric_uid1 = self._get_numeric_user_id(firebase_uid1)
        numeric_uid2 = self._get_numeric_user_id(firebase_uid2)
        
        for movie_id in candidate_movies:
            if len(candidates) >= needed:
                break
            
            movie_info = self.data_loader.get_movie_info(movie_id)
            if movie_info["title"] == "Unknown":
                continue
            
            movie_genres = set(movie_info["genres"].split('|'))
            genre_match = len(movie_genres.intersection(common_genres))
            
            if genre_match > 0:
                try:
                    pred1 = self.model.predict(numeric_uid1, movie_id)
                    pred2 = self.model.predict(numeric_uid2, movie_id)
                    
                    user1_rating = max(1, min(5, round(pred1.est)))
                    user2_rating = max(1, min(5, round(pred2.est)))
                    combined_score = (user1_rating + user2_rating) / 2.0
                    
                    candidates.append({
                        "movieId": movie_info["movieId"],
                        "title": movie_info["title"],
                        "genres": movie_info["genres"],
                        "user1_predicted_rating": user1_rating,
                        "user2_predicted_rating": user2_rating,
                        "combined_score": combined_score,
                        "recommendation_type": "genre_similar",
                        "genre_match_count": genre_match
                    })
                except Exception:
                    continue
        
        candidates.sort(key=lambda x: (x["genre_match_count"], x["combined_score"]), reverse=True)
        return candidates[:needed]
    
    def get_recommendations_for_firebase_user(
        self, 
        firebase_uid: str, 
        k: int = 10,
        min_rating: int = 3
    ) -> List[dict]:
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        can_recommend, current_count, required_count = self.data_loader.can_get_recommendations(firebase_uid)
        
        if not can_recommend:
            raise ValueError(
                f"User must rate at least {required_count} movies before getting recommendations. "
                f"Current count: {current_count}"
            )
        
        numeric_user_id = self._get_numeric_user_id(firebase_uid)
        
        all_movies = self.data_loader.get_all_movie_ids()
        rated_movies = self.data_loader.get_firebase_user_rated_movies(firebase_uid)
        unrated_movies = all_movies - rated_movies
        
        print(f"Firebase user {firebase_uid} (ID: {numeric_user_id}) has rated {len(rated_movies)} movies")
        print(f"Predicting ratings for {len(unrated_movies)} unrated movies...")
        
        predictions = []
        for movie_id in unrated_movies:
            try:
                pred = self.model.predict(numeric_user_id, movie_id)
                predicted_rating = max(1, min(5, round(pred.est)))
                if predicted_rating >= min_rating:
                    predictions.append({
                        "movie_id": int(movie_id),
                        "predicted_rating": predicted_rating
                    })
            except Exception as e:
                continue
        
        predictions.sort(key=lambda x: x["predicted_rating"], reverse=True)
        top_predictions = predictions[:k]
        
        recommendations = []
        for pred in top_predictions:
            movie_info = self.data_loader.get_movie_info(pred["movie_id"])
            recommendations.append({
                "movieId": movie_info["movieId"],
                "title": movie_info["title"],
                "genres": movie_info["genres"],
                "predictedRating": pred["predicted_rating"]
            })
        
        return recommendations
    
    def get_recommendations(
        self, 
        user_id: int, 
        k: int = 10,
        min_rating: int = 3
    ) -> List[dict]:
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        if not self.data_loader.user_exists(user_id):
            raise ValueError(f"User {user_id} not found in dataset")
        
        all_movies = self.data_loader.get_all_movie_ids()
        rated_movies = self.data_loader.get_user_rated_movies(user_id)
        unrated_movies = all_movies - rated_movies
        
        print(f"User {user_id} has rated {len(rated_movies)} movies")
        print(f"Predicting ratings for {len(unrated_movies)} unrated movies...")
        
        predictions = []
        for movie_id in unrated_movies:
            try:
                pred = self.model.predict(user_id, movie_id)
                predicted_rating = max(1, min(5, round(pred.est)))
                if predicted_rating >= min_rating:
                    predictions.append({
                        "movie_id": int(movie_id),
                        "predicted_rating": predicted_rating
                    })
            except Exception:
                continue
        
        predictions.sort(key=lambda x: x["predicted_rating"], reverse=True)
        top_predictions = predictions[:k]
        
        recommendations = []
        for pred in top_predictions:
            movie_info = self.data_loader.get_movie_info(pred["movie_id"])
            recommendations.append({
                "movieId": movie_info["movieId"],
                "title": movie_info["title"],
                "genres": movie_info["genres"],
                "predictedRating": pred["predicted_rating"]
            })
        
        return recommendations
    
    def predict_rating(self, user_id: int, movie_id: int) -> dict:
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        pred = self.model.predict(user_id, movie_id)
        movie_info = self.data_loader.get_movie_info(movie_id)
        
        predicted_rating = max(1, min(5, round(pred.est)))
        
        return {
            "userId": user_id,
            "movieId": movie_info["movieId"],
            "title": movie_info["title"],
            "genres": movie_info["genres"],
            "predictedRating": predicted_rating,
            "details": {
                "was_impossible": pred.details.get("was_impossible", False),
                "reason": pred.details.get("reason", "")
            }
        }