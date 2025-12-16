from pydantic import BaseModel, Field
from typing import List, Optional


class TrainResponse(BaseModel):
    success: bool
    message: str
    metrics: Optional[dict] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Model trained successfully",
                "metrics": {
                    "rmse": 0.8756,
                    "mae": 0.6721,
                    "model_type": "KNNWithMeans",
                    "k": 40,
                    "similarity": "cosine"
                }
            }
        }


class MovieRecommendation(BaseModel):
    movieId: int
    title: str
    genres: str
    predictedRating: int = Field(..., ge=1, le=5)
    
    class Config:
        json_schema_extra = {
            "example": {
                "movieId": 318,
                "title": "Shawshank Redemption, The (1994)",
                "genres": "Crime|Drama",
                "predictedRating": 5
            }
        }


class DualUserMovieRecommendation(BaseModel):
    movieId: int
    title: str
    genres: str
    user1_predicted_rating: int = Field(..., ge=1, le=5)
    user2_predicted_rating: int = Field(..., ge=1, le=5)
    combined_score: float
    recommendation_type: str  # "overlap", "knn_similar", "genre_similar"
    
    class Config:
        json_schema_extra = {
            "example": {
                "movieId": 318,
                "title": "Shawshank Redemption, The (1994)",
                "genres": "Crime|Drama",
                "user1_predicted_rating": 5,
                "user2_predicted_rating": 4,
                "combined_score": 4.5,
                "recommendation_type": "overlap"
            }
        }


class RecommendationRequest(BaseModel):
    k: int = Field(default=10, ge=1, le=100, description="Number of recommendations")
    min_rating: int = Field(
        default=3, 
        ge=1, 
        le=5, 
        description="Minimum predicted rating threshold (1-5)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "k": 10,
                "min_rating": 3
            }
        }


class DualUserRecommendationRequest(BaseModel):
    k: int = Field(default=10, ge=1, le=100, description="Number of recommendations")
    min_rating: int = Field(
        default=3, 
        ge=1, 
        le=5, 
        description="Minimum predicted rating threshold for each user (1-5)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "k": 10,
                "min_rating": 3
            }
        }


class RecommendationResponse(BaseModel):
    success: bool
    userId: int
    recommendations: List[MovieRecommendation]
    total_recommendations: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "userId": 1,
                "recommendations": [
                    {
                        "movieId": 318,
                        "title": "Shawshank Redemption, The (1994)",
                        "genres": "Crime|Drama",
                        "predictedRating": 5
                    }
                ],
                "total_recommendations": 10
            }
        }


class DualUserRecommendationResponse(BaseModel):
    success: bool
    user1_id: str
    user2_id: str
    recommendations: List[DualUserMovieRecommendation]
    total_recommendations: int
    overlap_count: int
    knn_similar_count: int
    genre_similar_count: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "user1_id": "firebase_user_1",
                "user2_id": "firebase_user_2",
                "recommendations": [
                    {
                        "movieId": 318,
                        "title": "Shawshank Redemption, The (1994)",
                        "genres": "Crime|Drama",
                        "user1_predicted_rating": 5,
                        "user2_predicted_rating": 4,
                        "combined_score": 4.5,
                        "recommendation_type": "overlap"
                    }
                ],
                "total_recommendations": 10,
                "overlap_count": 3,
                "knn_similar_count": 4,
                "genre_similar_count": 3
            }
        }


class PredictionRequest(BaseModel):
    movie_id: int = Field(..., description="Movie ID to predict rating for")
    
    class Config:
        json_schema_extra = {
            "example": {
                "movie_id": 318
            }
        }


class PredictionResponse(BaseModel):
    success: bool
    userId: int
    movieId: int
    title: str
    genres: str
    predictedRating: int
    details: Optional[dict] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "userId": 1,
                "movieId": 318,
                "title": "Shawshank Redemption, The (1994)",
                "genres": "Crime|Drama",
                "predictedRating": 5,
                "details": {
                    "was_impossible": False,
                    "reason": ""
                }
            }
        }


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "User not found in dataset"
            }
        }


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    dataset_stats: Optional[dict] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "dataset_stats": {
                    "total_movies": 9742,
                    "total_ratings": 100836,
                    "total_users": 610
                }
            }
        }


class AddRatingRequest(BaseModel):
    movie_id: int = Field(..., description="Movie ID to rate")
    rating: int = Field(..., ge=1, le=5, description="Rating value (1-5)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "movie_id": 318,
                "rating": 5
            }
        }


class AddRatingResponse(BaseModel):
    success: bool
    message: str
    current_rating_count: int
    can_get_recommendations: bool
    ratings_needed: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Rating added successfully",
                "current_rating_count": 3,
                "can_get_recommendations": False,
                "ratings_needed": 2
            }
        }


class UserStatusResponse(BaseModel):
    firebase_uid: str
    current_rating_count: int
    can_get_recommendations: bool
    ratings_needed: int
    rated_movies: List[dict]
    
    class Config:
        json_schema_extra = {
            "example": {
                "firebase_uid": "abc123xyz",
                "current_rating_count": 5,
                "can_get_recommendations": True,
                "ratings_needed": 0,
                "rated_movies": [
                    {"movieId": 318, "rating": 5}
                ]
            }
        }


class MoviesToRateResponse(BaseModel):
    movies: List[dict]
    total_movies: int
    already_rated: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "movies": [
                    {
                        "movieId": 318,
                        "title": "Shawshank Redemption, The (1994)",
                        "genres": "Crime|Drama"
                    }
                ],
                "total_movies": 20,
                "already_rated": 0
            }
        }