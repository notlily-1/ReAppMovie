from fastapi import FastAPI, HTTPException, Path, Body, Header
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional
import sys
from pathlib import Path as PathLib

sys.path.append(str(PathLib(__file__).resolve().parent.parent))

from app.model import MovieRecommender
from app.schemas import (
    TrainResponse,
    RecommendationRequest,
    RecommendationResponse,
    DualUserRecommendationRequest,
    DualUserRecommendationResponse,
    PredictionRequest,
    PredictionResponse,
    ErrorResponse,
    HealthResponse,
    AddRatingRequest,
    AddRatingResponse,
    UserStatusResponse,
    MoviesToRateResponse
)
from app.config import MIN_RATINGS_FOR_RECOMMENDATIONS

recommender = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender
    print("=" * 50)
    print("Starting Movie Recommendation System...")
    print("=" * 50)
    
    recommender = MovieRecommender()
    
    model_loaded = recommender.load_model()
    
    if not model_loaded:
        print("\nNo pre-trained model found.")
        print("Please train the model using POST /train endpoint")
        print("or access the Swagger UI at http://localhost:8000/docs")
    else:
        print("\n✓ Model loaded and ready for predictions!")
    
    print("=" * 50)
    print(f"API Documentation: http://localhost:8000/docs")
    print("=" * 50)
    
    yield
    
    print("\nShutting down...")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Movie Recommendation System API",
        "docs": "/docs",
        "min_ratings_required": MIN_RATINGS_FOR_RECOMMENDATIONS,
        "endpoints": {
            "health": "GET /health",
            "train": "POST /train",
            "recommendations": "POST /predict/{user_id}",
            "single_prediction": "POST /predict/{user_id}/movie",
            "firebase_user_status": "GET /firebase/user/status",
            "firebase_add_rating": "POST /firebase/user/rate",
            "firebase_get_movies_to_rate": "GET /firebase/movies/to-rate",
            "firebase_recommendations": "POST /firebase/user/recommendations",
            "dual_user_recommendations": "POST /firebase/dual-user/recommendations"
        }
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check"
)
async def health_check():
    model_loaded = recommender.model is not None
    
    dataset_stats = None
    if recommender.data_loader:
        try:
            dataset_stats = recommender.data_loader.get_dataset_stats()
        except Exception:
            pass
    
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        dataset_stats=dataset_stats
    )


@app.post(
    "/train",
    response_model=TrainResponse,
    tags=["Model Management"],
    summary="Train the KNN model"
)
async def train_model(test_size: float = 0.2):
    try:
        print("\n" + "=" * 50)
        print("TRAINING REQUEST RECEIVED")
        print("=" * 50)
        
        metrics = recommender.train_model(test_size=test_size)
        
        return TrainResponse(
            success=True,
            message="Model trained and saved successfully",
            metrics=metrics
        )
        
    except Exception as e:
        print(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/{user_id}",
    response_model=RecommendationResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Predictions"],
    summary="Get movie recommendations for a user"
)
async def get_recommendations(
    user_id: int = Path(..., description="User ID to get recommendations for", ge=1),
    request: RecommendationRequest = Body(default=RecommendationRequest())
):
    if recommender.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Please train the model first using POST /train"
        )
    
    try:
        recommendations = recommender.get_recommendations(
            user_id=user_id,
            k=request.k,
            min_rating=request.min_rating
        )
        
        return RecommendationResponse(
            success=True,
            userId=user_id,
            recommendations=recommendations,
            total_recommendations=len(recommendations)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/{user_id}/movie",
    response_model=PredictionResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Predictions"],
    summary="Predict rating for a specific movie"
)
async def predict_single_movie(
    user_id: int = Path(..., description="User ID", ge=1),
    request: PredictionRequest = Body(...)
):
    if recommender.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Please train the model first using POST /train"
        )
    
    try:
        prediction = recommender.predict_rating(
            user_id=user_id,
            movie_id=request.movie_id
        )
        
        return PredictionResponse(
            success=True,
            **prediction
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/users/{user_id}/stats",
    tags=["User Info"],
    summary="Get user statistics"
)
async def get_user_stats(user_id: int = Path(..., description="User ID", ge=1)):
    try:
        if not recommender.data_loader.user_exists(user_id):
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        
        rated_movies = recommender.data_loader.get_user_rated_movies(user_id)
        
        return {
            "userId": user_id,
            "total_ratings": len(rated_movies),
            "total_movies_available": len(recommender.data_loader.get_all_movie_ids())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/firebase/user/status",
    response_model=UserStatusResponse,
    tags=["Firebase Users"],
    summary="Get Firebase user status"
)
async def get_firebase_user_status(
    firebase_uid: str = Header(None, description="Firebase User ID from Authorization header"),
    test_uid: str = None  # NEW: Add query parameter option for testing
):
    """Get user status. Use header 'firebase_uid' or query param 'test_uid' for testing."""
    uid = firebase_uid or test_uid  # NEW: Use either header or query param
    if not uid:  # NEW: Validation
        raise HTTPException(
            status_code=400,
            detail="Provide firebase_uid header or test_uid query parameter"
        )
    
    try:
        can_recommend, current_count, required_count = recommender.data_loader.can_get_recommendations(uid)  # CHANGED: firebase_uid → uid
        
        ratings = recommender.data_loader.get_firebase_user_ratings(uid)  # CHANGED: firebase_uid → uid
        rated_movies = [
            {
                "movieId": r['movieId'],
                "rating": r['rating'],
                **recommender.data_loader.get_movie_info(r['movieId'])
            }
            for r in ratings
        ]
        
        return UserStatusResponse(
            firebase_uid=uid,  # CHANGED: firebase_uid → uid
            current_rating_count=current_count,
            can_get_recommendations=can_recommend,
            ratings_needed=max(0, required_count - current_count),
            rated_movies=rated_movies
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/firebase/movies/to-rate",
    response_model=MoviesToRateResponse,
    tags=["Firebase Users"],
    summary="Get movies for new users to rate"
)
async def get_movies_to_rate(
    firebase_uid: str = Header(..., description="Firebase User ID"),
    count: int = 20
):
    try:
        rated_movies = recommender.data_loader.get_firebase_user_rated_movies(firebase_uid)
        
        movies = recommender.data_loader.get_random_movies_for_rating(
            count=count,
            exclude_ids=rated_movies
        )
        
        return MoviesToRateResponse(
            movies=movies,
            total_movies=len(movies),
            already_rated=len(rated_movies)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/firebase/user/rate",
    response_model=AddRatingResponse,
    tags=["Firebase Users"],
    summary="Add a movie rating"
)
async def add_user_rating(
    firebase_uid: str = Header(..., description="Firebase User ID"),
    request: AddRatingRequest = Body(...)
):
    try:
        movie_info = recommender.data_loader.get_movie_info(request.movie_id)
        if movie_info["title"] == "Unknown":
            raise HTTPException(status_code=404, detail=f"Movie {request.movie_id} not found")
        
        recommender.data_loader.add_firebase_user_rating(
            firebase_uid=firebase_uid,
            movie_id=request.movie_id,
            rating=request.rating
        )
        
        can_recommend, current_count, required_count = recommender.data_loader.can_get_recommendations(firebase_uid)
        
        message = f"Rating added successfully for '{movie_info['title']}'"
        if can_recommend:
            message += ". You can now get recommendations!"
        else:
            message += f". Rate {required_count - current_count} more movies to get recommendations."
        
        return AddRatingResponse(
            success=True,
            message=message,
            current_rating_count=current_count,
            can_get_recommendations=can_recommend,
            ratings_needed=max(0, required_count - current_count)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/firebase/user/recommendations",
    response_model=RecommendationResponse,
    responses={403: {"model": ErrorResponse}},
    tags=["Firebase Users"],
    summary="Get recommendations for Firebase user"
)
async def get_firebase_recommendations(
    firebase_uid: str = Header(None, description="Firebase User ID"),  # CHANGED: ... → None
    test_uid: str = None,  # NEW: Add query parameter option for testing
    request: RecommendationRequest = Body(default=RecommendationRequest())
):
    """Get recommendations. Use header 'firebase_uid' or query param 'test_uid' for testing."""
    uid = firebase_uid or test_uid  # NEW: Use either header or query param
    if not uid:  # NEW: Validation
        raise HTTPException(
            status_code=400,
            detail="Provide firebase_uid header or test_uid query parameter"
        )
    
    if recommender.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Please wait for model training to complete."
        )
    
    try:
        recommendations = recommender.get_recommendations_for_firebase_user(
            firebase_uid=uid,  # CHANGED: firebase_uid → uid
            k=request.k,
            min_rating=request.min_rating
        )
        
        return RecommendationResponse(
            success=True,
            userId=0,  # firebase users dont have numeric IDs
            recommendations=recommendations,
            total_recommendations=len(recommendations)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/firebase/dual-user/recommendations",
    response_model=DualUserRecommendationResponse,
    responses={403: {"model": ErrorResponse}},
    tags=["Firebase Users - Dual User"],
    summary="Get recommendations for TWO Firebase users together"
)
async def get_dual_user_recommendations(
    firebase_uid1: str = Header(..., description="First Firebase User ID", alias="firebase-uid-1"),
    firebase_uid2: str = Header(..., description="Second Firebase User ID", alias="firebase-uid-2"),
    request: DualUserRecommendationRequest = Body(default=DualUserRecommendationRequest())
):
    """
    Get movie recommendations for 2 users watching together.
    
    Algorithm:
    1. Get individual recommendations for both users
    2. Find overlapping movies (both would enjoy)
    3. If less than k movies, fill with:
       - KNN-similar movies (similar to their top picks)
       - Genre-similar movies (matching their preferences)
    
    Headers required:
    - firebase-uid-1: First user's Firebase UID
    - firebase-uid-2: Second user's Firebase UID
    
    Both users must have rated at least 5 movies each.
    """
    if recommender.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Please wait for model training to complete."
        )
    
    try:
        recommendations, stats = recommender.get_dual_user_recommendations(
            firebase_uid1=firebase_uid1,
            firebase_uid2=firebase_uid2,
            k=request.k,
            min_rating=request.min_rating
        )
        
        return DualUserRecommendationResponse(
            success=True,
            user1_id=firebase_uid1,
            user2_id=firebase_uid2,
            recommendations=recommendations,
            total_recommendations=len(recommendations),
            overlap_count=stats["overlap_count"],
            knn_similar_count=stats["knn_similar_count"],
            genre_similar_count=stats["genre_similar_count"]
        )
        
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/test/setup",
    tags=["Testing"],
    summary="Setup test users with sample ratings"
)
async def setup_test_users():
    """
    Automatically create test Firebase users with sample ratings for local testing.
    This bypasses the need for a real Firebase frontend.
    """
    from app.config import ENABLE_TEST_MODE, TEST_FIREBASE_USERS
    
    if not ENABLE_TEST_MODE:
        raise HTTPException(
            status_code=403,
            detail="Test mode is disabled. Enable ENABLE_TEST_MODE in config.py"
        )
    
    try:
        results = []
        
        for firebase_uid in TEST_FIREBASE_USERS:
            # Get 10 random movies to rate
            movies_to_rate = recommender.data_loader.get_random_movies_for_rating(
                count=10,
                exclude_ids=set()
            )
            
            # Add random ratings (3-5 stars)
            import random
            for movie in movies_to_rate[:8]:  # Rate 8 movies (more than minimum of 5)
                rating = random.randint(3, 5)
                recommender.data_loader.add_firebase_user_rating(
                    firebase_uid=firebase_uid,
                    movie_id=movie['movieId'],
                    rating=rating
                )
            
            user_ratings = recommender.data_loader.get_firebase_user_ratings(firebase_uid)
            results.append({
                "firebase_uid": firebase_uid,
                "ratings_added": len(user_ratings),
                "can_get_recommendations": len(user_ratings) >= MIN_RATINGS_FOR_RECOMMENDATIONS,
                "sample_ratings": user_ratings[:3]
            })
        
        return {
            "success": True,
            "message": "Test users created successfully",
            "test_users": results,
            "next_steps": [
                f"GET /firebase/user/status with header 'firebase_uid: test_user_1'",
                f"POST /firebase/user/recommendations with header 'firebase_uid: test_user_1'"
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/test/train",
    tags=["Testing"],
    summary="Train model for testing"
)
async def train_model_for_testing():
    """Quick training endpoint for local testing"""
    from app.config import ENABLE_TEST_MODE
    
    if not ENABLE_TEST_MODE:
        raise HTTPException(
            status_code=403,
            detail="Test mode is disabled"
        )
    
    try:
        print("\n🧪 TEST MODE: Training model...")
        metrics = recommender.train_model(test_size=0.2)
        
        return {
            "success": True,
            "message": "Model trained successfully in TEST MODE",
            "metrics": metrics,
            "next_step": "Call GET /test/setup to create test users"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )