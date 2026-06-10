"""FastAPI backend service for the Music App"""
import os
from contextlib import asynccontextmanager
from pathlib import Path
import logging
import joblib
from fastapi import FastAPI, HTTPException
import pandas as pd


from app.backend.schemas import ClassificationInput, RegressionInput, RecommendationInput

pipelines = {}


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data"

MODEL_DIR_PATH = os.getenv("MODEL_DIR", str(DEFAULT_MODEL_DIR))
PIPELINES_DIR = Path(MODEL_DIR_PATH)

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load and remove pipelines on app start and close"""

    pipeline_paths = {
        "regression": PIPELINES_DIR / "regression_pipeline.pkl",
        "classification": PIPELINES_DIR / "classification_pipeline.pkl",
        "recommendation": PIPELINES_DIR / "recommendation_pipeline.pkl"
    }

    # Load all pipelines
    for name, path in pipeline_paths.items():
        if path.exists():
            try:
                with open(path, "rb") as f:
                    pipelines[name] = joblib.load(f)
                logger.info("Pipeline '%s' loaded successfully.", name)
            except Exception as e: # pylint: disable=broad-exception-caught
                logger.error("Error loading pipeline '%s': %s", name, e)
                pipelines[name] = None
        else:
            logger.warning("File for pipeline '%s' not found at path: %s", name, path)
            pipelines[name] = None

    yield

    # Clear on shutdown
    pipelines.clear()

app = FastAPI(
    title="Music App API",
    description="REST API for Music App.",
    version="1.0.0",
    lifespan=lifespan
)

def get_model_info(model_data, include_metrics=True):
    """Helper to safely extract model status and metrics."""
    if not model_data:
        return {"status": "error", "metrics": None} if include_metrics else {"status": "error"}

    result = {"status": "ok"}
    if include_metrics:
        result["metrics"] = getattr(model_data, "metrics_", None)

    return result

@app.get("/health")
def health_check():
    """Api health. Returns status and loaded models."""
    return {
        "status": "ok",
        "models": {
            "regression": get_model_info(pipelines.get("regression"), include_metrics=True),
            "classification": get_model_info(pipelines.get("classification"), include_metrics=True),
            "recommendation": get_model_info(pipelines.get("recommendation"), include_metrics=False),
        }
    }

@app.post("/predict")
def predict(data: RegressionInput):
    """Endpoint that performs prediction based on the provided track data."""
    pipeline = pipelines.get("regression")

    # 503 Service Unavailable, if pipeline is not loaded
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded.")

    try:
        input_df = pd.DataFrame([data.model_dump(by_alias=True)])

        prediction = pipeline.predict(input_df)

        return {"prediction": float(prediction[0])}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Data format error: {str(ve)}") from ve
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}") from e


@app.post("/classify")
def classify(data: ClassificationInput):
    """Endpoint that performs genre classification based on the provided track data."""
    pipeline = pipelines.get("classification")

    # 503 Service Unavailable, if pipeline is not loaded
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded.")

    try:
        input_df = pd.DataFrame([data.model_dump(by_alias=True)])

        prediction_int = pipeline.predict(input_df)

        # Match int to genre name
        genre_name = "Unknown"
        if hasattr(pipeline, "target_classes_"):
            genre_name = str(pipeline.target_classes_[prediction_int])

        return {"genre_name": genre_name}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Data format error: {str(ve)}") from ve
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}") from e

@app.post("/recommend")
def recommend(data: RecommendationInput, n_recommendations: int = 5):
    """Endpoint returning the most similar tracks from the training set based on the data."""
    pipeline = pipelines.get("recommendation")

    # 503 Service Unavailable, if pipeline is not loaded
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded.")

    try:
        input_df = pd.DataFrame([data.model_dump(by_alias=True)])

        # Run transformation from pipeline
        X_transformed = pipeline[:-1].transform(input_df)

        # Get nn model
        nn_model = pipeline.named_steps['recommender']

        distances, indices = nn_model.kneighbors(X_transformed, n_neighbors=n_recommendations)

        recommended_indices = indices[0].tolist()
        recommended_distances = distances[0].tolist()

        recommendations = []
        if hasattr(pipeline, "metadata_"):
            metadata_df = pipeline.metadata_
            recommended_metadata = metadata_df.iloc[recommended_indices]

            for i, (_, row) in enumerate(recommended_metadata.iterrows()):
                recommendations.append({
                    "track_id": str(row.get("track_id", "")),
                    "track_name": row["track_name"],
                    "artists": row["artists"],
                    "genre": row["track_genre"],
                    "cosine_distance": float(recommended_distances[i])
                })
        else:
            # Fallback
            recommendations = [
                {"dataset_index": idx, "distance": dist}
                for idx, dist in zip(recommended_indices, recommended_distances)
            ]

        return {
            "recommendations": recommendations
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Data format error: {str(ve)}") from ve
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}") from e

@app.get("/genres")
def get_available_genres():
    """Returns a list of genres supported by the current classification model"""
    cls_pipeline = pipelines.get("classification")

    if cls_pipeline is None or not hasattr(cls_pipeline, "target_classes_"):
        raise HTTPException(
            status_code=503,
            detail="Classification model is currently unavailable."
        )

    return {"genres": cls_pipeline.target_classes_.tolist()}
