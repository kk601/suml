import logging
import joblib
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder


from app.backend.schemas import TrackInput

pipelines = {}

BASE_DIR = Path(__file__).resolve().parent.parent
PIPELINES_DIR = BASE_DIR / "pipelines"

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
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
                logger.info(f"Pipeline '{name}' załadowany pomyślnie.")
            except Exception as e:
                logger.error(f"Błąd podczas ładowania pipeline '{name}': {e}")
                pipelines[name] = None
        else:
            logger.warning(f"Nie znaleziono pliku dla pipeline '{name}' pod ścieżką: {path}")
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
    """Helper to safely extract model status."""
    if not model_data:
        return {"status": "error", "metrics": None} if include_metrics else {"status": "error"}
    
    result = {"status": "ok"}
    if include_metrics:
        result["metrics"] = model_data.get("metrics")
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
def predict(data: TrackInput):
    """
    Endpoint wykonujący predykcję na podstawie przekazanych danych pasażera.
    """
    pipeline = pipelines.get("regression")
    
    # 503 Service Unavailable, jeśli pipeline nie jest załadowany
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded.")

    try:
        input_df = pd.DataFrame([data.model_dump(by_alias=True)])

        prediction = pipeline.predict(input_df)
        
        return {"prediction": float(prediction[0])}
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Data format error: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")