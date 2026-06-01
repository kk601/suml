import logging
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

pipelines = {}

BASE_DIR = Path(__file__).resolve().parent.parent
REG_PIPELINE_PATH = BASE_DIR / "pipelines" / "regression_pipeline.pkl"
CLF_PIPELINE_PATH = BASE_DIR / "pipelines" / "classification_pipeline.pkl"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Funkcja uruchamiana przy starcie i zamykaniu aplikacji ładująca i czyszcząca pipeline z modelami"""
    if REG_PIPELINE_PATH.exists():
        with open(REG_PIPELINE_PATH, "rb") as f:
            pipelines["regression"] = pickle.load(f)
        print(f"Pipeline załadowany pomyślnie z {REG_PIPELINE_PATH}")
    else:
        pipelines["regression"] = None
        print(f"Nie znaleziono pipeline pod ścieżką {REG_PIPELINE_PATH}")
    
    yield
    
    # Czyszczenie zasobów przy wyłączaniu serwera
    pipelines.clear()

app = FastAPI(
    title="Music App API",
    description="REST API for Music App.",
    version="1.0.0",
    lifespan=lifespan
)