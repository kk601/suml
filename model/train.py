import pickle
import logging
import os
import pandas as pd
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,accuracy_score,precision_score,recall_score,f1_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_data() -> pd.DataFrame:
    """Downloads the dataset from Hugging Face and returns a Pandas DataFrame."""    
    logger.info("Pobieranie zbioru danych z Hugging Face...")
    dataset = load_dataset("maharshipandya/spotify-tracks-dataset")

    df = dataset['train'].to_pandas()
    logger.info(f"Pomyślnie załadowano dane. Rozmiar: {df.shape}")
    return df

def preprocess_features(df: pd.DataFrame):
    """Preprocess data for regression and classification"""
    logger.info("Rozpoczęto wstępny preprocessing...")
    X = df.drop(["track_id", "artists", "album_name", "track_name"], axis=1)

    X['explicit'] = X['explicit'].astype(bool).map({False: 0, True: 1})

    return X


def prepare_regression(df: pd.DataFrame):
    """Prepare data for regression on popularity"""
    logger.info("Przygotowywanie danych pod regresję...")
    y = df['popularity'].astype(int)
    X = df.drop(["popularity"], axis=1) 

    X = pd.get_dummies(X, columns=['track_genre'], dtype=int)  

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


def prepare_classification(df: pd.DataFrame):
    """Prepare data for classification by track_genre"""
    logger.info("Przygotowywanie danych pod klasyfikację...")
    y_text = df['track_genre']
    le = LabelEncoder()
    y = le.fit_transform(y_text)

    X = df.drop(["track_genre"], axis=1) 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, le


def prepare_recommendation(df: pd.DataFrame):
    """Prepare data for Nearest Neighbors recommendation"""
    logger.info("Przygotowywanie danych pod system rekomendacji...")
    
    X = pd.get_dummies(df, columns=['track_genre'], dtype=int)  
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler

def split_data(X: pd.DataFrame, y: pd.DataFrame, stratify_col=None):
    """Split the dataset into train and test subsets."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=61,
        stratify=stratify_col
    )

    return X_train, X_test, y_train, y_test

def train_model(model: BaseEstimator, X_train: pd.DataFrame, y_train: pd.DataFrame = None):
    """Initialize and train model."""
    model_name = type(model).__name__
    logger.info(f"Rozpoczęto trenowanie modelu: {model_name}...")
    model.fit(X_train, y_train)
    logger.info(f"Zakończono trenowanie modelu: {model_name}.")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model quality metrics."""
    logger.info("Ewaluacja modelu na zbiorze testowym...")
    y_pred = model.predict(X_test)

    if is_regressor(model):
        metrics = {
            "mean_absolute_error": float(mean_absolute_error(y_test, y_pred)),
            "mean_squared_error": float(mean_squared_error(y_test, y_pred))
        }
        logger.info(f"Metryki Regresji: {metrics}")
    elif is_classifier(model):
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
        }
        logger.info(f"Metryki Klasyfikacji: {metrics}")
        
    return metrics

def save_pipeline(pipeline, output_path: str) -> None:
    """Save pipeline with model to .pkl file."""
    with open(output_path, 'wb') as f:
        pickle.dump(pipeline, f)
    logger.info(f"Zapisano pipeline do pliku: {output_path}")

def main():
    model_path = 'saved_model.pkl'

    df = load_data()
    df = preprocess_features(df)

    X_reg, y_reg, scaler_reg = prepare_regression(df)
    X_clf, y_clf, scaler_clf, le_clf = prepare_classification(df)
    X_nn, scaler_nn = prepare_recommendation(df)

    logger.info("Podział na zbiory treningowe i testowe...")
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = split_data(
        X_reg,
        y_reg
    )
    X_clf_train, X_clf_test, y_clf_train, y_clf_test = split_data(
        X_clf,
        y_clf,
        y_clf
    )

    model_reg = RandomForestRegressor(
        n_estimators=100,
        random_state=61,
        max_depth=10
    )

    model_clf = RandomForestClassifier(
        n_estimators=100,
        random_state=61,
        max_depth=10
    )

    model_nn = NearestNeighbors (
        n_neighbors=5,
        metric="cosine"
    )

    model_reg = train_model(model_reg,X_reg_train,y_reg_train)
    model_clf = train_model(model_clf,X_clf_train,y_clf_train)
    model_nn = train_model(model_nn, X_nn)

    metrics_reg = evaluate_model(model_reg, X_reg_test, y_reg_test)
    metrics_clf = evaluate_model(model_clf, X_clf_test, y_clf_test)

    pipeline_reg = {
        "model": model_reg,
        "scaler": scaler_reg,
        "metrics": metrics_reg
    }

    pipeline_clf = {
        "model": model_clf,
        "scaler": scaler_clf,
        "label_encoder": le_clf,
        "metrics": metrics_clf
    }

    pipeline_nn = {
        "model": model_nn,
        "scaler": scaler_nn
    }

    current_dir = os.path.dirname(os.path.abspath(__file__))    
    data_dir = os.path.join(current_dir, "..", "data")

    path_reg = os.path.join(data_dir, "regression_pipeline.pkl")
    path_clf = os.path.join(data_dir, "classification_pipeline.pkl")
    path_nn = os.path.join(data_dir, "recomendation_pipeline.pkl")

    save_pipeline(pipeline_reg,path_reg)
    save_pipeline(pipeline_clf,path_clf)
    save_pipeline(pipeline_nn,path_nn)

    logger.info("Wszystkie operacje zakończone sukcesem")

if __name__ == "__main__":
    main()