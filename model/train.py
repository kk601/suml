import joblib
import logging
import os
import pandas as pd
from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_data() -> pd.DataFrame:
    """Downloads the dataset from Hugging Face and returns a Pandas DataFrame."""    
    logger.info("Downloading dataset from Hugging Face...")
    dataset = load_dataset("maharshipandya/spotify-tracks-dataset")

    df = dataset['train'].to_pandas()
    logger.info(f"Successfully loaded data. Size: {df.shape}")
    return df

def preprocess_features(df: pd.DataFrame):
    """Initial preprocessing and filtering."""
    logger.info("Started initial preprocessing...")

    unwanted_genres = [
        'indian', 'bollywood', 'world-music', 'comedy'
    ]

    unwanted_track_ids = df[df['track_genre'].isin(unwanted_genres)]['track_id'].unique()
    df = df[~df['track_id'].isin(unwanted_track_ids)]
    
    df = df.drop_duplicates(subset=['track_name', 'artists']).reset_index(drop=True)

    logger.info(f"Rows remaining after removing duplicates and unwanted genres: {df.shape[0]}")

    df['explicit'] = df['explicit'].astype(bool).map({False: 0, True: 1})

    return df

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

    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model quality metrics."""
    logger.info("Evaluating the model on the test set...")
    y_pred = model.predict(X_test)

    if is_regressor(model):
        metrics = {
            "mean_absolute_error": float(mean_absolute_error(y_test, y_pred)),
            "mean_squared_error": float(mean_squared_error(y_test, y_pred))
        }
        logger.info(f"Regression Metrics: {metrics}")
    elif is_classifier(model):
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
        }
        logger.info(f"Classification Metrics: {metrics}")
        
    return metrics

def save_pipeline(pipeline, output_path: str) -> None:
    """Save pipeline to .pkl file."""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'wb') as f:
        joblib.dump(pipeline, f)
    logger.info(f"Saved pipeline to file: {output_path}")

def main():
    df = load_data()

    df = preprocess_features(df)

    # Store metadata for recommendation model
    metadata = df[["track_id",'track_name', 'artists', 'track_genre']].copy()

    # Drop columns before training
    df = df.drop(["track_id", "artists", "album_name", "track_name", "Unnamed: 0"], axis=1, errors='ignore')

    # Prepare data for regresion
    y_reg = df['popularity'].astype(int)
    X_reg = df.drop(["popularity"], axis=1)

    # Prepare data for classification
    le_clf = LabelEncoder()
    y_clf = le_clf.fit_transform(df['track_genre'])
    X_clf = df.drop(["track_genre","popularity"], axis=1)

    # Prepare data for recommendation
    X_nn = df.drop(["popularity"], axis=1)

    logger.info("Splitting into training and test sets...")
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = split_data(
        X_reg,
        y_reg
    )
    X_clf_train, X_clf_test, y_clf_train, y_clf_test = split_data(
        X_clf,
        y_clf,
        y_clf
    )

    reg_cat_cols = ['track_genre']
    reg_num_cols = X_reg.select_dtypes(include=['number']).columns.tolist()
    clf_num_cols = X_clf.select_dtypes(include=['number']).columns.tolist()


    preprocessor_reg_nn = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), reg_num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), reg_cat_cols)
        ]
    )

    preprocessor_clf = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), clf_num_cols)
        ]
    )

    pipeline_reg = Pipeline(
        steps=[
            ('preprocessor', preprocessor_reg_nn),
            ('regressor', RandomForestRegressor(
                n_estimators=100,
                random_state=61,
                max_depth=15,
                min_samples_split=5,
                n_jobs=-1
            )),
        ]
    )

    pipeline_clf = Pipeline(
        steps=[
            ('preprocessor', preprocessor_clf),
            ('classifier', RandomForestClassifier(
                n_estimators=100, 
                random_state=61, 
                max_depth=15,
                n_jobs=-1
            )),
        ]
    )
    
    pipeline_nn = Pipeline(
        steps=[
            ('preprocessor', preprocessor_reg_nn),
            ('recommender', NearestNeighbors(n_neighbors=5, metric="cosine")),
        ]
    )

    logger.info("Training the regression model...")
    pipeline_reg = train_model(pipeline_reg, X_reg_train, y_reg_train)
    logger.info("Training the classification model...")
    pipeline_clf = train_model(pipeline_clf, X_clf_train, y_clf_train)
    logger.info("Preparing the recommendation model...")
    pipeline_nn = train_model(pipeline_nn, X_nn)

    metrics_reg = evaluate_model(pipeline_reg, X_reg_test, y_reg_test)
    metrics_clf = evaluate_model(pipeline_clf, X_clf_test, y_clf_test)

    # Save metadata in recommendation nn model
    pipeline_nn.metadata_ = metadata

    # Save metrics
    pipeline_reg.metrics_ = metrics_reg
    pipeline_clf.metrics_ = metrics_clf

    # Save classes from encoder
    pipeline_clf.target_classes_ = le_clf.classes_

    current_dir = os.path.dirname(os.path.abspath(__file__))    
    data_dir = os.path.join(current_dir, "..", "data")

    path_reg = os.path.join(data_dir, "regression_pipeline.pkl")
    path_clf = os.path.join(data_dir, "classification_pipeline.pkl")
    path_nn = os.path.join(data_dir, "recommendation_pipeline.pkl")

    save_pipeline(pipeline_reg,path_reg)
    save_pipeline(pipeline_clf,path_clf)
    save_pipeline(pipeline_nn,path_nn)

    logger.info("All operations completed successfully")

if __name__ == "__main__":
    main()