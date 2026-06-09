# 🎵 Music App

A full-stack machine learning application to predict track popularity (Regression), classify genres (Classification), and recommend similar songs (Similarity Search) from dataset based on Spotify track features.

It is based on [Spotify Tracks Dataset by Maharshi Pandya](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)

## 🛠️ Built With

* **Language:** Python 3.11
* **Frontend:** Streamlit
* **Backend API:** FastAPI, Uvicorn, Pydantic
* **Machine Learning:** scikit-learn (Random Forest, K-Nearest Neighbors), pandas, Hugging Face `datasets`
* **Containerization:** Docker, Docker Compose

---

## Requirements
- **Docker**
- **Docker Compose**
- **(Optional) Python 3.11 for local development without Docker**

---

## 🐳 Quickstart with Docker Compose (Recommended)

The easiest way to run the application is using Docker Compose. This ensures all dependencies and environments are perfectly replicated.

### 1. Train the Models
Before starting the web services, you must download the dataset and train the machine learning models:
```bash
docker-compose run --rm train
```

### 2. Start the App

Once the models are generated, start the frontend and backend services:

```bash
docker-compose up --build
```

### 3. Access the Application
- Frontend: http://localhost:8501
- Backend API Docs: http://localhost:8000/docs

## 💻 Local Development (Without Docker)
**1. Create a virtual environment with python 3.11:**

*Linux:*
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

*Windows:*
```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate
```

**2. Install requirements:**
```bash
pip install -r model/requirements.txt
pip install -r app/backend/requirements.txt
pip install -r app/frontend/requirements.txt
```

**3. Train models:**
```bash
python model/train.py
```

**3. Run services**

*Backend*
```bash
uvicorn app.backend.main:app --reload
```

*Frontend*
```bash
streamlit run app/frontend/app.py
```

## 🧪 Sample Data for Testing
You can use the following JSON payloads to test the API directly via the FastAPI docs (http://127.0.0.1:8000/docs) or enter these values into the Streamlit UI.

**Classical**

```json
{
  "duration_ms": 150000,
  "explicit": false,
  "danceability": 0.150,
  "energy": 0.080,
  "key": 7,
  "loudness": -21.5,
  "mode": 0,
  "speechiness": 0.035,
  "acousticness": 0.990,
  "instrumentalness": 0.920,
  "liveness": 0.110,
  "valence": 0.120,
  "tempo": 75.5,
  "time_signature": 4,
  "track_genre": "classical"
}
```

**Energetic Rock / Grunge**
```json
{
  "duration_ms": 301920,
  "explicit": false,
  "danceability": 0.502,
  "energy": 0.912,
  "key": 1,
  "loudness": -4.556,
  "mode": 1,
  "speechiness": 0.0564,
  "acousticness": 0.000025,
  "instrumentalness": 0.000173,
  "liveness": 0.106,
  "valence": 0.720,
  "tempo": 116.761,
  "time_signature": 4,
  "track_genre": "rock"
}
```

**Club EDM / House**

```json
{
  "duration_ms": 210000,
  "explicit": false,
  "danceability": 0.820,
  "energy": 0.890,
  "key": 7,
  "loudness": -5.200,
  "mode": 1,
  "speechiness": 0.045,
  "acousticness": 0.005,
  "instrumentalness": 0.650,
  "liveness": 0.080,
  "valence": 0.550,
  "tempo": 128.000,
  "time_signature": 4,
  "track_genre": "edm"
}
```

**Synthwave**
```json
{
  "duration_ms": 255000,
  "explicit": false,
  "danceability": 0.610,
  "energy": 0.920,
  "key": 2,
  "loudness": -3.8,
  "mode": 0,
  "speechiness": 0.055,
  "acousticness": 0.0001,
  "instrumentalness": 0.890,
  "liveness": 0.150,
  "valence": 0.220,
  "tempo": 115.0,
  "time_signature": 4,
  "track_genre": "electro"
}
```