# Music app

Create venv(tested with python 3.11):  

Linux:  
    `python3.11 -m venv .venv`  
    `source .venv/bin/activate`

Windows:  
    `py -3.11 -m venv .venv`  
    `.venv\Scripts\activate`  


Install requirements:  
`pip install -r model/requirements.txt`  
`pip install -r app/backend/requirements.txt`

Create models:  
`python model/train.py`

Run backend:  
`uvicorn app.backend.main:app --reload`

Test  
http://127.0.0.1:8000/docs

Sample data:

- Classical

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

- Energetic Rock / Grunge
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

- Club EDM / House

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

- Synthwave
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