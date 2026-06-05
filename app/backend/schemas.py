from pydantic import BaseModel, Field

class TrackFeatures(BaseModel):
    duration_ms: int = Field(..., description="Czas trwania utworu w milisekundach", example=190000)
    explicit: bool = Field(..., description="Czy utwór zawiera wulgaryzmy", example=False)
    danceability: float = Field(..., description="Taneczność (0.0 - 1.0)", example=0.55)
    energy: float = Field(..., description="Energia (0.0 - 1.0)", example=0.35)
    key: int = Field(..., description="Tonacja utworu (0 - 11)", example=0)
    loudness: float = Field(..., description="Głośność w dB (zazwyczaj -60 do 0)", example=-10.5)
    mode: int = Field(..., description="Rodzaj skali (0 dla moll, 1 dla dur)", example=1)
    speechiness: float = Field(..., description="Mowność utworu (0.0 - 1.0)", example=0.04)
    acousticness: float = Field(..., description="Akustyczność (0.0 - 1.0)", example=0.75)
    instrumentalness: float = Field(..., description="Instrumentalność (0.0 - 1.0)", example=0.0)
    liveness: float = Field(..., description="Ewentualność nagrania na żywo (0.0 - 1.0)", example=0.1)
    valence: float = Field(..., description="Pozytywny ładunek emocjonalny (0.0 - 1.0)", example=0.45)
    tempo: float = Field(..., description="Tempo utworu w BPM", example=110.0)
    time_signature: int = Field(..., description="Metrum utworu (np. 3, 4, 5)", example=4)

class RegressionInput(TrackFeatures):
    track_genre: str = Field(..., example="pop")

class ClassificationInput(TrackFeatures):
    pass

class RecomendationInput(TrackFeatures):
    track_genre: str = Field(..., example="pop")
    pass