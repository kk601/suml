"""Schemas for the Music App API"""
from pydantic import BaseModel, Field


class TrackFeatures(BaseModel):
    duration_ms: int = Field(
        ..., title="Duration (ms)", description="Track duration in milliseconds",
        example=190000, ge=0, le=5237295
    )
    explicit: bool = Field(
        ..., title="Explicit", description="Whether the track contains explicit lyrics",
        example=False
    )
    danceability: float = Field(
        ..., title="Danceability", description="Danceability (0.0 - 1.0)",
        example=0.55, ge=0.0, le=1.0
    )
    energy: float = Field(
        ..., title="Energy", description="Energy (0.0 - 1.0)",
        example=0.35, ge=0.0, le=1.0
    )
    key: int = Field(
        ..., title="Key", description="Key of the track (0 - 11)",
        example=0, ge=0, le=11
    )
    loudness: float = Field(
        ..., title="Loudness (dB)", description="Loudness in dB (typically -60 to 0)",
        example=-10.5, ge=-49.5310, le=4.5320
    )
    mode: int = Field(
        ..., title="Mode", description="Scale mode (0 for minor, 1 for major)",
        example=1, ge=0, le=1
    )
    speechiness: float = Field(
        ..., title="Speechiness", description="Speechiness of the track (0.0 - 1.0)",
        example=0.04, ge=0.0, le=1.0
    )
    acousticness: float = Field(
        ..., title="Acousticness", description="Acousticness (0.0 - 1.0)",
        example=0.75, ge=0.0, le=1.0
    )
    instrumentalness: float = Field(
        ..., title="Instrumentalness", description="Instrumentalness (0.0 - 1.0)",
        example=0.0, ge=0.0, le=1.0
    )
    liveness: float = Field(
        ..., title="Liveness",description="Liveness (probability it was recorded live) (0.0 - 1.0)",
        example=0.1, ge=0.0, le=1.0
    )
    valence: float = Field(
        ..., title="Valence", description="Valence (positive emotional charge) (0.0 - 1.0)",
        example=0.45, ge=0.0, le=1.0
    )
    tempo: float = Field(
        ..., title="Tempo (BPM)", description="Track tempo in BPM",
        example=110.0, ge=0.0, le=243.3720
    )
    time_signature: int = Field(
        ..., title="Time Signature", description="Time signature (e.g., 3, 4, 5)",
        example=4, ge=1, le=5
    )


class RegressionInput(TrackFeatures):
    track_genre: str = Field(
        ..., title="Genre Context", example="pop")


class ClassificationInput(TrackFeatures):
    pass


class RecommendationInput(TrackFeatures):
    track_genre: str = Field(
        ..., title="Genre Context",
        example="pop")
