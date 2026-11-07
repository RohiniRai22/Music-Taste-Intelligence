
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(BASE_DIR / "data" / "spotify_tracks_cleaned.csv")

# Select numerical audio features
features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms"
]

X = df[features]

print("\n========== FEATURE SELECTION ==========")

print("\nSelected Features:")
print(features)

print("\nFeature Dataset Shape:")
print(X.shape)

print("\nFeature Summary:")
print(X.describe())

print("\nMissing Values:")
print(X.isnull().sum().sum())

print("\nFEATURE SELECTION COMPLETED!")
