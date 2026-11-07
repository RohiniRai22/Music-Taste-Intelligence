
import pandas as pd
from pathlib import Path

# Load dataset
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "spotify_tracks.csv"

df = pd.read_csv(DATA_PATH)

print("\n========== DATA QUALITY INVESTIGATION ==========")

# 1. Missing values
print("\nMissing Values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# 2. Duplicate records
print("\nExact Duplicate Rows:")
print(df.duplicated().sum())

# 3. Duplicate track IDs
print("\nDuplicate Track IDs:")
print(df["track_id"].duplicated().sum())

# 4. Zero duration
print("\nZero Duration Records:")
print((df["duration_ms"] == 0).sum())

# 5. Zero tempo
print("\nZero Tempo Records:")
print((df["tempo"] == 0).sum())

# 6. Extremely long tracks
print("\n10 Longest Tracks:")
print(
    df.nlargest(10, "duration_ms")[
        ["track_name", "artists", "duration_ms"]
    ]
)

# 7. Audio feature ranges
features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

print("\nAudio Feature Ranges:")
print(df[features].agg(["min", "max"]))

print("\nDATA QUALITY INVESTIGATION COMPLETED!")
