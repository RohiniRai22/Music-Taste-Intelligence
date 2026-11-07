
import pandas as pd
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load original dataset
df = pd.read_csv(BASE_DIR / "data" / "spotify_tracks.csv")

print("Original Dataset Shape:", df.shape)

# 1. Remove exact duplicate rows
df = df.drop_duplicates()

# 2. Remove records with zero duration
df = df[df["duration_ms"] > 0]

# 3. Fill missing text metadata
df["artists"] = df["artists"].fillna("Unknown Artist")
df["album_name"] = df["album_name"].fillna("Unknown Album")
df["track_name"] = df["track_name"].fillna("Unknown Track")

# 4. Check missing values after cleaning
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# 5. Check duplicate rows
print("\nExact Duplicate Rows:", df.duplicated().sum())

# 6. Final dataset shape
print("\nCleaned Dataset Shape:", df.shape)

# 7. Save cleaned dataset separately
output_path = BASE_DIR / "data" / "spotify_tracks_cleaned.csv"
df.to_csv(output_path, index=False)

print("\nCleaned dataset saved successfully!")
