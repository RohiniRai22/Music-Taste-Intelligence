
import pandas as pd
from pathlib import Path

# Load dataset
BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE_DIR / "data" / "spotify_tracks.csv")

print("\n========== DUPLICATE ID INVESTIGATION ==========")

# 1. Total repeated track IDs
print("\nRepeated Track ID Count:")
print(df["track_id"].duplicated().sum())

# 2. Number of unique tracks
print("\nUnique Track IDs:")
print(df["track_id"].nunique())

# 3. Track IDs appearing multiple times
repeated = df[df.duplicated("track_id", keep=False)]

print("\nSample Repeated Track IDs:")
print(
    repeated[
        ["track_id", "track_name", "artists", "track_genre"]
    ].head(15)
)

# 4. Check whether repeated IDs have different genres
genre_counts = df.groupby("track_id")["track_genre"].nunique()

print("\nTrack IDs Associated With Multiple Genres:")
print((genre_counts > 1).sum())

# 5. Investigate zero tempo
print("\nZero Tempo Records:")
print(
    df[df["tempo"] == 0][
        ["track_name", "artists", "track_genre", "tempo"]
    ].head(15)
)

# 6. Investigate zero duration
print("\nZero Duration Records:")
print(
    df[df["duration_ms"] == 0][
        ["track_name", "artists", "duration_ms"]
    ]
)

#print("\nINVESTIGATION COMPLETED!")
