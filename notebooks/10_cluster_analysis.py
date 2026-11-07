
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    BASE_DIR / "data" / "spotify_tracks_clustered.csv"
)

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

print("\n========== CLUSTER ANALYSIS ==========")

# Calculate average audio features per cluster
cluster_profiles = df.groupby("cluster")[features].mean()

print("\nAverage Audio Features Per Cluster:")
print(cluster_profiles.round(2))

# Count tracks in each cluster
print("\nCluster Sizes:")
print(df["cluster"].value_counts().sort_index())

# Display sample tracks from each cluster
for cluster in sorted(df["cluster"].unique()):

    print(f"\n========== CLUSTER {cluster} SAMPLE TRACKS ==========")

    sample = df[df["cluster"] == cluster][
        ["track_name", "artists", "track_genre"]
    ].head(5)

    print(sample.to_string(index=False))

# Save cluster profiles
cluster_profiles.to_csv(
    BASE_DIR / "data" / "cluster_profiles.csv"
)

print("\nCluster analysis completed!")
