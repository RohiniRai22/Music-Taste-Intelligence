
import pandas as pd
import numpy as np
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

print("\n========== DBSCAN CLUSTER ANALYSIS ==========")

# Load original cleaned dataset
df = pd.read_csv(
    BASE_DIR / "data" / "spotify_tracks_cleaned.csv"
)

# Load DBSCAN sample and cluster labels
dbscan_df = pd.read_csv(
    BASE_DIR / "data" / "dbscan_clustered_sample.csv"
)

print("\nOriginal Dataset Shape:", df.shape)
print("DBSCAN Dataset Shape:", dbscan_df.shape)

# Check required columns
if "cluster" not in dbscan_df.columns:
    raise ValueError("Cluster column not found in DBSCAN dataset.")

# Audio features
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

# Check whether feature columns exist
missing_features = [
    col for col in features if col not in dbscan_df.columns
]

if missing_features:
    raise ValueError(
        f"Missing audio feature columns: {missing_features}"
    )

# 1. Cluster distribution
print("\n========== CLUSTER DISTRIBUTION ==========")

cluster_counts = dbscan_df["cluster"].value_counts().sort_index()

print(cluster_counts)

# 2. Cluster percentages
print("\n========== CLUSTER PERCENTAGES ==========")

cluster_percentages = (
    dbscan_df["cluster"].value_counts(normalize=True).sort_index() * 100
)

print(cluster_percentages.round(2))

# 3. Noise analysis
print("\n========== NOISE ANALYSIS ==========")

noise_count = (dbscan_df["cluster"] == -1).sum()
total_tracks = len(dbscan_df)

noise_percentage = (noise_count / total_tracks) * 100

print("Total Tracks:", total_tracks)
print("Noise Points:", noise_count)
print("Noise Percentage:", round(noise_percentage, 2), "%")

# 4. Audio feature profiles
print("\n========== AUDIO FEATURE PROFILES ==========")

cluster_profiles = (
    dbscan_df[dbscan_df["cluster"] != -1]
    .groupby("cluster")[features]
    .mean()
    .round(3)
)

print(cluster_profiles)

# 5. Dominant genres in each cluster
print("\n========== TOP GENRES IN EACH CLUSTER ==========")

if "track_genre" in dbscan_df.columns:

    for cluster_id in sorted(
        dbscan_df["cluster"].unique()
    ):

        if cluster_id == -1:
            continue

        cluster_data = dbscan_df[
            dbscan_df["cluster"] == cluster_id
        ]

        print(f"\nCluster {cluster_id}")

        top_genres = (
            cluster_data["track_genre"]
            .value_counts()
            .head(10)
        )

        print(top_genres)

# 6. Representative tracks from each cluster
print("\n========== REPRESENTATIVE TRACKS ==========")

track_columns = [
    col for col in [
        "track_name",
        "artists",
        "track_genre",
        "cluster"
    ]
    if col in dbscan_df.columns
]

for cluster_id in sorted(
    dbscan_df["cluster"].unique()
):

    if cluster_id == -1:
        continue

    cluster_data = dbscan_df[
        dbscan_df["cluster"] == cluster_id
    ]

    print(f"\nCluster {cluster_id} - Sample Tracks")

    print(
        cluster_data[track_columns]
        .sample(
            n=min(10, len(cluster_data)),
            random_state=42
        )
        .to_string(index=False)
    )

# 7. Save cluster profiles
output_path = (
    BASE_DIR / "data" / "dbscan_cluster_profiles.csv"
)

cluster_profiles.to_csv(output_path)

print("\nCluster profiles saved successfully!")
print("File Location:", output_path)

print("\nDBSCAN CLUSTER ANALYSIS COMPLETED!")
