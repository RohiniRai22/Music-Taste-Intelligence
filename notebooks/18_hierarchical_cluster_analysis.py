
import pandas as pd
from pathlib import Path

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

HIERARCHICAL_PATH = BASE_DIR / "data" / "hierarchical_clustered_sample.csv"
CLEANED_PATH = BASE_DIR / "data" / "spotify_tracks_cleaned.csv"

OUTPUT_PATH = BASE_DIR / "data" / "hierarchical_cluster_profiles.csv"

# ==========================================
# 2. LOAD DATASETS
# ==========================================

print("\n========== HIERARCHICAL CLUSTER ANALYSIS ==========")

hierarchical_df = pd.read_csv(HIERARCHICAL_PATH)
cleaned_df = pd.read_csv(CLEANED_PATH)

print("Hierarchical Dataset Shape:", hierarchical_df.shape)
print("Cleaned Dataset Shape:", cleaned_df.shape)

# ==========================================
# 3. RECREATE THE SAME SAMPLE
# ==========================================

# Recreate the exact sample used in script 16.
# This preserves the original row indices for metadata matching.

pca_df = pd.read_csv(BASE_DIR / "data" / "spotify_features_pca.csv")

sample_indices = pca_df.sample(
    n=10000,
    random_state=42
).index

metadata_sample = cleaned_df.iloc[sample_indices].reset_index(drop=True)

# Attach hierarchical cluster labels.
metadata_sample["cluster"] = hierarchical_df["cluster"].values

print("\nMetadata successfully matched with cluster labels.")

# ==========================================
# 4. CLUSTER DISTRIBUTION
# ==========================================

print("\n========== CLUSTER DISTRIBUTION ==========")

cluster_counts = metadata_sample["cluster"].value_counts().sort_index()

print(cluster_counts)

# ==========================================
# 5. AUDIO FEATURE PROFILES
# ==========================================

print("\n========== AUDIO FEATURE PROFILES ==========")

audio_features = [
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

cluster_profiles = metadata_sample.groupby("cluster")[audio_features].mean()

print(cluster_profiles.round(3))

# ==========================================
# 6. TOP GENRES IN EACH CLUSTER
# ==========================================

print("\n========== TOP GENRES IN EACH CLUSTER ==========")

for cluster in sorted(metadata_sample["cluster"].unique()):

    print(f"\nCluster {cluster}:")

    cluster_data = metadata_sample[
        metadata_sample["cluster"] == cluster
    ]

    top_genres = cluster_data["track_genre"].value_counts().head(5)

    print(top_genres)

# ==========================================
# 7. REPRESENTATIVE TRACKS
# ==========================================

print("\n========== REPRESENTATIVE TRACKS ==========")

for cluster in sorted(metadata_sample["cluster"].unique()):

    print(f"\nCluster {cluster}:")

    cluster_data = metadata_sample[
        metadata_sample["cluster"] == cluster
    ]

    representative_tracks = cluster_data[
        ["track_name", "artists", "track_genre"]
    ].sample(
        n=min(5, len(cluster_data)),
        random_state=42
    )

    print(representative_tracks.to_string(index=False))

# ==========================================
# 8. SAVE CLUSTER PROFILES
# ==========================================

cluster_profiles.to_csv(OUTPUT_PATH)

print("\nCluster profiles saved successfully!")
print("Saved Location:", OUTPUT_PATH)

#print("\nHIERARCHICAL CLUSTER ANALYSIS COMPLETED!")
