
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.metrics import pairwise_distances


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

CLEANED_DATA_PATH = DATA_DIR / "spotify_tracks_cleaned.csv"
PCA_DATA_PATH = DATA_DIR / "spotify_features_pca.csv"

SCALER_PATH = MODEL_DIR / "standard_scaler.pkl"
PCA_MODEL_PATH = MODEL_DIR / "pca_model.pkl"
KMEANS_PATH = MODEL_DIR / "kmeans_model.pkl"

OUTPUT_PATH = DATA_DIR / "improved_music_recommendations.csv"


# ==========================================
# 2. LOAD DATA AND MODELS
# ==========================================

print("\n========== IMPROVED MUSIC RECOMMENDATION SYSTEM ==========")

df = pd.read_csv(CLEANED_DATA_PATH)
pca_reference = pd.read_csv(PCA_DATA_PATH)

scaler = joblib.load(SCALER_PATH)
pca_model = joblib.load(PCA_MODEL_PATH)
kmeans = joblib.load(KMEANS_PATH)

print("Dataset Loaded:", df.shape)


# ==========================================
# 3. SELECT AUDIO FEATURES
# ==========================================

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


# ==========================================
# 4. SCALE FEATURES
# ==========================================

X_scaled = scaler.transform(X)

print("Feature Scaling Completed!")


# ==========================================
# 5. APPLY PCA
# ==========================================

X_pca = pca_model.transform(X_scaled)

pca_df = pd.DataFrame(
    X_pca,
    columns=pca_reference.columns
)

print("PCA Transformation Completed!")


# ==========================================
# 6. PREDICT K-MEANS CLUSTERS
# ==========================================

clusters = kmeans.predict(pca_df)

df["cluster"] = clusters

print("Cluster Prediction Completed!")
print("Number of Clusters:", len(np.unique(clusters)))


# ==========================================
# 7. GET USER INPUT
# ==========================================

track_name = input("\nEnter track name: ").strip()

matches = df[
    df["track_name"].str.strip().str.lower()
    == track_name.lower()
]

if matches.empty:
    print("\nTrack Not Found!")
    raise SystemExit


# ==========================================
# 8. HANDLE MULTIPLE MATCHES
# ==========================================

if len(matches) > 1:

    print("\nMultiple matching tracks found:")

    for i, (_, track) in enumerate(matches.iterrows(), start=1):
        print(
            f"{i}. {track['track_name']} - "
            f"{track['artists']} - "
            f"{track['track_genre']}"
        )

    while True:
        try:
            choice = int(input("\nSelect track number: "))

            if 1 <= choice <= len(matches):
                selected_index = matches.index[choice - 1]
                break

            print("Invalid selection. Try again.")

        except ValueError:
            print("Please enter a valid number.")

else:
    selected_index = matches.index[0]


selected_track = df.loc[selected_index]

selected_cluster = selected_track["cluster"]

print("\n========== SELECTED TRACK ==========")

print("Track Name:", selected_track["track_name"])
print("Artist:", selected_track["artists"])
print("Genre:", selected_track["track_genre"])
print("Cluster:", selected_cluster)


# ==========================================
# 9. FIND TRACKS IN THE SAME CLUSTER
# ==========================================

candidate_indices = np.where(
    clusters == selected_cluster
)[0]

candidates = df.iloc[candidate_indices].copy()

candidates["_row_index"] = candidate_indices

# Exclude the selected track itself and its repeated track IDs.

selected_track_id = selected_track["track_id"]

if pd.notna(selected_track_id):
    candidates = candidates[
        candidates["track_id"] != selected_track_id
    ]
else:
    candidates = candidates[
        candidates["_row_index"] != selected_index
    ]

# Keep only one recommendation per Spotify track ID.

candidates = candidates.drop_duplicates(
    subset=["track_id"],
    keep="first"
)

print("\nUnique Candidate Tracks:", len(candidates))


# ==========================================
# 10. CALCULATE EUCLIDEAN DISTANCE
# ==========================================

candidate_positions = candidates["_row_index"].to_numpy()

selected_features = X_scaled[selected_index].reshape(1, -1)

candidate_features = X_scaled[candidate_positions]

distances = pairwise_distances(
    selected_features,
    candidate_features,
    metric="euclidean"
)[0]

candidates["distance"] = distances


# ==========================================
# 11. RANK RECOMMENDATIONS
# ==========================================

recommendations = candidates.sort_values(
    by="distance",
    ascending=True
).head(10).copy()

recommendations.insert(
    0,
    "rank",
    range(1, len(recommendations) + 1)
)


# ==========================================
# 12. DISPLAY RECOMMENDATIONS
# ==========================================

display_columns = [
    "rank",
    "track_name",
    "artists",
    "track_genre",
    "distance"
]

print("\n========== TOP MUSIC RECOMMENDATIONS ==========\n")

if recommendations.empty:
    print("No recommendations found in this cluster.")

else:
    print(
        recommendations[display_columns].to_string(
            index=False
        )
    )


# ==========================================
# 13. SAVE RECOMMENDATIONS
# ==========================================

recommendations.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nRecommendations Saved Successfully!")
print("Output File:", OUTPUT_PATH)

print("\n========== RECOMMENDATION SYSTEM COMPLETED ==========")
 