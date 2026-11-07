
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "spotify_tracks_cleaned.csv"
PCA_PATH = BASE_DIR / "data" / "spotify_features_pca.csv"

SCALER_PATH = BASE_DIR / "models" / "standard_scaler.pkl"
PCA_MODEL_PATH = BASE_DIR / "models" / "pca_model.pkl"
KMEANS_PATH = BASE_DIR / "models" / "kmeans_model.pkl"

OUTPUT_PATH = BASE_DIR / "data" / "music_recommendations.csv"

# ==========================================
# 2. LOAD DATA AND MODELS
# ==========================================

print("\n========== MUSIC RECOMMENDATION SYSTEM ==========")

df = pd.read_csv(DATA_PATH)
pca_df = pd.read_csv(PCA_PATH)

scaler = joblib.load(SCALER_PATH)
pca = joblib.load(PCA_MODEL_PATH)
kmeans = joblib.load(KMEANS_PATH)

print("Music Dataset Shape:", df.shape)
print("PCA Dataset Shape:", pca_df.shape)

print("Scaler loaded successfully!")
print("PCA model loaded successfully!")
print("K-Means model loaded successfully!")

# ==========================================
# 3. PREPARE PCA FEATURES
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

X_scaled = scaler.transform(X)

X_pca_array = pca.transform(X_scaled)

# Preserve the PCA feature names expected by K-Means.
X_pca = pd.DataFrame(
    X_pca_array,
    columns=pca_df.columns,
    index=df.index
)

print("\nPCA transformation completed!")

# ==========================================
# 4. PREDICT CLUSTERS
# ==========================================

df["cluster"] = kmeans.predict(X_pca)

print("Cluster assignment completed!")

# ==========================================
# 5. GET USER INPUT
# ==========================================

print("\n========== SEARCH MUSIC TRACK ==========")

track_name = input("Enter track name: ").strip()

if not track_name:
    raise ValueError("Track name cannot be empty.")

matches = df[
    df["track_name"].astype(str).str.casefold()
    == track_name.casefold()
]

if matches.empty:
    print("\nTrack not found in the dataset.")
    print("Please try another track name.")
    raise SystemExit

# ==========================================
# 6. HANDLE MULTIPLE MATCHES
# ==========================================

if len(matches) > 1:

    print("\nMultiple matching tracks found:")

    for i, (_, row) in enumerate(matches.iterrows(), start=1):
        print(f"{i}. {row['track_name']} - {row['artists']}")

    choice = int(input("\nSelect track number: "))

    if choice < 1 or choice > len(matches):
        raise ValueError("Invalid track selection.")

    selected_index = matches.index[choice - 1]

else:
    selected_index = matches.index[0]

selected_track = df.loc[selected_index]

# ==========================================
# 7. IDENTIFY TRACK CLUSTER
# ==========================================

selected_cluster = selected_track["cluster"]

print("\n========== SELECTED TRACK ==========")

print("Track Name:", selected_track["track_name"])
print("Artist:", selected_track["artists"])
print("Genre:", selected_track["track_genre"])
print("Cluster:", selected_cluster)

# ==========================================
# 8. FIND TRACKS IN SAME CLUSTER
# ==========================================

cluster_indices = df.index[
    df["cluster"] == selected_cluster
].to_numpy()

cluster_indices = cluster_indices[
    cluster_indices != selected_index
]

if len(cluster_indices) == 0:
    print("\nNo similar tracks found.")
    raise SystemExit

# ==========================================
# 9. CALCULATE COSINE SIMILARITY
# ==========================================

query_vector = X_pca.loc[[selected_index]]

candidate_vectors = X_pca.loc[cluster_indices]

similarity_scores = cosine_similarity(
    query_vector,
    candidate_vectors
)[0]

# ==========================================
# 10. GET TOP 10 RECOMMENDATIONS
# ==========================================

recommendation_indices = np.argsort(
    similarity_scores
)[::-1][:10]

recommended_indices = cluster_indices[
    recommendation_indices
]

recommended_scores = similarity_scores[
    recommendation_indices
]

recommendations = df.loc[
    recommended_indices,
    ["track_name", "artists", "track_genre", "cluster"]
].copy()

recommendations["similarity_score"] = (
    recommended_scores * 100
).round(2)

recommendations = recommendations.reset_index(drop=True)

# ==========================================
# 11. DISPLAY RECOMMENDATIONS
# ==========================================

print("\n========== TOP 10 MUSIC RECOMMENDATIONS ==========")

print(recommendations.to_string(index=False))

# ==========================================
# 12. SAVE RECOMMENDATIONS
# ==========================================

recommendations.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nRecommendations saved successfully!")
print("Saved Location:", OUTPUT_PATH)

#print("\nMUSIC RECOMMENDATION SYSTEM COMPLETED!")
