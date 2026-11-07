
import pandas as pd
import numpy as np
import joblib
import re

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

OUTPUT_PATH = DATA_DIR / "final_music_recommendations.csv"


# ==========================================
# 2. LOAD DATA AND MODELS
# ==========================================

print("\n========== FINAL MUSIC RECOMMENDATION ENGINE ==========")

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
# 4. FEATURE SCALING AND PCA
# ==========================================

X_scaled = scaler.transform(X)

X_pca = pca_model.transform(X_scaled)

pca_df = pd.DataFrame(
    X_pca,
    columns=pca_reference.columns
)

print("Feature Scaling Completed!")
print("PCA Transformation Completed!")


# ==========================================
# 5. K-MEANS CLUSTER PREDICTION
# ==========================================

clusters = kmeans.predict(pca_df)

df["cluster"] = clusters

print("Cluster Prediction Completed!")
print("Total Clusters:", len(np.unique(clusters)))


# ==========================================
# 6. NORMALIZE TEXT FOR DEDUPLICATION
# ==========================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    value = re.sub(r"\s+", " ", value)

    value = re.sub(r"[^\w\s]", "", value)

    return value


df["_normalized_name"] = df["track_name"].apply(
    normalize_text
)

df["_normalized_artist"] = df["artists"].apply(
    normalize_text
)


# ==========================================
# 7. SEARCH FOR TRACK
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

            print("Enter a valid number.")

else:

    selected_index = matches.index[0]


selected_track = df.loc[selected_index]

selected_cluster = selected_track["cluster"]
selected_genre = selected_track["track_genre"]

print("\n========== SELECTED TRACK ==========")

print("Track:", selected_track["track_name"])
print("Artist:", selected_track["artists"])
print("Genre:", selected_genre)
print("Cluster:", selected_cluster)


# ==========================================
# 9. ASK FOR GENRE PREFERENCE
# ==========================================

print("\n========== GENRE PREFERENCE ==========")

print("1. Same genre recommendations")
print("2. Similar audio features (all genres)")

while True:

    genre_choice = input("\nChoose option (1 or 2): ").strip()

    if genre_choice in ["1", "2"]:
        break

    print("Please enter 1 or 2.")


# ==========================================
# 10. SELECT SAME-CLUSTER CANDIDATES
# ==========================================

candidate_indices = np.where(
    clusters == selected_cluster
)[0]

candidates = df.iloc[candidate_indices].copy()

candidates["_row_index"] = candidate_indices

# Exclude the selected track itself.

candidates = candidates[
    candidates["_row_index"] != selected_index
]


# ==========================================
# 11. APPLY OPTIONAL GENRE FILTER
# ==========================================

if genre_choice == "1":

    candidates = candidates[
        candidates["track_genre"] == selected_genre
    ]

    print("\nGenre Filter:", selected_genre)

else:

    print("\nGenre Filter: All genres")


# ==========================================
# 12. REMOVE DUPLICATE SONGS
# ==========================================

candidates = candidates.drop_duplicates(
    subset=[
        "_normalized_name",
        "_normalized_artist"
    ],
    keep="first"
)

print("Unique Candidate Tracks:", len(candidates))


# ==========================================
# 13. CALCULATE EUCLIDEAN DISTANCE
# ==========================================

candidate_positions = candidates["_row_index"].to_numpy()

selected_features = X_scaled[
    selected_index
].reshape(1, -1)

candidate_features = X_scaled[
    candidate_positions
]

if len(candidates) == 0:

    print("\nNo recommendations available for this selection.")
    raise SystemExit


distances = pairwise_distances(
    selected_features,
    candidate_features,
    metric="euclidean"
)[0]

candidates["distance"] = distances


# ==========================================
# 14. RANK RECOMMENDATIONS
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
# 15. DISPLAY RECOMMENDATIONS
# ==========================================

display_columns = [
    "rank",
    "track_name",
    "artists",
    "track_genre",
    "distance"
]

print("\n========== TOP 10 RECOMMENDATIONS ==========\n")

print(
    recommendations[display_columns].to_string(
        index=False
    )
)


# ==========================================
# 16. SAVE RECOMMENDATIONS
# ==========================================

recommendations.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nRecommendations Saved Successfully!")
print("File:", OUTPUT_PATH)


# ==========================================
# 17. FINAL SUMMARY
# ==========================================

print("\n========== FINAL SUMMARY ==========")

print("Selected Track:", selected_track["track_name"])
print("Selected Genre:", selected_genre)
print("Selected Cluster:", selected_cluster)
print("Genre Filtering:", genre_choice)
print("Recommendations Generated:", len(recommendations))

print("\n========== SYSTEM COMPLETED ==========")
