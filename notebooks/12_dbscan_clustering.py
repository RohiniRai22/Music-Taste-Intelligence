
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# ==========================================
# 1. PROJECT DIRECTORY
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

# ==========================================
# 2. LOAD PCA DATASET
# ==========================================

X = pd.read_csv(
    BASE_DIR / "data" / "spotify_features_pca.csv"
)

df = pd.read_csv(
    BASE_DIR / "data" / "spotify_tracks_cleaned.csv"
)

print("\n========== DBSCAN CLUSTERING ==========")

print("\nOriginal PCA Dataset Shape:", X.shape)

# ==========================================
# 3. SELECT SAMPLE
# ==========================================

sample_size = min(10000, len(X))

sample_indices = X.sample(
    n=sample_size,
    random_state=42
).index

X_sample = X.loc[sample_indices]

print("\nSample Dataset Shape:", X_sample.shape)

# ==========================================
# 4. APPLY DBSCAN
# ==========================================

# Initial experimental parameters
dbscan = DBSCAN(
    eps=1.5,
    min_samples=10,
    metric="euclidean",
    n_jobs=-1
)

labels = dbscan.fit_predict(X_sample)

# ==========================================
# 5. COUNT CLUSTERS AND NOISE
# ==========================================

df_sample = df.loc[sample_indices].copy()

df_sample["cluster"] = labels

n_clusters = len(set(labels) - {-1})

n_noise = np.sum(labels == -1)

print("\n========== DBSCAN RESULTS ==========")

print("\nNumber of Clusters:", n_clusters)

print("Number of Noise Points:", n_noise)

print(
    "Noise Percentage:",
    round((n_noise / len(labels)) * 100, 2),
    "%"
)

print("\nCluster Distribution:")
print(
    df_sample["cluster"]
    .value_counts()
    .sort_index()
)

# ==========================================
# 6. SILHOUETTE SCORE
# ==========================================

# Exclude noise points when calculating silhouette
non_noise_mask = labels != -1

valid_labels = labels[non_noise_mask]
valid_data = X_sample.iloc[np.where(non_noise_mask)[0]]

unique_clusters = np.unique(valid_labels)

print("\n========== SILHOUETTE ANALYSIS ==========")

if len(unique_clusters) >= 2 and len(valid_labels) > 2:

    score_sample_size = min(2000, len(valid_labels))

    silhouette = silhouette_score(
        valid_data,
        valid_labels,
        sample_size=score_sample_size,
        random_state=42
    )

    print("Silhouette Score:", round(silhouette, 4))

else:
    print("Silhouette Score cannot be calculated.")
    print("DBSCAN did not identify at least two non-noise clusters.")

# ==========================================
# 7. SAVE CLUSTER ASSIGNMENTS
# ==========================================

output_path = (
    BASE_DIR / "data" / "dbscan_clustered_sample.csv"
)

df_sample.to_csv(
    output_path,
    index=False
)

print("\nCluster assignments saved successfully!")

# ==========================================
# 8. SAVE DBSCAN MODEL
# ==========================================

model_path = BASE_DIR / "models" / "dbscan_model.pkl"

joblib.dump(dbscan, model_path)

print("DBSCAN model saved successfully!")

print("\nDBSCAN CLUSTERING COMPLETED!")
