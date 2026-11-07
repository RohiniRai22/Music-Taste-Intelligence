
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

print("\n========== K-MEANS VS DBSCAN COMPARISON ==========")

# 1. Load PCA dataset
X = pd.read_csv(
    BASE_DIR / "data" / "spotify_features_pca.csv"
)

# Use the same sample for both algorithms
X_sample = X.sample(
    n=10000,
    random_state=42
)

print("\nOriginal Dataset Shape:", X.shape)
print("Sample Dataset Shape:", X_sample.shape)

# 2. Initialize results
results = []

# 3. K-Means clustering
print("\n========== K-MEANS ==========")

kmeans = KMeans(
    n_clusters=7,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X_sample)

kmeans_clusters = len(set(kmeans_labels))

kmeans_silhouette = silhouette_score(
    X_sample,
    kmeans_labels,
    sample_size=2000,
    random_state=42
)

print("Number of Clusters:", kmeans_clusters)
print("Noise Points: 0")
print("Noise Percentage: 0%")
print("Silhouette Score:", round(kmeans_silhouette, 4))

results.append({
    "algorithm": "K-Means",
    "clusters": kmeans_clusters,
    "noise_points": 0,
    "noise_percentage": 0,
    "silhouette_score": round(kmeans_silhouette, 4)
})

# 4. DBSCAN clustering
print("\n========== DBSCAN ==========")

dbscan = DBSCAN(
    eps=1.5,
    min_samples=10,
    metric="euclidean",
    n_jobs=-1
)

dbscan_labels = dbscan.fit_predict(X_sample)

dbscan_clusters = len(set(dbscan_labels) - {-1})

noise_points = np.sum(dbscan_labels == -1)

noise_percentage = (
    noise_points / len(dbscan_labels)
) * 100

# Exclude noise points for silhouette calculation
mask = dbscan_labels != -1

valid_labels = dbscan_labels[mask]
valid_data = X_sample.iloc[np.where(mask)[0]]

if len(np.unique(valid_labels)) >= 2:

    dbscan_silhouette = silhouette_score(
        valid_data,
        valid_labels,
        sample_size=min(2000, len(valid_labels)),
        random_state=42
    )

else:
    dbscan_silhouette = np.nan

print("Number of Clusters:", dbscan_clusters)
print("Noise Points:", noise_points)
print("Noise Percentage:", round(noise_percentage, 2), "%")

if not np.isnan(dbscan_silhouette):
    print(
        "Silhouette Score:",
        round(dbscan_silhouette, 4)
    )
else:
    print("Silhouette Score: N/A")

results.append({
    "algorithm": "DBSCAN",
    "clusters": dbscan_clusters,
    "noise_points": noise_points,
    "noise_percentage": round(noise_percentage, 2),
    "silhouette_score": (
        round(dbscan_silhouette, 4)
        if not np.isnan(dbscan_silhouette)
        else np.nan
    )
})

# 5. Display comparison
print("\n========== FINAL COMPARISON ==========")

comparison_df = pd.DataFrame(results)

print(comparison_df.to_string(index=False))

# 6. Save comparison
output_path = (
    BASE_DIR / "data" / "clustering_model_comparison.csv"
)

comparison_df.to_csv(
    output_path,
    index=False
)

print("\nComparison results saved successfully!")
print("File Location:", output_path)

# 7. Display cluster distributions
print("\n========== K-MEANS CLUSTER DISTRIBUTION ==========")

print(
    pd.Series(kmeans_labels)
    .value_counts()
    .sort_index()
)

print("\n========== DBSCAN CLUSTER DISTRIBUTION ==========")

print(
    pd.Series(dbscan_labels)
    .value_counts()
    .sort_index()
)

#print("\nMODEL COMPARISON COMPLETED!")
