
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import joblib

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

print("\n========== HIERARCHICAL CLUSTERING ==========")

# 1. Load PCA dataset
X = pd.read_csv(
    BASE_DIR / "data" / "spotify_features_pca.csv"
)

print("Original PCA Dataset Shape:", X.shape)

# 2. Create a sample
X_sample = X.sample(
    n=10000,
    random_state=42
).reset_index(drop=True)

print("Sample Dataset Shape:", X_sample.shape)

# 3. Apply Agglomerative Clustering
print("\n========== AGGLOMERATIVE CLUSTERING ==========")

hierarchical = AgglomerativeClustering(
    n_clusters=7,
    metric="euclidean",
    linkage="ward"
)

labels = hierarchical.fit_predict(X_sample)

# 4. Count clusters
n_clusters = len(np.unique(labels))

print("Number of Clusters:", n_clusters)

# 5. Cluster distribution
print("\n========== CLUSTER DISTRIBUTION ==========")

cluster_counts = pd.Series(labels).value_counts().sort_index()

print(cluster_counts)

# 6. Calculate silhouette score
print("\n========== SILHOUETTE ANALYSIS ==========")

score = silhouette_score(
    X_sample,
    labels,
    sample_size=2000,
    random_state=42
)

print("Silhouette Score:", round(score, 4))

# 7. Save cluster assignments
clustered_df = X_sample.copy()
clustered_df["cluster"] = labels

output_path = (
    BASE_DIR / "data" / "hierarchical_clustered_sample.csv"
)

clustered_df.to_csv(
    output_path,
    index=False
)

print("\nCluster assignments saved successfully!")
print("File Location:", output_path)

# 8. Save model
model_path = (
    BASE_DIR / "models" / "hierarchical_model.pkl"
)

joblib.dump(hierarchical, model_path)

print("Hierarchical model saved successfully!")
print("Model Location:", model_path)

print("\nHIERARCHICAL CLUSTERING COMPLETED!")
