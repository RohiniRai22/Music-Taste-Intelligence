
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.metrics import silhouette_score

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

PCA_PATH = BASE_DIR / "data" / "spotify_features_pca.csv"

KMEANS_PATH = BASE_DIR / "data" / "spotify_tracks_clustered.csv"
DBSCAN_PATH = BASE_DIR / "data" / "dbscan_clustered_sample.csv"
HIERARCHICAL_PATH = BASE_DIR / "data" / "hierarchical_clustered_sample.csv"

OUTPUT_PATH = BASE_DIR / "data" / "final_algorithm_comparison.csv"

# ==========================================
# 2. LOAD DATASETS
# ==========================================

print("\n========== FINAL ALGORITHM COMPARISON ==========")

pca_df = pd.read_csv(PCA_PATH)

# Recreate the same 10,000-row sample used in
# DBSCAN and hierarchical clustering.
sample_df = pca_df.sample(
    n=10000,
    random_state=42
).reset_index(drop=True)

kmeans_df = pd.read_csv(KMEANS_PATH)
dbscan_df = pd.read_csv(DBSCAN_PATH)
hierarchical_df = pd.read_csv(HIERARCHICAL_PATH)

print("Comparison Sample Shape:", sample_df.shape)

# ==========================================
# 3. ALIGN K-MEANS LABELS
# ==========================================

# K-Means was fitted on the complete PCA dataset.
# Select labels for the same sampled original rows.

sample_indices = pca_df.sample(
    n=10000,
    random_state=42
).index

kmeans_labels = kmeans_df.iloc[sample_indices]["cluster"].to_numpy()

dbscan_labels = dbscan_df["cluster"].to_numpy()

hierarchical_labels = hierarchical_df["cluster"].to_numpy()

# ==========================================
# 4. DEFINE EVALUATION FUNCTION
# ==========================================

def evaluate_clustering(name, X, labels, exclude_noise=False):

    labels = np.asarray(labels)

    if exclude_noise:
        valid_mask = labels != -1
        evaluation_X = X[valid_mask]
        evaluation_labels = labels[valid_mask]
    else:
        evaluation_X = X
        evaluation_labels = labels

    unique_clusters = np.unique(evaluation_labels)

    n_clusters = len(unique_clusters)

    noise_count = np.sum(labels == -1)

    noise_percentage = (noise_count / len(labels)) * 100

    if n_clusters >= 2 and len(evaluation_labels) > n_clusters:

        score = silhouette_score(
            evaluation_X,
            evaluation_labels,
            sample_size=min(2000, len(evaluation_labels)),
            random_state=42
        )

    else:
        score = np.nan

    print(f"\n{name}")
    print("-" * 40)
    print("Number of Clusters:", n_clusters)
    print("Noise Points:", noise_count)
    print("Noise Percentage:", round(noise_percentage, 2), "%")
    print(
        "Silhouette Score:",
        round(score, 4) if not np.isnan(score) else "N/A"
    )

    return {
        "Algorithm": name,
        "Clusters": n_clusters,
        "Noise Points": noise_count,
        "Noise Percentage": round(noise_percentage, 2),
        "Silhouette Score": round(score, 4) if not np.isnan(score) else np.nan
    }

# ==========================================
# 5. EVALUATE ALL THREE ALGORITHMS
# ==========================================

results = []

results.append(
    evaluate_clustering(
        "K-Means",
        sample_df,
        kmeans_labels
    )
)

results.append(
    evaluate_clustering(
        "DBSCAN",
        sample_df,
        dbscan_labels,
        exclude_noise=True
    )
)

results.append(
    evaluate_clustering(
        "Hierarchical",
        sample_df,
        hierarchical_labels
    )
)

# ==========================================
# 6. CLUSTER DISTRIBUTION
# ==========================================

print("\n========== CLUSTER DISTRIBUTION ==========")

for name, labels in [
    ("K-Means", kmeans_labels),
    ("DBSCAN", dbscan_labels),
    ("Hierarchical", hierarchical_labels)
]:

    print(f"\n{name}:")

    print(
        pd.Series(labels)
        .value_counts()
        .sort_index()
    )

# ==========================================
# 7. FINAL COMPARISON TABLE
# ==========================================

comparison_df = pd.DataFrame(results)

print("\n========== FINAL COMPARISON TABLE ==========")

print(comparison_df.to_string(index=False))

# ==========================================
# 8. SAVE RESULTS
# ==========================================

comparison_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nComparison results saved successfully!")
print("Saved Location:", OUTPUT_PATH)

#print("\nFINAL ALGORITHM COMPARISON COMPLETED!")
