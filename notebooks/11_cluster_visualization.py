
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. PROJECT DIRECTORY
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

# ==========================================
# 2. LOAD DATASETS
# ==========================================

X = pd.read_csv(
    BASE_DIR / "data" / "spotify_features_pca.csv"
)

df = pd.read_csv(
    BASE_DIR / "data" / "spotify_tracks_clustered.csv"
)

print("\n========== K-MEANS CLUSTER VISUALIZATION ==========")

print("\nPCA Dataset Shape:", X.shape)
print("Clustered Dataset Shape:", df.shape)

# ==========================================
# 3. CHECK DATA
# ==========================================

if len(X) != len(df):
    raise ValueError("PCA data and cluster labels have different row counts!")

# ==========================================
# 4. SELECT SAMPLE DATA
# ==========================================

sample_size = min(10000, len(X))

sample_indices = X.sample(
    n=sample_size,
    random_state=42
).index

X_sample = X.loc[sample_indices]
cluster_labels = df.loc[sample_indices, "cluster"]

print("\nSample Size:", sample_size)

# ==========================================
# 5. CREATE CLUSTER VISUALIZATION
# ==========================================

plt.figure(figsize=(12, 8))

scatter = plt.scatter(
    X_sample["PC1"],
    X_sample["PC2"],
    c=cluster_labels,
    cmap="viridis",
    alpha=0.5,
    s=12
)

plt.title(
    "Music Track Clusters - K-Means",
    fontsize=16
)

plt.xlabel("Principal Component 1", fontsize=12)
plt.ylabel("Principal Component 2", fontsize=12)

plt.colorbar(
    scatter,
    label="Cluster"
)

plt.grid(alpha=0.2)

plt.tight_layout()

# ==========================================
# 6. SAVE VISUALIZATION
# ==========================================

output_path = BASE_DIR / "data" / "kmeans_clusters.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

# Close instead of displaying the graph window
plt.close()

# ==========================================
# 7. COMPLETION
# ==========================================

print("\nVisualization saved successfully!")
print("File Location:", output_path)

print("\nK-MEANS VISUALIZATION COMPLETED!")
