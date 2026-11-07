
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.cluster.hierarchy import linkage, dendrogram

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

PCA_PATH = BASE_DIR / "data" / "spotify_features_pca.csv"
OUTPUT_PATH = BASE_DIR / "data" / "hierarchical_dendrogram.png"

# ==========================================
# 2. LOAD PCA DATA
# ==========================================

print("\n========== HIERARCHICAL DENDROGRAM ANALYSIS ==========")

df = pd.read_csv(PCA_PATH)

print("Original PCA Dataset Shape:", df.shape)

# ==========================================
# 3. SELECT SAMPLE
# ==========================================

sample_df = df.sample(
    n=500,
    random_state=42
)

print("Sample Dataset Shape:", sample_df.shape)

# ==========================================
# 4. GENERATE LINKAGE MATRIX
# ==========================================

print("\nGenerating hierarchical linkage matrix...")

linked = linkage(
    sample_df,
    method="ward",
    metric="euclidean"
)

print("Linkage matrix generated successfully!")

# ==========================================
# 5. CREATE DENDROGRAM
# ==========================================

plt.figure(figsize=(16, 8))

dendrogram(
    linked,
    truncate_mode="lastp",
    p=30,
    show_leaf_counts=True,
    leaf_rotation=90,
    leaf_font_size=9
)

plt.title("Hierarchical Clustering Dendrogram (500 Track Sample)")
plt.xlabel("Cluster Groups")
plt.ylabel("Ward Distance")

plt.tight_layout()

# ==========================================
# 6. SAVE VISUALIZATION
# ==========================================

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nDendrogram generated successfully!")
print("Saved Location:", OUTPUT_PATH)

print("\nHIERARCHICAL DENDROGRAM ANALYSIS COMPLETED!")
