
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

BASE_DIR = Path(__file__).resolve().parent.parent

# Load scaled features
X_scaled = pd.read_csv(
    BASE_DIR / "data" / "spotify_features_scaled.csv"
)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance
explained_variance = pca.explained_variance_ratio_

print("\n========== PCA ANALYSIS ==========")

print("\nExplained Variance Ratio:")
print(explained_variance.round(4))

print("\nCumulative Explained Variance:")
print(explained_variance.cumsum().round(4))

# Find components needed for 95% variance
pca_95 = PCA(n_components=0.95)
X_pca_95 = pca_95.fit_transform(X_scaled)

print("\nOriginal Number of Features:", X_scaled.shape[1])
print("Components Retaining 95% Variance:", X_pca_95.shape[1])

# Save PCA-transformed data
pca_df = pd.DataFrame(
    X_pca_95,
    columns=[f"PC{i+1}" for i in range(X_pca_95.shape[1])]
)

pca_df.to_csv(
    BASE_DIR / "data" / "spotify_features_pca.csv",
    index=False
)

print("\nPCA Dataset Shape:", pca_df.shape)
print("PCA analysis completed!")
