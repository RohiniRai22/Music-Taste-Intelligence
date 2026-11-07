
import pandas as pd
import joblib
from pathlib import Path
from sklearn.cluster import KMeans

BASE_DIR = Path(__file__).resolve().parent.parent

# Load PCA features
X = pd.read_csv(
    BASE_DIR / "data" / "spotify_features_pca.csv"
)

# Train final K-Means model
kmeans = KMeans(
    n_clusters=7,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(X)

# Save cluster assignments
df = pd.read_csv(
    BASE_DIR / "data" / "spotify_tracks_cleaned.csv"
)

df["cluster"] = labels

df.to_csv(
    BASE_DIR / "data" / "spotify_tracks_clustered.csv",
    index=False
)

# Save model
joblib.dump(
    kmeans,
    BASE_DIR / "models" / "kmeans_model.pkl"
)

print("\n========== FINAL K-MEANS MODEL ==========")

print("\nCluster Distribution:")
print(df["cluster"].value_counts().sort_index())

print("\nClustered Dataset Shape:", df.shape)

print("\nModel saved successfully!")
print("Cluster assignments saved successfully!")
