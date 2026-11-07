
import pandas as pd
import joblib
import numpy as np

from pathlib import Path
from sklearn.decomposition import PCA

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "spotify_tracks_cleaned.csv"
SCALED_PATH = BASE_DIR / "data" / "spotify_features_scaled.csv"

SCALER_PATH = BASE_DIR / "models" / "standard_scaler.pkl"
KMEANS_PATH = BASE_DIR / "models" / "kmeans_model.pkl"

PCA_MODEL_PATH = BASE_DIR / "models" / "pca_model.pkl"

# ==========================================
# 2. SELECT AUDIO FEATURES
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

print("\n========== FINAL MODEL PREPARATION ==========")

# ==========================================
# 3. LOAD DATA AND TRAINED MODELS
# ==========================================

df = pd.read_csv(DATA_PATH)

scaler = joblib.load(SCALER_PATH)
kmeans = joblib.load(KMEANS_PATH)

print("Cleaned Dataset Shape:", df.shape)
print("Scaler loaded successfully!")
print("K-Means model loaded successfully!")

# ==========================================
# 4. PREPARE AUDIO FEATURES
# ==========================================

X = df[features]

print("\nOriginal Feature Shape:", X.shape)

# ==========================================
# 5. APPLY SAVED SCALER
# ==========================================

X_scaled = scaler.transform(X)

print("Scaled Feature Shape:", X_scaled.shape)

# ==========================================
# 6. RECREATE PCA TRANSFORMATION
# ==========================================

pca = PCA(n_components=8)

X_pca = pca.fit_transform(X_scaled)

print("PCA Feature Shape:", X_pca.shape)

print(
    "Total Explained Variance:",
    round(sum(pca.explained_variance_ratio_) * 100, 2),
    "%"
)

# ==========================================
# 7. VERIFY PCA AGAINST EXISTING DATA
# ==========================================

existing_pca = pd.read_csv(
    BASE_DIR / "data" / "spotify_features_pca.csv"
).to_numpy()

if existing_pca.shape != X_pca.shape:
    raise ValueError(
        "Recreated PCA shape does not match existing PCA dataset!"
    )

if not np.allclose(existing_pca, X_pca, atol=1e-5):
    raise ValueError(
        "Recreated PCA values do not match the original PCA dataset. "
        "Check the PCA settings and preprocessing used in script 07."
    )

print("PCA verification successful!")

# ==========================================
# 8. VERIFY K-MEANS MODEL
# ==========================================

if X_pca.shape[1] != kmeans.n_features_in_:
    raise ValueError("PCA feature count does not match K-Means model!")

print("K-Means input feature verification successful!")

# ==========================================
# 9. SAVE PCA MODEL
# ==========================================

joblib.dump(pca, PCA_MODEL_PATH)

print("\nPCA model saved successfully!")
print("Saved Location:", PCA_MODEL_PATH)

# ==========================================
# 10. TEST CLUSTER ASSIGNMENT
# ==========================================

sample_tracks = X_pca[:5]

predictions = kmeans.predict(sample_tracks)

print("\n========== SAMPLE TRACK CLUSTER PREDICTIONS ==========")

for i, cluster in enumerate(predictions):

    print(f"Track {i + 1} assigned to Cluster {cluster}")

# ==========================================
# 11. FINAL STATUS
# ==========================================

print("\n========== MODEL STATUS ==========")

print("Scaler: Ready")
print("PCA Model: Ready")
print("K-Means Model: Ready")
print("Number of Clusters:", kmeans.n_clusters)

print("\nFINAL MODEL PREPARATION COMPLETED!")
