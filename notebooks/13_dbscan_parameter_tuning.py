
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

BASE_DIR = Path(__file__).resolve().parent.parent

# Load PCA features
X = pd.read_csv(
    BASE_DIR / "data" / "spotify_features_pca.csv"
)

# Use the same 10,000-track sample
X_sample = X.sample(
    n=10000,
    random_state=42
)

print("\n========== DBSCAN PARAMETER TUNING ==========")

# Experiment with different parameters
eps_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
min_samples_values = [5, 10, 15, 20]

results = []

for eps in eps_values:

    for min_samples in min_samples_values:

        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric="euclidean",
            n_jobs=-1
        )

        labels = dbscan.fit_predict(X_sample)

        n_clusters = len(set(labels) - {-1})
        n_noise = np.sum(labels == -1)

        noise_percentage = (
            n_noise / len(labels)
        ) * 100

        # Calculate silhouette excluding noise
        mask = labels != -1
        valid_labels = labels[mask]
        valid_data = X_sample.iloc[np.where(mask)[0]]

        if len(np.unique(valid_labels)) >= 2:

            score = silhouette_score(
                valid_data,
                valid_labels,
                sample_size=min(2000, len(valid_labels)),
                random_state=42
            )

        else:
            score = np.nan

        results.append({
            "eps": eps,
            "min_samples": min_samples,
            "clusters": n_clusters,
            "noise_points": n_noise,
            "noise_percentage": round(noise_percentage, 2),
            "silhouette_score": (
                round(score, 4)
                if not np.isnan(score)
                else None
            )
        })

        print(
            f"eps={eps}, min_samples={min_samples}, "
            f"clusters={n_clusters}, "
            f"noise={n_noise}, "
            f"silhouette={score:.4f}"
            if not np.isnan(score)
            else
            f"eps={eps}, min_samples={min_samples}, "
            f"clusters={n_clusters}, "
            f"noise={n_noise}, silhouette=N/A"
        )

# Create results table
results_df = pd.DataFrame(results)

print("\n========== TUNING RESULTS ==========")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv(
    BASE_DIR / "data" / "dbscan_tuning_results.csv",
    index=False
)

print("\nTuning results saved successfully!")
