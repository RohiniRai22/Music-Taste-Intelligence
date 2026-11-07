
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

BASE_DIR = Path(__file__).resolve().parent.parent

# Load PCA dataset
X = pd.read_csv(
    BASE_DIR / "data" / "spotify_features_pca.csv"
)

print("\n========== K-MEANS CLUSTERING ==========")

# Test different cluster counts
k_values = [3, 4, 5, 6, 7, 8]

results = []

for k in k_values:

    print(f"\nTesting K = {k}")

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X)

    # Calculate silhouette score using a sample
    score = silhouette_score(
        X,
        labels,
        sample_size=5000,
        random_state=42
    )

    results.append({
        "K": k,
        "Silhouette Score": score
    })

# Display results
results_df = pd.DataFrame(results)

print("\n========== K-MEANS RESULTS ==========")
print(results_df)

# Select best K based on silhouette score
best_result = results_df.loc[
    results_df["Silhouette Score"].idxmax()
]

print("\nBest Number of Clusters:", int(best_result["K"]))
print("Best Silhouette Score:", round(best_result["Silhouette Score"], 4))

print("\nK-MEANS EXPERIMENT COMPLETED!")
