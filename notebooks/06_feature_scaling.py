
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent

# Load cleaned dataset
df = pd.read_csv(BASE_DIR / "data" / "spotify_tracks_cleaned.csv")

# Select audio features
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

X = df[features]

# Initialize StandardScaler
scaler = StandardScaler()

# Scale features
X_scaled = scaler.fit_transform(X)

# Convert scaled data into DataFrame
X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=features
)

print("\n========== FEATURE SCALING ==========")

print("\nScaled Dataset Shape:")
print(X_scaled_df.shape)

print("\nScaled Feature Means:")
print(X_scaled_df.mean().round(3))

print("\nScaled Feature Standard Deviations:")
print(X_scaled_df.std().round(3))

# Save scaled features
X_scaled_df.to_csv(
    BASE_DIR / "data" / "spotify_features_scaled.csv",
    index=False
)

# Save scaler for future use
(BASE_DIR / "models").mkdir(exist_ok=True)

joblib.dump(
    scaler,
    BASE_DIR / "models" / "standard_scaler.pkl"
)

print("\nScaled dataset saved successfully!")
print("Scaler saved successfully!")
