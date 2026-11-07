
import pandas as pd
import numpy as np

# Load dataset
#df = pd.read_csv("../data/spotify_tracks.csv")
df = pd.read_csv(
    r"D:\ALML_PROJECTS\Music_Taste_Intelligence\data\spotify_tracks.csv"
)

# 1. Dataset overview
print("\n========== DATASET OVERVIEW ==========")
print("Shape:", df.shape)
print("\nFirst 5 Records:")
print(df.head())

# 2. Column information
print("\n========== DATA TYPES ==========")
print(df.info())

# 3. Missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# 4. Duplicate records
print("\n========== DUPLICATES ==========")
print("Duplicate rows:", df.duplicated().sum())

# 5. Statistical summary
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

# 6. Genre distribution
print("\n========== GENRE DISTRIBUTION ==========")
print(df["track_genre"].value_counts().head(10))

print("\nDATA EXPLORATION COMPLETED!")
