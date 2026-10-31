
from pathlib import Path
import pandas as pd


# 1. Define the seasons
seasons = ["2023_2024", "2024_2025", "2025_2026"]

# 2. Create an empty list to store our DataFrames
dataframes = []

# 3. Read each season's CSV file
for season in seasons:
    file_path = Path(
        f"data/processed/ucl_matches_{season}.csv"
    )

    df = pd.read_csv(file_path)

    dataframes.append(df)

# 4. Combine all DataFrames into one
all_matches = pd.concat(dataframes, ignore_index=True)

# 5. Save the combined dataset
output_path = Path("data/processed/ucl_matches_all.csv")
all_matches.to_csv(output_path, index=False)

print("All seasons combined successfully!")
print("Dataset shape:", all_matches.shape)
print("Saved to:", output_path)

# 5. Inspect the combined dataset

print("\n--- DATASET OVERVIEW ---")

print("\nDataset shape:")
print(all_matches.shape)

print("\nMatches per season:")
print(all_matches["season"].value_counts().sort_index())

print("\nOutcome distribution:")
print(all_matches["outcome"].value_counts())

print("\nMissing values:")
print(all_matches.isnull().sum())

print("\nFirst five matches:")
print(all_matches.head())

# Some matches may have identical recorded information but represent different fixtures.
print("\nDuplicate rows:")
print(all_matches.duplicated().sum())