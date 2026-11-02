from pathlib import Path
import pandas as pd

# Configuration
INITIAL_RATING = 1500
K_FACTOR = 20

# Load the original dataset
matches = pd.read_csv("data/processed/ucl_matches_all.csv")
matches["match_date"] = pd.to_datetime(matches["match_date"])

# Sort matches chronologically
matches = matches.sort_values("match_date").reset_index(drop=True)

# Every team starts with the same rating
ratings = {}

elo_records = []

for _, match in matches.iterrows():

    home_team = match["home_team"]
    away_team = match["away_team"]

    # Give new teams an initial rating
    if home_team not in ratings:
        ratings[home_team] = INITIAL_RATING

    if away_team not in ratings:
        ratings[away_team] = INITIAL_RATING

    # Read ratings BEFORE the match
    home_elo = ratings[home_team]
    away_elo = ratings[away_team]

    # Calculate expected results
    expected_home = 1 / (
        1 + 10 ** ((away_elo - home_elo) / 400)
    )

    expected_away = 1 - expected_home

    # Determine actual results
    if match["outcome"] == "home_win":
        actual_home = 1
        actual_away = 0

    elif match["outcome"] == "away_win":
        actual_home = 0
        actual_away = 1

    else:
        actual_home = 0.5
        actual_away = 0.5

    # Save pre-match Elo features
    elo_records.append({
        "match_date": match["match_date"],
        "home_team": home_team,
        "away_team": away_team,
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_difference": home_elo - away_elo
    })

    # Update ratings AFTER the match
    ratings[home_team] += K_FACTOR * (
        actual_home - expected_home
    )

    ratings[away_team] += K_FACTOR * (
        actual_away - expected_away
    )


# Convert Elo records into a DataFrame
elo_df = pd.DataFrame(elo_records)

# Load our existing V2 feature dataset
features = pd.read_csv(
    "data/processed/ucl_matches_features.csv"
)

features["match_date"] = pd.to_datetime(
    features["match_date"]
)

# Merge Elo features into the existing dataset
merged = features.merge(
    elo_df,
    on=["match_date", "home_team", "away_team"],
    how="left",
    validate="one_to_one"
)

# Verify the result
elo_columns = [
    "home_elo",
    "away_elo",
    "elo_difference"
]

print("\n--- ELO FEATURE ENGINEERING ---")

print("\nDataset shape:")
print(merged.shape)

print("\nMissing Elo values:")
print(merged[elo_columns].isnull().sum())

print("\nFirst five Elo records:")
print(
    merged[
        [
            "match_date",
            "home_team",
            "away_team",
            "home_elo",
            "away_elo",
            "elo_difference"
        ]
    ].head()
)

# Save the new dataset
output_path = Path(
    "data/processed/ucl_matches_features_elo.csv"
)

merged.to_csv(output_path, index=False)

print("\nElo features created successfully!")
print(f"Saved to: {output_path}")