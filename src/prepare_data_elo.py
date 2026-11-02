import pandas as pd
from pathlib import Path

# Load dataset with Elo features
df = pd.read_csv(
    "data/processed/ucl_matches_features_elo.csv"
)

df["match_date"] = pd.to_datetime(df["match_date"])

# Define the 15 model features
feature_columns = [
    "home_avg_goals_scored",
    "home_avg_goals_conceded",
    "home_win_rate",
    "away_avg_goals_scored",
    "away_avg_goals_conceded",
    "away_win_rate",
    "home_points_per_game",
    "away_points_per_game",
    "home_goal_difference",
    "away_goal_difference",
    "home_venue_points_per_game",
    "away_venue_points_per_game",
    "home_elo",
    "away_elo",
    "elo_difference"
]

# Keep the features, target and date
df = df[
    ["match_date", "outcome"] + feature_columns
].copy()

# Sort chronologically
df = df.sort_values("match_date").reset_index(drop=True)

# Split by unique dates to avoid separating matches on the same date
unique_dates = sorted(df["match_date"].unique())

split_index = int(len(unique_dates) * 0.80)
split_date = unique_dates[split_index]

train_df = df[df["match_date"] < split_date].copy()
test_df = df[df["match_date"] >= split_date].copy()

# Display information
print("\n--- ELO DATA PREPARATION ---")

print("\nTotal matches:", len(df))
print("Number of features:", len(feature_columns))
print("Split date:", split_date)

print("\nTraining matches:", len(train_df))
print("Testing matches:", len(test_df))

print("\nTraining period:")
print(train_df["match_date"].min(), "to", train_df["match_date"].max())

print("\nTesting period:")
print(test_df["match_date"].min(), "to", test_df["match_date"].max())

print("\nTraining outcomes:")
print(train_df["outcome"].value_counts())

print("\nTesting outcomes:")
print(test_df["outcome"].value_counts())

# Verify no date overlap
train_dates = set(train_df["match_date"])
test_dates = set(test_df["match_date"])

print("\nOverlapping dates:", len(train_dates.intersection(test_dates)))

# Save datasets
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

train_df.to_csv(output_dir / "ucl_train_elo.csv", index=False)
test_df.to_csv(output_dir / "ucl_test_elo.csv", index=False)

print("\nElo training and testing datasets saved successfully!")