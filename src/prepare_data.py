
from pathlib import Path
import pandas as pd

# 1. Load the enhanced dataset
data_dir = Path("data/processed")

df = pd.read_csv(data_dir / "ucl_matches_features.csv")

df["match_date"] = pd.to_datetime(df["match_date"])

df = df.sort_values("match_date").reset_index(drop=True)


# 2. Define our 12 model features
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
    "away_venue_points_per_game"
]


# 3. Separate features and target
X = df[feature_columns]
y = df["outcome"]


# 4. Split chronologically using unique dates
unique_dates = sorted(df["match_date"].unique())

split_index = int(len(unique_dates) * 0.80)

split_date = unique_dates[split_index]

train_df = df[df["match_date"] < split_date].copy()

test_df = df[df["match_date"] >= split_date].copy()


# 5. Display dataset information
print("\n--- UPDATED TRAIN/TEST SPLIT ---")

print(f"\nTotal matches: {len(df)}")

print(f"Number of features: {len(feature_columns)}")

print(f"\nSplit date: {split_date}")

print(f"\nTraining matches: {len(train_df)}")
print(f"Testing matches: {len(test_df)}")

print("\nTraining date range:")
print(
    train_df["match_date"].min(),
    "to",
    train_df["match_date"].max()
)

print("\nTesting date range:")
print(
    test_df["match_date"].min(),
    "to",
    test_df["match_date"].max()
)

print("\nTraining outcome distribution:")
print(train_df["outcome"].value_counts())

print("\nTesting outcome distribution:")
print(test_df["outcome"].value_counts())


# 6. Verify that no dates overlap
overlapping_dates = set(train_df["match_date"]) & set(
    test_df["match_date"]
)

print(f"\nOverlapping dates: {len(overlapping_dates)}")

if len(overlapping_dates) > 0:
    raise ValueError("Training and testing dates overlap!")


# 7. Save the new datasets
train_df.to_csv(data_dir / "ucl_train.csv", index=False)

test_df.to_csv(data_dir / "ucl_test.csv", index=False)

print("\nUpdated training and testing datasets saved successfully!")
