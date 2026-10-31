
from pathlib import Path
import pandas as pd

# 1. Load and prepare the original dataset
file_path = Path("data/processed/ucl_matches_all.csv")

df = pd.read_csv(file_path)

df["match_date"] = pd.to_datetime(df["match_date"])

df = df.sort_values("match_date").reset_index(drop=True)

df["match_id"] = range(len(df))


# 2. Create home-team match history
home_matches = df[
    [
        "match_id",
        "match_date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals"
    ]
].copy()

home_matches.columns = [
    "match_id",
    "match_date",
    "team",
    "opponent",
    "goals_scored",
    "goals_conceded"
]

home_matches["venue"] = "home"


# 3. Create away-team match history
away_matches = df[
    [
        "match_id",
        "match_date",
        "away_team",
        "home_team",
        "away_goals",
        "home_goals"
    ]
].copy()

away_matches.columns = [
    "match_id",
    "match_date",
    "team",
    "opponent",
    "goals_scored",
    "goals_conceded"
]

away_matches["venue"] = "away"


# 4. Combine both histories
team_history = pd.concat(
    [home_matches, away_matches],
    ignore_index=True
)

team_history = team_history.sort_values(
    ["team", "match_date", "match_id"]
).reset_index(drop=True)


# 5. Check for multiple matches by the same team on one date
duplicate_team_dates = team_history.duplicated(
    subset=["team", "match_date"]
).sum()

print(f"Duplicate team-date records: {duplicate_team_dates}")

if duplicate_team_dates > 0:
    raise ValueError("A team has multiple matches on the same date.")


# 6. Calculate each team's match result
team_history["result"] = "draw"

team_history.loc[
    team_history["goals_scored"] > team_history["goals_conceded"],
    "result"
] = "win"

team_history.loc[
    team_history["goals_scored"] < team_history["goals_conceded"],
    "result"
] = "loss"


# 7. Calculate points and goal difference
team_history["points"] = 1

team_history.loc[
    team_history["result"] == "win",
    "points"
] = 3

team_history.loc[
    team_history["result"] == "loss",
    "points"
] = 0

team_history["goal_difference"] = (
    team_history["goals_scored"]
    - team_history["goals_conceded"]
)


# 8. Calculate general form over the previous five matches

# Average goals scored
team_history["avg_goals_scored_last5"] = (
    team_history.groupby("team")["goals_scored"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=5,
            min_periods=5
        ).mean()
    )
)

# Average goals conceded
team_history["avg_goals_conceded_last5"] = (
    team_history.groupby("team")["goals_conceded"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=5,
            min_periods=5
        ).mean()
    )
)

# Win rate
team_history["is_win"] = (
    team_history["result"] == "win"
).astype(float)

team_history["win_rate_last5"] = (
    team_history.groupby("team")["is_win"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=5,
            min_periods=5
        ).mean()
    )
)

# Average points per game
team_history["points_per_game_last5"] = (
    team_history.groupby("team")["points"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=5,
            min_periods=5
        ).mean()
    )
)

# Average goal difference
team_history["goal_difference_last5"] = (
    team_history.groupby("team")["goal_difference"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=5,
            min_periods=5
        ).mean()
    )
)


# 9. Calculate venue-specific form
# A minimum of three previous matches at that venue is required.

team_history["venue_points_per_game_last5"] = (
    team_history.groupby(["team", "venue"])["points"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=5,
            min_periods=3
        ).mean()
    )
)


# 10. Select all features
feature_columns = [
    "avg_goals_scored_last5",
    "avg_goals_conceded_last5",
    "win_rate_last5",
    "points_per_game_last5",
    "goal_difference_last5",
    "venue_points_per_game_last5"
]


# 11. Prepare home-team features
home_form = team_history[
    ["match_id", "team", "venue"] + feature_columns
].copy()

home_form = home_form[
    home_form["venue"] == "home"
].drop(columns="venue")

home_form = home_form.rename(
    columns={
        "team": "home_team",
        "avg_goals_scored_last5": "home_avg_goals_scored",
        "avg_goals_conceded_last5": "home_avg_goals_conceded",
        "win_rate_last5": "home_win_rate",
        "points_per_game_last5": "home_points_per_game",
        "goal_difference_last5": "home_goal_difference",
        "venue_points_per_game_last5": "home_venue_points_per_game"
    }
)


# 12. Prepare away-team features
away_form = team_history[
    ["match_id", "team", "venue"] + feature_columns
].copy()

away_form = away_form[
    away_form["venue"] == "away"
].drop(columns="venue")

away_form = away_form.rename(
    columns={
        "team": "away_team",
        "avg_goals_scored_last5": "away_avg_goals_scored",
        "avg_goals_conceded_last5": "away_avg_goals_conceded",
        "win_rate_last5": "away_win_rate",
        "points_per_game_last5": "away_points_per_game",
        "goal_difference_last5": "away_goal_difference",
        "venue_points_per_game_last5": "away_venue_points_per_game"
    }
)


# 13. Merge features into the original match dataset
match_features = df.merge(
    home_form,
    on=["match_id", "home_team"],
    how="left",
    validate="one_to_one"
)

match_features = match_features.merge(
    away_form,
    on=["match_id", "away_team"],
    how="left",
    validate="one_to_one"
)


# 14. Define the final model features
model_features = [
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


# 15. Remove matches with incomplete historical features
match_features_clean = match_features.dropna(
    subset=model_features
).copy()


# 16. Display dataset information
print("\n--- FEATURE ENGINEERING V2 ---")

print("\nOriginal dataset shape:")
print(df.shape)

print("\nDataset after feature engineering:")
print(match_features.shape)

print("\nClean dataset shape:")
print(match_features_clean.shape)

print("\nMatches removed:")
print(len(match_features) - len(match_features_clean))

print("\nMissing values in model features:")
print(match_features_clean[model_features].isnull().sum())

print("\nOutcome distribution:")
print(match_features_clean["outcome"].value_counts())

print("\nFirst five matches:")
print(
    match_features_clean[
        ["match_date", "home_team", "away_team", "outcome"]
        + model_features
    ].head().to_string(index=False)
)


# 17. Save the enhanced dataset
output_path = Path("data/processed/ucl_matches_features.csv")

output_path.parent.mkdir(parents=True, exist_ok=True)

match_features_clean.to_csv(output_path, index=False)

print("\nEnhanced dataset saved successfully!")
print(f"Saved to: {output_path}")
