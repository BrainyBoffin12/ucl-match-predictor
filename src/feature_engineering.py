
from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ucl_matches_with_team_ids.csv"
)

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    by=["date", "season"]
).reset_index(drop=True)


# --------------------------------------------------
# 2. Dataset overview
# --------------------------------------------------

print("\nFEATURE ENGINEERING PREPARATION")
print("--------------------------------")

print(f"Total matches: {len(df)}")
print(f"First match date: {df['date'].min()}")
print(f"Last match date: {df['date'].max()}")


# --------------------------------------------------
# 3. Calculate team participation
# --------------------------------------------------

home_counts = df["home_team_id"].value_counts()
away_counts = df["away_team_id"].value_counts()

team_counts = (
    home_counts.add(away_counts, fill_value=0)
    .sort_values(ascending=False)
)

print(f"\nNumber of teams: {len(team_counts)}")

print("\nTop 10 most frequent teams:")
print(team_counts.head(10))


# --------------------------------------------------
# 4. Initialize team histories
# --------------------------------------------------

team_history = {}
home_history = {}
away_history = {}

match_features = []


# --------------------------------------------------
# 5. Calculate pre-match rolling features
# --------------------------------------------------

for match_date, daily_matches in df.groupby("date", sort=True):

    daily_features = []

    # Calculate features using only previous matchdays
    for _, row in daily_matches.iterrows():

        home_team = row["home_team_id"]
        away_team = row["away_team_id"]

        # Overall team history
        home_previous = team_history.get(home_team, [])
        away_previous = team_history.get(away_team, [])

        home_last_5 = home_previous[-5:]
        away_last_5 = away_previous[-5:]

        # Overall points
        home_last_5_points = sum(
            match["points"] for match in home_last_5
        )

        away_last_5_points = sum(
            match["points"] for match in away_last_5
        )

        # Overall goal difference
        home_last_5_gd = sum(
            match["goal_difference"] for match in home_last_5
        )

        away_last_5_gd = sum(
            match["goal_difference"] for match in away_last_5
        )

        # Overall goals scored and conceded
        home_last_5_goals_scored = sum(
            match["goals_scored"] for match in home_last_5
        )

        home_last_5_goals_conceded = sum(
            match["goals_conceded"] for match in home_last_5
        )

        away_last_5_goals_scored = sum(
            match["goals_scored"] for match in away_last_5
        )

        away_last_5_goals_conceded = sum(
            match["goals_conceded"] for match in away_last_5
        )

        # Venue-specific histories
        home_previous_home = home_history.get(home_team, [])
        away_previous_away = away_history.get(away_team, [])

        home_last_5_home = home_previous_home[-5:]
        away_last_5_away = away_previous_away[-5:]

        # Points in previous home/away matches
        home_last_5_home_points = sum(
            match["points"] for match in home_last_5_home
        )

        away_last_5_away_points = sum(
            match["points"] for match in away_last_5_away
        )

        # Goal difference in previous home/away matches
        home_last_5_home_gd = sum(
            match["goal_difference"] for match in home_last_5_home
        )

        away_last_5_away_gd = sum(
            match["goal_difference"] for match in away_last_5_away
        )

        # Number of previous matches at each venue
        home_previous_home_matches = len(home_previous_home)
        away_previous_away_matches = len(away_previous_away)

        # Store all features for the current match
        daily_features.append({

            "home_last_5_points": home_last_5_points,
            "away_last_5_points": away_last_5_points,

            "home_last_5_gd": home_last_5_gd,
            "away_last_5_gd": away_last_5_gd,

            "home_last_5_goals_scored": home_last_5_goals_scored,
            "home_last_5_goals_conceded": home_last_5_goals_conceded,

            "away_last_5_goals_scored": away_last_5_goals_scored,
            "away_last_5_goals_conceded": away_last_5_goals_conceded,

            "home_last_5_home_points": home_last_5_home_points,
            "away_last_5_away_points": away_last_5_away_points,

            "home_last_5_home_gd": home_last_5_home_gd,
            "away_last_5_away_gd": away_last_5_away_gd,

            "home_previous_home_matches": home_previous_home_matches,
            "away_previous_away_matches": away_previous_away_matches

        })

    # Save today's features before updating histories
    match_features.extend(daily_features)

    # Update histories only after all matches on this date
    # have had their features calculated

    for _, row in daily_matches.iterrows():

        home_team = row["home_team_id"]
        away_team = row["away_team_id"]

        home_goals = row["home_goals"]
        away_goals = row["away_goals"]

        # Calculate match points
        if home_goals > away_goals:
            home_points, away_points = 3, 0

        elif home_goals < away_goals:
            home_points, away_points = 0, 3

        else:
            home_points, away_points = 1, 1

        # Calculate goal differences
        home_gd = home_goals - away_goals
        away_gd = away_goals - home_goals

        # Store home team's match information
        home_match = {
            "points": home_points,
            "goal_difference": home_gd,
            "goals_scored": home_goals,
            "goals_conceded": away_goals
        }

        # Store away team's match information
        away_match = {
            "points": away_points,
            "goal_difference": away_gd,
            "goals_scored": away_goals,
            "goals_conceded": home_goals
        }

        # Update overall histories
        team_history.setdefault(home_team, []).append(home_match)
        team_history.setdefault(away_team, []).append(away_match)

        # Update venue-specific histories
        home_history.setdefault(home_team, []).append(home_match)
        away_history.setdefault(away_team, []).append(away_match)


# --------------------------------------------------
# 6. Add features to the dataset
# --------------------------------------------------

features_df = pd.DataFrame(match_features)

df = pd.concat(
    [df.reset_index(drop=True), features_df],
    axis=1
)


# --------------------------------------------------
# 7. Validate generated features
# --------------------------------------------------

feature_columns = [
    "home_last_5_points",
    "away_last_5_points",

    "home_last_5_gd",
    "away_last_5_gd",

    "home_last_5_goals_scored",
    "home_last_5_goals_conceded",

    "away_last_5_goals_scored",
    "away_last_5_goals_conceded",

    "home_last_5_home_points",
    "away_last_5_away_points",

    "home_last_5_home_gd",
    "away_last_5_away_gd",

    "home_previous_home_matches",
    "away_previous_away_matches"
]

print("\nROLLING FORM FEATURES")
print("---------------------")

print(f"Total matches: {len(df)}")
print(f"Total features: {len(feature_columns)}")

print("\nFirst 10 matches:")

print(
    df[
        ["date", "home_team_id", "away_team_id"]
        + feature_columns
    ]
    .head(10)
    .to_string(index=False)
)

print("\nFeature statistics:")

print(df[feature_columns].describe())

print("\nMissing feature values:")

print(df[feature_columns].isna().sum())

print("\nFirst matchday validation:")

first_matchday = df[df["date"] == df["date"].min()]

print(
    first_matchday[
        [
            "home_last_5_points",
            "away_last_5_points",
            "home_last_5_home_points",
            "away_last_5_away_points",
            "home_previous_home_matches",
            "away_previous_away_matches"
        ]
    ].to_string(index=False)
)

print("\nFEATURE ENGINEERING COMPLETED!")

# --------------------------------------------------
# 8. Save feature-engineered dataset
# --------------------------------------------------

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ucl_matches_with_features.csv"
)

df.to_csv(OUTPUT_PATH, index=False)

print(f"\nFeature-engineered dataset saved to: {OUTPUT_PATH}")