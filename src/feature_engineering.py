
from pathlib import Path

import pandas as pd


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ucl_matches_with_team_ids.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ucl_matches_with_features.csv"
)


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(INPUT_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    by=["date"]
).reset_index(drop=True)

print("FEATURE ENGINEERING PREPARATION")
print("--------------------------------")
print(f"Total matches: {len(df)}")
print(f"First match date: {df['date'].min()}")
print(f"Last match date: {df['date'].max()}")

print(
    f"\nNumber of teams: "
    f"{pd.concat([df['home_team_id'], df['away_team_id']]).nunique()}"
)


# ==========================================
# TEAM HISTORIES
# ==========================================

# All previous UCL matches for each team.
team_history = {}

# Previous home matches for each team.
home_history = {}

# Previous away matches for each team.
away_history = {}


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_points(goals_for, goals_against):

    if goals_for > goals_against:
        return 3

    elif goals_for == goals_against:
        return 1

    return 0


def get_last_five_points(history):

    recent_matches = history[-5:]

    return sum(
        match["points"]
        for match in recent_matches
    )


def get_last_five_goal_difference(history):

    recent_matches = history[-5:]

    return sum(
        match["goal_difference"]
        for match in recent_matches
    )


def get_last_five_goals_scored(history):

    recent_matches = history[-5:]

    return sum(
        match["goals_scored"]
        for match in recent_matches
    )


def get_last_five_goals_conceded(history):

    recent_matches = history[-5:]

    return sum(
        match["goals_conceded"]
        for match in recent_matches
    )


# ==========================================
# FEATURE ENGINEERING
# ==========================================

engineered_rows = []

# Process each date separately.
# Results from the current date are not available
# until every match on that date has been processed.

for match_date, daily_matches in df.groupby("date", sort=True):

    daily_updates = []

    # --------------------------------------
    # CALCULATE FEATURES BEFORE MATCHES
    # --------------------------------------

    for index, match in daily_matches.iterrows():

        home_id = match["home_team_id"]
        away_id = match["away_team_id"]

        # Retrieve previous histories.
        home_team_matches = team_history.get(home_id, [])
        away_team_matches = team_history.get(away_id, [])

        home_matches = home_history.get(home_id, [])
        away_matches = away_history.get(away_id, [])

        # ----------------------------------
        # GENERAL FORM FEATURES
        # ----------------------------------

        home_last_5_points = get_last_five_points(
            home_team_matches
        )

        away_last_5_points = get_last_five_points(
            away_team_matches
        )

        home_last_5_gd = get_last_five_goal_difference(
            home_team_matches
        )

        away_last_5_gd = get_last_five_goal_difference(
            away_team_matches
        )

        home_last_5_goals_scored = get_last_five_goals_scored(
            home_team_matches
        )

        home_last_5_goals_conceded = get_last_five_goals_conceded(
            home_team_matches
        )

        away_last_5_goals_scored = get_last_five_goals_scored(
            away_team_matches
        )

        away_last_5_goals_conceded = get_last_five_goals_conceded(
            away_team_matches
        )

        # ----------------------------------
        # VENUE-SPECIFIC FORM FEATURES
        # ----------------------------------

        home_last_5_home_points = get_last_five_points(
            home_matches
        )

        away_last_5_away_points = get_last_five_points(
            away_matches
        )

        home_last_5_home_gd = get_last_five_goal_difference(
            home_matches
        )

        away_last_5_away_gd = get_last_five_goal_difference(
            away_matches
        )

        # ----------------------------------
        # PREVIOUS VENUE APPEARANCES
        # ----------------------------------

        home_previous_home_matches = len(home_matches)

        away_previous_away_matches = len(away_matches)

        # ----------------------------------
        # STORE FEATURES
        # ----------------------------------

        features = {
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
        }

        engineered_rows.append({
            "index": index,
            **features
        })

        # ----------------------------------
        # PREPARE HISTORY UPDATES
        # ----------------------------------

        home_goals = int(match["home_goals"])
        away_goals = int(match["away_goals"])

        home_points = get_points(
            home_goals,
            away_goals
        )

        away_points = get_points(
            away_goals,
            home_goals
        )

        home_goal_difference = home_goals - away_goals
        away_goal_difference = away_goals - home_goals

        daily_updates.append({
            "home_id": home_id,
            "away_id": away_id,
            "home_record": {
                "points": home_points,
                "goal_difference": home_goal_difference,
                "goals_scored": home_goals,
                "goals_conceded": away_goals
            },
            "away_record": {
                "points": away_points,
                "goal_difference": away_goal_difference,
                "goals_scored": away_goals,
                "goals_conceded": home_goals
            }
        })

    # --------------------------------------
    # UPDATE HISTORIES AFTER THE ENTIRE DATE
    # --------------------------------------

    for update in daily_updates:

        home_id = update["home_id"]
        away_id = update["away_id"]

        home_record = update["home_record"]
        away_record = update["away_record"]

        # Initialize histories if necessary.
        team_history.setdefault(home_id, [])
        team_history.setdefault(away_id, [])

        home_history.setdefault(home_id, [])
        away_history.setdefault(away_id, [])

        # Update general team histories.
        team_history[home_id].append(home_record)
        team_history[away_id].append(away_record)

        # Update venue-specific histories.
        home_history[home_id].append(home_record)
        away_history[away_id].append(away_record)


# ==========================================
# MERGE FEATURES INTO DATASET
# ==========================================

features_df = pd.DataFrame(engineered_rows)

features_df = features_df.set_index("index")

df = df.join(features_df)

# Restore the original chronological order.
df = df.sort_values("date").reset_index(drop=True)


# ==========================================
# FORM DIFFERENCE
# ==========================================

df["last_5_points_difference"] = (
    df["home_last_5_points"]
    - df["away_last_5_points"]
)


# ==========================================
# FEATURE VALIDATION
# ==========================================

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
    "away_previous_away_matches",
    "last_5_points_difference"
]

print("\nROLLING FORM FEATURES")
print("---------------------")

print(f"Total matches: {len(df)}")
print(f"Total features: {len(feature_columns)}")

print("\nFirst 10 matches:")
print(
    df[
        [
            "date",
            "home_team",
            "away_team",
            "home_last_5_points",
            "away_last_5_points",
            "last_5_points_difference"
        ]
    ].head(10).to_string(index=False)
)

print("\nFeature statistics:")
print(df[feature_columns].describe().to_string())

print("\nMissing feature values:")
print(df[feature_columns].isnull().sum().to_string())


# ==========================================
# FIRST MATCHDAY VALIDATION
# ==========================================

first_date = df["date"].min()

first_day = df[df["date"] == first_date]

print("\nFIRST MATCHDAY VALIDATION")
print("-------------------------")

first_day_features = first_day[feature_columns]

if (first_day_features == 0).all().all():
    print("All first-matchday feature values are zero.")
else:
    print("WARNING: Unexpected non-zero first-matchday values.")


# ==========================================
# SAVE ENGINEERED DATASET
# ==========================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nFEATURE ENGINEERING COMPLETED!")
print(f"Updated dataset saved to: {OUTPUT_PATH}")
