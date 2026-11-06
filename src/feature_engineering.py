
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
# ELO SETTINGS
# ==========================================

INITIAL_ELO = 1500
K_FACTOR = 20
HOME_ADVANTAGE = 100

# Current Elo rating for every team.
elo_ratings = {}


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


def calculate_expected_score(home_elo, away_elo):

    effective_home_elo = home_elo + HOME_ADVANTAGE

    expected_home = 1 / (
        1 + 10 ** (
            (away_elo - effective_home_elo) / 400
        )
    )

    return expected_home


# ==========================================
# FEATURE ENGINEERING
# ==========================================

engineered_rows = []

# Process each date separately.
# All matches on the same date use ratings and
# histories from before that date.

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
        # ELO FEATURES BEFORE THE MATCH
        # ----------------------------------

        home_elo = elo_ratings.get(
            home_id,
            INITIAL_ELO
        )

        away_elo = elo_ratings.get(
            away_id,
            INITIAL_ELO
        )

        elo_difference = home_elo - away_elo

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
            "away_previous_away_matches": away_previous_away_matches,

            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_difference": elo_difference
        }

        engineered_rows.append({
            "index": index,
            **features
        })

        # ----------------------------------
        # PREPARE HISTORY AND ELO UPDATES
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

        # Actual result for the home team:
        # 1 = win, 0.5 = draw, 0 = defeat.

        if home_goals > away_goals:
            home_score = 1

        elif home_goals == away_goals:
            home_score = 0.5

        else:
            home_score = 0

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
            },

            # Save pre-match ratings.
            "home_elo": home_elo,
            "away_elo": away_elo,

            "home_score": home_score
        })

    # --------------------------------------
    # UPDATE HISTORIES AFTER THE ENTIRE DATE
    # --------------------------------------

    # Accumulate Elo changes before applying them.
    # This prevents matches on the same date from
    # influencing each other's pre-match ratings.

    elo_daily_deltas = {}

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

        # ----------------------------------
        # ELO RATING UPDATE
        # ----------------------------------

        expected_home = calculate_expected_score(
            update["home_elo"],
            update["away_elo"]
        )

        actual_home = update["home_score"]

        rating_change = K_FACTOR * (
            actual_home - expected_home
        )

        elo_daily_deltas[home_id] = (
            elo_daily_deltas.get(home_id, 0)
            + rating_change
        )

        elo_daily_deltas[away_id] = (
            elo_daily_deltas.get(away_id, 0)
            - rating_change
        )

    # Apply all Elo changes after processing the date.
    for team_id, delta in elo_daily_deltas.items():

        current_elo = elo_ratings.get(
            team_id,
            INITIAL_ELO
        )

        elo_ratings[team_id] = current_elo + delta


# ==========================================
# MERGE FEATURES INTO DATASET
# ==========================================

features_df = pd.DataFrame(engineered_rows)

features_df = features_df.set_index("index")

df = df.join(features_df)

# Restore chronological order.
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

    "last_5_points_difference",

    # New Elo features
    "home_elo",
    "away_elo",
    "elo_difference"
]


print("\nFEATURE ENGINEERING RESULTS")
print("---------------------------")

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
            "home_elo",
            "away_elo",
            "elo_difference"
        ]
    ].head(10).to_string(index=False)
)


# ==========================================
# FEATURE STATISTICS
# ==========================================

print("\nFeature statistics:")

print(
    df[feature_columns].describe().to_string()
)


# ==========================================
# MISSING VALUES VALIDATION
# ==========================================

print("\nMissing feature values:")

print(
    df[feature_columns].isnull().sum().to_string()
)


# ==========================================
# FIRST MATCHDAY VALIDATION
# ==========================================

first_date = df["date"].min()

first_day = df[df["date"] == first_date]

print("\nFIRST MATCHDAY VALIDATION")
print("-------------------------")

# Rolling-form features should start at zero.
rolling_columns = [
    col for col in feature_columns
    if col not in [
        "home_elo",
        "away_elo",
        "elo_difference"
    ]
]

if (first_day[rolling_columns] == 0).all().all():
    print("Rolling features: OK")
else:
    print("WARNING: Unexpected rolling feature values.")

# All teams start with an Elo rating of 1500.
if (
    (first_day["home_elo"] == INITIAL_ELO).all()
    and
    (first_day["away_elo"] == INITIAL_ELO).all()
):
    print("Initial Elo ratings: OK")
else:
    print("WARNING: Initial Elo ratings are incorrect.")

# Check that Elo difference is calculated correctly.
if (
    df["elo_difference"]
    == df["home_elo"] - df["away_elo"]
).all():
    print("Elo difference calculation: OK")
else:
    print("WARNING: Elo difference calculation is incorrect.")

# Check for missing values.
if df[feature_columns].isnull().sum().sum() == 0:
    print("Missing values: None")
else:
    print("WARNING: Missing feature values detected.")


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

print(
    f"Updated dataset saved to: {OUTPUT_PATH}"
)
