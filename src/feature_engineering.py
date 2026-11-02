
from pathlib import Path
import pandas as pd


# ==========================================
# 1. LOAD AND PREPARE THE DATASET
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
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

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    by=["date", "season"]
).reset_index(drop=True)

print("FEATURE ENGINEERING PREPARATION")
print("--------------------------------")
print(f"Total matches: {len(df)}")
print(f"First match date: {df['date'].min()}")
print(f"Last match date: {df['date'].max()}")


# ==========================================
# 2. INITIALIZE TEAM HISTORIES
# ==========================================

# Each history entry contains:
# points, goal difference, goals scored, goals conceded

team_history = {}
home_history = {}
away_history = {}

feature_records = []


# ==========================================
# 3. DEFINE FEATURE CALCULATION
# ==========================================

def calculate_features(history):
    """
    Calculate statistics using the team's
    previous five matches.
    """

    last_five = history[-5:]

    points = sum(match[0] for match in last_five)
    goal_difference = sum(match[1] for match in last_five)
    goals_scored = sum(match[2] for match in last_five)
    goals_conceded = sum(match[3] for match in last_five)

    return points, goal_difference, goals_scored, goals_conceded


# ==========================================
# 4. PROCESS MATCHES CHRONOLOGICALLY
# ==========================================

# Important:
# All matches on the same date receive their
# features BEFORE any results from that date
# are added to team histories.

for match_date, day_matches in df.groupby("date", sort=True):

    # --------------------------------------
    # A. Calculate features for the whole day
    # --------------------------------------

    for _, match in day_matches.iterrows():

        home_team = match["home_team_id"]
        away_team = match["away_team_id"]

        # Get previous team histories
        home_team_matches = team_history.get(home_team, [])
        away_team_matches = team_history.get(away_team, [])

        # Get previous home/away appearances
        previous_home_matches = home_history.get(home_team, [])
        previous_away_matches = away_history.get(away_team, [])

        # General last-five form
        home_form = calculate_features(home_team_matches)
        away_form = calculate_features(away_team_matches)

        # Venue-specific last-five form
        home_venue_form = calculate_features(previous_home_matches)
        away_venue_form = calculate_features(previous_away_matches)

        # Store the 14 features
        features = {
            "home_last_5_points": home_form[0],
            "away_last_5_points": away_form[0],

            "home_last_5_gd": home_form[1],
            "away_last_5_gd": away_form[1],

            "home_last_5_goals_scored": home_form[2],
            "home_last_5_goals_conceded": home_form[3],

            "away_last_5_goals_scored": away_form[2],
            "away_last_5_goals_conceded": away_form[3],

            "home_last_5_home_points": home_venue_form[0],
            "away_last_5_away_points": away_venue_form[0],

            "home_last_5_home_gd": home_venue_form[1],
            "away_last_5_away_gd": away_venue_form[1],

            "home_previous_home_matches": len(previous_home_matches),
            "away_previous_away_matches": len(previous_away_matches)
        }

        feature_records.append(features)

    # --------------------------------------
    # B. Update histories AFTER the whole day
    # --------------------------------------

    for _, match in day_matches.iterrows():

        home_team = match["home_team_id"]
        away_team = match["away_team_id"]

        home_goals = int(match["home_goals"])
        away_goals = int(match["away_goals"])

        outcome = match["outcome"]

        # Determine points earned by each team
        if outcome == "home_win":
            home_points = 3
            away_points = 0

        elif outcome == "away_win":
            home_points = 0
            away_points = 3

        elif outcome == "draw":
            home_points = 1
            away_points = 1

        else:
            raise ValueError(f"Unexpected outcome: {outcome}")

        # Calculate goal difference from each team's perspective
        home_gd = home_goals - away_goals
        away_gd = away_goals - home_goals

        # Create history entries
        home_entry = (
            home_points,
            home_gd,
            home_goals,
            away_goals
        )

        away_entry = (
            away_points,
            away_gd,
            away_goals,
            home_goals
        )

        # Initialize histories for new teams
        team_history.setdefault(home_team, [])
        team_history.setdefault(away_team, [])

        home_history.setdefault(home_team, [])
        away_history.setdefault(away_team, [])

        # Update general team histories
        team_history[home_team].append(home_entry)
        team_history[away_team].append(away_entry)

        # Update venue-specific histories
        home_history[home_team].append(home_entry)
        away_history[away_team].append(away_entry)


# ==========================================
# 5. ADD FEATURES TO THE DATASET
# ==========================================

feature_df = pd.DataFrame(feature_records)

df = pd.concat(
    [df.reset_index(drop=True), feature_df],
    axis=1
)

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


# ==========================================
# 6. VALIDATE AND SAVE THE DATASET
# ==========================================

print("\nROLLING FORM FEATURES")
print("---------------------")

print(f"Total matches: {len(df)}")
print(f"Total features: {len(feature_columns)}")

print("\nFirst 10 matches:")
print(df[["date", "home_team", "away_team"] + feature_columns].head(10))

print("\nFeature statistics:")
print(df[feature_columns].describe())

print("\nMissing feature values:")
print(df[feature_columns].isnull().sum())

# First matchday should have zero previous form
first_matchday = df[df["date"] == df["date"].min()]

print("\nFIRST MATCHDAY VALIDATION")
print("------------------------")

print(
    "All feature values are zero:",
    (first_matchday[feature_columns] == 0).all().all()
)

# Save the complete engineered dataset
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nDataset saved to: {OUTPUT_PATH}")


# ==========================================
# 7. TRAIN / VALIDATION / TEST SPLIT
# ==========================================

train_seasons = [
    "2011-2012",
    "2012-2013",
    "2013-2014",
    "2014-2015",
    "2015-2016",
    "2016-2017",
    "2017-2018",
    "2018-2019",
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023"
]

validation_seasons = ["2023-2024"]

test_seasons = ["2024-2025", "2025-2026"]

train_df = df[df["season"].isin(train_seasons)].copy()

validation_df = df[
    df["season"].isin(validation_seasons)
].copy()

test_df = df[df["season"].isin(test_seasons)].copy()

print("\nDATASET SPLIT")
print("-------------")

print(f"Training matches: {len(train_df)}")
print(f"Validation matches: {len(validation_df)}")
print(f"Testing matches: {len(test_df)}")

total_split = (
    len(train_df)
    + len(validation_df)
    + len(test_df)
)

print(f"Total matches: {total_split}")


# ==========================================
# 8. SEPARATE FEATURES (X) AND TARGET (y)
# ==========================================

target_column = "outcome"

X_train = train_df[feature_columns]
y_train = train_df[target_column]

X_validation = validation_df[feature_columns]
y_validation = validation_df[target_column]

X_test = test_df[feature_columns]
y_test = test_df[target_column]


# ==========================================
# 9. CHECK THE FINAL DATA
# ==========================================

print("\nFEATURES AND TARGET")
print("-------------------")

print("Training:", X_train.shape, y_train.shape)
print("Validation:", X_validation.shape, y_validation.shape)
print("Testing:", X_test.shape, y_test.shape)

print("\nTraining outcome distribution:")
print(y_train.value_counts())

print("\nValidation outcome distribution:")
print(y_validation.value_counts())

print("\nTesting outcome distribution:")
print(y_test.value_counts())

print("\nFEATURE ENGINEERING COMPLETED!")
