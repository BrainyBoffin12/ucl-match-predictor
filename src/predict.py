
from pathlib import Path

import pandas as pd
import joblib
import re


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ucl_matches_with_team_ids.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ucl_model.joblib"
)

DRAW_THRESHOLD = 0.225


# ==========================================
# 2. LOAD MODEL AND HISTORICAL DATA
# ==========================================

print("Loading prediction system...")

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

model = joblib.load(MODEL_PATH)

print("Prediction system ready!")


# ==========================================
# 3. FEATURE LIST
# ==========================================

FEATURES = [
    "home_last_5_points",
    "away_last_5_points",
    "last_5_points_difference",
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
# 4. TEAM NAME MANAGEMENT
# ==========================================

def normalize_name(name):
    name = str(name).lower()
    return re.sub(r"[^a-z0-9]", "", name)


def find_team_id(team_name):

    teams = {}

    for _, match in df.iterrows():

        home_id = match["home_team_id"]
        away_id = match["away_team_id"]

        teams.setdefault(home_id, set()).update([
            str(home_id),
            str(match["home_team"])
        ])

        teams.setdefault(away_id, set()).update([
            str(away_id),
            str(match["away_team"])
        ])

    search_name = normalize_name(team_name)

    # First, look for an exact match.
    exact_matches = []

    for team_id, aliases in teams.items():

        for alias in aliases:

            if normalize_name(alias) == search_name:
                exact_matches.append(team_id)

    if len(set(exact_matches)) == 1:
        return exact_matches[0]

    # Then, look for a unique partial match.
    partial_matches = []

    for team_id, aliases in teams.items():

        for alias in aliases:

            normalized_alias = normalize_name(alias)

            if (
                search_name in normalized_alias
                or normalized_alias in search_name
            ):
                partial_matches.append(team_id)
                break

    partial_matches = list(set(partial_matches))

    if len(partial_matches) == 1:
        return partial_matches[0]

    if len(partial_matches) > 1:
        raise ValueError(
            f"Ambiguous team name: {team_name}. "
            "Please use a more specific name."
        )

    raise ValueError(
        f"Team not found: {team_name}"
    )


# ==========================================
# 5. CALCULATE TEAM HISTORIES
# ==========================================

team_history = {}
home_history = {}
away_history = {}

for match_date, daily_matches in df.groupby("date", sort=True):

    daily_updates = []

    # First, prepare all match records for this date.

    for _, match in daily_matches.iterrows():

        home_id = match["home_team_id"]
        away_id = match["away_team_id"]

        home_goals = int(match["home_goals"])
        away_goals = int(match["away_goals"])

        if home_goals > away_goals:
            home_points, away_points = 3, 0
        elif home_goals == away_goals:
            home_points, away_points = 1, 1
        else:
            home_points, away_points = 0, 3

        home_record = {
            "points": home_points,
            "goal_difference": home_goals - away_goals,
            "goals_scored": home_goals,
            "goals_conceded": away_goals
        }

        away_record = {
            "points": away_points,
            "goal_difference": away_goals - home_goals,
            "goals_scored": away_goals,
            "goals_conceded": home_goals
        }

        daily_updates.append(
            (home_id, away_id, home_record, away_record)
        )

    # Update histories only after processing the entire date.

    for home_id, away_id, home_record, away_record in daily_updates:

        team_history.setdefault(home_id, []).append(home_record)
        team_history.setdefault(away_id, []).append(away_record)

        home_history.setdefault(home_id, []).append(home_record)
        away_history.setdefault(away_id, []).append(away_record)




# ==========================================
# 6. HELPER FUNCTIONS
# ==========================================

def get_last_five(history, column):

    recent_matches = history[-5:]

    return sum(
        match[column]
        for match in recent_matches
    )


# ==========================================
# 7. PREDICTION FUNCTION
# ==========================================

def predict_match(home_team, away_team):

    # Find the teams' canonical IDs.

    home_id = find_team_id(home_team)
    away_id = find_team_id(away_team)

    if home_id == away_id:
        raise ValueError("A team cannot play against itself.")

    # Retrieve historical records.

    home_matches = team_history.get(home_id, [])
    away_matches = team_history.get(away_id, [])

    previous_home_matches = home_history.get(home_id, [])
    previous_away_matches = away_history.get(away_id, [])

    # Calculate the 15 features.

    features = {

        "home_last_5_points":
            get_last_five(home_matches, "points"),

        "away_last_5_points":
            get_last_five(away_matches, "points"),

        "last_5_points_difference":
            get_last_five(home_matches, "points")
            - get_last_five(away_matches, "points"),

        "home_last_5_gd":
            get_last_five(home_matches, "goal_difference"),

        "away_last_5_gd":
            get_last_five(away_matches, "goal_difference"),

        "home_last_5_goals_scored":
            get_last_five(home_matches, "goals_scored"),

        "home_last_5_goals_conceded":
            get_last_five(home_matches, "goals_conceded"),

        "away_last_5_goals_scored":
            get_last_five(away_matches, "goals_scored"),

        "away_last_5_goals_conceded":
            get_last_five(away_matches, "goals_conceded"),

        "home_last_5_home_points":
            get_last_five(previous_home_matches, "points"),

        "away_last_5_away_points":
            get_last_five(previous_away_matches, "points"),

        "home_last_5_home_gd":
            get_last_five(previous_home_matches, "goal_difference"),

        "away_last_5_away_gd":
            get_last_five(previous_away_matches, "goal_difference"),

        "home_previous_home_matches":
            len(previous_home_matches),

        "away_previous_away_matches":
            len(previous_away_matches)
    }

    # Convert features into a DataFrame.

    X = pd.DataFrame([features], columns=FEATURES)

    # Get prediction probabilities.

    probabilities = model.predict_proba(X)[0]

    probability_dict = dict(
        zip(model.classes_, probabilities)
    )

    home_probability = probability_dict["home_win"]
    draw_probability = probability_dict["draw"]
    away_probability = probability_dict["away_win"]

    # Apply the draw threshold.

    if draw_probability >= DRAW_THRESHOLD:
        prediction = "draw"

    elif home_probability > away_probability:
        prediction = "home_win"

    else:
        prediction = "away_win"

    # Display results.

    print("\n====================================")
    print("          MATCH PREDICTION")
    print("====================================")

    print(f"Home team: {home_team}")
    print(f"Away team: {away_team}")

    print("\nProbabilities:")

    print(f"Home win: {home_probability:.2%}")
    print(f"Draw:     {draw_probability:.2%}")
    print(f"Away win: {away_probability:.2%}")

    print("\nPredicted outcome:")

    if prediction == "home_win":
        print(f"{home_team} wins!")

    elif prediction == "draw":
        print("Draw!")

    else:
        print(f"{away_team} wins!")

    print("====================================")

    return {
        "prediction": prediction,
        "home_win_probability": home_probability,
        "draw_probability": draw_probability,
        "away_win_probability": away_probability
    }


# ==========================================
# 8. FIRST TEST
# ==========================================

if __name__ == "__main__":

    predict_match(
        "SSC Napoli",
        "Arsenal FC"
    )
