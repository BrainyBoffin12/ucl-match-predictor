
from pathlib import Path
import pandas as pd


# ============================================================
# 1. LOAD AND PREPARE THE DATASET
# ============================================================

file_path = Path("data/processed/ucl_matches_all.csv")
df = pd.read_csv(file_path)

df["match_date"] = pd.to_datetime(df["match_date"])

# Sort matches chronologically
df = df.sort_values("match_date").reset_index(drop=True)

# Create a unique ID for every match
df["match_id"] = range(len(df))

print("\n--- ORIGINAL DATASET ---")
print("Dataset shape:", df.shape)


# ============================================================
# 2. CREATE HOME-TEAM MATCH HISTORY
# ============================================================

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

home_matches["result"] = "draw"

home_matches.loc[
    home_matches["goals_scored"] > home_matches["goals_conceded"],
    "result"
] = "win"

home_matches.loc[
    home_matches["goals_scored"] < home_matches["goals_conceded"],
    "result"
] = "loss"


# ============================================================
# 3. CREATE AWAY-TEAM MATCH HISTORY
# ============================================================

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

away_matches["result"] = "draw"

away_matches.loc[
    away_matches["goals_scored"] > away_matches["goals_conceded"],
    "result"
] = "win"

away_matches.loc[
    away_matches["goals_scored"] < away_matches["goals_conceded"],
    "result"
] = "loss"


# ============================================================
# 4. COMBINE BOTH TEAM HISTORIES
# ============================================================

team_history = pd.concat(
    [home_matches, away_matches],
    ignore_index=True
)

team_history = team_history.sort_values(
    ["team", "match_date", "match_id"]
).reset_index(drop=True)

print("\n--- TEAM MATCH HISTORY ---")
print("Total team records:", len(team_history))
print("Unique teams:", team_history["team"].nunique())


# Check that no team has multiple matches on the same date
duplicate_team_dates = team_history.duplicated(
    subset=["team", "match_date"]
).sum()

print("\nDuplicate team-date records:", duplicate_team_dates)

if duplicate_team_dates > 0:
    raise ValueError(
        "A team has multiple matches on the same date. "
        "Check the match data before calculating recent form."
    )


# ============================================================
# 5. CALCULATE RECENT FORM USING PREVIOUS 5 MATCHES
# ============================================================

# Create a numerical win indicator:
# win = 1, draw/loss = 0
team_history["is_win"] = (
    team_history["result"] == "win"
).astype(float)


# Average goals scored in the previous 5 matches
team_history["avg_goals_scored_last5"] = (
    team_history.groupby("team")["goals_scored"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=5,
            min_periods=5
        ).mean()
    )
)


# Average goals conceded in the previous 5 matches
team_history["avg_goals_conceded_last5"] = (
    team_history.groupby("team")["goals_conceded"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=5,
            min_periods=5
        ).mean()
    )
)


# Win rate in the previous 5 matches
# shift(1) excludes the current match
# NaN values remain NaN until 5 previous matches exist
team_history["win_rate_last5"] = (
    team_history.groupby("team")["is_win"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=5,
            min_periods=5
        ).mean()
    )
)


# ============================================================
# 6. INSPECT ARSENAL'S RECENT FORM
# ============================================================

arsenal_history = team_history[
    team_history["team"] == "Arsenal FC"
]

print("\n--- ARSENAL RECENT FORM ---")

arsenal_columns = [
    "match_date",
    "opponent",
    "goals_scored",
    "goals_conceded",
    "result",
    "avg_goals_scored_last5",
    "avg_goals_conceded_last5",
    "win_rate_last5"
]

print(
    arsenal_history[arsenal_columns].to_string(
        index=False,
        formatters={
            "avg_goals_scored_last5": lambda x: (
                f"{x:.2f}" if pd.notna(x) else "NaN"
            ),
            "avg_goals_conceded_last5": lambda x: (
                f"{x:.2f}" if pd.notna(x) else "NaN"
            ),
            "win_rate_last5": lambda x: (
                f"{x:.2f}" if pd.notna(x) else "NaN"
            )
        }
    )
)


# ============================================================
# 7. PREPARE HOME-TEAM FEATURES
# ============================================================

feature_columns = [
    "avg_goals_scored_last5",
    "avg_goals_conceded_last5",
    "win_rate_last5"
]

home_form = team_history[
    ["match_id", "team"] + feature_columns
].copy()

home_form = home_form.rename(
    columns={
        "team": "home_team",
        "avg_goals_scored_last5": "home_avg_goals_scored",
        "avg_goals_conceded_last5": "home_avg_goals_conceded",
        "win_rate_last5": "home_win_rate"
    }
)


# ============================================================
# 8. PREPARE AWAY-TEAM FEATURES
# ============================================================

away_form = team_history[
    ["match_id", "team"] + feature_columns
].copy()

away_form = away_form.rename(
    columns={
        "team": "away_team",
        "avg_goals_scored_last5": "away_avg_goals_scored",
        "avg_goals_conceded_last5": "away_avg_goals_conceded",
        "win_rate_last5": "away_win_rate"
    }
)


# ============================================================
# 9. MERGE FEATURES WITH ORIGINAL MATCHES
# ============================================================

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


# ============================================================
# 10. REMOVE MATCHES WITHOUT COMPLETE RECENT-FORM FEATURES
# ============================================================

model_features = [
    "home_avg_goals_scored",
    "home_avg_goals_conceded",
    "home_win_rate",
    "away_avg_goals_scored",
    "away_avg_goals_conceded",
    "away_win_rate"
]

match_features_clean = match_features.dropna(
    subset=model_features
).copy()


# ============================================================
# 11. INSPECT THE FINAL FEATURE DATASET
# ============================================================

print("\n--- FINAL MATCH FEATURE DATASET ---")

print("\nOriginal dataset shape:")
print(df.shape)

print("\nDataset shape after merging:")
print(match_features.shape)

print("\nDataset shape after removing missing features:")
print(match_features_clean.shape)

print("\nMatches removed:")
print(len(match_features) - len(match_features_clean))

print("\nMissing values in model features:")
print(match_features_clean[model_features].isnull().sum())

print("\nFirst five rows:")

columns_to_show = [
    "match_date",
    "home_team",
    "away_team"
] + model_features + ["outcome"]

print(
    match_features_clean[columns_to_show]
    .head()
    .to_string(index=False)
)

print("\nOutcome distribution:")
print(match_features_clean["outcome"].value_counts())

print("\nNumber of unique teams:")
print(match_features_clean["home_team"].nunique())


# ============================================================
# 12. SAVE THE FINAL FEATURE DATASET
# ============================================================

output_path = Path(
    "data/processed/ucl_matches_features.csv"
)

output_path.parent.mkdir(parents=True, exist_ok=True)

match_features_clean.to_csv(
    output_path,
    index=False
)

print("\nFeature engineering completed successfully!")
print(f"Final dataset saved to: {output_path}")
