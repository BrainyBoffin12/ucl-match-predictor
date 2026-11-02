
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Generate charts without opening a Tkinter window

import matplotlib.pyplot as plt
import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ucl_matches.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


# --------------------------------------------------
# 2. Load the dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

print("\nDATASET OVERVIEW")
print("----------------")
print(f"Total matches: {len(df)}")
print(f"Total seasons: {df['season'].nunique()}")

print("\nFirst 5 matches:")
print(df.head().to_string(index=False))

print("\nDataset information:")
df.info()


# --------------------------------------------------
# 3. Outcome distribution
# --------------------------------------------------

outcome_counts = df["outcome"].value_counts()

print("\nOUTCOME DISTRIBUTION")
print("--------------------")
print(outcome_counts)

print("\nOutcome percentages:")
print(
    (df["outcome"].value_counts(normalize=True) * 100).round(2)
)


# --------------------------------------------------
# 4. Goals analysis
# --------------------------------------------------

df["total_goals"] = df["home_goals"] + df["away_goals"]

print("\nGOALS ANALYSIS")
print("--------------")

print(f"Average goals per match: {df['total_goals'].mean():.2f}")
print(f"Average home goals: {df['home_goals'].mean():.2f}")
print(f"Average away goals: {df['away_goals'].mean():.2f}")

print(f"Highest scoring match: {df['total_goals'].max()} goals")
print(f"Matches with 0 goals: {(df['total_goals'] == 0).sum()}")


# --------------------------------------------------
# 5. Outcome distribution by season
# --------------------------------------------------

season_outcomes = pd.crosstab(
    df["season"],
    df["outcome"],
    normalize="index"
) * 100

print("\nOUTCOME DISTRIBUTION BY SEASON (%)")
print("----------------------------------")
print(season_outcomes.round(2))


# --------------------------------------------------
# 6. Visualize outcome distribution by season
# --------------------------------------------------

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

season_outcomes.plot(
    kind="bar",
    stacked=True,
    figsize=(13, 6)
)

plt.title("UCL Match Outcomes by Season")
plt.xlabel("Season")
plt.ylabel("Percentage of Matches (%)")
plt.xticks(rotation=45)
plt.legend(title="Outcome")
plt.tight_layout()

chart_path = FIGURES_DIR / "outcomes_by_season.png"

plt.savefig(
    chart_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"\nChart saved successfully: {chart_path}")


# --------------------------------------------------
# 7. Visualize overall outcome distribution
# --------------------------------------------------

outcome_counts.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Overall UCL Match Outcome Distribution")
plt.xlabel("Match Outcome")
plt.ylabel("Number of Matches")
plt.xticks(rotation=0)
plt.tight_layout()

chart_path = FIGURES_DIR / "overall_outcome_distribution.png"

plt.savefig(
    chart_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Chart saved successfully: {chart_path}")


# --------------------------------------------------
# 8. Team name inspection
# --------------------------------------------------

all_teams = pd.concat(
    [df["home_team"], df["away_team"]],
    ignore_index=True
)

team_appearances = all_teams.value_counts().sort_index()

print("\nTEAM NAME INSPECTION")
print("--------------------")
print(f"Unique team names: {team_appearances.nunique() if False else len(team_appearances)}")

print("\nAll teams and their total appearances:")
print(team_appearances.to_string())

# --------------------------------------------------
# 9. Inspect possible duplicate team identities
# --------------------------------------------------

team_seasons = pd.concat(
    [
        df[["season", "home_team"]].rename(
            columns={"home_team": "team"}
        ),
        df[["season", "away_team"]].rename(
            columns={"away_team": "team"}
        )
    ],
    ignore_index=True
)

team_seasons = team_seasons.drop_duplicates()

candidate_groups = [
    ["Bayern München (GER)", "FC Bayern München (GER)"],
    ["Atlético Madrid (ESP)", "Club Atlético de Madrid (ESP)"],
    ["AS Monaco (FRA)", "AS Monaco FC (MCO)"],
    ["Atalanta (ITA)", "Atalanta BC (ITA)"],
    ["Real Madrid (ESP)", "Real Madrid CF (ESP)"],
    ["Manchester City (ENG)", "Manchester City FC (ENG)"],
    ["Juventus (ITA)", "Juventus FC (ITA)"],
    ["Paris Saint-Germain (FRA)", "Paris Saint-Germain FC (FRA)"],
    ["PSV (NED)", "PSV Eindhoven (NED)"]
]

print("\nPOSSIBLE DUPLICATE TEAM IDENTITIES")
print("----------------------------------")

for group in candidate_groups:
    print(f"\n{group[0]} / {group[1]}")

    for team in group:
        seasons = team_seasons.loc[
            team_seasons["team"] == team, "season"
        ].tolist()

        print(f"  {team}: {', '.join(seasons)}")

print("\nEDA COMPLETED SUCCESSFULLY!")
