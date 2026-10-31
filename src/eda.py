
from pathlib import Path
import pandas as pd
import matplotlib

# Use a non-interactive backend to avoid the Windows Tkinter issue
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# 1. Load the combined dataset
file_path = Path("data/processed/ucl_matches_all.csv")
df = pd.read_csv(file_path)


# 2. Match outcome distribution
outcome_counts = df["outcome"].value_counts()

outcome_percentages = (
    df["outcome"].value_counts(normalize=True) * 100
)

print("\n--- MATCH OUTCOME DISTRIBUTION ---")

print("\nNumber of matches:")
print(outcome_counts)

print("\nPercentage of matches:")
print(outcome_percentages.round(2))


# 3. Goal statistics
df["total_goals"] = df["home_goals"] + df["away_goals"]

print("\n--- GOAL STATISTICS ---")

print("\nAverage goals per match:")
print(round(df["total_goals"].mean(), 2))

print("\nAverage home goals:")
print(round(df["home_goals"].mean(), 2))

print("\nAverage away goals:")
print(round(df["away_goals"].mean(), 2))

print("\nHighest-scoring match:")
print(df.loc[df["total_goals"].idxmax()])


# 4. Visualize average home and away goals
average_goals = [
    df["home_goals"].mean(),
    df["away_goals"].mean()
]

labels = ["Home teams", "Away teams"]

plt.figure(figsize=(7, 5))
plt.bar(labels, average_goals)

plt.title("Average Goals: Home vs Away")
plt.ylabel("Average Goals")

plt.savefig("data/processed/average_goals.png")
plt.close()


# 5. Compare statistics across seasons
print("\n--- SEASON COMPARISON ---")

print("\nMatches per season:")
print(df.groupby("season").size())

print("\nAverage goals per match:")
print(
    df.groupby("season")["total_goals"]
    .mean()
    .round(2)
)

print("\nOutcome percentages per season:")

outcome_by_season = pd.crosstab(
    df["season"],
    df["outcome"],
    normalize="index"
) * 100

print(outcome_by_season.round(2))


# 6. Top 10 teams by total goals scored

home_stats = df[["home_team", "home_goals"]].copy()
home_stats.columns = ["team", "goals_scored"]

away_stats = df[["away_team", "away_goals"]].copy()
away_stats.columns = ["team", "goals_scored"]

team_goals = pd.concat(
    [home_stats, away_stats],
    ignore_index=True
)

top_scorers = (
    team_goals.groupby("team")["goals_scored"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\n--- TOP 10 TEAMS BY TOTAL GOALS SCORED ---")
print(top_scorers)


# 7. Team attacking performance

team_stats = pd.concat(
    [home_stats, away_stats],
    ignore_index=True
)

team_performance = (
    team_stats.groupby("team")
    .agg(
        matches_played=("goals_scored", "count"),
        total_goals=("goals_scored", "sum"),
        average_goals=("goals_scored", "mean")
    )
    .sort_values("average_goals", ascending=False)
)

print("\n--- TOP 10 TEAMS BY AVERAGE GOALS SCORED ---")

print(
    team_performance[
        team_performance["matches_played"] >= 15
    ].head(10).round(2)
)


# 8. Team defensive performance

home_defense = df[["home_team", "away_goals"]].copy()
home_defense.columns = ["team", "goals_conceded"]

away_defense = df[["away_team", "home_goals"]].copy()
away_defense.columns = ["team", "goals_conceded"]

defense_stats = pd.concat(
    [home_defense, away_defense],
    ignore_index=True
)

team_defense = (
    defense_stats.groupby("team")
    .agg(
        matches_played=("goals_conceded", "count"),
        total_conceded=("goals_conceded", "sum"),
        average_conceded=("goals_conceded", "mean")
    )
    .sort_values("average_conceded", ascending=True)
)

print("\n--- TOP 10 TEAMS BY DEFENSIVE PERFORMANCE ---")

print(
    team_defense[
        team_defense["matches_played"] >= 15
    ].head(10).round(2)
)


print("\nEDA analysis completed successfully!")
