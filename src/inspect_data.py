
from pathlib import Path
import re
import pandas as pd


# 1. Define the seasons we want to process
seasons = ["2023-2024", "2024-2025", "2025-2026"]


# 2. Extract the actual match score
def extract_score(line):
    # Prefer the score after extra time, excluding penalty goals
    score = re.search(r"(\d+)-(\d+)\s+a\.e\.t\.", line)

    # Otherwise, extract the first score
    if not score:
        score = re.search(r"(\d+)-(\d+)", line)

    if not score:
        return None

    home_goals = int(score.group(1))
    away_goals = int(score.group(2))

    return home_goals, away_goals


# 3. Determine the match outcome
def get_outcome(home_goals, away_goals):
    if home_goals > away_goals:
        return "home_win"
    elif home_goals == away_goals:
        return "draw"
    else:
        return "away_win"


# 4. Process each season
for season in seasons:
    file_path = Path(f"data/raw/{season}-ucl.txt")

    with file_path.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    # Detect match lines
    match_lines = [
        line for line in lines
        if " v " in line and re.search(r"\d+-\d+", line)
    ]

    matches = []

    # Extract information from each match
    for line in match_lines:
        teams = line.split(" v ", maxsplit=1)

        home_team = teams[0].strip()
        away_section = teams[1]

        # Remove kickoff time and country codes
        home_team = re.sub(r"^\d{1,2}:\d{2}\s+", "", home_team)
        home_team = re.sub(r"\s+\([A-Z]{3}\)$", "", home_team)
        away_team = re.sub(
            r"\s+\([A-Z]{3}\).*$", "", away_section
        ).strip()

        score = extract_score(line)

        if score is None:
            continue

        home_goals, away_goals = score
        outcome = get_outcome(home_goals, away_goals)

        matches.append({
            "season": season,
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "outcome": outcome
        })

    # Convert matches into a DataFrame
    df = pd.DataFrame(matches)

    # Save the processed season
    output_path = Path(
        f"data/processed/ucl_matches_{season.replace('-', '_')}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Print a summary
    print(f"\nSeason: {season}")
    print(f"Detected match lines: {len(match_lines)}")
    print(f"Processed matches: {len(df)}")
    print(f"Outcome counts:\n{df['outcome'].value_counts()}")
    print(f"Saved to: {output_path}")


print("\nAll seasons processed!")
