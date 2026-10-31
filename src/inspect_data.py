
from pathlib import Path
from datetime import datetime
import re
import pandas as pd


# 1. Define the seasons
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

    matches = []

    current_date = None
    current_year = int(season[:4])

    # 5. Process every line to detect dates and matches
    for line in lines:

        # Detect date lines
        date_match = re.match(
            r"^\s*(?:[A-Z][a-z]{2})\s+"
            r"([A-Z][a-z]{2})\s+(\d{1,2})"
            r"(?:\s+(\d{4}))?\s*$",
            line
        )

        if date_match:

            month, day, year = date_match.groups()

            if year:
                current_year = int(year)

            else:
                # Handle December-to-January transitions
                if current_date is not None:

                    previous_month = current_date.month

                    new_month = datetime.strptime(
                        month, "%b"
                    ).month

                    if new_month < previous_month:
                        current_year += 1

            current_date = datetime.strptime(
                f"{month} {day} {current_year}",
                "%b %d %Y"
            ).date()

            continue

        # Ignore lines that aren't match records
        if " v " not in line:
            continue

        # Extract the match score
        score = extract_score(line)

        if score is None or current_date is None:
            continue

        # Extract team names
        teams = line.split(" v ", maxsplit=1)

        home_team = teams[0].strip()
        away_section = teams[1]

        # Remove kickoff time and country codes
        home_team = re.sub(
            r"^\d{1,2}:\d{2}\s+", "", home_team
        )

        home_team = re.sub(
            r"\s+\([A-Z]{3}\)$", "", home_team
        )

        away_team = re.sub(
            r"\s+\([A-Z]{3}\).*$", "", away_section
        ).strip()

        # Determine the outcome
        home_goals, away_goals = score

        outcome = get_outcome(home_goals, away_goals)

        # Store the match information
        matches.append({
            "season": season,
            "match_date": current_date,
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "outcome": outcome
        })

    # 6. Convert matches into a DataFrame
    df = pd.DataFrame(matches)

    # 7. Save the processed season
    output_path = Path(
        f"data/processed/ucl_matches_{season.replace('-', '_')}.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    # 8. Print a summary
    print(f"\nSeason: {season}")
    print(f"Processed matches: {len(df)}")

    print("\nMatches per date:")
    print(df["match_date"].value_counts().sort_index().head())

    print("\nFirst five matches:")
    print(df.head())

    print(f"\nSaved to: {output_path}")


print("\nAll seasons processed successfully!")
