
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# --------------------------------------------------
# 2. Regular expressions
# --------------------------------------------------

DATE_PATTERN = re.compile(
    r"^[A-Za-z]{3}\s+([A-Za-z]{3})\s+(\d{1,2})(?:\s+(\d{4}))?$"
)

TIME_PATTERN = re.compile(r"^\d{1,2}:\d{2}\s+")

# Accepts:
# 1-1 (0-0)
# 4-1 a.e.t. (3-1, 1-0)
# 4-3 pen. 1-0 a.e.t. (1-0, 1-0)

RESULT_PATTERN = re.compile(
    r"(?P<result>"
    r"(?:\d+-\d+\s+pen\.?\s+)?"
    r"\d+-\d+(?:\s+a\.e\.t\.)?"
    r")"
    r"(?:\s+\([^)]*\))?$"
)


# --------------------------------------------------
# 3. Determine match outcome
# --------------------------------------------------

def get_outcome(home_goals, away_goals):

    if home_goals > away_goals:
        return "home_win"

    if home_goals < away_goals:
        return "away_win"

    return "draw"


# --------------------------------------------------
# 4. Parse one season
# --------------------------------------------------

def parse_season(file_path):

    matches = []

    file_path = Path(file_path)

    season = file_path.stem.replace("-ucl", "")

    # Initialize the year from the filename.
    current_year = int(season.split("-")[0])

    current_date = None
    previous_month = None

    with open(file_path, "r", encoding="utf-8-sig") as file:

        for line_number, raw_line in enumerate(file, start=1):

            line = raw_line.strip()

            if not line:
                continue

            # --------------------------------------
            # A. Detect dates
            # --------------------------------------

            date_match = DATE_PATTERN.match(line)

            if date_match:

                month, day, year = date_match.groups()

                month_number = datetime.strptime(
                    month, "%b"
                ).month

                if year:
                    current_year = int(year)

                # Only advance the year when crossing
                # from late in one year into early next year.
                elif (
                    previous_month is not None
                    and previous_month >= 11
                    and month_number <= 2
                    and month_number < previous_month
                ):
                    current_year += 1

                current_date = datetime.strptime(
                    f"{month} {day} {current_year}",
                    "%b %d %Y"
                ).date()

                previous_month = month_number

                continue

            # --------------------------------------
            # B. Detect match lines
            # --------------------------------------

            if " v " not in line:
                continue

            # Remove kickoff time when present.
            line = TIME_PATTERN.sub("", line, count=1)

            # --------------------------------------
            # C. Extract the result
            # --------------------------------------

            result_match = RESULT_PATTERN.search(line)

            if not result_match:
                continue

            result_text = result_match.group("result")

            # Extract the score after extra time.
            # Penalty scores are deliberately excluded.
            scores = re.findall(r"\d+-\d+", result_text)

            score_text = scores[-1]

            home_goals, away_goals = map(
                int, score_text.split("-")
            )

            # Remove the result and annotations.
            teams_text = line[:result_match.start()].strip()

            # --------------------------------------
            # D. Separate the teams
            # --------------------------------------

            teams = re.split(
                r"\s+v\s+",
                teams_text,
                maxsplit=1
            )

            if len(teams) != 2:
                continue

            home_team = teams[0].strip()
            away_team = teams[1].strip()

            # --------------------------------------
            # E. Validate date
            # --------------------------------------

            if current_date is None:
                raise ValueError(
                    f"No date found before match at "
                    f"{file_path.name}, line {line_number}"
                )

            # --------------------------------------
            # F. Store match
            # --------------------------------------

            matches.append({
                "season": season,
                "date": current_date,
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "outcome": get_outcome(
                    home_goals,
                    away_goals
                )
            })

    return pd.DataFrame(matches)


# --------------------------------------------------
# 5. Test on 2011-2012
# --------------------------------------------------


# --------------------------------------------------
# 5. Parse all seasons
# --------------------------------------------------

if __name__ == "__main__":

    all_matches = []

    raw_files = sorted(RAW_DATA_DIR.glob("*-ucl.txt"))

    print(f"Found {len(raw_files)} season files.\n")

    for file_path in raw_files:

        df = parse_season(file_path)

        all_matches.append(df)

        print(
            f"{file_path.name}: "
            f"{len(df)} matches parsed"
        )

    # Combine all seasons
    combined_df = pd.concat(
        all_matches,
        ignore_index=True
    )

    print("\n" + "=" * 45)
    print("COMBINED DATASET SUMMARY")
    print("=" * 45)

    print(f"Total seasons: {combined_df['season'].nunique()}")
    print(f"Total matches: {len(combined_df)}")

    print("\nMatches per season:")
    print(combined_df.groupby("season").size())

    print("\nOutcome distribution:")
    print(combined_df["outcome"].value_counts())

    print("\nMissing values:")
    print(combined_df.isna().sum())

    # Save combined dataset
    output_path = PROJECT_ROOT / "data" / "processed" / "ucl_matches.csv"

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    combined_df.to_csv(
        output_path,
        index=False
    )

    print(f"\nDataset saved to: {output_path}")


    # Check for duplicate matches
    duplicates = combined_df.duplicated(
        subset=[
            "season",
            "date",
            "home_team",
            "away_team"
        ]
    )

    print("\nDUPLICATE CHECK")
    print("----------------")
    print(f"Duplicate matches: {duplicates.sum()}")