
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ucl_matches.csv"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ucl_matches_with_team_ids.csv"
)


# --------------------------------------------------
# 2. Load the original dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)


# --------------------------------------------------
# 3. Define canonical team-name mappings
# --------------------------------------------------

TEAM_NAME_MAPPING = {

    # First 9 confirmed mappings
    "Bayern München (GER)": "FC Bayern München (GER)",
    "Atlético Madrid (ESP)": "Club Atlético de Madrid (ESP)",
    "AS Monaco (FRA)": "AS Monaco FC (MCO)",
    "Atalanta (ITA)": "Atalanta BC (ITA)",
    "Real Madrid (ESP)": "Real Madrid CF (ESP)",
    "Manchester City (ENG)": "Manchester City FC (ENG)",
    "Juventus (ITA)": "Juventus FC (ITA)",
    "Paris Saint-Germain (FRA)": "Paris Saint-Germain FC (FRA)",
    "PSV Eindhoven (NED)": "PSV (NED)",

    # Additional 19 confirmed mappings
    "Bayer Leverkusen (GER)": "Bayer 04 Leverkusen (GER)",
    "Crvena Zvezda (SRB)": "FK Crvena Zvezda (SRB)",
    "Dinamo Zagreb (CRO)": "GNK Dinamo Zagreb (CRO)",
    "Feyenoord (NED)": "Feyenoord Rotterdam (NED)",
    "Galatasaray (TUR)": "Galatasaray SK (TUR)",
    "Inter (ITA)": "FC Internazionale Milano (ITA)",
    "Manchester United (ENG)": "Manchester United FC (ENG)",
    "Olympique Marseille (FRA)": "Olympique de Marseille (FRA)",
    "Olympiakos Piraeus (GRE)": "PAE Olympiakos SFP (GRE)",
    "Qarabağ FK (AZE)": "Qarabağ Ağdam FK (AZE)",
    "Real Sociedad (ESP)": "Real Sociedad de Fútbol (ESP)",
    "Slavia Praha (CZE)": "SK Slavia Praha (CZE)",
    "SL Benfica (POR)": "Sport Lisboa e Benfica (POR)",
    "Sporting Braga (POR)": "Sporting Clube de Braga (POR)",
    "Sporting CP (POR)": "Sporting Clube de Portugal (POR)",
    "Tottenham Hotspur (ENG)": "Tottenham Hotspur FC (ENG)",
    "RB Salzburg (AUT)": "FC Red Bull Salzburg (AUT)",
    "Shakhtar Donetsk (UKR)": "FK Shakhtar Donetsk (UKR)",
    "Lazio Roma (ITA)": "SS Lazio (ITA)",
}


# --------------------------------------------------
# 4. Create canonical team columns
# --------------------------------------------------

df["home_team_id"] = df["home_team"].replace(TEAM_NAME_MAPPING)

df["away_team_id"] = df["away_team"].replace(TEAM_NAME_MAPPING)


# --------------------------------------------------
# 5. Validate the mapping
# --------------------------------------------------

original_teams = pd.concat(
    [df["home_team"], df["away_team"]],
    ignore_index=True
)

canonical_teams = pd.concat(
    [df["home_team_id"], df["away_team_id"]],
    ignore_index=True
)

print("\nTEAM MAPPING VALIDATION")
print("-----------------------")

print(f"Original unique team names: {original_teams.nunique()}")

print(f"Canonical unique team identities: {canonical_teams.nunique()}")

print(f"Mappings applied: {len(TEAM_NAME_MAPPING)}")


# --------------------------------------------------
# 6. Display the mappings
# --------------------------------------------------

print("\nAPPLIED TEAM MAPPINGS")
print("---------------------")

for original_name, canonical_name in TEAM_NAME_MAPPING.items():

    print(f"{original_name} -> {canonical_name}")


# --------------------------------------------------
# 7. Check for unmapped original names
# --------------------------------------------------

unmapped_teams = sorted(
    set(original_teams.unique()) - set(TEAM_NAME_MAPPING.keys())
)

print("\nUNMAPPED TEAM NAMES")
print("-------------------")

for team in unmapped_teams:
    print(team)

print(f"\nNumber of unmapped names: {len(unmapped_teams)}")


# --------------------------------------------------
# 8. Save the cleaned dataset
# --------------------------------------------------

df.to_csv(OUTPUT_PATH, index=False)

print("\nUPDATED DATASET SAVED")
print("---------------------")

print(OUTPUT_PATH)

# --------------------------------------------------
# 9. Validate dataset integrity
# --------------------------------------------------

print("\nDATASET INTEGRITY CHECK")
print("-----------------------")

print(f"Original rows: {len(df)}")
print(f"Output rows: {len(pd.read_csv(OUTPUT_PATH))}")

print(
    "Missing canonical home IDs:",
    df["home_team_id"].isna().sum()
)

print(
    "Missing canonical away IDs:",
    df["away_team_id"].isna().sum()
)

print(
    "Original home team columns preserved:",
    df["home_team"].equals(pd.read_csv(DATA_PATH)["home_team"])
)

print(
    "Original away team columns preserved:",
    df["away_team"].equals(pd.read_csv(DATA_PATH)["away_team"])
)

print("\nTEAM MAPPING COMPLETED!")
