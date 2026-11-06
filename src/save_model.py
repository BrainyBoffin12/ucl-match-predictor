
from pathlib import Path

import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ucl_matches_with_features.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ucl_model.joblib"
)


# ==========================================
# 2. FEATURES
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
# 3. LOAD DATA
# ==========================================

print("Loading historical data...")

df = pd.read_csv(DATA_PATH)

X = df[FEATURES]
y = df["outcome"]

print(f"Total matches: {len(df)}")


# ==========================================
# 4. CREATE MODEL
# ==========================================

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])


# ==========================================
# 5. TRAIN MODEL
# ==========================================

print("\nTraining model...")

model.fit(X, y)

print("Training completed!")


# ==========================================
# 6. SAVE MODEL
# ==========================================

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully!")
print(MODEL_PATH)
