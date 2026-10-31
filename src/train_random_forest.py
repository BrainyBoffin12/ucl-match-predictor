
from pathlib import Path
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load the updated datasets
data_dir = Path("data/processed")

train_df = pd.read_csv(data_dir / "ucl_train.csv")
test_df = pd.read_csv(data_dir / "ucl_test.csv")

# 2. Select the 12 features
feature_columns = [
    "home_avg_goals_scored",
    "home_avg_goals_conceded",
    "home_win_rate",
    "away_avg_goals_scored",
    "away_avg_goals_conceded",
    "away_win_rate",
    "home_points_per_game",
    "away_points_per_game",
    "home_goal_difference",
    "away_goal_difference",
    "home_venue_points_per_game",
    "away_venue_points_per_game"
]

X_train = train_df[feature_columns]
y_train = train_df["outcome"]

X_test = test_df[feature_columns]
y_test = test_df["outcome"]

# 3. Create Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# 4. Train
model.fit(X_train, y_train)

# 5. Predict
y_pred = model.predict(X_test)

# 6. Evaluate
labels = ["home_win", "draw", "away_win"]

print("\n--- RANDOM FOREST V2 (12 FEATURES) ---")

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2%}")

print(
    f"Balanced accuracy: "
    f"{balanced_accuracy_score(y_test, y_pred):.2%}"
)

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    labels=labels,
    zero_division=0
))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred, labels=labels))

# 7. Feature importance
importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print("\nFeature importance:")
print(importance_df.to_string(index=False))
