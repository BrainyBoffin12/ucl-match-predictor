import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    classification_report,
    confusion_matrix
)

# Load datasets
train_df = pd.read_csv("data/processed/ucl_train_elo.csv")
test_df = pd.read_csv("data/processed/ucl_test_elo.csv")

# Define features
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
    "away_venue_points_per_game",
    "home_elo",
    "away_elo",
    "elo_difference"
]

# Separate features and target
X_train = train_df[feature_columns]
y_train = train_df["outcome"]

X_test = test_df[feature_columns]
y_test = test_df["outcome"]

# Create model pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# Evaluate model
print("\n--- LOGISTIC REGRESSION WITH ELO ---")

print("\nAccuracy:")
print(f"{accuracy_score(y_test, y_pred):.2%}")

print("\nBalanced Accuracy:")
print(f"{balanced_accuracy_score(y_test, y_pred):.2%}")

print("\nMacro F1:")
print(f"{f1_score(y_test, y_pred, average='macro'):.4f}")

print("\nLog Loss:")
print(f"{log_loss(y_test, y_prob, labels=model.classes_):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

labels = ["home_win", "draw", "away_win"]

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=labels))

print("\nModel classes:")
print(model.classes_)