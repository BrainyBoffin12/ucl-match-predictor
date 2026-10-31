
from pathlib import Path
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss
)


# 1. Load the datasets
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


# 3. Create the standard Logistic Regression pipeline
model = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000
    ))
])


# 4. Train the model
model.fit(X_train, y_train)


# 5. Make predictions
y_pred = model.predict(X_test)
probabilities = model.predict_proba(X_test)


# 6. Evaluate the model
labels = ["home_win", "draw", "away_win"]

print("\n--- STANDARD LOGISTIC REGRESSION V2 ---")

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2%}")

print(
    f"Balanced accuracy: "
    f"{balanced_accuracy_score(y_test, y_pred):.2%}"
)

print(
    f"Macro F1: "
    f"{f1_score(y_test, y_pred, average='macro'):.2%}"
)

print(
    f"Log loss: "
    f"{log_loss(y_test, probabilities, labels=model.named_steps['classifier'].classes_):.4f}"
)

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    labels=labels,
    zero_division=0
))

print("\nConfusion matrix:")
print(confusion_matrix(
    y_test,
    y_pred,
    labels=labels
))


# 7. Inspect predicted probabilities
probability_df = pd.DataFrame(
    probabilities,
    columns=model.named_steps["classifier"].classes_
)

probability_df.insert(
    0, "actual", y_test.reset_index(drop=True)
)

probability_df.insert(1, "predicted", y_pred)

print("\n--- ACTUAL DRAW MATCHES ---")

draw_matches = probability_df[
    probability_df["actual"] == "draw"
]

print(draw_matches.to_string(index=False))


print("\nStandard Logistic Regression V2 completed successfully!")
