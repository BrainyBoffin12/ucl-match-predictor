
from pathlib import Path
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. LOAD TRAINING AND TESTING DATA
# ============================================================

data_dir = Path("data/processed")

train_df = pd.read_csv(data_dir / "ucl_train.csv")
test_df = pd.read_csv(data_dir / "ucl_test.csv")


# ============================================================
# 2. DEFINE FEATURES AND TARGET
# ============================================================

feature_columns = [
    "home_avg_goals_scored",
    "home_avg_goals_conceded",
    "home_win_rate",
    "away_avg_goals_scored",
    "away_avg_goals_conceded",
    "away_win_rate"
]

X_train = train_df[feature_columns]
y_train = train_df["outcome"]

X_test = test_df[feature_columns]
y_test = test_df["outcome"]


# ============================================================
# 3. CREATE THE BASELINE MODEL
# ============================================================

baseline_model = DummyClassifier(
    strategy="most_frequent"
)


# ============================================================
# 4. TRAIN THE BASELINE MODEL
# ============================================================

baseline_model.fit(X_train, y_train)


# ============================================================
# 5. MAKE PREDICTIONS
# ============================================================

y_pred = baseline_model.predict(X_test)


# ============================================================
# 6. EVALUATE THE MODEL
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n--- BASELINE MODEL RESULTS ---")

print("\nMost frequent training outcome:")
print(baseline_model.classes_[baseline_model.class_prior_.argmax()])

print("\nBaseline accuracy:")
print(f"{accuracy * 100:.2f}%")


# Classification report
print("\nClassification report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=["home_win", "draw", "away_win"],
        zero_division=0
    )
)


# Confusion matrix
print("\nConfusion matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=["home_win", "draw", "away_win"]
    )
)


# ============================================================
# 7. INSPECT ACTUAL VS PREDICTED OUTCOMES
# ============================================================

comparison = pd.DataFrame({
    "actual": y_test.values,
    "predicted": y_pred
})

print("\n--- ACTUAL VS PREDICTED ---")
print(comparison.head(15).to_string(index=False))


print("\nBaseline evaluation completed successfully!")
