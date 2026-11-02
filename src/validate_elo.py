
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    confusion_matrix
)

# Load both training datasets
v2_df = pd.read_csv("data/processed/ucl_train.csv")
elo_df = pd.read_csv("data/processed/ucl_train_elo.csv")

v2_df["match_date"] = pd.to_datetime(v2_df["match_date"])
elo_df["match_date"] = pd.to_datetime(elo_df["match_date"])

# Original 12 features
v2_features = [
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

# New 15 features, including Elo
elo_features = v2_features + [
    "home_elo",
    "away_elo",
    "elo_difference"
]

# Verify both datasets contain the same matches
# Team names are not available in these prepared datasets,
# so we compare dates and outcomes instead.
match_columns = ["match_date", "outcome"]

v2_matches = v2_df[match_columns].sort_values(
    match_columns
).reset_index(drop=True)

elo_matches = elo_df[match_columns].sort_values(
    match_columns
).reset_index(drop=True)

assert v2_matches.equals(elo_matches), (
    "The two datasets do not contain identical matches!"
)

print("\nBoth datasets contain identical dates and outcomes.")

# Create the same chronological split for both datasets
unique_dates = sorted(v2_df["match_date"].unique())

split_index = int(len(unique_dates) * 0.80)
split_date = unique_dates[split_index]

v2_train = v2_df[v2_df["match_date"] < split_date]
v2_val = v2_df[v2_df["match_date"] >= split_date]

elo_train = elo_df[elo_df["match_date"] < split_date]
elo_val = elo_df[elo_df["match_date"] >= split_date]

print("\n--- CHRONOLOGICAL VALIDATION ---")
print("Split date:", split_date)
print("Training matches:", len(v2_train))
print("Validation matches:", len(v2_val))

print("\nTraining period:")
print(
    v2_train["match_date"].min(),
    "to",
    v2_train["match_date"].max()
)

print("\nValidation period:")
print(
    v2_val["match_date"].min(),
    "to",
    v2_val["match_date"].max()
)

# Define both experiments
experiments = {
    "Logistic Regression V2": (
        v2_train, v2_val, v2_features
    ),
    "Logistic Regression V2 + Elo": (
        elo_train, elo_val, elo_features
    )
}

results = []

labels = ["home_win", "draw", "away_win"]

# Train and evaluate both models
for name, (train_df, val_df, features) in experiments.items():

    X_train = train_df[features]
    y_train = train_df["outcome"]

    X_val = val_df[features]
    y_val = val_df["outcome"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)

    accuracy = accuracy_score(y_val, predictions)

    balanced_accuracy = balanced_accuracy_score(
        y_val, predictions
    )

    macro_f1 = f1_score(
        y_val, predictions, average="macro"
    )

    loss = log_loss(
        y_val,
        probabilities,
        labels=model.classes_
    )

    matrix = confusion_matrix(
        y_val,
        predictions,
        labels=labels
    )

    print(f"\n--- {name} ---")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Balanced accuracy: {balanced_accuracy:.2%}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Log loss: {loss:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    results.append({
        "model": name,
        "features": len(features),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "log_loss": loss
    })

# Compare results
results_df = pd.DataFrame(results)

print("\n--- FINAL COMPARISON ---")
print(results_df.to_string(index=False))

# Save results
output_path = "data/processed/elo_validation_comparison.csv"

results_df.to_csv(output_path, index=False)

print(f"\nResults saved successfully to: {output_path}")
