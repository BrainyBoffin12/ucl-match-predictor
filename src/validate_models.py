
from pathlib import Path
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    log_loss
)


# 1. Load the original training dataset
data_dir = Path("data/processed")

train_df = pd.read_csv(data_dir / "ucl_train.csv")


# 2. Define our 12 features
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


# 3. Sort matches chronologically
train_df["match_date"] = pd.to_datetime(train_df["match_date"])

train_df = train_df.sort_values("match_date").reset_index(drop=True)


# 4. Create a chronological training/validation split
unique_dates = sorted(train_df["match_date"].unique())

split_index = int(len(unique_dates) * 0.80)

split_date = unique_dates[split_index]

training_df = train_df[
    train_df["match_date"] < split_date
].copy()

validation_df = train_df[
    train_df["match_date"] >= split_date
].copy()


# 5. Prepare X and y
X_train = training_df[feature_columns]
y_train = training_df["outcome"]

X_val = validation_df[feature_columns]
y_val = validation_df["outcome"]


# 6. Display the split information
print("\n--- VALIDATION DATASET OVERVIEW ---")

print(f"\nTraining matches: {len(training_df)}")
print(f"Validation matches: {len(validation_df)}")

print(f"\nTraining period:")
print(
    f"{training_df['match_date'].min().date()} "
    f"to {training_df['match_date'].max().date()}"
)

print(f"\nValidation period:")
print(
    f"{validation_df['match_date'].min().date()} "
    f"to {validation_df['match_date'].max().date()}"
)

print("\nTraining outcome distribution:")
print(y_train.value_counts())

print("\nValidation outcome distribution:")
print(y_val.value_counts())


# 7. Define the draw weights we want to experiment with
draw_weights = [1.0, 1.5, 2.0, 2.5]

results = []

labels = ["home_win", "draw", "away_win"]


# 8. Train and evaluate each model
for draw_weight in draw_weights:

    print("\n" + "=" * 55)
    print(f"TESTING DRAW WEIGHT: {draw_weight}")
    print("=" * 55)

    # Give the draw class a custom weight
    class_weights = {
        "home_win": 1.0,
        "draw": draw_weight,
        "away_win": 1.0
    }

    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight=class_weights
        ))
    ])

    # Train using only the training portion
    model.fit(X_train, y_train)

    # Predict validation outcomes
    y_pred = model.predict(X_val)

    probabilities = model.predict_proba(X_val)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_val, y_pred)

    balanced_accuracy = balanced_accuracy_score(
        y_val,
        y_pred
    )

    macro_f1 = f1_score(
        y_val,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0
    )

    validation_log_loss = log_loss(
        y_val,
        probabilities,
        labels=model.named_steps["classifier"].classes_
    )

    # Save the results
    results.append({
        "draw_weight": draw_weight,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "log_loss": validation_log_loss
    })

    # Display detailed evaluation
    print(f"\nAccuracy: {accuracy:.2%}")
    print(f"Balanced accuracy: {balanced_accuracy:.2%}")
    print(f"Macro F1: {macro_f1:.2%}")
    print(f"Log loss: {validation_log_loss:.4f}")

    print("\nClassification report:")

    print(classification_report(
        y_val,
        y_pred,
        labels=labels,
        zero_division=0
    ))

    print("\nConfusion matrix:")

    print(confusion_matrix(
        y_val,
        y_pred,
        labels=labels
    ))


# 9. Compare all experiments
results_df = pd.DataFrame(results)

print("\n\n--- FINAL VALIDATION COMPARISON ---")

print(
    results_df.sort_values(
        by="balanced_accuracy",
        ascending=False
    ).to_string(index=False)
)


# 10. Save the comparison results
output_path = data_dir / "validation_results.csv"

results_df.to_csv(output_path, index=False)

print(f"\nValidation results saved to: {output_path}")

print("\nValidation experiments completed successfully!")
