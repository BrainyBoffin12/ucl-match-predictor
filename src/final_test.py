
# ============================================================
# FINAL TEST - UCL MATCH OUTCOME PREDICTOR
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib

# Use a non-interactive backend to avoid Tkinter errors
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/ucl_matches_with_features.csv"

OUTPUT_DIR = "reports/final_test"

DRAW_THRESHOLD = 0.225

RANDOM_STATE = 42

CLASSES = ["home_win", "draw", "away_win"]

CLASS_NAMES = ["Home Win", "Draw", "Away Win"]


# Baseline features (15 features)
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


TRAIN_SEASONS = [
    "2011-2012",
    "2012-2013",
    "2013-2014",
    "2014-2015",
    "2015-2016",
    "2016-2017",
    "2017-2018",
    "2018-2019",
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024"
]

TEST_SEASONS = [
    "2024-2025",
    "2025-2026"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_metrics(y_true, y_pred):
    """
    Calculate the main classification metrics.
    """

    return {
        "Accuracy": accuracy_score(y_true, y_pred),

        "Balanced Accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),

        "Macro F1": f1_score(
            y_true,
            y_pred,
            labels=CLASSES,
            average="macro",
            zero_division=0
        ),

        "Draw Precision": precision_score(
            y_true,
            y_pred,
            labels=["draw"],
            average="macro",
            zero_division=0
        ),

        "Draw Recall": recall_score(
            y_true,
            y_pred,
            labels=["draw"],
            average="macro",
            zero_division=0
        ),

        "Draw F1": f1_score(
            y_true,
            y_pred,
            labels=["draw"],
            average="macro",
            zero_division=0
        ),

        "Predicted Draws": int(
            np.sum(np.array(y_pred) == "draw")
        )
    }


def predict_with_draw_threshold(probabilities, model_classes, threshold):
    """
    Predict a draw if its probability reaches the threshold.

    Otherwise, choose the more likely result between
    home win and away win.
    """

    class_indices = {
        class_name: list(model_classes).index(class_name)
        for class_name in CLASSES
    }

    home_prob = probabilities[:, class_indices["home_win"]]
    draw_prob = probabilities[:, class_indices["draw"]]
    away_prob = probabilities[:, class_indices["away_win"]]

    predictions = []

    for i in range(len(probabilities)):

        if draw_prob[i] >= threshold:
            predictions.append("draw")

        elif home_prob[i] >= away_prob[i]:
            predictions.append("home_win")

        else:
            predictions.append("away_win")

    return np.array(predictions)


def print_metrics(title, metrics):
    """
    Display metrics in a readable format.
    """

    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)

    for metric, value in metrics.items():

        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")

        else:
            print(f"{metric}: {value}")


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n" + "=" * 65)
print("LOADING DATA")
print("=" * 65)

df = pd.read_csv(DATA_PATH)

print(f"Total matches: {len(df)}")
print(f"Available seasons: {df['season'].nunique()}")


# ============================================================
# 2. VALIDATE DATA
# ============================================================

required_columns = ["season", "outcome"] + FEATURES

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if df[required_columns].isnull().any().any():
    raise ValueError(
        "Missing values detected in the target or feature columns. "
        "Check the feature engineering output before training."
    )

if not set(df["outcome"].unique()).issubset(set(CLASSES)):
    raise ValueError(
        "Unexpected outcome labels detected in the dataset."
    )


# ============================================================
# 3. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 65)
print("DATA SPLIT")
print("=" * 65)

train_df = df[df["season"].isin(TRAIN_SEASONS)].copy()

test_df = df[df["season"].isin(TEST_SEASONS)].copy()

if train_df.empty or test_df.empty:
    raise ValueError("Training or test dataset is empty.")

if set(train_df["season"]) & set(test_df["season"]):
    raise ValueError("Training and test seasons overlap.")

X_train = train_df[FEATURES]
y_train = train_df["outcome"]

X_test = test_df[FEATURES]
y_test = test_df["outcome"]

print(f"Training matches: {len(train_df)}")
print(f"Test matches: {len(test_df)}")

print("\nTraining seasons:")
print(sorted(train_df["season"].unique()))

print("\nTest seasons:")
print(sorted(test_df["season"].unique()))

print("\nActual test outcome distribution:")
print(y_test.value_counts())


# ============================================================
# 4. CREATE FINAL MODEL
# ============================================================

print("\n" + "=" * 65)
print("CREATING FINAL MODEL")
print("=" * 65)

model = Pipeline([
    ("scaler", StandardScaler()),

    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    ))
])

print("\nTraining Standard Logistic Regression...")

model.fit(X_train, y_train)

print("Training completed successfully.")


# ============================================================
# 5. GENERATE TEST PREDICTIONS
# ============================================================

print("\n" + "=" * 65)
print("GENERATING TEST PREDICTIONS")
print("=" * 65)

probabilities = model.predict_proba(X_test)

model_classes = model.named_steps["classifier"].classes_

# Predictions using the frozen draw threshold
y_pred = predict_with_draw_threshold(
    probabilities,
    model_classes,
    DRAW_THRESHOLD
)

# Standard argmax predictions for comparison
y_pred_argmax = model.predict(X_test)


# ============================================================
# 6. FINAL TEST RESULTS
# ============================================================

final_metrics = calculate_metrics(y_test, y_pred)

print_metrics(
    "FINAL TEST RESULTS - ALL SEASONS",
    final_metrics
)


# ============================================================
# 7. THRESHOLD VS ARGMAX
# ============================================================

threshold_metrics = calculate_metrics(y_test, y_pred)

argmax_metrics = calculate_metrics(y_test, y_pred_argmax)

comparison_df = pd.DataFrame([
    {
        "Strategy": f"Draw threshold {DRAW_THRESHOLD}",
        **threshold_metrics
    },
    {
        "Strategy": "Standard argmax",
        **argmax_metrics
    }
])

print("\n" + "=" * 65)
print("THRESHOLD VS ARGMAX")
print("=" * 65)

print(comparison_df.to_string(index=False))


# ============================================================
# 8. PER-SEASON TEST RESULTS
# ============================================================

season_results = []

for season in TEST_SEASONS:

    season_mask = test_df["season"].values == season

    season_y_true = y_test.iloc[
        np.where(season_mask)[0]
    ]

    season_y_pred = y_pred[season_mask]

    metrics = calculate_metrics(
        season_y_true,
        season_y_pred
    )

    season_results.append({
        "Season": season,
        "Matches": len(season_y_true),
        **metrics
    })


season_results_df = pd.DataFrame(season_results)

print("\n" + "=" * 65)
print("PER-SEASON TEST RESULTS")
print("=" * 65)

print(season_results_df.to_string(index=False))


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=CLASSES
)

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual Home Win",
        "Actual Draw",
        "Actual Away Win"
    ],
    columns=[
        "Predicted Home Win",
        "Predicted Draw",
        "Predicted Away Win"
    ]
)

print("\n" + "=" * 65)
print("CONFUSION MATRIX")
print("=" * 65)

print(cm_df)


# ============================================================
# 10. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    labels=CLASSES,
    target_names=CLASS_NAMES,
    zero_division=0
)

print("\n" + "=" * 65)
print("CLASSIFICATION REPORT")
print("=" * 65)

print(report)


# ============================================================
# 11. SAVE TEST PREDICTIONS
# ============================================================

predictions_df = test_df.copy()

predictions_df["actual_outcome"] = y_test.values

predictions_df["predicted_outcome"] = y_pred

predictions_df["argmax_prediction"] = y_pred_argmax

predictions_df["correct_prediction"] = (
    predictions_df["actual_outcome"]
    == predictions_df["predicted_outcome"]
)

# Save the probability of each class
for i, class_name in enumerate(model_classes):

    predictions_df[f"prob_{class_name}"] = probabilities[:, i]

predictions_df["draw_threshold"] = DRAW_THRESHOLD

predictions_path = os.path.join(
    OUTPUT_DIR,
    "test_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# 12. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame([
    {
        "Metric": metric,
        "Value": value
    }
    for metric, value in final_metrics.items()
])

metrics_path = os.path.join(
    OUTPUT_DIR,
    "test_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# 13. SAVE THRESHOLD COMPARISON
# ============================================================

comparison_path = os.path.join(
    OUTPUT_DIR,
    "threshold_vs_argmax.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# 14. SAVE PER-SEASON RESULTS
# ============================================================

season_path = os.path.join(
    OUTPUT_DIR,
    "per_season_metrics.csv"
)

season_results_df.to_csv(
    season_path,
    index=False
)


# ============================================================
# 15. SAVE CONFUSION MATRIX
# ============================================================

confusion_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.csv"
)

cm_df.to_csv(confusion_path)


# ============================================================
# 16. GENERATE CONFUSION MATRIX PLOT
# ============================================================

print("\nGenerating confusion matrix plot...")

fig, ax = plt.subplots(figsize=(8, 6))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

display.plot(
    ax=ax,
    values_format="d",
    cmap="Blues"
)

plt.title("UCL Match Outcome Predictor - Final Test")

plt.tight_layout()

plot_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("Confusion matrix saved successfully.")


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("FINAL TEST COMPLETED")
print("=" * 65)

print(f"Training matches: {len(train_df)}")
print(f"Test matches: {len(test_df)}")

print(f"Frozen draw threshold: {DRAW_THRESHOLD}")

print(f"Accuracy: {final_metrics['Accuracy']:.4f}")

print(
    f"Balanced Accuracy: "
    f"{final_metrics['Balanced Accuracy']:.4f}"
)

print(f"Macro F1: {final_metrics['Macro F1']:.4f}")

print(f"Draw F1: {final_metrics['Draw F1']:.4f}")


print("\nGenerated files:")

print(predictions_path)
print(metrics_path)
print(comparison_path)
print(season_path)
print(confusion_path)
print(plot_path)

print("\nFinal test finished successfully.")
