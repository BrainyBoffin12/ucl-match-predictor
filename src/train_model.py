
from pathlib import Path

import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ucl_matches_with_features.csv"
)

REPORTS_PATH = PROJECT_ROOT / "reports"


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(DATA_PATH)

print("DATASET LOADED")
print("--------------")
print(f"Total matches: {len(df)}")


# ==========================================
# DEFINE FEATURE SETS
# ==========================================

# Original 15 features: without Elo

baseline_features = [
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


# New 18 features: original 15 + 3 Elo features

elo_features = baseline_features + [
    "home_elo",
    "away_elo",
    "elo_difference"
]

target_column = "outcome"


# Check that all required columns exist

required_columns = (
    baseline_features
    + ["home_elo", "away_elo", "elo_difference", target_column]
)

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )

if df[required_columns].isnull().any().any():
    raise ValueError("Missing values detected in required columns.")


print("\nFEATURE SETS")
print("------------")
print(f"Baseline features: {len(baseline_features)}")
print(f"Elo features: {len(elo_features)}")


# ==========================================
# CHRONOLOGICAL DATA SPLIT
# ==========================================

train_seasons = [
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
    "2022-2023"
]

validation_seasons = ["2023-2024"]

test_seasons = ["2024-2025", "2025-2026"]

train_mask = df["season"].isin(train_seasons)
validation_mask = df["season"].isin(validation_seasons)
test_mask = df["season"].isin(test_seasons)

y_train = df.loc[train_mask, target_column]
y_validation = df.loc[validation_mask, target_column]

print("\nDATA SPLIT")
print("----------")
print(f"Training matches: {len(y_train)}")
print(f"Validation matches: {len(y_validation)}")
print(f"Testing matches reserved for final evaluation: {test_mask.sum()}")


# ==========================================
# DEFINE MODELS
# ==========================================

def create_models():

    return {
        "Standard Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]),

        "Balanced Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            ))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    }


# ==========================================
# EVALUATION FUNCTION
# ==========================================

def evaluate_model(model, X_train, y_train, X_val, y_val):

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    return {
        "Accuracy": accuracy_score(y_val, predictions),
        "Balanced Accuracy": balanced_accuracy_score(
            y_val, predictions
        ),
        "Macro F1": f1_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0
        )
    }


# ==========================================
# DUMMY CLASSIFIER
# ==========================================

# This model always predicts the most frequent class.
# It provides a simple reference point.

dummy_model = DummyClassifier(strategy="most_frequent")

dummy_X_train = df.loc[train_mask, baseline_features]
dummy_X_val = df.loc[validation_mask, baseline_features]

dummy_model.fit(dummy_X_train, y_train)

dummy_predictions = dummy_model.predict(dummy_X_val)

dummy_results = {
    "Feature Set": "Reference",
    "Model": "Dummy Classifier",
    "Accuracy": accuracy_score(y_validation, dummy_predictions),
    "Balanced Accuracy": balanced_accuracy_score(
        y_validation, dummy_predictions
    ),
    "Macro F1": f1_score(
        y_validation,
        dummy_predictions,
        average="macro",
        zero_division=0
    )
}


# ==========================================
# TRAIN AND COMPARE BOTH FEATURE SETS
# ==========================================

all_results = [dummy_results]

feature_sets = {
    "Baseline (15 features)": baseline_features,
    "With Elo (18 features)": elo_features
}

for feature_set_name, features in feature_sets.items():

    print("\n" + "=" * 55)
    print(f"EXPERIMENT: {feature_set_name}")
    print("=" * 55)

    X_train = df.loc[train_mask, features]
    X_val = df.loc[validation_mask, features]

    models = create_models()

    for model_name, model in models.items():

        metrics = evaluate_model(
            model,
            X_train,
            y_train,
            X_val,
            y_validation
        )

        result = {
            "Feature Set": feature_set_name,
            "Model": model_name,
            **metrics
        }

        all_results.append(result)

        print(f"\n{model_name}")
        print("-" * len(model_name))
        print(f"Accuracy: {metrics['Accuracy']:.4f}")
        print(
            f"Balanced Accuracy: "
            f"{metrics['Balanced Accuracy']:.4f}"
        )
        print(f"Macro F1: {metrics['Macro F1']:.4f}")

        # Show the detailed classification report
        predictions = model.predict(X_val)

        print("\nClassification report:")
        print(
            classification_report(
                y_validation,
                predictions,
                zero_division=0
            )
        )


# ==========================================
# FINAL COMPARISON TABLE
# ==========================================

comparison_df = pd.DataFrame(all_results)

comparison_df = comparison_df.sort_values(
    by=["Model", "Feature Set"]
)

print("\n" + "=" * 65)
print("BASELINE VS ELO: FINAL COMPARISON")
print("=" * 65)

print(comparison_df.to_string(index=False))


# ==========================================
# CALCULATE THE IMPACT OF ELO
# ==========================================

baseline_results = comparison_df[
    comparison_df["Feature Set"] == "Baseline (15 features)"
].set_index("Model")

elo_results = comparison_df[
    comparison_df["Feature Set"] == "With Elo (18 features)"
].set_index("Model")

print("\n" + "=" * 65)
print("IMPACT OF ADDING ELO FEATURES")
print("=" * 65)

for model_name in baseline_results.index:

    if model_name not in elo_results.index:
        continue

    print(f"\n{model_name}")

    for metric in [
        "Accuracy",
        "Balanced Accuracy",
        "Macro F1"
    ]:

        baseline_value = baseline_results.loc[model_name, metric]
        elo_value = elo_results.loc[model_name, metric]

        difference = elo_value - baseline_value

        print(
            f"{metric}: "
            f"{baseline_value:.4f} -> {elo_value:.4f} "
            f"(change: {difference:+.4f})"
        )


# ==========================================
# SAVE RESULTS
# ==========================================

REPORTS_PATH.mkdir(parents=True, exist_ok=True)

output_path = REPORTS_PATH / "baseline_vs_elo_comparison.csv"

comparison_df.to_csv(output_path, index=False)

print("\nRESULTS SAVED")
print("-------------")
print(output_path)

print("\nEXPERIMENT COMPLETED!")
