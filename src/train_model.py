
from pathlib import Path

import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
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


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(DATA_PATH)

print("DATASET LOADED")
print("--------------")
print(f"Total matches: {len(df)}")


# ==========================================
# DEFINE FEATURES AND TARGET
# ==========================================

feature_columns = [
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

target_column = "outcome"

X = df[feature_columns]
y = df[target_column]


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

X_train = X[train_mask]
y_train = y[train_mask]

X_validation = X[validation_mask]
y_validation = y[validation_mask]

X_test = X[test_mask]
y_test = y[test_mask]

print("\nDATA SPLIT")
print("----------")
print(f"Training matches: {len(X_train)}")
print(f"Validation matches: {len(X_validation)}")
print(f"Testing matches: {len(X_test)}")


# ==========================================
# MODEL EVALUATION FUNCTION
# ==========================================

def evaluate_model(model, X_val, y_val, model_name):

    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)

    balanced_accuracy = balanced_accuracy_score(
        y_val,
        predictions
    )

    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
        zero_division=0
    )

    print(f"\n{model_name}")
    print("-" * len(model_name))

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nClassification report:")

    print(
        classification_report(
            y_val,
            predictions,
            zero_division=0
        )
    )

    labels = ["away_win", "draw", "home_win"]

    print("Confusion matrix:")

    print(
        confusion_matrix(
            y_val,
            predictions,
            labels=labels
        )
    )

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
        "Macro F1": macro_f1
    }


# ==========================================
# 1. DUMMY CLASSIFIER
# ==========================================

dummy_model = DummyClassifier(
    strategy="most_frequent"
)

dummy_model.fit(X_train, y_train)

dummy_results = evaluate_model(
    dummy_model,
    X_validation,
    y_validation,
    "Dummy Classifier"
)


# ==========================================
# 2. STANDARD LOGISTIC REGRESSION
# ==========================================

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("logisticregression", LogisticRegression(
        max_iter=1000
    ))
])

logistic_model.fit(X_train, y_train)

logistic_results = evaluate_model(
    logistic_model,
    X_validation,
    y_validation,
    "Standard Logistic Regression"
)


# ==========================================
# LOGISTIC REGRESSION COEFFICIENTS
# ==========================================

coefficients = (
    logistic_model
    .named_steps["logisticregression"]
    .coef_
)

coefficient_df = pd.DataFrame(
    coefficients.T,
    index=feature_columns,
    columns=logistic_model
    .named_steps["logisticregression"]
    .classes_
)

print("\nLOGISTIC REGRESSION COEFFICIENTS")
print("--------------------------------")

print(coefficient_df.to_string())


# ==========================================
# LOGISTIC REGRESSION CLASS PROBABILITIES
# ==========================================

probabilities = logistic_model.predict_proba(X_validation)

print("\nLOGISTIC REGRESSION CLASS PROBABILITIES")
print("---------------------------------------")

print("Class order:")
print(logistic_model.classes_)

probability_df = pd.DataFrame(
    probabilities,
    columns=logistic_model.classes_
)

print("\nFirst 10 validation predictions:")
print(probability_df.head(10).to_string(index=False))

print("\nAverage predicted probability per class:")

print(
    probability_df.mean().to_string()
)

print("\nActual validation outcome distribution:")

print(
    y_validation.value_counts(normalize=True)
    .reindex(logistic_model.classes_)
    .to_string()
)


# ==========================================
# DRAW PROBABILITY ANALYSIS
# ==========================================

validation_probability_analysis = probability_df.copy()

validation_probability_analysis["actual_outcome"] = (
    y_validation.reset_index(drop=True)
)

print("\nDRAW PROBABILITY BY ACTUAL OUTCOME")
print("----------------------------------")

print(
    validation_probability_analysis
    .groupby("actual_outcome")["draw"]
    .agg(["count", "mean", "median"])
    .to_string()
)


# ==========================================
# 3. BALANCED LOGISTIC REGRESSION
# ==========================================

balanced_logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("logisticregression", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

balanced_logistic_model.fit(X_train, y_train)

balanced_logistic_results = evaluate_model(
    balanced_logistic_model,
    X_validation,
    y_validation,
    "Balanced Logistic Regression"
)


# ==========================================
# 4. RANDOM FOREST
# ==========================================

random_forest_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

random_forest_model.fit(X_train, y_train)

random_forest_results = evaluate_model(
    random_forest_model,
    X_validation,
    y_validation,
    "Random Forest"
)


# ==========================================
# RANDOM FOREST FEATURE IMPORTANCE
# ==========================================

importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": random_forest_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print("\nRANDOM FOREST FEATURE IMPORTANCE")
print("--------------------------------")

print(importance_df.to_string(index=False))


# ==========================================
# PERMUTATION FEATURE IMPORTANCE
# ==========================================

permutation_result = permutation_importance(
    random_forest_model,
    X_validation,
    y_validation,
    n_repeats=30,
    random_state=42,
    scoring="balanced_accuracy",
    n_jobs=-1
)

permutation_df = pd.DataFrame({
    "feature": feature_columns,
    "importance_mean": permutation_result.importances_mean,
    "importance_std": permutation_result.importances_std
})

permutation_df = permutation_df.sort_values(
    by="importance_mean",
    ascending=False
)

print("\nPERMUTATION FEATURE IMPORTANCE")
print("--------------------------------")

print(permutation_df.to_string(index=False))


# ==========================================
# MODEL COMPARISON
# ==========================================

results = [
    dummy_results,
    logistic_results,
    balanced_logistic_results,
    random_forest_results
]

comparison_df = pd.DataFrame(results)

comparison_df = comparison_df.sort_values(
    by="Macro F1",
    ascending=False
)

print("\nMODEL COMPARISON")
print("----------------")

print(comparison_df.to_string(index=False))


print("\nMODEL TRAINING AND EVALUATION COMPLETED!")
