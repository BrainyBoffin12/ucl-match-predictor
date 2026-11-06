# UEFA Champions League Match Predictor

A Machine Learning project that predicts the outcome of UEFA Champions League football matches using historical match data and team performance statistics.

The model predicts one of three possible outcomes:

- Home Win
- Draw
- Away Win

## Project Overview

This project uses historical Champions League matches to build a classification model capable of predicting match outcomes.

The predictions are based on each team's recent performance, including points earned, goals scored, goals conceded, and home/away form over the previous five matches.

## Dataset

- **Source:** OpenFootball
- **Total matches:** 1,997
- **Seasons covered:** 2011–2012 to 2025–2026
- **Number of teams:** 108
- **Last available match:** May 30, 2026

The original match data is parsed, cleaned, and prepared before feature engineering.

## Machine Learning Approach

**Algorithm:** Logistic Regression

**Preprocessing:** StandardScaler

**Features:** The model uses 15 numerical features, including:

- Points earned in the last five matches.
- Goals scored and conceded.
- Goal difference.
- Recent home performance.
- Recent away performance.

Features are calculated using previous matches only, avoiding the use of future match results.

A draw probability threshold of 0.225 is used to help the model identify draws.

## Model Evaluation

The model was evaluated using a chronological test set containing 378 matches from the 2024–2025 and 2025–2026 seasons.

| Metric | Result |
|---|---:|
| Accuracy | 50.00% |
| Balanced Accuracy | 44.68% |
| Macro F1-Score | 44.74% |
| Correct Predictions | 189 / 378 |

The model also attempts to identify draws, which are generally difficult to predict.

The draw threshold improves Macro F1 compared with the standard argmax approach, but reduces overall accuracy. The results show that predicting football match outcomes remains challenging.

## Project Structure

```text
ucl-match-predictor/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── ucl_model.joblib
│
├── notebooks/
│
├── reports/
│   ├── figures/
│   └── final_test/
│
├── src/
│   ├── parse_data.py
│   ├── team_mapping.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── final_test.py
│   ├── save_model.py
│   └── predict.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/BrainyBoffin12/ucl-match-predictor.git
cd ucl-match-predictor
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Usage

Run the prediction script:

```bash
python -m src.predict
```

The script loads the saved model and asks for the home and away teams.

Example:

```text
Home team: Real Madrid
Away team: Manchester City

Probabilities:
Home win: 72.39%
Draw:     12.45%
Away win: 15.16%

Predicted outcome:
Real Madrid wins!
```

*Example prediction from the trained model.*

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Joblib

## Limitations

This is an educational Machine Learning project. Predictions are based on historical data and recent team performance.

The model does not consider live information such as injuries, lineups, betting odds, or other real-time factors.

Its predictions should not be interpreted as guaranteed match results.

## Author

Developed as a personal Machine Learning project to practice data preprocessing, feature engineering, model training, and evaluation.