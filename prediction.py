import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from itertools import product
import joblib
import os

# ----------------------------- #
#        Data Preparation       #
# ----------------------------- #

def load_data(path):
    df = pd.read_excel(path)
    df = df.dropna()
    df['squadName'] = df['squadName'].str.strip()
    df['dateTime'] = pd.to_datetime(df['dateTime']).dt.tz_localize(None)
    return df

def calculate_form(team, date, form_history):
    date = pd.to_datetime(date)
    recent = [x[1] for x in form_history[team] if x[0] < date][-5:]
    return np.mean(recent) if recent else 1.5

def update_elo(home_elo, away_elo, home_goals, away_goals, K=30):
    exp_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
    act_home = 1 if home_goals > away_goals else 0.5 if home_goals == home_goals else 0
    return home_elo + K * (act_home - exp_home), away_elo + K * ((1 - act_home) - (1 - exp_home))

# ----------------------------- #
#         Model Training        #
# ----------------------------- #

def build_training_data(df):
    # Match pairing
    home = df.iloc[::2].reset_index(drop=True)
    away = df.iloc[1::2].reset_index(drop=True)
    matches = pd.merge(home, away, on='matchId', suffixes=('_home', '_away'))

    # Outcome
    matches['outcome'] = np.select(
        [matches['GOALS_home'] > matches['GOALS_away'],
         matches['GOALS_home'] == matches['GOALS_away']],
        [1.0, 0.5], default=0.0
    )

    teams = set(matches['squadName_home']).union(matches['squadName_away'])
    elo = {team: 1500 for team in teams}
    form_history = {team: [] for team in teams}
    xga_dict = {team: [] for team in teams}

    matches = matches.sort_values('dateTime_home').reset_index(drop=True)
    matches['home_elo'] = 0.0
    matches['away_elo'] = 0.0

    for i, row in matches.iterrows():
        h, a = row['squadName_home'], row['squadName_away']
        matches.at[i, 'home_elo'] = elo[h]
        matches.at[i, 'away_elo'] = elo[a]
        elo[h], elo[a] = update_elo(elo[h], elo[a], row['GOALS_home'], row['GOALS_away'])

        form_history[h].append((row['dateTime_home'], 3.0 if row['outcome'] == 1.0 else (1.0 if row['outcome'] == 0.5 else 0.0)))
        form_history[a].append((row['dateTime_home'], 3.0 if row['outcome'] == 0.0 else (1.0 if row['outcome'] == 0.5 else 0.0)))

        xga_dict[h].append(row['SHOT_XG_home'])
        xga_dict[a].append(row['SHOT_XG_away'])

    avg_xga = {team: np.mean(values) for team, values in xga_dict.items()}

    matches['form_home'] = matches.apply(lambda r: calculate_form(r['squadName_home'], r['dateTime_home'], form_history), axis=1)
    matches['form_away'] = matches.apply(lambda r: calculate_form(r['squadName_away'], r['dateTime_home'], form_history), axis=1)

    points = {team: 0 for team in teams}
    match_status_home, match_status_away = [], []

    for _, row in matches.iterrows():
        sorted_teams = sorted(points.items(), key=lambda x: -x[1])
        standings = {team: rank + 1 for rank, (team, _) in enumerate(sorted_teams)}
        h, a = row['squadName_home'], row['squadName_away']
        h_rank, a_rank = standings[h], standings[a]

        match_status_home.append(1.0 if h_rank <= 4 or h_rank >= len(teams) - 3 else 0.5)
        match_status_away.append(1.0 if a_rank <= 4 or a_rank >= len(teams) - 3 else 0.5)

        points[h] += 3 if row['outcome'] == 1.0 else 1 if row['outcome'] == 0.5 else 0
        points[a] += 3 if row['outcome'] == 0.0 else 1 if row['outcome'] == 0.5 else 0

    matches['match_status_home'] = match_status_home
    matches['match_status_away'] = match_status_away

    matches['elo_diff'] = matches['home_elo'] - matches['away_elo']
    matches['form_diff'] = matches['form_home'] - matches['form_away']
    matches['status_diff'] = matches['match_status_home'] - matches['match_status_away']
    matches['xGA_home'] = matches['squadName_home'].map(avg_xga)
    matches['xGA_away'] = matches['squadName_away'].map(avg_xga)
    matches['xGA_diff'] = matches['xGA_away'] - matches['xGA_home']

    matches['label'] = matches['outcome'].apply(lambda x: 0 if x == 0.0 else (1 if x == 0.5 else 2))

    return matches, list(teams), avg_xga, form_history, elo

def train_model(matches, features):
    split_date = matches['dateTime_home'].quantile(0.8)
    train = matches[matches['dateTime_home'] < split_date]
    X_train = train[features]
    y_train = train['label']

    # Check for NaNs
    if X_train.isnull().any().any():
        raise ValueError("Training features contain NaN values.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        max_depth=4,
        learning_rate=0.05,
        n_estimators=150,
        reg_alpha=0.4,
        reg_lambda=0.4,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric='mlogloss'
    )
    model.fit(X_scaled, y_train)

    # Save model
    joblib.dump(model, "model.joblib")
    joblib.dump(scaler, "scaler.joblib")

    return model, scaler

# ----------------------------- #
#       Prediction Utility      #
# ----------------------------- #

def predict_outcome(elo_diff, form_diff, status_diff, xGA_diff):
    if not os.path.exists("model.joblib") or not os.path.exists("scaler.joblib"):
        raise FileNotFoundError("Model and scaler files not found. Run training first.")

    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")

    features = np.array([[elo_diff, form_diff, status_diff, xGA_diff]])
    scaled = scaler.transform(features)

    return model.predict_proba(scaled)[0]
