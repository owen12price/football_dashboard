import pandas as pd
import numpy as np
import joblib

model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

def load_data(path):
    if path is None:
        raise ValueError("No file uploaded")
    df = pd.read_excel(path)
    df['squadName'] = df['squadName'].str.strip()
    df['dateTime'] = pd.to_datetime(df['dateTime']).dt.tz_localize(None)
    return df

def update_elo(home_elo, away_elo, home_goals, away_goals, K=30):
    exp_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
    act_home = 1 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0
    return home_elo + K * (act_home - exp_home), away_elo + K * ((1 - act_home) - (1 - exp_home))

def build_training_data(df):
    home = df.iloc[::2].reset_index(drop=True)
    away = df.iloc[1::2].reset_index(drop=True)
    matches = pd.merge(home, away, on='matchId', suffixes=('_home', '_away'))

    matches['outcome'] = np.select(
        [matches['GOALS_home'] > matches['GOALS_away'], matches['GOALS_home'] == matches['GOALS_away']],
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

        form_history[h].append((row['dateTime_home'], 3 if row['outcome'] == 1.0 else 1 if row['outcome'] == 0.5 else 0))
        form_history[a].append((row['dateTime_home'], 3 if row['outcome'] == 0.0 else 1 if row['outcome'] == 0.5 else 0))

        xga_dict[h].append(row['SHOT_XG_home'])
        xga_dict[a].append(row['SHOT_XG_away'])

    def calc_form(team, date):
        recent = [x[1] for x in form_history[team] if x[0] < date][-5:]
        return np.mean(recent) if recent else 1.5

    matches['form_home'] = matches.apply(lambda r: calc_form(r['squadName_home'], r['dateTime_home']), axis=1)
    matches['form_away'] = matches.apply(lambda r: calc_form(r['squadName_away'], r['dateTime_home']), axis=1)

    points = {team: 0 for team in teams}
    match_status_home, match_status_away = [], []
    for _, row in matches.iterrows():
        sorted_teams = sorted(points.items(), key=lambda x: -x[1])
        ranks = {team: rank + 1 for rank, (team, _) in enumerate(sorted_teams)}
        h, a = row['squadName_home'], row['squadName_away']
        match_status_home.append(1.0 if ranks[h] <= 4 or ranks[h] >= len(teams) - 3 else 0.5)
        match_status_away.append(1.0 if ranks[a] <= 4 or ranks[a] >= len(teams) - 3 else 0.5)
        points[h] += 3 if row['outcome'] == 1.0 else 1 if row['outcome'] == 0.5 else 0
        points[a] += 3 if row['outcome'] == 0.0 else 1 if row['outcome'] == 0.5 else 0

    matches['match_status_home'] = match_status_home
    matches['match_status_away'] = match_status_away

    avg_xga = {team: np.mean(xga_dict[team]) for team in xga_dict}
    matches['xGA_home'] = matches['squadName_home'].map(avg_xga)
    matches['xGA_away'] = matches['squadName_away'].map(avg_xga)

    matches['elo_diff'] = matches['home_elo'] - matches['away_elo']
    matches['form_diff'] = matches['form_home'] - matches['form_away']
    matches['status_diff'] = matches['match_status_home'] - matches['match_status_away']
    matches['xGA_diff'] = matches['xGA_away'] - matches['xGA_home']

    return matches, teams, avg_xga, form_history, elo

def predict_outcome(elo_diff, form_diff, status_diff, xGA_diff):
    X = np.array([[elo_diff, form_diff, status_diff, xGA_diff]])
    X_scaled = scaler.transform(X)
    return model.predict_proba(X_scaled)[0]
