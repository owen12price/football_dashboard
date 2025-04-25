import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from itertools import combinations
from prediction import predict_outcome

st.set_page_config(
    page_title="Football Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------- Helper -------------------
def calculate_form(team, date, form_history):
    date = pd.to_datetime(date)
    recent = [x[1] for x in form_history[team] if x[0] < date][-5:]
    return np.mean(recent) if recent else 1.5

# ------------------- Load and Prepare -------------------
@st.cache_data

def load_data(uploaded_file):
    from prediction import load_data as raw_loader, build_training_data, train_model

    if uploaded_file is None:
        st.warning("📂 Please upload a match dataset (.xlsx) using the sidebar.")
        st.stop()

    # Correctly load uploaded file using prediction.py's loader
    df = raw_loader(uploaded_file)

    matches, teams, avg_xga, form_history, elo = build_training_data(df)
    features = ['elo_diff', 'form_diff', 'status_diff', 'xGA_diff']
    model, scaler = train_model(matches, features)

    # Calculate current standings
    standings = pd.DataFrame(0, index=list(teams), columns=['Pts', 'W', 'D', 'L', 'GF', 'GA'])
    for _, row in matches.iterrows():
        h, a = row['squadName_home'], row['squadName_away']
        h_goals, a_goals = row['GOALS_home'], row['GOALS_away']
        standings.loc[h, 'GF'] += h_goals
        standings.loc[a, 'GF'] += a_goals
        standings.loc[h, 'GA'] += a_goals
        standings.loc[a, 'GA'] += h_goals
        if row['outcome'] == 1.0:
            standings.loc[h, ['Pts', 'W']] += [3, 1]
            standings.loc[a, 'L'] += 1
        elif row['outcome'] == 0.5:
            standings.loc[[h, a], ['Pts', 'D']] += [1, 1]
        else:
            standings.loc[a, ['Pts', 'W']] += [3, 1]
            standings.loc[h, 'L'] += 1

    return {
        'matches': matches,
        'teams': list(teams),
        'form_history': form_history,
        'elo_dict': elo,
        'avg_xga': avg_xga,
        'standings': standings.sort_values('Pts', ascending=False)
    }

# ------------------- Load File -------------------
uploaded_file = st.sidebar.file_uploader("Upload Match Data", type=["xlsx"])
data = load_data(uploaded_file)

# ------------------- Sidebar + Header -------------------
selected_team = st.sidebar.selectbox("Select Team", data['teams'], index=0)
st.title(f"⚽ {selected_team} Performance Dashboard")

# ------------------- Team Stats -------------------
st.metric("Current ELO Rating", f"{int(data['elo_dict'][selected_team])}")
form_value = calculate_form(selected_team, pd.Timestamp.now(), data['form_history'])
st.metric("Recent Form (last 5)", f"{form_value:.2f}")

# ------------------- Cumulative Points -------------------
st.subheader("📈 Cumulative Points")
team_matches = data['matches'][
    (data['matches']['squadName_home'] == selected_team) |
    (data['matches']['squadName_away'] == selected_team)
].sort_values('dateTime_home')

points, dates, total = [], [], 0
for m in team_matches.itertuples():
    if m.squadName_home == selected_team:
        total += 3 if m.outcome == 1.0 else 1 if m.outcome == 0.5 else 0
    elif m.squadName_away == selected_team:
        total += 3 if m.outcome == 0.0 else 1 if m.outcome == 0.5 else 0
    points.append(total)
    dates.append(m.dateTime_home)

fig_points = px.line(x=dates, y=points, labels={'x': 'Date', 'y': 'Points'}, title='Cumulative Points')
st.plotly_chart(fig_points, use_container_width=True)

# ------------------- xG Trend -------------------
st.subheader("📊 xG vs xGA Trend")
xg_for, xg_against, xg_dates = [], [], []
for m in team_matches.itertuples():
    if m.squadName_home == selected_team:
        xg_for.append(m.SHOT_XG_home)
        xg_against.append(m.SHOT_XG_away)
    else:
        xg_for.append(m.SHOT_XG_away)
        xg_against.append(m.SHOT_XG_home)
    xg_dates.append(m.dateTime_home)

fig_xg = go.Figure([
    go.Scatter(x=xg_dates, y=xg_for, name='xG For'),
    go.Scatter(x=xg_dates, y=xg_against, name='xG Against')
])
fig_xg.update_layout(title='Expected Goals (xG vs xGA)', xaxis_title='Date', yaxis_title='Goals')
st.plotly_chart(fig_xg, use_container_width=True)

# ------------------- League Position -------------------
st.subheader("📉 League Position Over Time")
pos_history = []
for date in sorted(data['matches']['dateTime_home'].unique()):
    partial = data['matches'][data['matches']['dateTime_home'] <= date]
    pts = {team: 0 for team in data['teams']}
    for m in partial.itertuples():
        if m.outcome == 1.0:
            pts[m.squadName_home] += 3
        elif m.outcome == 0.5:
            pts[m.squadName_home] += 1
            pts[m.squadName_away] += 1
        else:
            pts[m.squadName_away] += 3
    sorted_teams = sorted(pts.items(), key=lambda x: -x[1])
    rank = [team for team, _ in sorted_teams].index(selected_team) + 1
    pos_history.append((date, rank))

pdts, pranks = zip(*pos_history)
fig_rank = px.line(x=pdts, y=pranks, labels={'x': 'Date', 'y': 'Position'}, title='League Position Over Time')
fig_rank.update_yaxes(autorange='reversed')
st.plotly_chart(fig_rank, use_container_width=True)

# ------------------- Predict Any Match -------------------
st.subheader("🔮 Predict Match Outcome")
home_team = st.selectbox("Home Team", data['teams'], key='home')
away_team = st.selectbox("Away Team", [t for t in data['teams'] if t != home_team], key='away')

if home_team and away_team:
    elo_diff = data['elo_dict'][home_team] - data['elo_dict'][away_team]
    form_diff = calculate_form(home_team, pd.Timestamp.now(), data['form_history']) - \
                calculate_form(away_team, pd.Timestamp.now(), data['form_history'])

    home_status = 1.0 if home_team in data['standings'].head(4).index or home_team in data['standings'].tail(3).index else 0.5
    away_status = 1.0 if away_team in data['standings'].head(4).index or away_team in data['standings'].tail(3).index else 0.5
    status_diff = home_status - away_status

    xGA_diff = data['avg_xga'][away_team] - data['avg_xga'][home_team]
    probs = predict_outcome(elo_diff, form_diff, status_diff, xGA_diff)

    col1, col2, col3 = st.columns(3)
    col1.metric("Home Win", f"{probs[2]*100:.1f}%")
    col2.metric("Draw", f"{probs[1]*100:.1f}%")
    col3.metric("Away Win", f"{probs[0]*100:.1f}%")

    fig_pred = go.Figure(go.Bar(x=["Home Win", "Draw", "Away Win"], y=[probs[2]*100, probs[1]*100, probs[0]*100]))
    fig_pred.update_layout(title=f"Predicted Outcome: {home_team} vs {away_team}", yaxis_title="Probability (%)")
    st.plotly_chart(fig_pred, use_container_width=True)
