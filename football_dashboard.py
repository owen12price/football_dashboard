import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Football Team Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and process data (cached)
@st.cache_data
def load_and_process_data():
    # Load data (replace with your actual path)
    try:
        df = pd.read_excel(r'C:\Users\Ryan\Downloads\squadMatchsums_L2_24_251021.xlsx')
    except:
        st.error("Could not load data file. Using sample data instead.")
        # Create sample data if real data not available
        dates = pd.date_range(start='2023-08-01', end='2024-01-01', freq='W')
        teams = ['Crewe Alexandra', 'Stockport County', 'Wrexham', 'Mansfield Town', 
                'Barrow', 'Crawley Town', 'AFC Wimbledon', 'Gillingham',
                'Swindon Town', 'Notts County', 'Morecambe', 'Accrington Stanley']
        
        data = []
        for i in range(len(dates)):
            home = teams[i % len(teams)]
            away = teams[(i + 1) % len(teams)]
            data.append({
                'matchId': i,
                'dateTime': dates[i],
                'squadName': home,
                'GOALS': np.random.randint(0, 3),
                'OPPONENT_GOALS': np.random.randint(0, 3),
                'SHOT_XG': round(np.random.uniform(0.5, 2.5), 2),
                'BYPASSED_OPPONENTS': np.random.randint(50, 150),
                'SUCCESSFUL_PASSES': np.random.randint(200, 500),
                'BALL_WIN_NUMBER': np.random.randint(40, 100)
            })
            data.append({
                'matchId': i,
                'dateTime': dates[i],
                'squadName': away,
                'GOALS': data[-1]['OPPONENT_GOALS'],
                'OPPONENT_GOALS': data[-1]['GOALS'],
                'SHOT_XG': round(np.random.uniform(0.5, 2.5), 2),
                'BYPASSED_OPPONENTS': np.random.randint(50, 150),
                'SUCCESSFUL_PASSES': np.random.randint(200, 500),
                'BALL_WIN_NUMBER': np.random.randint(40, 100)
            })
        df = pd.DataFrame(data)
    
    # Process data as in original code
    columns_to_keep = [
        'matchId', 'dateTime', 'squadName', 'GOALS', 'OPPONENT_GOALS',
        'SHOT_XG', 'BYPASSED_OPPONENTS', 'SUCCESSFUL_PASSES', 'BALL_WIN_NUMBER'
    ]
    
    df_clean = df[columns_to_keep].dropna().copy()
    df_clean['squadName'] = df_clean['squadName'].str.strip()
    df_clean['dateTime'] = pd.to_datetime(df_clean['dateTime']).dt.tz_localize(None)
    
    # Pair teams in matches
    home_teams = df_clean.iloc[::2].reset_index(drop=True)
    away_teams = df_clean.iloc[1::2].reset_index(drop=True)
    
    matches = pd.merge(home_teams, away_teams, on='matchId', suffixes=('_home', '_away'))
    
    # Add outcomes
    matches['outcome'] = np.select(
        condlist=[
            matches['GOALS_home'] > matches['GOALS_away'],  # Home win condition
            matches['GOALS_home'] == matches['GOALS_away']  # Draw condition
        ],
        choicelist=[
            1.0,  # Value if home wins
            0.5   # Value if draw
        ],
        default=0.0  # Value if away wins (only one default allowed)
    )
    
    # Elo calculation
    teams = set(matches['squadName_home']).union(set(matches['squadName_away']))
    elo = {team: 1500 for team in teams}
    form_history = {team: [] for team in teams}
    
    matches = matches.sort_values('dateTime_home').reset_index(drop=True)
    matches['home_elo'] = 0.0
    matches['away_elo'] = 0.0
    
    def update_elo(home_elo, away_elo, home_goals, away_goals, K=30):
        exp_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        act_home = 1 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0
        return home_elo + K * (act_home - exp_home), away_elo + K * ((1 - act_home) - (1 - exp_home))
    
    for i, row in matches.iterrows():
        h, a = row['squadName_home'], row['squadName_away']
        matches.at[i, 'home_elo'] = elo[h]
        matches.at[i, 'away_elo'] = elo[a]
        elo[h], elo[a] = update_elo(elo[h], elo[a], row['GOALS_home'], row['GOALS_away'])
        
        form_history[h].append((row['dateTime_home'], 3.0 if row['outcome'] == 1.0 else (1.0 if row['outcome'] == 0.5 else 0.0)))
        form_history[a].append((row['dateTime_home'], 3.0 if row['outcome'] == 0.0 else (1.0 if row['outcome'] == 0.5 else 0.0)))
    
    # Calculate form
    def calculate_form(team, date):
        date = pd.to_datetime(date)
        recent = [x[1] for x in form_history[team] if x[0] < date][-5:]
        return np.mean(recent) if recent else 1.5
    
    matches['form_home'] = matches.apply(lambda r: calculate_form(r['squadName_home'], r['dateTime_home']), axis=1)
    matches['form_away'] = matches.apply(lambda r: calculate_form(r['squadName_away'], r['dateTime_home']), axis=1)
    
    # Match status
    points = {team: 0 for team in teams}
    match_status_home = []
    match_status_away = []
    
    for _, row in matches.iterrows():
        sorted_teams = sorted(points.items(), key=lambda x: -x[1])
        standings = {team: rank + 1 for rank, (team, _) in enumerate(sorted_teams)}
        
        h, a = row['squadName_home'], row['squadName_away']
        h_rank, a_rank = standings[h], standings[a]
        
        def match_importance(rank):
            if rank <= 4 or rank >= len(teams) - 3:
                return 1.0
            return 0.5
        
        match_status_home.append(match_importance(h_rank))
        match_status_away.append(match_importance(a_rank))
        
        points[h] += 3 if row['outcome'] == 1.0 else 1 if row['outcome'] == 0.5 else 0
        points[a] += 3 if row['outcome'] == 0.0 else 1 if row['outcome'] == 0.5 else 0
    
    matches['match_status_home'] = match_status_home
    matches['match_status_away'] = match_status_away
    
    # Feature engineering
    matches['elo_diff'] = matches['home_elo'] - matches['away_elo']
    matches['form_diff'] = matches['form_home'] - matches['form_away']
    matches['status_diff'] = matches['match_status_home'] - matches['match_status_away']
    
    # xGA feature
    xga_dict = {}
    for _, row in matches.iterrows():
        if row['squadName_home'] not in xga_dict:
            xga_dict[row['squadName_home']] = []
        if row['squadName_away'] not in xga_dict:
            xga_dict[row['squadName_away']] = []
        
        xga_dict[row['squadName_home']].append(row['SHOT_XG_home'])
        xga_dict[row['squadName_away']].append(row['SHOT_XG_away'])
    
    avg_xga = {team: np.mean(vals) for team, vals in xga_dict.items()}
    matches['xGA_home'] = matches['squadName_home'].map(avg_xga)
    matches['xGA_away'] = matches['squadName_away'].map(avg_xga)
    matches['xGA_diff'] = matches['xGA_away'] - matches['xGA_home']
    
    # Prepare training data
    features = ['elo_diff', 'form_diff', 'status_diff', 'xGA_diff']
    matches['label'] = matches['outcome'].apply(lambda x: 0 if x == 0.0 else (1 if x == 0.5 else 2))
    
    split_date = matches['dateTime_home'].quantile(0.8)
    train = matches[matches['dateTime_home'] < split_date]
    test = matches[matches['dateTime_home'] >= split_date]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features])
    X_test = scaler.transform(test[features])
    y_train = train['label']
    y_test = test['label']
    
    # Train model
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
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)
    
    # Current standings
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
            standings.loc[h, ['Pts', 'D']] += [1, 1]
            standings.loc[a, ['Pts', 'D']] += [1, 1]
        else:
            standings.loc[a, ['Pts', 'W']] += [3, 1]
            standings.loc[h, 'L'] += 1
    
    standings = standings.sort_values(by='Pts', ascending=False)
    
    # Simulate season
    num_simulations = 1000
    crewe_positions = []
    
    for _ in range(num_simulations):
        future_fixtures = pd.DataFrame(
            [(h, a) for h, a in product(teams, teams) if h != a],
            columns=['home', 'away']
        )
        
        future_fixtures['elo_diff'] = future_fixtures.apply(lambda x: elo[x['home']] - elo[x['away']], axis=1)
        future_fixtures['form_diff'] = future_fixtures.apply(lambda x: calculate_form(x['home'], pd.Timestamp.now()) - calculate_form(x['away'], pd.Timestamp.now()), axis=1)
        future_fixtures['status_diff'] = future_fixtures.apply(lambda x: 1.0 if x['home'] in standings.head(4).index or x['home'] in standings.tail(3).index else 0.5, axis=1) - \
                                        future_fixtures.apply(lambda x: 1.0 if x['away'] in standings.head(4).index or x['away'] in standings.tail(3).index else 0.5, axis=1)
        future_fixtures['xGA_diff'] = future_fixtures['away'].map(avg_xga) - future_fixtures['home'].map(avg_xga)
        
        X_future = scaler.transform(future_fixtures[features])
        preds = model.predict(X_future)
        
        simulated = pd.DataFrame(0, index=list(teams), columns=['Pts', 'W', 'D', 'L'])
        
        for idx, row in future_fixtures.iterrows():
            h, a = row['home'], row['away']
            outcome = preds[idx]
            if outcome == 2:
                simulated.loc[h, ['Pts', 'W']] += [3, 1]
                simulated.loc[a, 'L'] += 1
            elif outcome == 1:
                simulated.loc[h, ['Pts', 'D']] += [1, 1]
                simulated.loc[a, ['Pts', 'D']] += [1, 1]
            else:
                simulated.loc[a, ['Pts', 'W']] += [3, 1]
                simulated.loc[h, 'L'] += 1
        
        simulated_ranked = simulated.sort_values('Pts', ascending=False).reset_index()
        crewe_position = simulated_ranked[simulated_ranked['index'] == 'Crewe Alexandra'].index[0] + 1
        crewe_positions.append(crewe_position)
    
    # Calculate probabilities
    position_counts = pd.Series(crewe_positions).value_counts().sort_index()
    probabilities = (position_counts / num_simulations * 100).round(1)
    
    return {
        'matches': matches,
        'standings': standings,
        'probabilities': probabilities,
        'teams': list(teams),
        'elo_dict': elo,
        'form_history': form_history,
        'model': model,
        'scaler': scaler,
        'features': features,
        'avg_xga': avg_xga
    }

data = load_and_process_data()

# Sidebar controls
st.sidebar.header("Dashboard Controls")
selected_team = st.sidebar.selectbox("Select Team", data['teams'], index=data['teams'].index('Crewe Alexandra') if 'Crewe Alexandra' in data['teams'] else 0)
time_range = st.sidebar.selectbox("Time Period", ["Last 5 Games", "Last 10 Games", "Season to Date"])
show_advanced = st.sidebar.checkbox("Show Advanced Metrics")

# Dashboard Header
st.title(f"{selected_team} Performance Dashboard")
st.markdown("---")

# Key Metrics Row
st.header("Current Status")
col1, col2, col3, col4 = st.columns(4)

with col1:
    current_pos = data['standings'].index.get_loc(selected_team) + 1
    pos_change = "→"
    if len(data['matches']) > 10:
        prev_pos = data['matches'].sort_values('dateTime_home', ascending=False).iloc[:10]
        prev_standings = prev_pos.groupby('squadName_home')['outcome'].apply(lambda x: sum(3 if o == 1.0 else 1 if o == 0.5 else 0 for o in x))
        prev_pos = prev_standings.sort_values(ascending=False).index.get_loc(selected_team) + 1
        pos_change = "↑" if current_pos < prev_pos else "↓" if current_pos > prev_pos else "→"
    
    st.metric("League Position", f"{current_pos}th {pos_change}", 
              f"{len(data['teams'])-current_pos} pts from safety" if current_pos > len(data['teams'])-3 else f"{current_pos} pts from playoffs")

with col2:
    team_form = []
    for match in data['matches'].sort_values('dateTime_home', ascending=False).itertuples():
        if match.squadName_home == selected_team:
            team_form.append('W' if match.outcome == 1.0 else 'D' if match.outcome == 0.5 else 'L')
        elif match.squadName_away == selected_team:
            team_form.append('W' if match.outcome == 0.0 else 'D' if match.outcome == 0.5 else 'L')
        if len(team_form) >= 5:
            break
    st.metric("Current Form", " ".join(team_form[:5]))

with col3:
    goals_for = data['standings'].loc[selected_team, 'GF']
    goals_against = data['standings'].loc[selected_team, 'GA']
    gd = goals_for - goals_against
    st.metric("Goal Difference", f"+{gd}" if gd > 0 else gd,
              f"{goals_for} for, {goals_against} against")

with col4:
    elo_rating = data['elo_dict'][selected_team]
    league_avg_elo = np.mean(list(data['elo_dict'].values()))
    st.metric("Team Rating (ELO)", f"{int(elo_rating)}", 
              f"{'Above' if elo_rating > league_avg_elo else 'Below'} league average")

# Performance Charts Row
st.header("Performance Analysis")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    # Form over time chart
    form_data = []
    dates = []
    points = 0
    for match in data['matches'].sort_values('dateTime_home').itertuples():
        if match.squadName_home == selected_team:
            points += 3 if match.outcome == 1.0 else 1 if match.outcome == 0.5 else 0
            form_data.append(points)
            dates.append(match.dateTime_home)
        elif match.squadName_away == selected_team:
            points += 3 if match.outcome == 0.0 else 1 if match.outcome == 0.5 else 0
            form_data.append(points)
            dates.append(match.dateTime_home)
    
    fig = px.line(x=dates, y=form_data, 
                 title="Cumulative Points Over Time",
                 labels={'x': 'Date', 'y': 'Points'})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with chart_col2:
    # xG performance
    xg_data = []
    xga_data = []
    match_dates = []
    for match in data['matches'].sort_values('dateTime_home').itertuples():
        if match.squadName_home == selected_team:
            xg_data.append(match.SHOT_XG_home)
            xga_data.append(match.SHOT_XG_away)
            match_dates.append(match.dateTime_home)
        elif match.squadName_away == selected_team:
            xg_data.append(match.SHOT_XG_away)
            xga_data.append(match.SHOT_XG_home)
            match_dates.append(match.dateTime_home)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=match_dates, y=xg_data, name='xG For', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=match_dates, y=xga_data, name='xG Against', line=dict(color='red')))
    fig.update_layout(title="Expected Goals (xG) Trend",
                     height=400,
                     xaxis_title="Date",
                     yaxis_title="Expected Goals")
    st.plotly_chart(fig, use_container_width=True)

# Season Projections Row
st.header("Season Projections")
proj_col1, proj_col2 = st.columns(2)

with proj_col1:
    # Position probability chart
    fig = px.bar(data['probabilities'], 
                x=data['probabilities'].index,
                y=data['probabilities'].values,
                title="Probability of Final League Position",
                labels={'x': 'Position', 'y': 'Probability (%)'})
    
    # Highlight promotion/relegation zones
    promotion_cutoff = 3
    relegation_cutoff = len(data['teams']) - 3
    
    fig.add_vrect(x0=0.5, x1=promotion_cutoff + 0.5, 
                 fillcolor="green", opacity=0.1, line_width=0)
    fig.add_vrect(x0=relegation_cutoff + 0.5, x1=len(data['teams']) + 0.5,
                 fillcolor="red", opacity=0.1, line_width=0)
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with proj_col2:
    # Key probabilities
    promotion_prob = data['probabilities'].loc[1:3].sum()
    playoff_prob = data['probabilities'].loc[4:7].sum()
    relegation_prob = data['probabilities'].loc[len(data['teams'])-2:].sum()
    
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode="number",
        value=promotion_prob,
        number={'suffix': "%"},
        title={"text": "Promotion Probability (Top 3)"},
        domain={'row': 0, 'column': 0}))
    
    fig.add_trace(go.Indicator(
        mode="number",
        value=playoff_prob,
        number={'suffix': "%"},
        title={"text": "Playoff Probability (4th-7th)"},
        domain={'row': 0, 'column': 1}))
    
    fig.add_trace(go.Indicator(
        mode="number",
        value=relegation_prob,
        number={'suffix': "%"},
        title={"text": "Relegation Probability (Bottom 2)"},
        domain={'row': 1, 'column': 0}))
    
    fig.update_layout(
        grid={'rows': 2, 'columns': 2, 'pattern': "independent"},
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# Upcoming Fixtures
st.header("Upcoming Fixtures")

# Simulate next 5 matches
all_teams = data['teams']
future_opponents = [t for t in all_teams if t != selected_team][:5]

predictions = []
for opponent in future_opponents:
    # Create feature vector
    elo_diff = data['elo_dict'][selected_team] - data['elo_dict'][opponent]
    form_diff = calculate_form(selected_team, pd.Timestamp.now()) - calculate_form(opponent, pd.Timestamp.now())
    
    # Determine match status (simplified)
    home_status = 1.0 if selected_team in data['standings'].head(4).index or selected_team in data['standings'].tail(3).index else 0.5
    away_status = 1.0 if opponent in data['standings'].head(4).index or opponent in data['standings'].tail(3).index else 0.5
    status_diff = home_status - away_status
    
    xGA_diff = data['avg_xga'][opponent] - data['avg_xga'][selected_team]
    
    features = np.array([[elo_diff, form_diff, status_diff, xGA_diff]])
    features_scaled = data['scaler'].transform(features)
    
    # Predict
    probs = model.predict_proba(features_scaled)[0]
    predictions.append({
        'opponent': opponent,
        'win_prob': probs[2] * 100,
        'draw_prob': probs[1] * 100,
        'loss_prob': probs[0] * 100
    })

# Display predictions
for pred in predictions:
    with st.expander(f"vs {pred['opponent']}"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Win Probability", f"{pred['win_prob']:.1f}%")
        col2.metric("Draw Probability", f"{pred['draw_prob']:.1f}%")
        col3.metric("Loss Probability", f"{pred['loss_prob']:.1f}%")
        
        fig = go.Figure(go.Bar(
            x=['Win', 'Draw', 'Loss'],
            y=[pred['win_prob'], pred['draw_prob'], pred['loss_prob']],
            marker_color=['green', 'gray', 'red']
        ))
        fig.update_layout(title="Outcome Probabilities",
                         yaxis_title="Probability (%)",
                         height=300)
        st.plotly_chart(fig, use_container_width=True)

# Advanced Metrics (conditional)
if show_advanced:
    st.header("Advanced Metrics")
    adv_col1, adv_col2 = st.columns(2)
    
    with adv_col1:
        # Key performance indicators
        team_matches = data['matches'][(data['matches']['squadName_home'] == selected_team) | 
                                      (data['matches']['squadName_away'] == selected_team)]
        
        home_matches = team_matches[team_matches['squadName_home'] == selected_team]
        away_matches = team_matches[team_matches['squadName_away'] == selected_team]
        
        avg_xg_for = np.mean(home_matches['SHOT_XG_home'].tolist() + away_matches['SHOT_XG_away'].tolist())
        avg_xg_against = np.mean(home_matches['SHOT_XG_away'].tolist() + away_matches['SHOT_XG_home'].tolist())
        avg_possession = np.mean(home_matches['SUCCESSFUL_PASSES_home'].tolist() + away_matches['SUCCESSFUL_PASSES_away'].tolist())
        avg_def_actions = np.mean(home_matches['BALL_WIN_NUMBER_home'].tolist() + away_matches['BALL_WIN_NUMBER_away'].tolist())
        
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="number",
            value=avg_xg_for,
            title={"text": "Avg xG For"},
            domain={'row': 0, 'column': 0}))
        
        fig.add_trace(go.Indicator(
            mode="number",
            value=avg_xg_against,
            title={"text": "Avg xG Against"},
            domain={'row': 0, 'column': 1}))
        
        fig.add_trace(go.Indicator(
            mode="number",
            value=avg_possession,
            title={"text": "Avg Successful Passes"},
            domain={'row': 1, 'column': 0}))
        
        fig.add_trace(go.Indicator(
            mode="number",
            value=avg_def_actions,
            title={"text": "Avg Defensive Actions"},
            domain={'row': 1, 'column': 1}))
        
        fig.update_layout(
            grid={'rows': 2, 'columns': 2, 'pattern': "independent"},
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with adv_col2:
        # Performance by game state
        st.write("Performance by Game State:")
        
        # Calculate points per game in different scenarios
        leading = team_matches[((team_matches['squadName_home'] == selected_team) & 
                              (team_matches['GOALS_home'] > team_matches['GOALS_away'])) |
                             ((team_matches['squadName_away'] == selected_team) & 
                              (team_matches['GOALS_away'] > team_matches['GOALS_home']))]
        
        drawing = team_matches[team_matches['GOALS_home'] == team_matches['GOALS_away']]
        
        trailing = team_matches[((team_matches['squadName_home'] == selected_team) & 
                               (team_matches['GOALS_home'] < team_matches['GOALS_away'])) |
                              ((team_matches['squadName_away'] == selected_team) & 
                               (team_matches['GOALS_away'] < team_matches['GOALS_home']))]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['When Leading', 'When Drawing', 'When Trailing'],
            y=[len(leading), len(drawing), len(trailing)],
            name='Number of Games',
            marker_color=['green', 'gray', 'red']
        ))
        
        fig.update_layout(
            title="Performance by Game State",
            yaxis_title="Number of Games",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
**Dashboard Features:**
- Real-time performance metrics
- Predictive analytics for upcoming matches
- Season-long projections
- Advanced team statistics
""")
