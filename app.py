import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.title("College Baseball Roster Analysis")

# Reset button
st.sidebar.header("Filters")
if st.sidebar.button("Reset All Filters"):
    st.session_state.clear()
    st.rerun()

@st.cache_data
def load_data():
    # Fixed encoding + suppress low_memory warning
    try:
        pitchers = pd.read_csv('novick_college_pitching_history_all.csv', encoding='latin1', low_memory=False)
    except UnicodeDecodeError:
        pitchers = pd.read_csv('novick_college_pitching_history_all.csv', encoding='cp1252', low_memory=False)

    try:
        hitters = pd.read_csv('novick_college_batting_history_all.csv', encoding='latin1', low_memory=False)
    except UnicodeDecodeError:
        hitters = pd.read_csv('novick_college_batting_history_all.csv', encoding='cp1252', low_memory=False)

    pitchers['role'] = 'Pitcher'
    hitters['role'] = 'Hitter'
    df = pd.concat([pitchers, hitters], ignore_index=True, sort=False)

    # Critical fixes for sorting/multiselect
    df['LeagueAbbr'] = df['LeagueAbbr'].astype(str).replace('nan', '')  # Fix mixed types + NaN
    df['teamName'] = df['teamName'].astype(str).replace('nan', 'Unknown Team')
    df['leagueName'] = df['leagueName'].astype(str).replace('nan', 'Unknown Conference')

    # State
    df['state'] = df['hsplace'].str.split(',').str[-1].str.strip().str.upper()
    us_states = ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']
    df = df[df['state'].isin(us_states)]

    # Draft cleanup
    df['draft_year'] = pd.to_numeric(df['draft_year'], errors='coerce')
    df['draft_Round'] = pd.to_numeric(df['draft_Round'], errors='coerce').fillna(0)
    df['is_drafted'] = df['draft_year'].notna()

    # Region mapping (unchanged)
    region_map = {
        'East': ['KY','OH','PA','TN','WV'],
        'Mid Atlantic': ['DE','MD','NJ','NY','VA'],
        'Midwest I': ['IL','IN','IA','KS','MI','MN','MO','NE','ND','SD','WI'],
        'Midwest II': ['AR','OK','TX'],
        'New England': ['CT','ME','MA','NH','RI','VT'],
        'South': ['AL','FL','GA','LA','MS','NC','SC'],
        'West': ['AK','AZ','CA','CO','HI','ID','MT','NV','NM','OR','UT','WA','WY'],
    }
    def get_region(s):
        for r, states in region_map.items():
            if s in states:
                return r
        return 'Other'
    df['region'] = df['state'].apply(get_region)

    # T90s and T90/PA — only for hitters
    df['T90s'] = np.nan
    df['T90/PA'] = np.nan
    hitter_mask = df['role'] == 'Hitter'
    if hitter_mask.any():
        df.loc[hitter_mask, 'Singles'] = (df.loc[hitter_mask, 'H'] - df.loc[hitter_mask, 'Dbl'] - df.loc[hitter_mask, 'Tpl'] - df.loc[hitter_mask, 'HR']).fillna(0)
        df.loc[hitter_mask, 'TotalBases'] = (df.loc[hitter_mask, 'Singles'] + 2*df.loc[hitter_mask, 'Dbl'].fillna(0) + 3*df.loc[hitter_mask, 'Tpl'].fillna(0) + 4*df.loc[hitter_mask, 'HR'].fillna(0))
        df.loc[hitter_mask, 'T90s'] = (df.loc[hitter_mask, 'TotalBases'] + df.loc[hitter_mask, 'SB'].fillna(0) + df.loc[hitter_mask, 'BB'].fillna(0) + df.loc[hitter_mask, 'HBP'].fillna(0))
        df.loc[hitter_mask, 'PA'] = (df.loc[hitter_mask, 'AB'].fillna(0) + df.loc[hitter_mask, 'BB'].fillna(0) + df.loc[hitter_mask, 'HBP'].fillna(0) + df.loc[hitter_mask, 'SF'].fillna(0) + df.loc[hitter_mask, 'SH'].fillna(0))
        df.loc[hitter_mask, 'T90/PA'] = df.loc[hitter_mask, 'T90s'] / df.loc[hitter_mask, 'PA'].replace(0, np.nan)

    # Clean Bats/Throws/Position
    df['Bats'] = df['Bats'].str.upper().replace('B', 'S')
    df['Throws'] = df['Throws'].str.upper()
    df['posit'] = df['posit'].str.upper().str.strip()

    # Miami → Miami-Ohio fix
    df.loc[(df['teamName'] == 'Miami') & (df['LeagueAbbr'] == 'MAC'), 'teamName'] = 'Miami-Ohio'

    # Conference type
    power = ['Atlantic Coast Conference','Big 12 Conference','Big Ten Conference','Pacific-10 Conference','Pacific-12 Conference','Southeastern Conference']
    low_major = ['Big South Conference','Patriot League','Ivy League','America East Conference','Metro Atlantic Athletic Conference','Northeast Conference','Southwest Athletic Conference','Horizon League']
    df['conference_type'] = 'Mid Major'
    df.loc[df['leagueName'].isin(power), 'conference_type'] = 'Power Conference'
    df.loc[df['leagueName'].isin(low_major), 'conference_type'] = 'Low Major'

    # Academic School flag
    academic_schools = ['Air Force','Army','Boston College','Brown','Bryant','Bryant University','Bucknell','California','Columbia','Cornell','Dartmouth','Davidson','Davidson College','Duke','Fordham','Georgetown','Georgia Tech','Harvard','Holy Cross','Lafayette','Lafayette College','Lehigh','Maryland','Massachusetts','Michigan','Navy','New Jersey Tech','North Carolina','Northeastern','Northwestern','Notre Dame','Penn','Pennsylvania','Princeton','Purdue','Rice','Richmond','Stanford','Tulane','UC Davis','UC Irvine','UC San Diego','UC Santa Barbara','UCLA','USC','Vanderbilt','Villanova','Virginia','Wake Forest','Washington','William and Mary','Wofford','Yale']
    df['is_academic_school'] = df['teamName'].isin(academic_schools)

    return df

data = load_data()

# Filters
year_filter = st.sidebar.slider("Year Range", int(data['year'].min()), int(data['year'].max()), (2015, int(data['year'].max())), key="year")
role_filter = st.sidebar.multiselect("Role", ['Pitcher','Hitter'], default=['Pitcher','Hitter'], key="role")

# Good Players Only toggle + description
good_players_only = st.sidebar.checkbox("Good Players Only", key="good_players")
if good_players_only:
    st.sidebar.caption("Pitchers: IP > 30, WHIP < 1.35 & Hitters: T90/PA > .550")
    
league_filter = st.sidebar.multiselect("Conference", sorted(data['LeagueAbbr'].unique()), key="league")
conference_type_filter = st.sidebar.multiselect("Conference Type", options=['Power Conference', 'Mid Major', 'Low Major'], key="conference_type")

# NEW: Level filter - right under Conference Type
level_options = sorted([x for x in data['Level'].astype(str).unique() if x not in ['nan', 'None', '']])
level_filter = st.sidebar.multiselect("Level", options=level_options, key="level")

academic_school_filter = st.sidebar.radio("School Academic Level", ["All", "Top 60 Academic"], key="academic_school")
team_filter = st.sidebar.multiselect("Team", sorted(data['teamName'].unique()), key="team")
state_filter = st.sidebar.multiselect("State", sorted(data['state'].unique()), key="state")
region_filter = st.sidebar.multiselect("Region", sorted(data['region'].unique()), key="region")

min_games = st.sidebar.slider("Minimum Games Played", 0, int(data['G'].max()), 5, key="min_games")
position_filter = st.sidebar.multiselect("Position", options=sorted(data['posit'].dropna().unique()), key="posit")
bats_filter = st.sidebar.multiselect("Bats", options=['L', 'R', 'S'], key="bats")
throws_filter = st.sidebar.multiselect("Throws", options=['L', 'R'], key="throws")

draft_round_range = st.sidebar.slider("Draft Round Range", 0, 70, (0,70), key="draft_round")

available_stats = ['ERA','OPS','W','L','SO','BB','HR','RBI','SB','CS','Bavg','Slg','obp','WHIP','IP','H','R','ER','G','GS','T90s','T90/PA']
stat1 = st.sidebar.selectbox("Custom Stat Filter 1", ['None']+available_stats, key="stat1")
if stat1 != 'None':
    direction1 = st.sidebar.radio(f"{stat1} comparison", ["Greater than or equal to", "Less than or equal to"], key="dir1")
    step1 = 0.1 if stat1 in ['ERA','OPS','Bavg','Slg','obp','WHIP','T90/PA'] else 1.0
    value1 = st.sidebar.number_input(f"{stat1} value", value=0.0, step=step1, key="val1")

stat2 = 'None'
if stat1 != 'None':
    remaining = [s for s in available_stats if s != stat1]
    stat2 = st.sidebar.selectbox("Custom Stat Filter 2", ['None']+remaining, key="stat2")
if stat2 != 'None':
    direction2 = st.sidebar.radio(f"{stat2} comparison", ["Greater than or equal to", "Less than or equal to"], key="dir2")
    step2 = 0.1 if stat2 in ['ERA','OPS','Bavg','Slg','obp','WHIP','T90/PA'] else 1.0
    value2 = st.sidebar.number_input(f"{stat2} value", value=0.0, step=step2, key="val2")

name_search = st.sidebar.text_input("Search Player Name", key="name_search")

# Base filtering
filtered = data[
    data['role'].isin(role_filter) &
    data['year'].between(*year_filter) &
    (data['G'] >= min_games)
]

if league_filter: filtered = filtered[filtered['LeagueAbbr'].isin(league_filter)]
if team_filter: filtered = filtered[filtered['teamName'].isin(team_filter)]
if state_filter: filtered = filtered[filtered['state'].isin(state_filter)]
if region_filter: filtered = filtered[filtered['region'].isin(region_filter)]
if position_filter: filtered = filtered[filtered['posit'].isin(position_filter)]
if bats_filter: filtered = filtered[filtered['Bats'].isin(bats_filter)]
if throws_filter: filtered = filtered[filtered['Throws'].isin(throws_filter)]
if name_search:
    filtered = filtered[filtered['firstname'].str.contains(name_search, case=False, na=False) |
                      filtered['lastname'].str.contains(name_search, case=False, na=False)]

if conference_type_filter:
    filtered = filtered[filtered['conference_type'].isin(conference_type_filter)]

# NEW: Apply Level filter
if level_filter:
    filtered = filtered[filtered['Level'].astype(str).isin(level_filter)]

if academic_school_filter == "Top 60 Academic":
    filtered = filtered[filtered['is_academic_school']]

filtered = filtered[filtered['draft_Round'].between(*draft_round_range)]

# Good Players Only filter
if good_players_only:
    hitters_good = (filtered['role'] == 'Hitter') & (filtered['G'] > 30) & (filtered['T90/PA'] > 0.550)
    pitchers_good = (filtered['role'] == 'Pitcher') & (filtered['IP'] > 30) & (filtered['WHIP'] < 1.35)
    filtered = filtered[hitters_good | pitchers_good]

# Custom stat filters
if stat1 != 'None' and stat1 in filtered.columns:
    filtered = filtered[filtered[stat1] >= value1] if direction1 == "Greater than or equal to" else filtered[filtered[stat1] <= value1]
if stat2 != 'None' and stat2 in filtered.columns:
    filtered = filtered[filtered[stat2] >= value2] if direction2 == "Greater than or equal to" else filtered[filtered[stat2] <= value2]

# Column selector
with st.expander("Columns to show (click to expand)", expanded=False):
    default_cols = ['lastname','firstname','teamName','year','Age','state','LeagueAbbr','experience','G','T90s','OPS','draft_Round','ERA','W','SV','IP','WHIP']
    available_default = [c for c in default_cols if c in filtered.columns]
    cols = st.multiselect("", options=filtered.columns.tolist(), default=available_default, key="cols")

# Export
csv = filtered.to_csv(index=False).encode('utf-8')
st.download_button("Export Filtered Data as CSV", data=csv, file_name='college_baseball_filtered.csv', mime='text/csv')

st.subheader(f"Filtered Players – {len(filtered):,} rows")
st.dataframe(filtered[cols] if cols else filtered.head(100), use_container_width=True, hide_index=True)

# The rest of your original code (maps, charts, leaderboards) is unchanged — paste it here exactly as before
# (Everything from "# State map" to the end of the file)
