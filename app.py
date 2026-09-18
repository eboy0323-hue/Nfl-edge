import datetime as dt
import pandas as pd
import streamlit as st
from data import load_nfl_data
from model import (build_team_form,project_game,spread_cover_prob,over_prob,
                   american_to_implied,expected_value_per_100,first_td_proxy)

st.set_page_config(page_title='NFL EDGE V2',layout='wide')
st.title('🏈 NFL EDGE V2 — Research Preview')
st.warning('EXPERIMENTAL: no calibrated probabilities, verified edge, live odds, or betting recommendations. Paper-track only. Pregame model must not be used for in-play betting.')
@st.cache_data(ttl=3600)
def get_data(): return load_nfl_data()
try: schedules,players=get_data()
except Exception as exc:
    st.error(f'Data unavailable: {exc}');st.stop()
now=dt.datetime.now(dt.timezone(dt.timedelta(hours=-4)))
today=now.date()
season=int(schedules.season.max())
# Date-scoped, scores must exist, excludes games on today's date to prevent look-ahead.
form=build_team_form(schedules,season,asof=today)
schedules=schedules.copy();schedules['gameday']=pd.to_datetime(schedules.gameday,errors='coerce')
future=schedules[(schedules.season==season)&(schedules.gameday>=pd.Timestamp(today))].sort_values(['gameday','gametime'])
t1,t2,t3,t4,t5=st.tabs(['Game research','Player props','First TD','SGP Lab','Paper log'])
with t1:
    st.caption('Pregame only • completed games before today • prior-season shrinkage • fixed, uncalibrated error scales')
    if future.empty: st.info('No scheduled games found in data.')
    else:
        games=future.head(64).reset_index(drop=True)
        i=st.selectbox('Choose matchup',range(len(games)),format_func=lambda j:f'{games.iloc[j].away_team} @ {games.iloc[j].home_team} — {games.iloc[j].gameday.date()}')
        g=games.iloc[i];p=project_game(g.home_team,g.away_team,form)
        st.metric('Research score',f"{g.away_team} {p['away_points']:.1f} – {g.home_team} {p['home_points']:.1f}")
        st.metric('Research margin (home)',f"{p['projected_margin']:+.1f}")
        st.metric('Research total',f"{p['projected_total']:.1f}")
        st.caption('Enter current Fanatics prices manually. These illustrative calculations are NOT betting signals.')
        a,b,c=st.columns(3)
        with a:
            spread=st.number_input('Home spread',value=-2.5,step=.5)
            spreadodds=st.number_input('Home spread odds',value=-110,step=5)
            probability=spread_cover_prob(p['projected_margin'],spread)
            st.write(f'Illustrative cover chance: {probability:.1%}; EV/$100: ${expected_value_per_100(probability,spreadodds):+.2f}')
        with b:
            total=st.number_input('Total line',value=45.5,step=.5)
            overodds=st.number_input('Over odds',value=-110,step=5)
            probability=over_prob(p['projected_total'],total)
            st.write(f'Illustrative over chance: {probability:.1%}; EV/$100: ${expected_value_per_100(probability,overodds):+.2f}')
        with c:
            ml=st.number_input('Home moneyline',value=-130,step=5)
            probability=p['home_win_prob']
            st.write(f'Illustrative win chance: {probability:.1%}; EV/$100: ${expected_value_per_100(probability,ml):+.2f}')
        st.info('The EV figures are arithmetic demonstrations based on unvalidated assumptions. Positive EV here does not establish a real edge.')
with t2:
    st.subheader('Player stats explorer — not a prop predictor')
    if players.empty: st.info('Player stats not available.')
    else:
        ps=players[players.season==season]
        cols=[x for x in ['player_display_name','recent_team','week','passing_yards','rushing_yards','receiving_yards','targets','carries'] if x in ps.columns]
        if cols: st.dataframe(ps[cols].tail(200),use_container_width=True)
        st.info('A trustworthy player-prop model requires injury/usage context, opponent adjustments, verified dates and walk-forward validation. Not implemented.')
with t3:
    st.subheader('Touchdown-share explorer — NOT first-TD odds')
    st.caption('Only compares rushing + receiving TD counts by team; no drive order, passing TD credit, no-TD event or first-TD probability.')
    if players.empty: st.info('Player stats unavailable.')
    else:
        teams=sorted(players.loc[players.season==season,'recent_team'].dropna().unique())
        if teams:
            team=st.selectbox('Team',teams)
            completed=schedules[(schedules.season==season)&(schedules.gameday<pd.Timestamp(today))&schedules.home_score.notna()]
            week=int(completed.week.max()) if not completed.empty else 0
            if week==0: st.info('No completed games yet this season.')
            else:
                shares=first_td_proxy(players,team,season,week)
                if shares.empty: st.info('TD statistics unavailable.')
                else: st.dataframe(shares,use_container_width=True)
        st.warning('Do not compare these shares with First TD sportsbook odds. They are not first-TD probabilities.')
with t4:
    st.subheader('SGP Lab — not implemented')
    st.info('Joint-outcome simulation and correlation calibration are not available. Multiplying independent probabilities is not a valid substitute.')
with t5:
    st.subheader('Paper log — session only')
    st.caption('Export CSV before closing the app. Data is not permanently stored.')
    cols=['date','game','market','selection','odds','notes']
    if 'log' not in st.session_state: st.session_state.log=pd.DataFrame(columns=cols)
    with st.form('paper'):
        game=st.text_input('Game');market=st.selectbox('Market',['Spread','Total','Moneyline','Prop','First TD','SGP'])
        selection=st.text_input('Selection');odds=st.number_input('American odds',value=-110,step=5)
        notes=st.text_input('Notes');submit=st.form_submit_button('Add paper entry')
        if submit:
            row=dict(date=today.isoformat(),game=game,market=market,selection=selection,odds=odds,notes=notes)
            st.session_state.log=pd.concat([st.session_state.log,pd.DataFrame([row])],ignore_index=True)
    st.dataframe(st.session_state.log,use_container_width=True)
    st.download_button('Export paper log CSV',st.session_state.log.to_csv(index=False).encode(),'nfl_edge_paper_log.csv','text/csv')
