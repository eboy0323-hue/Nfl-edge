import pandas as pd
import streamlit as st
from data import load_nfl_data
from model import build_team_form,project_game,spread_cover_prob,over_prob,american_to_implied,expected_value_per_100,walk_forward_backtest,multi_season_backtest,validation_summary
st.set_page_config(page_title='NFL EDGE V3.1',layout='wide'); st.title('🏈 NFL EDGE V3.1 — Historical Validation'); st.caption('Walk-forward backtesting • 2026 held out live • Fanatics prices entered manually')
@st.cache_data(ttl=3600)
def get_data():return load_nfl_data()
@st.cache_data(ttl=3600)
def get_bt(schedules,season):return walk_forward_backtest(schedules,season)
@st.cache_data(ttl=3600)
def get_hist(schedules,seasons):return multi_season_backtest(schedules,seasons)
try:schedules,player_stats=get_data()
except Exception as e:st.error(f'Could not load nflverse data: {e}');st.stop()
season=int(schedules.season.max()); today=pd.Timestamp.now().normalize(); form=build_team_form(schedules,season,as_of=today); future=schedules[(schedules.season==season)&(pd.to_datetime(schedules.gameday)>=today-pd.Timedelta(days=1))].copy().sort_values(['gameday','gametime'])
t1,t2,t3,t4=st.tabs(['Game Model','2026 Holdout','Historical Test','Player Data'])
with t1:
 st.subheader('Pregame research'); st.warning('Research output until historical + 2026 holdout gates pass.')
 if future.empty:st.info('No upcoming games found.')
 else:
  games=future.head(32); labels=[f'{r.away_team} @ {r.home_team} — W{int(r.week)} — {r.gameday}' for _,r in games.iterrows()]; i=st.selectbox('Game',range(len(labels)),format_func=lambda x:labels[x]); g=games.iloc[i]; p=project_game(g.home_team,g.away_team,form)
  favorite=g.home_team if p['projected_margin']>=0 else g.away_team; fav_line=-abs(p['projected_margin'])
  c1,c2,c3=st.columns(3); c1.metric('Projected score',f"{g.away_team} {p['away_points']} – {g.home_team} {p['home_points']}"); c2.metric('Model fair spread',f'{favorite} {fav_line:.1f}'); c3.metric('Projected total',f"{p['projected_total']:.1f}")
  st.caption(f"Internal home-margin convention: {g.home_team} {p['projected_margin']:+.1f} (positive = home team projected ahead).")
  a,b,c=st.columns(3)
  with a:
   hs=st.number_input(f'{g.home_team} Fanatics spread (e.g. -3.5 if home favored)',value=-2.5,step=.5); odds=st.number_input('Spread odds',value=-110,step=5); pc=spread_cover_prob(p['projected_margin'],hs,p['margin_sd']); imp=american_to_implied(odds); st.metric('Raw cover probability',f'{pc:.1%}',f'{pc-imp:+.1%} vs implied');st.metric('Raw EV / $100',f"${expected_value_per_100(pc,odds):.2f}");st.info('RESEARCH — gate not passed')
  with b:
   tl=st.number_input('Game total',value=45.5,step=.5);oo=st.number_input('Over odds',value=-110,step=5);po=over_prob(p['projected_total'],tl,p['total_sd']);imp=american_to_implied(oo);st.metric('Raw Over probability',f'{po:.1%}',f'{po-imp:+.1%} vs implied');st.metric('Raw EV / $100',f"${expected_value_per_100(po,oo):.2f}");st.info('PASS — totals rebuild pending')
  with c:
   ml=st.number_input(f'{g.home_team} moneyline',value=-130,step=5);imp=american_to_implied(ml);st.metric('Raw win probability',f"{p['home_win_prob']:.1%}",f"{p['home_win_prob']-imp:+.1%} vs implied");st.metric('Raw EV / $100',f"${expected_value_per_100(p['home_win_prob'],ml):.2f}");st.info('RESEARCH — gate not passed')
with t2:
 st.subheader(f'{season} live holdout — walk-forward'); st.write('Every game uses only earlier games from the same season. Week 1 is skipped. This season is not used to tune parameters.')
 bt=get_bt(schedules,season); s=validation_summary(bt)
 if not s:st.warning('No completed games available.')
 else:
  cols=st.columns(6);cols[0].metric('Games',s['games']);cols[1].metric('Margin MAE',f"{s['margin_mae']:.1f}");cols[2].metric('Total MAE',f"{s['total_mae']:.1f}");cols[3].metric('Winner',f"{s['winner_accuracy']:.1%}");cols[4].metric('ATS',f"{s.get('ats_accuracy',float('nan')):.1%}");cols[5].metric('ATS games',s.get('ats_games',0))
  by=bt.groupby('week').agg(games=('week','size'),margin_mae=('margin_error','mean'),total_mae=('total_error','mean'),winner_accuracy=('winner_correct','mean')).reset_index();st.dataframe(by,use_container_width=True,hide_index=True)
  if 'spread_edge_abs' in bt:
   bins=pd.cut(bt.spread_edge_abs,[0,1.5,3,5,7,100],right=False);edge=bt.assign(edge_bucket=bins).groupby('edge_bucket',observed=True).agg(bets=('model_side_win','count'),ats=('model_side_win','mean')).reset_index();st.markdown('#### ATS by model-vs-market edge');st.dataframe(edge,use_container_width=True,hide_index=True)
  st.download_button('Download 2026 backtest CSV',bt.to_csv(index=False).encode(),'nfl_edge_2026_holdout.csv','text/csv')
with t3:
 st.subheader('Historical walk-forward test — 2022–2025'); hist_seasons=tuple(s for s in [2022,2023,2024,2025] if s in set(pd.to_numeric(schedules.season,errors='coerce').dropna().astype(int))); hist=get_hist(schedules,hist_seasons); hs=validation_summary(hist)
 st.caption('Each season is reset and tested chronologically; Week 1 is skipped. 2026 is excluded from this table so it remains the live holdout.')
 if not hs:st.warning('Historical seasons could not be tested.')
 else:
  cols=st.columns(6);cols[0].metric('Games',hs['games']);cols[1].metric('Margin MAE',f"{hs['margin_mae']:.1f}");cols[2].metric('Total MAE',f"{hs['total_mae']:.1f}");cols[3].metric('Winner',f"{hs['winner_accuracy']:.1%}");cols[4].metric('ATS',f"{hs.get('ats_accuracy',float('nan')):.1%}");cols[5].metric('ATS games',hs.get('ats_games',0))
  bys=hist.groupby('season').agg(games=('season','size'),margin_mae=('margin_error','mean'),total_mae=('total_error','mean'),winner_accuracy=('winner_correct','mean'),ats_games=('model_side_win','count'),ats=('model_side_win','mean')).reset_index();st.dataframe(bys,use_container_width=True,hide_index=True)
  if 'spread_edge_abs' in hist:
   bins=pd.cut(hist.spread_edge_abs,[0,1.5,3,5,7,10,100],right=False);edge=hist.assign(edge_bucket=bins).groupby('edge_bucket',observed=True).agg(bets=('model_side_win','count'),ats=('model_side_win','mean'),margin_mae=('margin_error','mean')).reset_index();st.markdown('#### Historical ATS by edge size');st.dataframe(edge,use_container_width=True,hide_index=True)
  st.download_button('Download historical backtest CSV',hist.to_csv(index=False).encode(),'nfl_edge_2022_2025_backtest.csv','text/csv')
with t4:
 st.subheader('Player data foundation');st.write('Props and First TD remain disabled until separate player-level models are validated.')
 if not player_stats.empty:
  ps=player_stats.copy()
  if 'recent_team' not in ps.columns:
   for alt in ('team','team_abbr','club'):
    if alt in ps.columns:ps['recent_team']=ps[alt];break
  wanted=['player_display_name','recent_team','week','passing_yards','rushing_yards','receiving_yards','targets','carries'];available=[c for c in wanted if c in ps.columns]
  if 'season' in ps.columns:ps=ps[pd.to_numeric(ps.season,errors='coerce')==season]
  if available:st.dataframe(ps.reindex(columns=available).tail(100),use_container_width=True,hide_index=True)
