import pandas as pd
import streamlit as st
from data import load_nfl_data
from model import build_team_form,project_game,spread_cover_prob,over_prob,american_to_implied,expected_value_per_100,walk_forward_backtest,multi_season_backtest,validation_summary,parameter_search,roi_at_minus110
st.set_page_config(page_title='NFL EDGE V3.2',layout='wide'); st.title('🏈 NFL EDGE V3.2 — Model Lab'); st.caption('Train 2022–24 • validate 2025 • 2026 live holdout • no automatic BET NOW claims')
@st.cache_data(ttl=3600)
def get_data():return load_nfl_data()
@st.cache_data(ttl=3600)
def get_bt(schedules,season,decay=.82,dw=.5,hfa=1.8):return walk_forward_backtest(schedules,season,decay,hfa,dw)
@st.cache_data(ttl=3600)
def get_hist(schedules,seasons,decay=.82,dw=.5,hfa=1.8):return multi_season_backtest(schedules,seasons,decay,hfa,dw)
@st.cache_data(ttl=3600,show_spinner='Testing model parameter grid on 2022–2024…')
def get_search(schedules):return parameter_search(schedules)
try:schedules,player_stats=get_data()
except Exception as e:st.error(f'Could not load nflverse data: {e}');st.stop()
season=int(schedules.season.max()); today=pd.Timestamp.now().normalize()
# Defaults remain frozen V1 until Model Lab produces a candidate that survives 2025 validation.
ACTIVE={'decay':.82,'def_weight':.5,'home_field':1.8}
form=build_team_form(schedules,season,as_of=today,decay=ACTIVE['decay']); future=schedules[(schedules.season==season)&(pd.to_datetime(schedules.gameday)>=today-pd.Timedelta(days=1))].copy().sort_values(['gameday','gametime'])
t1,t2,t3,t4,t5=st.tabs(['Game Model','2026 Holdout','Historical','Model Lab','Player Data'])
with t1:
 st.subheader('Pregame research'); st.warning('Active model stays frozen until a candidate passes 2025 validation and 2026 holdout review.')
 if future.empty:st.info('No upcoming games found.')
 else:
  games=future.head(32); labels=[f'{r.away_team} @ {r.home_team} — W{int(r.week)} — {r.gameday}' for _,r in games.iterrows()]; i=st.selectbox('Game',range(len(labels)),format_func=lambda x:labels[x]); g=games.iloc[i]; p=project_game(g.home_team,g.away_team,form,ACTIVE['home_field'],ACTIVE['def_weight'])
  favorite=g.home_team if p['projected_margin']>=0 else g.away_team; fav_line=-abs(p['projected_margin']); c1,c2,c3=st.columns(3); c1.metric('Projected score',f"{g.away_team} {p['away_points']} – {g.home_team} {p['home_points']}"); c2.metric('Model fair spread',f'{favorite} {fav_line:.1f}'); c3.metric('Projected total',f"{p['projected_total']:.1f}")
  st.caption(f"Internal home-margin: {g.home_team} {p['projected_margin']:+.1f}. Positive means home projected ahead.")
with t2:
 st.subheader(f'{season} untouched live holdout'); bt=get_bt(schedules,season); s=validation_summary(bt)
 if s:
  c=st.columns(6);c[0].metric('Games',s['games']);c[1].metric('Margin MAE',f"{s['margin_mae']:.1f}");c[2].metric('Total MAE',f"{s['total_mae']:.1f}");c[3].metric('Winner',f"{s['winner_accuracy']:.1%}");c[4].metric('ATS',f"{s.get('ats_accuracy',float('nan')):.1%}");c[5].metric('ATS games',s.get('ats_games',0))
  st.info('2026 is displayed here but is NOT used by Model Lab to choose parameters.')
with t3:
 st.subheader('Baseline historical test — 2022–2025'); hist=get_hist(schedules,(2022,2023,2024,2025)); hs=validation_summary(hist)
 if hs:
  c=st.columns(5);c[0].metric('Games',hs['games']);c[1].metric('Margin MAE',f"{hs['margin_mae']:.1f}");c[2].metric('Winner',f"{hs['winner_accuracy']:.1%}");c[3].metric('ATS',f"{hs.get('ats_accuracy',float('nan')):.1%}");c[4].metric('ATS games',hs.get('ats_games',0))
  bys=hist.groupby('season').agg(games=('season','size'),margin_mae=('margin_error','mean'),total_mae=('total_error','mean'),winner=('winner_correct','mean'),ats=('model_side_win','mean')).reset_index();st.dataframe(bys,use_container_width=True,hide_index=True)
with t4:
 st.subheader('Model Lab — development → validation')
 st.write('Parameter selection uses **2022–2024 only**. The ranking is based on margin prediction error plus a stability penalty across seasons. ATS/ROI are shown for diagnosis but are **not used to choose the winner**.')
 if st.button('Run / refresh parameter search',type='primary'):
  st.cache_data.clear(); st.rerun()
 search=get_search(schedules)
 if search.empty:st.warning('Parameter search returned no candidates.')
 else:
  show=search.head(10).copy(); show['winner_accuracy']=show.winner_accuracy.map(lambda x:f'{x:.1%}');show['ats']=show.ats.map(lambda x:f'{x:.1%}');show['edge_3_7_roi']=show.edge_3_7_roi.map(lambda x:f'{x:.1%}' if pd.notna(x) else '—');st.markdown('#### Top 10 on 2022–2024 development data');st.dataframe(show,use_container_width=True,hide_index=True)
  best=search.iloc[0]; params={'decay':float(best.decay),'def_weight':float(best.def_weight),'home_field':float(best.home_field)}
  st.markdown('#### Frozen candidate → 2025 validation');st.write(f"Candidate: decay **{params['decay']:.2f}**, defense weight **{params['def_weight']:.2f}**, home field **{params['home_field']:.1f}**")
  base25=get_bt(schedules,2025,.82,.5,1.8); cand25=get_bt(schedules,2025,params['decay'],params['def_weight'],params['home_field']); sb=validation_summary(base25); sc=validation_summary(cand25)
  def row(name,s,bt):
   e=roi_at_minus110(bt,3,7);return {'model':name,'games':s.get('games',0),'margin_mae':s.get('margin_mae',float('nan')),'winner_accuracy':s.get('winner_accuracy',float('nan')),'ats_all':s.get('ats_accuracy',float('nan')),'edge_3_7_bets':e['bets'],'edge_3_7_W':e['wins'],'edge_3_7_L':e['losses'],'edge_3_7_roi_-110':e['roi']}
  comp=pd.DataFrame([row('V1 baseline',sb,base25),row('Model Lab candidate',sc,cand25)]);st.dataframe(comp,use_container_width=True,hide_index=True)
  mae_pass=sc.get('margin_mae',999)<sb.get('margin_mae',-999); stable=abs(sc.get('margin_mae',999)-best.margin_mae)<=2.0
  if mae_pass and stable:st.success('VALIDATION STATUS: Candidate clears the first predictive gate on 2025. Do not promote yet — next compare it prospectively with the frozen 2026 holdout.')
  else:st.error('VALIDATION STATUS: Candidate does not clear the 2025 predictive gate. Keep V1 frozen and revise the feature/model design rather than promoting these parameters.')
  st.caption('No candidate is promoted automatically. This prevents the app from silently tuning itself to favorable betting results.')
with t5:
 st.subheader('Player model status');st.info('Props / First TD remain separate future models. Game-model validation does not validate player markets.')
