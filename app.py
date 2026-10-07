import math
import itertools
import numpy as np
import pandas as pd
import streamlit as st



def american_to_implied(odds: float) -> float:
    odds=float(odds); return (-odds)/((-odds)+100.0) if odds<0 else 100.0/(odds+100.0)

def prob_to_american(p: float) -> int:
    p=min(max(float(p),1e-6),1-1e-6); return int(round(-100*p/(1-p))) if p>=.5 else int(round(100*(1-p)/p))

def expected_value_per_100(p_win, american_odds):
    odds=float(american_odds); profit=odds if odds>0 else 10000.0/abs(odds); return p_win*profit-(1-p_win)*100.0

def normal_cdf(x): return .5*(1+math.erf(x/math.sqrt(2)))

def build_team_form(schedule, season, as_of=None, decay=.82):
    df=schedule.copy(); as_of=pd.Timestamp.now().normalize() if as_of is None else pd.Timestamp(as_of)
    gd=pd.to_datetime(df['gameday'])
    df=df[(df['season']==season)&(gd<as_of)&df.home_score.notna()&df.away_score.notna()].copy()
    if df.empty: return pd.DataFrame(columns=['team','off_rating','def_rating','net_rating','avg_total','games'])
    rows=[]
    for _,g in df.sort_values(['gameday','week']).iterrows():
        rows += [{'team':g.home_team,'week':g.week,'pf':g.home_score,'pa':g.away_score}, {'team':g.away_team,'week':g.week,'pf':g.away_score,'pa':g.home_score}]
    x=pd.DataFrame(rows); league_pf=x.pf.mean(); out=[]
    for team,t in x.groupby('team'):
        t=t.sort_values('week'); n=len(t); w=np.array([decay**(n-1-i) for i in range(n)],float); w/=w.sum()
        pf=np.average(t.pf,weights=w); pa=np.average(t.pa,weights=w)
        out.append({'team':team,'off_rating':pf-league_pf,'def_rating':league_pf-pa,'net_rating':pf-pa,'avg_total':np.average(t.pf+t.pa,weights=w),'games':n})
    return pd.DataFrame(out)

def project_game(home_team, away_team, form, home_field=1.8, def_weight=.5):
    f=form.set_index('team') if not form.empty else pd.DataFrame()
    def get(team,col,default=0.):
        try:return float(f.loc[team,col])
        except:return default
    ho,hd,ao,ad=get(home_team,'off_rating'),get(home_team,'def_rating'),get(away_team,'off_rating'),get(away_team,'def_rating')
    home_pts=22.5+ho-def_weight*ad+home_field/2
    away_pts=22.5+ao-def_weight*hd-home_field/2
    margin=home_pts-away_pts; total=home_pts+away_pts; ms,ts=13.4,13.0; p=normal_cdf(margin/ms)
    return {'home_points':round(home_pts,1),'away_points':round(away_pts,1),'projected_margin':round(margin,2),'projected_total':round(total,2),'home_win_prob':p,'home_fair_ml':prob_to_american(p),'away_fair_ml':prob_to_american(1-p),'margin_sd':ms,'total_sd':ts}

def spread_cover_prob(projected_margin, home_spread, margin_sd=13.4): return normal_cdf((projected_margin+float(home_spread))/margin_sd)
def over_prob(projected_total, market_total, total_sd=13.): return normal_cdf((projected_total-float(market_total))/total_sd)

def walk_forward_backtest(schedule, season, decay=.82, home_field=1.8, def_weight=.5):
    df=schedule[(schedule.season==season)&schedule.home_score.notna()&schedule.away_score.notna()].copy(); df['gameday_dt']=pd.to_datetime(df.gameday); df=df.sort_values(['gameday_dt','week','gametime']); rows=[]
    for _,g in df.iterrows():
        form=build_team_form(schedule,season,as_of=g.gameday_dt,decay=decay)
        if form.empty or g.home_team not in set(form.team) or g.away_team not in set(form.team): continue
        p=project_game(g.home_team,g.away_team,form,home_field=home_field,def_weight=def_weight); am=float(g.home_score-g.away_score); at=float(g.home_score+g.away_score)
        r={'season':int(season),'week':int(g.week),'gameday':str(g.gameday),'away':g.away_team,'home':g.home_team,'projected_margin':p['projected_margin'],'actual_margin':am,'margin_error':abs(p['projected_margin']-am),'projected_total':p['projected_total'],'actual_total':at,'total_error':abs(p['projected_total']-at),'winner_correct':(p['projected_margin']>0)==(am>0) if am!=0 else np.nan}
        if 'spread_line' in g.index and pd.notna(g.get('spread_line')):
            market_home_margin=float(g.get('spread_line')); r['market_home_margin']=market_home_margin
            edge=p['projected_margin']-market_home_margin; r['spread_edge']=edge; r['spread_edge_abs']=abs(edge); r['model_side']='HOME' if edge>0 else ('AWAY' if edge<0 else 'PASS')
            home_ats=am-market_home_margin
            r['ats_result']='PUSH' if abs(home_ats)<1e-9 else ('WIN' if (home_ats>0 and r['model_side']=='HOME') or (home_ats<0 and r['model_side']=='AWAY') else 'LOSS') if r['model_side']!='PASS' else 'PASS'
            r['model_side_win']=True if r['ats_result']=='WIN' else False if r['ats_result']=='LOSS' else np.nan
        if 'total_line' in g.index and pd.notna(g.get('total_line')):
            tl=float(g.get('total_line')); r['market_total']=tl; edge=p['projected_total']-tl; r['total_edge']=edge; r['total_edge_abs']=abs(edge); r['model_total_side']='OVER' if edge>0 else ('UNDER' if edge<0 else 'PASS')
            diff=at-tl; r['total_result']='PUSH' if abs(diff)<1e-9 else ('WIN' if (diff>0 and r['model_total_side']=='OVER') or (diff<0 and r['model_total_side']=='UNDER') else 'LOSS') if r['model_total_side']!='PASS' else 'PASS'
            r['model_total_win']=True if r['total_result']=='WIN' else False if r['total_result']=='LOSS' else np.nan
        rows.append(r)
    return pd.DataFrame(rows)

def multi_season_backtest(schedule,seasons,decay=.82,home_field=1.8,def_weight=.5):
    frames=[walk_forward_backtest(schedule,int(s),decay,home_field,def_weight) for s in seasons]; frames=[x for x in frames if not x.empty]; return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

def validation_summary(bt):
    if bt.empty:return {}
    out={'games':len(bt),'margin_mae':bt.margin_error.mean(),'total_mae':bt.total_error.mean(),'winner_accuracy':bt.winner_correct.dropna().mean()}
    if 'model_side_win' in bt:
        x=bt.model_side_win.dropna(); out['ats_games']=len(x); out['ats_accuracy']=x.mean() if len(x) else np.nan
    if 'model_total_win' in bt:
        x=bt.model_total_win.dropna(); out['total_games']=len(x); out['total_side_accuracy']=x.mean() if len(x) else np.nan
    return out

def roi_at_minus110(bt, min_edge=0.0, max_edge=999.0):
    if bt.empty or 'model_side_win' not in bt:return {'bets':0,'wins':0,'losses':0,'roi':np.nan}
    x=bt[(bt.spread_edge_abs>=min_edge)&(bt.spread_edge_abs<max_edge)&bt.model_side_win.notna()].copy()
    wins=int(x.model_side_win.sum()); losses=int(len(x)-wins); profit=wins*(100/110)-losses
    return {'bets':len(x),'wins':wins,'losses':losses,'roi':profit/len(x) if len(x) else np.nan}

def precompute_decay_games(schedule, seasons, decays):
    """Precompute walk-forward game inputs once per season/decay.
    Defense weight and HFA are applied later, so the grid avoids rebuilding forms 27x.
    """
    out = {}
    for season in seasons:
        sdf = schedule[(schedule.season==season)&schedule.home_score.notna()&schedule.away_score.notna()].copy()
        sdf['gameday_dt']=pd.to_datetime(sdf.gameday)
        sdf=sdf.sort_values(['gameday_dt','week','gametime'])
        for decay in decays:
            rows=[]
            for _,g in sdf.iterrows():
                form=build_team_form(schedule,season,as_of=g.gameday_dt,decay=decay)
                if form.empty: continue
                f=form.set_index('team')
                if g.home_team not in f.index or g.away_team not in f.index: continue
                def gv(team,col):
                    try:return float(f.loc[team,col])
                    except:return 0.0
                r={
                    'season':int(season),'week':int(g.week),'gameday':str(g.gameday),
                    'home':g.home_team,'away':g.away_team,
                    'ho':gv(g.home_team,'off_rating'),'hd':gv(g.home_team,'def_rating'),
                    'ao':gv(g.away_team,'off_rating'),'ad':gv(g.away_team,'def_rating'),
                    'actual_margin':float(g.home_score-g.away_score),
                    'actual_total':float(g.home_score+g.away_score),
                }
                if 'spread_line' in g.index and pd.notna(g.get('spread_line')):
                    r['market_home_margin']=float(g.get('spread_line'))
                if 'total_line' in g.index and pd.notna(g.get('total_line')):
                    r['market_total']=float(g.get('total_line'))
                rows.append(r)
            out[(int(season),float(decay))]=pd.DataFrame(rows)
    return out

def evaluate_precomputed(pre, seasons, decay, def_weight, home_field):
    frames=[]
    for season in seasons:
        x=pre.get((int(season),float(decay)),pd.DataFrame()).copy()
        if x.empty: continue
        x['projected_home_points']=22.5+x.ho-def_weight*x.ad+home_field/2
        x['projected_away_points']=22.5+x.ao-def_weight*x.hd-home_field/2
        x['projected_margin']=x.projected_home_points-x.projected_away_points
        x['projected_total']=x.projected_home_points+x.projected_away_points
        x['margin_error']=(x.projected_margin-x.actual_margin).abs()
        x['total_error']=(x.projected_total-x.actual_total).abs()
        x['winner_correct']=np.where(x.actual_margin!=0,(x.projected_margin>0)==(x.actual_margin>0),np.nan)
        if 'market_home_margin' in x.columns:
            x['spread_edge']=x.projected_margin-x.market_home_margin
            x['spread_edge_abs']=x.spread_edge.abs()
            side=np.where(x.spread_edge>0,'HOME',np.where(x.spread_edge<0,'AWAY','PASS'))
            home_ats=x.actual_margin-x.market_home_margin
            win=np.where(side=='PASS',np.nan,
                np.where(abs(home_ats)<1e-9,np.nan,
                    ((home_ats>0)&(side=='HOME'))|((home_ats<0)&(side=='AWAY'))))
            x['model_side_win']=win
        frames.append(x)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

def parameter_search(schedule, train_seasons=(2022,2023,2024), decays=(.72,.82,.90), def_weights=(.35,.50,.65), home_fields=(1.2,1.8,2.4)):
    pre=precompute_decay_games(schedule,train_seasons,decays)
    rows=[]
    for decay,dw,hfa in itertools.product(decays,def_weights,home_fields):
        bt=evaluate_precomputed(pre,train_seasons,decay,dw,hfa)
        if bt.empty:continue
        s=validation_summary(bt); by=bt.groupby('season').margin_error.mean(); edge=roi_at_minus110(bt,3,7)
        score=float(s['margin_mae']) + .20*float(by.std(ddof=0) if len(by)>1 else 0)
        rows.append({'decay':decay,'def_weight':dw,'home_field':hfa,'train_games':s['games'],'margin_mae':s['margin_mae'],'season_mae_sd':float(by.std(ddof=0) if len(by)>1 else 0),'selection_score':score,'winner_accuracy':s['winner_accuracy'],'ats':s.get('ats_accuracy',np.nan),'edge_3_7_bets':edge['bets'],'edge_3_7_roi':edge['roi']})
    return pd.DataFrame(rows).sort_values(['selection_score','margin_mae']).reset_index(drop=True)


def load_nfl_data():
    import nflreadpy as nfl
    schedules = nfl.load_schedules(seasons=True).to_pandas()
    try:
        player_stats = nfl.load_player_stats(seasons=True).to_pandas()
    except Exception:
        player_stats = pd.DataFrame()
    return schedules, player_stats

st.set_page_config(page_title='NFL EDGE V3.2',layout='wide'); st.title('🏈 NFL EDGE V3.2 — Model Lab'); st.caption('Train 2022–24 • validate 2025 • 2026 live holdout • no automatic BET NOW claims')
@st.cache_data(ttl=3600)
def get_data():return load_nfl_data()
@st.cache_data(ttl=3600)
def get_bt(schedules,season,decay=.82,dw=.5,hfa=1.8):return walk_forward_backtest(schedules,season,decay,hfa,dw)
@st.cache_data(ttl=3600)
def get_hist(schedules,seasons,decay=.82,dw=.5,hfa=1.8):return multi_season_backtest(schedules,seasons,decay,hfa,dw)
@st.cache_data(ttl=3600,show_spinner='Fast parameter search: precomputing 2022–2024 once…')
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
