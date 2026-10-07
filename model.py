import math
import numpy as np
import pandas as pd


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

def project_game(home_team, away_team, form, home_field=1.8):
    f=form.set_index('team') if not form.empty else pd.DataFrame()
    def get(team,col,default=0.):
        try:return float(f.loc[team,col])
        except:return default
    ho,hd,ao,ad=get(home_team,'off_rating'),get(home_team,'def_rating'),get(away_team,'off_rating'),get(away_team,'def_rating')
    home_pts=22.5+ho-.5*ad+home_field/2; away_pts=22.5+ao-.5*hd-home_field/2
    margin=home_pts-away_pts; total=home_pts+away_pts; ms,ts=13.4,13.0; p=normal_cdf(margin/ms)
    return {'home_points':round(home_pts,1),'away_points':round(away_pts,1),'projected_margin':round(margin,2),'projected_total':round(total,2),'home_win_prob':p,'home_fair_ml':prob_to_american(p),'away_fair_ml':prob_to_american(1-p),'margin_sd':ms,'total_sd':ts}

def spread_cover_prob(projected_margin, home_spread, margin_sd=13.4): return normal_cdf((projected_margin+float(home_spread))/margin_sd)
def over_prob(projected_total, market_total, total_sd=13.): return normal_cdf((projected_total-float(market_total))/total_sd)

def walk_forward_backtest(schedule, season, decay=.82, home_field=1.8):
    df=schedule[(schedule.season==season)&schedule.home_score.notna()&schedule.away_score.notna()].copy(); df['gameday_dt']=pd.to_datetime(df.gameday); df=df.sort_values(['gameday_dt','week','gametime']); rows=[]
    for _,g in df.iterrows():
        form=build_team_form(schedule,season,as_of=g.gameday_dt,decay=decay)
        if form.empty or g.home_team not in set(form.team) or g.away_team not in set(form.team): continue
        p=project_game(g.home_team,g.away_team,form,home_field=home_field); am=float(g.home_score-g.away_score); at=float(g.home_score+g.away_score)
        r={'season':int(season),'week':int(g.week),'gameday':str(g.gameday),'away':g.away_team,'home':g.home_team,'projected_margin':p['projected_margin'],'actual_margin':am,'margin_error':abs(p['projected_margin']-am),'projected_total':p['projected_total'],'actual_total':at,'total_error':abs(p['projected_total']-at),'winner_correct':(p['projected_margin']>0)==(am>0) if am!=0 else np.nan}
        # nflverse spread_line = number of points HOME team is favored by. +3.5 means home -3.5.
        if 'spread_line' in g.index and pd.notna(g.get('spread_line')):
            fav=float(g.get('spread_line')); market_home_margin=fav; r['spread_line_home_favored_by']=fav; r['market_home_margin']=market_home_margin
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

def multi_season_backtest(schedule,seasons,decay=.82,home_field=1.8):
    frames=[walk_forward_backtest(schedule,int(s),decay,home_field) for s in seasons]; frames=[x for x in frames if not x.empty]; return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

def validation_summary(bt):
    if bt.empty:return {}
    out={'games':len(bt),'margin_mae':bt.margin_error.mean(),'total_mae':bt.total_error.mean(),'winner_accuracy':bt.winner_correct.dropna().mean()}
    if 'model_side_win' in bt:
        x=bt.model_side_win.dropna(); out['ats_games']=len(x); out['ats_accuracy']=x.mean() if len(x) else np.nan
    if 'model_total_win' in bt:
        x=bt.model_total_win.dropna(); out['total_games']=len(x); out['total_side_accuracy']=x.mean() if len(x) else np.nan
    return out
