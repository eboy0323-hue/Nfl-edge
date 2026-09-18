"""Transparent research baselines, not validated wagering probabilities."""
import math
import numpy as np
import pandas as pd

def american_to_implied(odds):
    odds=float(odds)
    if odds == 0: raise ValueError('American odds cannot be zero')
    return -odds/(100-odds) if odds < 0 else 100/(100+odds)

def expected_value_per_100(p, odds):
    if not 0 <= p <= 1: raise ValueError('Probability outside [0,1]')
    profit=100* (float(odds)/100 if odds>0 else 100/abs(float(odds)))
    return p*profit-(1-p)*100

def cdf(x): return .5*(1+math.erf(x/math.sqrt(2)))

def completed_games(schedules, asof):
    d=schedules.copy()
    d['gameday']=pd.to_datetime(d['gameday'], errors='coerce')
    return d[(d.gameday < pd.Timestamp(asof).normalize()) & d.home_score.notna() & d.away_score.notna()].copy()

def build_team_form(schedules, season, asof=None, decay=.82, prior_games=6):
    """Prior-season information is strongly shrunk; current season added as it arrives."""
    if asof is None: asof=pd.Timestamp.now(tz='America/New_York').date()
    d=completed_games(schedules, asof)
    d=d[d.season.isin([season-1,season])].sort_values(['gameday','week'])
    if d.empty: return pd.DataFrame(columns=['team','off_rating','def_rating','games','prior_games'])
    rows=[]
    for g in d.itertuples():
        rows.extend([(g.home_team,g.season,g.home_score,g.away_score,g.gameday),
                     (g.away_team,g.season,g.away_score,g.home_score,g.gameday)])
    x=pd.DataFrame(rows,columns=['team','season','pf','pa','date'])
    league=float(x.pf.mean())
    out=[]
    for team,t in x.groupby('team'):
        prev=t[t.season==season-1].sort_values('date').tail(8)
        cur=t[t.season==season].sort_values('date')
        # Prior season is a small, decayed prior, not eight full-weight games.
        prior_off=float(prev.pf.mean()-league) if len(prev) else 0.
        prior_def=float(league-prev.pa.mean()) if len(prev) else 0.
        n=len(cur)
        w=np.array([decay**(n-1-i) for i in range(n)]) if n else np.array([])
        effective=float(w.sum())
        denom=prior_games+effective
        off=(prior_games*.35*prior_off+float(np.dot(w,cur.pf-league)))/denom
        defense=(prior_games*.35*prior_def+float(np.dot(w,league-cur.pa)))/denom
        out.append(dict(team=team,off_rating=off,def_rating=defense,games=n,prior_games=len(prev)))
    return pd.DataFrame(out)

def project_game(home_team,away_team,form,home_field=1.5,league_points=22.5):
    f=form.set_index('team') if not form.empty else pd.DataFrame()
    def val(team,col): return float(f.loc[team,col]) if team in f.index else 0.
    # Defensive rating is positive for stingy defenses: subtract it, never double-count it.
    hp=league_points+val(home_team,'off_rating')-val(away_team,'def_rating')+home_field/2
    ap=league_points+val(away_team,'off_rating')-val(home_team,'def_rating')-home_field/2
    hp=max(7.,min(42.,hp));ap=max(7.,min(42.,ap))
    margin=hp-ap; total=hp+ap
    return dict(home_points=hp,away_points=ap,projected_margin=margin,projected_total=total,
                home_win_prob=cdf(margin/13.4),margin_sd=13.4,total_sd=13.)

def spread_cover_prob(margin,home_spread,sd=13.4): return cdf((margin+home_spread)/sd)
def over_prob(total,line,sd=13.): return cdf((total-line)/sd)

def player_projection(stats, player, metric, season, asof, recent=6):
    """Descriptive rolling mean/SD; normal approximation is uncalibrated."""
    d=stats.copy()
    if metric not in d.columns: raise ValueError('Stat not available in source')
    d=d[(d.player_display_name==player)&(d.season==season)&(d[metric].notna())]
    if 'week' in d.columns: d=d.sort_values('week')
    if 'game_date' in d.columns:
        dates=pd.to_datetime(d.game_date,errors='coerce');d=d[dates<pd.Timestamp(asof)]
    elif 'week' in d.columns:
        # Caller must supply completed week bound when player stats lack dates.
        raise ValueError('Player stats lack game dates; cannot prevent future-data leakage safely.')
    d=d.tail(recent)
    if len(d)<3: return None
    a=d[metric].astype(float).to_numpy()
    return dict(mean=float(a.mean()),sd=max(float(a.std(ddof=1)),max(1.,float(a.mean())*.25)),games=len(a))

def first_td_proxy(stats,team,season,completed_week):
    """Uncalibrated relative TD shares, NOT first-TD probabilities or odds."""
    d=stats[(stats.season==season)&(stats.recent_team==team)&(stats.week<=completed_week)].copy()
    tdcols=[c for c in ['rushing_tds','receiving_tds'] if c in d.columns]
    if not tdcols: return pd.DataFrame()
    d['td']=d[tdcols].fillna(0).sum(axis=1)
    agg=d.groupby('player_display_name',as_index=False).agg(td=('td','sum'),games=('week','nunique'))
    agg=agg[agg.games>=1].copy()
    agg['relative_td_share']=(agg.td+.25)/(agg.td.sum()+.25*len(agg))
    return agg.sort_values('relative_td_share',ascending=False)
