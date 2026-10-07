import math
import numpy as np
import pandas as pd


def american_to_implied(odds: float) -> float:
    odds = float(odds)
    return (-odds) / ((-odds) + 100.0) if odds < 0 else 100.0 / (odds + 100.0)


def prob_to_american(p: float) -> int:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


def expected_value_per_100(p_win: float, american_odds: float) -> float:
    odds = float(american_odds)
    profit = odds if odds > 0 else 10000.0 / abs(odds)
    return p_win * profit - (1 - p_win) * 100.0


def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def build_team_form(schedule: pd.DataFrame, season: int, as_of=None, decay: float = 0.82) -> pd.DataFrame:
    """Confirmed V1 logic, with no future leakage and the defense adjustment preserved."""
    df = schedule.copy()
    if as_of is None:
        as_of = pd.Timestamp.now().normalize()
    as_of = pd.Timestamp(as_of)
    gameday = pd.to_datetime(df["gameday"])
    df = df[(df["season"] == season) & (gameday < as_of) & df["home_score"].notna() & df["away_score"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=["team","off_rating","def_rating","net_rating","avg_total","games"])

    rows = []
    for _, g in df.sort_values(["week"]).iterrows():
        rows.append({"team":g.home_team,"week":g.week,"pf":g.home_score,"pa":g.away_score})
        rows.append({"team":g.away_team,"week":g.week,"pf":g.away_score,"pa":g.home_score})
    x = pd.DataFrame(rows).sort_values(["team","week"])
    league_pf = x["pf"].mean()
    out = []
    for team, t in x.groupby("team"):
        t = t.sort_values("week")
        n = len(t)
        weights = np.array([decay ** (n-1-i) for i in range(n)], dtype=float)
        weights /= weights.sum()
        pf = np.average(t["pf"], weights=weights)
        pa = np.average(t["pa"], weights=weights)
        out.append({"team": team, "off_rating": pf-league_pf, "def_rating": league_pf-pa,
                    "net_rating": pf-pa, "avg_total": np.average(t["pf"]+t["pa"], weights=weights), "games": n})
    return pd.DataFrame(out)


def project_game(home_team, away_team, form: pd.DataFrame, home_field=1.8):
    f = form.set_index("team") if not form.empty else pd.DataFrame()
    def get(team, col, default=0.0):
        try: return float(f.loc[team, col])
        except Exception: return default
    ho, hd = get(home_team,"off_rating"), get(home_team,"def_rating")
    ao, ad = get(away_team,"off_rating"), get(away_team,"def_rating")
    league_team_pts = 22.5
    # Confirmed phone edits: defense influence reduced to 50%.
    home_pts = league_team_pts + ho - 0.5 * ad + home_field/2
    away_pts = league_team_pts + ao - 0.5 * hd - home_field/2
    margin, total = home_pts-away_pts, home_pts+away_pts
    margin_sd, total_sd = 13.4, 13.0
    p = normal_cdf(margin/margin_sd)
    return {"home_points":round(home_pts,1),"away_points":round(away_pts,1),"projected_margin":round(margin,2),
            "projected_total":round(total,2),"home_win_prob":p,"home_fair_ml":prob_to_american(p),
            "away_fair_ml":prob_to_american(1-p),"margin_sd":margin_sd,"total_sd":total_sd}


def spread_cover_prob(projected_margin, home_spread, margin_sd=13.4):
    return normal_cdf((projected_margin + float(home_spread))/margin_sd)


def over_prob(projected_total, market_total, total_sd=13.0):
    return normal_cdf((projected_total-float(market_total))/total_sd)


def walk_forward_backtest(schedule: pd.DataFrame, season: int, decay=0.82) -> pd.DataFrame:
    """True walk-forward: each game uses only earlier dates from the same season."""
    df = schedule[(schedule["season"]==season) & schedule["home_score"].notna() & schedule["away_score"].notna()].copy()
    df["gameday_dt"] = pd.to_datetime(df["gameday"])
    df = df.sort_values(["gameday_dt","week","gametime"])
    rows=[]
    for _, g in df.iterrows():
        form = build_team_form(schedule, season, as_of=g.gameday_dt, decay=decay)
        # Skip Week 1 / games with no prior current-season evidence.
        if form.empty or g.home_team not in set(form.team) or g.away_team not in set(form.team):
            continue
        p=project_game(g.home_team,g.away_team,form)
        actual_margin=float(g.home_score-g.away_score); actual_total=float(g.home_score+g.away_score)
        r={"week":int(g.week),"gameday":str(g.gameday),"away":g.away_team,"home":g.home_team,
           "projected_margin":p["projected_margin"],"actual_margin":actual_margin,"margin_error":abs(p["projected_margin"]-actual_margin),
           "projected_total":p["projected_total"],"actual_total":actual_total,"total_error":abs(p["projected_total"]-actual_total),
           "winner_correct": (p["projected_margin"]>0)==(actual_margin>0) if actual_margin != 0 else np.nan}
        # nflverse schedules normally expose home-side spread_line and total_line; use only if present.
        if "spread_line" in g.index and pd.notna(g.get("spread_line")):
            market_home_spread=float(g.get("spread_line"))
            r["market_home_spread"]=market_home_spread
            r["model_side"]="HOME" if p["projected_margin"] > -market_home_spread else "AWAY"
            home_cover = actual_margin + market_home_spread > 0
            away_cover = actual_margin + market_home_spread < 0
            r["model_side_win"] = home_cover if r["model_side"]=="HOME" else away_cover
        if "total_line" in g.index and pd.notna(g.get("total_line")):
            tl=float(g.get("total_line")); r["market_total"]=tl
            r["model_total_side"]="OVER" if p["projected_total"] > tl else "UNDER"
            r["model_total_win"]=(actual_total>tl) if r["model_total_side"]=="OVER" else (actual_total<tl)
        rows.append(r)
    return pd.DataFrame(rows)


def validation_summary(bt: pd.DataFrame) -> dict:
    if bt.empty: return {}
    out={"games":len(bt),"margin_mae":bt.margin_error.mean(),"total_mae":bt.total_error.mean(),"winner_accuracy":bt.winner_correct.dropna().mean()}
    if "model_side_win" in bt: out["ats_accuracy"]=bt.model_side_win.dropna().mean()
    if "model_total_win" in bt: out["total_side_accuracy"]=bt.model_total_win.dropna().mean()
    return out
