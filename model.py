import math
import numpy as np
import pandas as pd

TEAM_MAP = {
    "LA": "LAR",
    "LV": "LV",
    "WSH": "WAS",
}

def american_to_implied(odds: float) -> float:
    odds = float(odds)
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    return 100.0 / (odds + 100.0)

def prob_to_american(p: float) -> int:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    if p >= 0.5:
        return int(round(-100 * p / (1 - p)))
    return int(round(100 * (1 - p) / p))

def expected_value_per_100(p_win: float, american_odds: float) -> float:
    odds = float(american_odds)
    if odds > 0:
        profit = odds
    else:
        profit = 10000.0 / abs(odds)
    return p_win * profit - (1 - p_win) * 100.0

def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def build_team_form(schedule: pd.DataFrame, season: int, decay: float = 0.82) -> pd.DataFrame:
    df = schedule.copy()
    df = df[pd.to_datetime(df["gameday"]) < pd.Timestamp.now().normalize()]
    df = df[(df["season"] == season) & df["home_score"].notna() & df["away_score"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=["team","off_rating","def_rating","net_rating","avg_total","games"])

    rows = []
    for _, g in df.sort_values(["week"]).iterrows():
        rows.append({"team":g.home_team,"week":g.week,"pf":g.home_score,"pa":g.away_score})
        rows.append({"team":g.away_team,"week":g.week,"pf":g.away_score,"pa":g.home_score})

    x = pd.DataFrame(rows).sort_values(["team","week"])
    out = []
    league_pf = x["pf"].mean()

    for team, t in x.groupby("team"):
        t = t.sort_values("week").copy()
        n = len(t)
        weights = np.array([decay ** (n-1-i) for i in range(n)], dtype=float)
        weights /= weights.sum()
        pf = np.average(t["pf"], weights=weights)
        pa = np.average(t["pa"], weights=weights)
        out.append({
            "team": team,
            "off_rating": pf - league_pf,
            "def_rating": league_pf - pa,
            "net_rating": pf - pa,
            "avg_total": np.average(t["pf"] + t["pa"], weights=weights),
            "games": n,
        })
    return pd.DataFrame(out)

def project_game(home_team, away_team, form: pd.DataFrame, home_field=1.8):
    f = form.set_index("team") if not form.empty else pd.DataFrame()
    def get(team, col, default=0.0):
        try:
            return float(f.loc[team, col])
        except Exception:
            return default

    home_off = get(home_team, "off_rating")
    home_def = get(home_team, "def_rating")
    away_off = get(away_team, "off_rating")
    away_def = get(away_team, "def_rating")

    # Baseline league scoring assumption for first pass.
    league_team_pts = 22.5
    home_pts = league_team_pts + home_off - away_def * 0.5 + home_field/2
    away_pts = league_team_pts + away_off - home_def * 0.5 - home_field/2

    projected_margin = home_pts - away_pts
    projected_total = home_pts + away_pts

    # Empirical NFL-ish residual scales as a V1 baseline.
    margin_sd = 13.4
    total_sd = 13.0

    home_win_p = normal_cdf(projected_margin / margin_sd)
    return {
        "home_points": round(home_pts, 1),
        "away_points": round(away_pts, 1),
        "projected_margin": round(projected_margin, 2),
        "projected_total": round(projected_total, 2),
        "home_win_prob": home_win_p,
        "home_fair_ml": prob_to_american(home_win_p),
        "away_fair_ml": prob_to_american(1-home_win_p),
        "margin_sd": margin_sd,
        "total_sd": total_sd,
    }

def spread_cover_prob(projected_margin: float, home_spread: float, margin_sd=13.4):
    # home covers when actual margin + home_spread > 0
    threshold = -float(home_spread)
    z = (projected_margin - threshold) / margin_sd
    return normal_cdf(z)

def over_prob(projected_total: float, market_total: float, total_sd=13.0):
    z = (projected_total - market_total) / total_sd
    return normal_cdf(z)
