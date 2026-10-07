import pandas as pd
import streamlit as st
from data import load_nfl_data
from model import build_team_form, project_game, spread_cover_prob, over_prob, american_to_implied, expected_value_per_100, walk_forward_backtest, validation_summary

st.set_page_config(page_title="NFL EDGE V3", layout="wide")
st.title("🏈 NFL EDGE V3 — Validation Build")
st.caption("Walk-forward validation first • Fanatics prices entered manually • no automatic BET NOW claims")

@st.cache_data(ttl=3600)
def get_data(): return load_nfl_data()
@st.cache_data(ttl=3600)
def get_backtest(schedules, season): return walk_forward_backtest(schedules, season)

try: schedules, player_stats = get_data()
except Exception as e:
    st.error(f"Could not load nflverse data: {e}"); st.stop()
season=int(schedules["season"].max()); today=pd.Timestamp.now().normalize()
form=build_team_form(schedules,season,as_of=today)
future=schedules[(schedules.season==season)&(pd.to_datetime(schedules.gameday)>=today-pd.Timedelta(days=1))].copy().sort_values(["gameday","gametime"])

t1,t2,t3,t4=st.tabs(["Game Model","Validation","Player Data","Roadmap"])
with t1:
    st.subheader("Pregame research")
    st.warning("V3 validation build: projections are research outputs. A positive calculated EV is not a verified betting edge until calibration and holdout testing pass.")
    if future.empty: st.info("No upcoming games found.")
    else:
        games=future.head(32); labels=[f"{r.away_team} @ {r.home_team} — W{int(r.week)} — {r.gameday}" for _,r in games.iterrows()]
        i=st.selectbox("Game",range(len(labels)),format_func=lambda x:labels[x]); g=games.iloc[i]; p=project_game(g.home_team,g.away_team,form)
        c1,c2,c3=st.columns(3); c1.metric("Projected score",f"{g.away_team} {p['away_points']} – {g.home_team} {p['home_points']}"); c2.metric("Projected margin",f"{g.home_team} {p['projected_margin']:+.1f}"); c3.metric("Projected total",f"{p['projected_total']:.1f}")
        st.markdown("### Manual Fanatics comparison")
        a,b,c=st.columns(3)
        with a:
            hs=st.number_input(f"{g.home_team} spread",value=-2.5,step=.5); odds=st.number_input("Spread odds",value=-110,step=5)
            pc=spread_cover_prob(p['projected_margin'],hs,p['margin_sd']); imp=american_to_implied(odds); ev=expected_value_per_100(pc,odds)
            st.metric("Raw cover probability",f"{pc:.1%}",f"{pc-imp:+.1%} vs implied"); st.metric("Raw EV / $100",f"${ev:.2f}"); st.info("RESEARCH — validation gate not passed")
        with b:
            tl=st.number_input("Game total",value=45.5,step=.5); oo=st.number_input("Over odds",value=-110,step=5)
            po=over_prob(p['projected_total'],tl,p['total_sd']); imp=american_to_implied(oo); ev=expected_value_per_100(po,oo)
            st.metric("Raw Over probability",f"{po:.1%}",f"{po-imp:+.1%} vs implied"); st.metric("Raw EV / $100",f"${ev:.2f}"); st.info("PASS — totals engine requires rebuild/validation")
        with c:
            ml=st.number_input(f"{g.home_team} moneyline",value=-130,step=5); imp=american_to_implied(ml); ev=expected_value_per_100(p['home_win_prob'],ml)
            st.metric("Raw win probability",f"{p['home_win_prob']:.1%}",f"{p['home_win_prob']-imp:+.1%} vs implied"); st.metric("Raw EV / $100",f"${ev:.2f}"); st.info("RESEARCH — validation gate not passed")
with t2:
    st.subheader("2026 walk-forward scoreboard")
    st.write("Every game below is projected using only earlier games from the same season. Week 1 is skipped because V1 has no current-season evidence before Week 1.")
    bt=get_backtest(schedules,season); s=validation_summary(bt)
    if not s: st.warning("No completed games available for backtest.")
    else:
        cols=st.columns(5); cols[0].metric("Games",s['games']); cols[1].metric("Margin MAE",f"{s['margin_mae']:.1f}"); cols[2].metric("Total MAE",f"{s['total_mae']:.1f}"); cols[3].metric("Winner accuracy",f"{s['winner_accuracy']:.1%}"); cols[4].metric("ATS*",f"{s.get('ats_accuracy',float('nan')):.1%}" if 'ats_accuracy' in s else "N/A")
        by=bt.groupby('week').agg(games=('week','size'),margin_mae=('margin_error','mean'),total_mae=('total_error','mean'),winner_accuracy=('winner_correct','mean')).reset_index()
        st.dataframe(by,use_container_width=True,hide_index=True)
        st.download_button("Download full backtest CSV",bt.to_csv(index=False).encode(),"nfl_edge_v3_backtest.csv","text/csv")
        st.caption("*ATS/total-side fields appear only when nflverse supplies historical market lines. Pushes are excluded by pandas when represented as missing values.")
with t3:
    st.subheader("Player data foundation")
    st.write("Props and First TD remain disabled until a separate player-usage model is validated.")
    if not player_stats.empty:
        cs=[c for c in ['player_display_name','recent_team','week','passing_yards','rushing_yards','receiving_yards','targets','carries'] if c in player_stats.columns]
        if cs: st.dataframe(player_stats[player_stats.season==season][cs].tail(100),use_container_width=True)
with t4:
    st.subheader("V3 gates")
    st.markdown("**Spread/ML:** preserve V1 rolling team-strength engine, measure walk-forward error and ATS behavior, then calibrate probabilities on historical seasons.\n\n**Totals:** current engine stays PASS until rebuilt and validated.\n\n**Props / First TD:** require player-level usage + opponent + game-script models.\n\n**SGP:** requires joint simulation after individual legs are calibrated.\n\n**Fanatics:** manual verified price entry remains mandatory; the app never invents a Fanatics price.")
