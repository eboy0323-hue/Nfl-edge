import datetime as dt
import pandas as pd
import streamlit as st

from data import load_nfl_data
from model import (
    build_team_form, project_game, spread_cover_prob, over_prob,
    american_to_implied, expected_value_per_100
)

st.set_page_config(page_title="NFL EDGE", layout="wide")
st.title("🏈 NFL EDGE")
st.caption("Free-data NFL betting research dashboard • Fanatics manual-line workflow • V1")

@st.cache_data(ttl=3600)
def get_data():
    return load_nfl_data()

with st.sidebar:
    st.header("Settings")
    edge_threshold = st.slider("Minimum probability edge", 0.0, 0.10, 0.025, 0.005)
    min_ev = st.slider("Minimum EV / $100", -10.0, 25.0, 2.0, 0.5)
    st.caption("V1 default thresholds are intentionally conservative.")

try:
    schedules, player_stats = get_data()
except Exception as e:
    st.error(f"Could not load nflverse data: {e}")
    st.stop()

season = int(schedules["season"].max())
today = pd.Timestamp.now().normalize()
form = build_team_form(schedules, season)

future = schedules[
    (schedules["season"] == season) &
    (pd.to_datetime(schedules["gameday"]) >= today - pd.Timedelta(days=1))
].copy().sort_values(["gameday","gametime"])

if future.empty:
    st.warning("No upcoming games found.")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Game Model", "Player Props", "First TD", "SGP Lab", "Bet Log"]
)

with tab1:
    st.subheader("Game markets")
    game_labels = [
        f"{r.away_team} @ {r.home_team} — W{int(r.week)} — {r.gameday}"
        for _, r in future.head(32).iterrows()
    ]
    idx = st.selectbox("Game", range(len(game_labels)), format_func=lambda i: game_labels[i])
    g = future.head(32).iloc[idx]

    proj = project_game(g.home_team, g.away_team, form)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Projected score", f"{g.away_team} {proj['away_points']} – {g.home_team} {proj['home_points']}")
    c2.metric("Projected margin", f"{g.home_team} {proj['projected_margin']:+.1f}")
    c3.metric("Projected total", f"{proj['projected_total']:.1f}")
    c4.metric("Home fair ML", f"{proj['home_fair_ml']:+d}")

    st.divider()
    st.markdown("### Enter Fanatics lines")

    a,b,c = st.columns(3)
    with a:
        home_spread = st.number_input(f"{g.home_team} spread", value=-2.5, step=0.5)
        spread_odds = st.number_input("Spread odds", value=-110, step=5)
        p_cover = spread_cover_prob(proj["projected_margin"], home_spread, proj["margin_sd"])
        implied = american_to_implied(spread_odds)
        ev = expected_value_per_100(p_cover, spread_odds)
        st.metric("Model cover probability", f"{p_cover:.1%}", f"{(p_cover-implied):+.1%} edge")
        st.metric("Spread EV / $100", f"${ev:.2f}")
        if (p_cover-implied) >= edge_threshold and ev >= min_ev:
            st.success("BET SIGNAL")
        else:
            st.info("NO BET")

    with b:
        market_total = st.number_input("Game total", value=45.5, step=0.5)
        over_odds = st.number_input("Over odds", value=-110, step=5)
        p_over = over_prob(proj["projected_total"], market_total, proj["total_sd"])
        implied_o = american_to_implied(over_odds)
        ev_o = expected_value_per_100(p_over, over_odds)
        st.metric("Model Over probability", f"{p_over:.1%}", f"{(p_over-implied_o):+.1%} edge")
        st.metric("Over EV / $100", f"${ev_o:.2f}")
        if (p_over-implied_o) >= edge_threshold and ev_o >= min_ev:
            st.success("BET SIGNAL")
        else:
            st.info("NO BET")

    with c:
        home_ml = st.number_input(f"{g.home_team} moneyline", value=-130, step=5)
        p_ml = proj["home_win_prob"]
        implied_ml = american_to_implied(home_ml)
        ev_ml = expected_value_per_100(p_ml, home_ml)
        st.metric("Model win probability", f"{p_ml:.1%}", f"{(p_ml-implied_ml):+.1%} edge")
        st.metric("ML EV / $100", f"${ev_ml:.2f}")
        if (p_ml-implied_ml) >= edge_threshold and ev_ml >= min_ev:
            st.success("BET SIGNAL")
        else:
            st.info("NO BET")

    st.caption("V1 projections are baseline rolling-score models. Do not treat them as production-grade until walk-forward backtesting and calibration are added.")

with tab2:
    st.subheader("Player props — V1 data foundation")
    st.write("Player-stat ingestion is wired in. The next model layer will estimate distributions for passing, rushing and receiving volume.")
    if not player_stats.empty:
        cols = [c for c in ["player_display_name","recent_team","week","passing_yards","rushing_yards","receiving_yards","targets","carries"] if c in player_stats.columns]
        if cols:
            st.dataframe(player_stats[player_stats["season"]==season][cols].tail(100), use_container_width=True)
    st.info("V2: usage + opponent + game-script model, then Fanatics line/price comparison.")

with tab3:
    st.subheader("First touchdown model")
    st.write("Planned probability inputs:")
    st.markdown("- red-zone opportunity share\n- inside-10 carries/targets\n- snap/route participation\n- team implied TD expectation\n- opponent TD allowance by position\n- opening-drive usage")
    st.info("V2 will normalize player TD hazards so probabilities are coherent within each game.")

with tab4:
    st.subheader("Same-game parlay lab")
    st.write("SGPs cannot be priced correctly by multiplying independent probabilities because legs are correlated.")
    st.markdown("- team result ↔ QB/receiver overs\n- rushing overs ↔ favorite game scripts\n- unders ↔ rushing volume / clock effects\n- TD scorer ↔ team scoring expectation")
    st.info("V3 will use simulation to estimate joint probabilities and compare them with Fanatics SGP prices.")

with tab5:
    st.subheader("Bet log")
    cols = ["date","game","market","selection","fanatics_odds","model_prob","implied_prob","edge","ev100","result","closing_line"]
    if "betlog" not in st.session_state:
        st.session_state.betlog = pd.DataFrame(columns=cols)

    with st.form("add_bet"):
        game = st.text_input("Game")
        market = st.selectbox("Market", ["Spread","Total","Moneyline","Player prop","First TD","SGP"])
        selection = st.text_input("Selection")
        odds = st.number_input("Fanatics odds", value=-110, step=5)
        model_prob = st.number_input("Model probability", value=0.55, min_value=0.0, max_value=1.0, step=0.01)
        submit = st.form_submit_button("Add bet")
        if submit:
            imp = american_to_implied(odds)
            ev = expected_value_per_100(model_prob, odds)
            row = {
                "date": dt.date.today().isoformat(),
                "game": game, "market": market, "selection": selection,
                "fanatics_odds": odds, "model_prob": model_prob,
                "implied_prob": imp, "edge": model_prob-imp,
                "ev100": ev, "result": "", "closing_line": ""
            }
            st.session_state.betlog = pd.concat([st.session_state.betlog, pd.DataFrame([row])], ignore_index=True)

    st.dataframe(st.session_state.betlog, use_container_width=True)
    st.download_button(
        "Download bet log CSV",
        st.session_state.betlog.to_csv(index=False).encode(),
        "nfl_edge_betlog.csv",
        "text/csv"
    )
