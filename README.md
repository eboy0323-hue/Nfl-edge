# NFL EDGE V1

A free-data NFL betting dashboard focused on Fanatics Sportsbook.

## What V1 does
- Pulls free NFL schedule/team/player data from nflverse through `nflreadpy`
- Builds rolling team power features
- Produces baseline projections for:
  - spread
  - total
  - moneyline
- Lets you enter Fanatics prices manually in the dashboard
- Calculates:
  - model edge
  - implied probability
  - fair odds
  - expected value
  - bet/no-bet signal
- Includes placeholders for:
  - player props
  - first touchdown
  - same-game parlays
- Stores a bet log so the model can be evaluated over time

## Install
1. Install Python 3.11+
2. Open a terminal in this folder
3. Run:
   `pip install -r requirements.txt`
4. Start:
   `streamlit run app.py`

## Important
This is a research/betting-assistance model, not a guarantee of profit.
V1 deliberately avoids scraping Fanatics. Enter Fanatics lines manually in the dashboard.

## V2 roadmap
- walk-forward historical backtest
- calibrated cover/over probabilities
- player opportunity + usage models
- anytime/first-TD hazard model
- correlation-aware SGP simulator
- CLV tracker
- automatic weekly report export
