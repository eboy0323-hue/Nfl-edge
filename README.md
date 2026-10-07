# NFL EDGE V3 — Validation Build

This build turns NFL EDGE from a demo signal generator into a validation-first research app.

## What changed
- Preserves the confirmed V1 phone edits: no same-day/future leakage and 50% defensive adjustment.
- Adds true walk-forward backtesting: each game is projected only from games completed before it.
- Adds a Validation tab with margin MAE, total MAE, winner accuracy, week-by-week results, optional ATS/total-side grading when nflverse exposes market lines, and CSV export.
- Removes automatic BET SIGNAL claims. Raw EV remains visible for research but is explicitly unvalidated.
- Totals are hard-labeled PASS pending a rebuilt totals engine.
- Player props / First TD are not presented as models until their own validation exists.

## Deploy safely
Use the existing `v2-test` branch or another test branch. Replace `app.py` and `model.py`; `data.py` and `requirements.txt` can remain, or upload the full package. Do not merge to `main` until the Validation tab runs successfully in Streamlit.

## Important
A backtest can reveal whether a model deserves further work; it cannot guarantee future profit. Historical sportsbook fields also need to be checked for their exact line convention before treating ATS results as final.
