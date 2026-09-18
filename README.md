# NFL EDGE V2 — experimental research preview

Deploy on Streamlit Community Cloud using `app.py` and the included `requirements.txt`.

Changes from V1: excludes games on/after current local date from form calculations; includes previous-season prior with strong shrinkage; corrects defensive sign and removes prior ad-hoc 0.5 defensive patch; expands matchup selection to 64 games; removes misleading BET SIGNAL labels; first-TD tab now shows *descriptive TD shares only*, explicitly not first-TD probabilities. Props remain raw stats; SGP remains unavailable. Paper log is session-only, export manually.

IMPORTANT: The original V1 GitHub edits were reconstructed from the user's confirmation, not downloaded from their live repository. The V1 0.5 defensive multiplier is deliberately superseded by a coherent formula, not silently copied. There is no verified Fanatics odds feed, live-game model, first-TD predictor, prop predictor, calibration, or backtest. Do not wager using this model. Current-day games remain visible for selection, but their projections are pregame snapshots, NOT in-play projections. Data cache refreshes hourly. Timezone in this preview uses fixed UTC-4 and should be replaced with zoneinfo America/New_York before winter.

To update GitHub on iPhone, upload all V2 files to repository root, replacing files with the same names. Streamlit will redeploy. Keep a backup of V1 first.
