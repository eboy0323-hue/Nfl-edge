import pandas as pd

def load_nfl_data():
    import nflreadpy as nfl
    schedules = nfl.load_schedules(seasons=True).to_pandas()
    try:
        player_stats = nfl.load_player_stats(seasons=True).to_pandas()
    except Exception:
        player_stats = pd.DataFrame()
    return schedules, player_stats
