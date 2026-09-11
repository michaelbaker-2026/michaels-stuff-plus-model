import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

os.makedirs("woba_outputs", exist_ok=True)

print("Loading Statcast Parquet: ")

statcast = pd.read_parquet("data/statcast_with_rv.parquet")

if "game_year" not in statcast.columns:
    statcast["game_year"] = pd.to_datetime(statcast['game_date']).dt.year

print(f"Rows: {len(statcast):} | Games: {statcast["game_pk"].nunique()}")

print("\nComputing season-specific wOBA Weights from constructed RE24: ")

pa = statcast[statcast["events"].notna() & (statcast["events"] != "")]

event_rv = (
    pa.groupby(["game_year", "events"])["run_value"]
    .agg(["mean", "count"])
    .reset_index()
    .rename(columns = {"mean": "avg_rv", "count": "n"})
)

woba_events = {
    "walk": "BB",
    "hit_by_pitch": "HBP",
    "single": "1B",
    "double": "2B",
    "triple": "3B",
    "home_run": "HR"
}

# Filter to wOBA events

woba_rv = event_rv[event_rv["events"].isin(woba_events.keys())].copy()

woba_rv["event_label"] = woba_rv["events"].map(woba_events)

print("\nRun-Value by Event and Season: ")
print(
    woba_rv.pivot_table(
        index= "event_label",
        columns= "game_year",
        values = "avg_rv"
    ).round(4).to_string()
)

# Compute wOBA scale factor per season
# wOBA is scaled so that league wOBA = ~league OBP (0.320)
# Scale factor = target_woba / weighted_avg_run_value_per_pa

# League OBP per season from 2022-2025

league_obp = (
    pa.groupby("game_year").apply(lambda x: pd.Series({
        "obp_num": (
            (x["events"] == "single").sum() +
            (x["events"] == "double").sum() +
            (x["events"] == "triple").sum() +
            (x["events"] == "home_run").sum() +
            (x["events"] == "walk").sum() +
            (x["events"] == "hit_by_pitch").sum()
        ),
        "obp_denom": (
            (x["events"] == "single").sum() +
            (x["events"] == "double").sum() +
            (x["events"] == "triple").sum() +
            (x["events"] == "home_run").sum() +
            (x["events"] == "walk").sum() + 
            (x["events"] == "hit_by_pitch").sum() +
            (x["events"].isin(["strikeout", "field_out", "grounded_into_double_play",
                               "double_play", "force_out", "fielders_choice_out",
                               "strikeout_double_play", "other_out", "truncated_pa"])).sum() +
            (x["events"].isin(["sac_fly"])).sum()
        )
    }))
    .reset_index()
)


league_obp["league_obp"] = (
    league_obp["obp_num"]/league_obp["obp_denom"]
).round(4)

print("\nLeague OBP per Season: ")
print(league_obp[["game_year", "league_obp"]].to_string(index = False))

# Compute weighted AVG run value per PA per season
# using raw run values before scaling
woba_weights = {}

for year in sorted(statcast["game_year"].unique()):
    year_rv = woba_rv[woba_rv["game_year"] == year].set_index("event_label")["avg_rv"]
    year_pa = pa[pa["game_year"] == year]
    year_obp = league_obp[league_obp["game_year"] == year]["league_obp"].values[0]

    # Count events
    event_counts = {
        "BB": (year_pa["events"] == "walk").sum(),
        "HBP": (year_pa["events"] == "hit_by_pitch").sum(),
        "1B": (year_pa["events"] == "single").sum(),
        "2B": (year_pa["events"] == "double").sum(),
        "3B": (year_pa["events"] == "triple").sum(),
        "HR": (year_pa["events"] == "home_run").sum()
    }

    total_pa = len(year_pa)

    #Weighted avg run value per PA
    weighted_rv = sum(
        event_counts.get(label, 0) * year_rv.get(label, 0)
        for label in event_counts
    ) / total_pa

    # Scale factor to match league OBP

    scale = year_obp / weighted_rv if weighted_rv != 0 else 1.0

    # Scaled weights

    weights = {
        label: round(year_rv.get(label, 0) * scale, 4)
        for label in ["BB", "HBP", "1B", "2B", "3B", "HR"]
    }

    woba_weights[year] = {
        "scale_factor": round(scale, 4),
        "league_obp": round(year_obp, 4),
        **weights
    }

weights_df = pd.DataFrame(woba_weights).T

weights_df.index.name = "game_year"

print("\nSeason Specific wOBA Weights: ")
print(weights_df.to_string())

# Compute wOBAA per pitcher per season

print("\nComputing wOBAA: ")

def compute_woba(df, weights_by_year, level_cols):
    """Compute wOBAA at any group level"""

    results = []
    for (keys, group) in df.groupby(level_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)

        year = keys[level_cols.index("game_year")]
        w = weights_by_year.get(year, {})

        if not w:
            continue

        # Numerator -- weighted outcomes
        woba_num = (
            w.get("BB", 0) * (group["events"] == "walk").sum() + 
            w.get("HBP", 0) * (group["events"] == "hit_by_pitch").sum() +
            w.get("1B", 0) * (group["events"] == "single").sum() +
            w.get("2B", 0) * (group["events"] == "double").sum() +
            w.get("3B", 0) * (group["events"] == "triple").sum() +
            w.get("HR", 0) * (group["events"] == "home_run").sum()
        )

        woba_denom = (
            (group["events"] == "walk").sum() +
            (group["events"] == "hit_by_pitch").sum() +
            (group["events"].isin(["single", "double", "triple", "home_run"])).sum() +
            (group["events"].isin(["strikeout", "field_out", "grounded_into_double_play",
                                    "double_play", "force_out", "fielders_choice_out",
                                    "strikeout_double_play", "other_out"])).sum() +
            (group["events"] == "sac_fly").sum()
        )

        woba = round(woba_num / woba_denom, 4) if woba_denom > 0 else np.nan

        # xwOBA - using Statcast's estimated_woba_using_speedangle

        xwOBA = group["estimated_woba_using_speedangle"].mean()

        row = dict(zip(level_cols, keys))

        row.update({
            "woba_num": round(woba_num, 4),
            "woba_denom": int(woba_denom),
            "wOBAA": woba,
            "xwOBAA": round(xwOBA, 4) if not np.isnan(xwOBA) else np.nan,
            "BB": (group["events"] == "walk").sum(),
            "HBP": (group["events"] == "hit_by_pitch").sum(),
            "singles": (group["events"] == "single").sum(),
            "doubles": (group["events"] == "double").sum(),
            "triples": (group["events"] == "triple").sum(),
            "HR": (group["events"] == "home_run").sum(),
            "K": (group["events"] == "strikeout").sum()
        })

        results.append(row)

    return pd.DataFrame(results)

# Pitcher Level
print("Computing pitcher-level wOBAA: ")
woba_pitcher = compute_woba(
    pa, 
    woba_weights,
    ["pitcher", "player_name", "game_year"]
)

# Min PA Threshold

woba_pitcher = woba_pitcher[woba_pitcher["woba_denom"] >= 100]

print(f"Pitchers w/ 100+ Batters Faced Events: {len(woba_pitcher):}")

# Pitcher + Pitch-Type Level

print("Computing Pitcher + Pitch Type wOBAA: ")

woba_pitch_type = compute_woba(
    pa,
    woba_weights,
    ["pitcher", "player_name", "pitch_type", "game_year"]
)

woba_pitch_type = woba_pitch_type[woba_pitch_type["woba_denom"] >= 60]

print(f"Pitcher-Pitch Type Combos w/ 60+ Batters Faced Events: {len(woba_pitch_type):}")

# League Average wOBAA per season

league_woba = (
    woba_pitcher.groupby("game_year")["wOBAA"]
    .mean()
    .round(4)
    .reset_index()
    .rename(columns = {"wOBAA": "league_wOBAA"})
)

print("\nLeague Average wOBAA by Season: ")

print(league_woba.to_string(index = False))

# wOBAA+ - normalized to 100 similar to ERA- 
# wOBAA+ < 100 better than average

woba_pitcher = woba_pitcher.merge(league_woba, on = "game_year", how = "left")

woba_pitcher["wOBAA_minus"] = (
    100 * woba_pitcher["wOBAA"] / woba_pitcher["league_wOBAA"]
).round(1)

# Results

print("\n Best wOBAA (lowest = best, min 100 PA)")
print(
    woba_pitcher
    .sort_values("wOBAA")
    .head(20)
    [["player_name", "game_year", "woba_denom", "wOBAA",
      "xwOBAA", "wOBAA_minus", "HR", "K", "BB"]]
    .to_string(index = False)
)

print("\n Worst wOBAA")
print(
    woba_pitcher
    .sort_values("wOBAA", ascending=False)
    .head(10)
    [["player_name", "game_year", "woba_denom", "wOBAA",
      "xwOBAA", "wOBAA_minus", "HR", "K", "BB"]]
      .to_string(index = False)
)

print("\nBest xwOBAA (contact quality): ")
print(
    woba_pitcher
    .sort_values("xwOBAA")
    .head(20)
    [["player_name", "game_year", "xwOBAA", "woba_denom", 
      "wOBAA", "wOBAA_minus", "HR", "K", "BB"]]
)

print("\nwOBAA vs. xwOBAA - Biggest Gaps (Regression Candidates): ")

woba_pitcher["woba_minus_xwoba"] = (woba_pitcher["wOBAA"] - woba_pitcher["xwOBAA"]).round(4)

print("Pitchers Outperforming xwOBAA (Good Luck): ")

print(
    woba_pitcher
    .sort_values("woba_minus_xwoba")
    .head(10)
    [["player_name", "game_year", "wOBAA", "xwOBAA", "woba_minus_xwoba"]]
    .to_string(index = False)
)

print("\nPitchers Underperforming xwOBAA (Bad Luck): ")

print(
    woba_pitcher
    .sort_values("woba_minus_xwoba", ascending = False)
    .head(10)
    [["player_name", "game_year", "wOBAA", "xwOBAA", "woba_minus_xwoba"]]
    .to_string(index = False)
)

print("\nBest xwOBAA Pitcher-Pitch Type Combos: ")
print(
    woba_pitch_type
    .sort_values("xwOBAA", ascending=True)
    .head(20)
    [["player_name", "pitch_type", "game_year", "wOBAA", "xwOBAA"]]
    .to_string(index = False)
)

print("\nWorst xwOBAA Pitcher-Pitch Type Combos: ")
print(
    woba_pitch_type
    .sort_values("xwOBAA", ascending=False)
    .head(10)
    [["player_name", "pitch_type", "game_year", "wOBAA", "xwOBAA"]]
    .to_string(index = False)
)

# Save

woba_pitcher.to_csv("outputs/woba_pitcher.csv", index = False)
woba_pitch_type.to_csv("outputs/woba_pitch_type.csv", index = False)
weights_df.to_csv("outputs/woba_weights_by_season.csv")

print("\nSaved")
print("outputs/woba_pitcher.csv")
print("outputs/woba_pitch_type.csv")
print("outputs/woba_weights.csv")


