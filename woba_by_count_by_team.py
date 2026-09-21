import pandas as pd
import numpy as np
import os

statcast = pd.read_parquet("data/statcast_with_rv.parquet")
woba = pd.read_csv("outputs/woba_pitcher.csv")

if "game_year" not in statcast.columns:
    statcast["game_year"] = pd.to_datetime(statcast["game_date"]).dt.year

print(statcast[["balls", "strikes", "description", "events", "pitch_type",
                "estimated_woba_using_speedangle"]].head(10))

print("\nNull Rates:")
for col in ["balls", "strikes", "pitch_type",
            "estimated_woba_using_speedangle"]:
    pct = statcast[col].isna().mean() * 100
    print(f"{col}: {pct:.1f}% is null")

statcast["count"] = statcast["balls"].astype(str) + "-" + statcast["strikes"].astype(str)

print("\nMost common count states: ")
print(statcast["count"].value_counts().head(12))

print("Building Features: ")

statcast["count"] = (
    statcast["balls"].astype(int).astype(str) + "-" +
    statcast["strikes"].astype(int).astype(str)
)

statcast["pitching_team"] = np.where(
    statcast["inning_topbot"] == "Top",
    statcast["home_team"],
    statcast["away_team"]
)

statcast["pitching_team"] = statcast["pitching_team"].replace("ATH", "OAK")

weights_df = pd.read_csv("outputs/woba_weights_by_season.csv")

weights_df = weights_df.set_index("game_year")

woba_weights = {int(yr): row.to_dict() for yr, row in weights_df.iterrows()}

print(list(woba_weights.keys()))  # should show [2022, 2023, 2024, 2025]
print(woba_weights[2022])

print("wOBA weights loaded: ")
print(weights_df.reset_index()[["game_year", "BB", "HBP", "1B", "2B", "3B", "HR"]].round(4))

pa = statcast[
    statcast["events"].notna() & (statcast["events"] != "")
].copy()

print(f"\nTotal Pitches: {len(statcast):}")
print(f"PA-ending Pitches: {len(pa):}")

# Map events to wOBA components

print("\nMapping wOBA components: ")

pa["is_bb"] = (pa["events"] == "walk").astype(int)
pa["is_hbp"] = (pa["events"] == "hit_by_pitch").astype(int)
pa["is_1b"] = (pa["events"] == "single").astype(int)
pa["is_2b"] = (pa["events"] == "double").astype(int)
pa["is_3b"] = (pa["events"] == "triple").astype(int)
pa["is_hr"] = (pa["events"] == "home_run").astype(int)

out_events = {
    "strikeout", "strikeout_double_play", "grounded_into_double_play",
    "double_play", "triple_play", "field_out", "fielders_choice_out",
    "other_out", "sac_fly"
}

pa["is_out"] = pa["events"].isin(out_events).astype(int)

# PA denominator = hits + walks + outs + sac fly

pa["is_pa_denom"] = (
    pa["is_out"] + pa["is_1b"] + pa["is_2b"] + pa["is_3b"] + pa["is_hr"] + pa["is_bb"] + pa["is_hbp"]
)

# Force consistent types before apply
pa["game_year"] = pa["game_year"].astype(int)

# Also the apply is very slow on millions of rows
# Use vectorized approach instead
def compute_woba_vectorized(pa, woba_weights):
    pa = pa.copy()
    pa["woba_numerator"] = 0.0
    
    for year, w in woba_weights.items():
        mask = pa["game_year"] == year
        pa.loc[mask, "woba_numerator"] = (
            w.get("BB",  0) * pa.loc[mask, "is_bb"]  +
            w.get("HBP", 0) * pa.loc[mask, "is_hbp"] +
            w.get("1B",  0) * pa.loc[mask, "is_1b"]  +
            w.get("2B",  0) * pa.loc[mask, "is_2b"]  +
            w.get("3B",  0) * pa.loc[mask, "is_3b"]  +
            w.get("HR",  0) * pa.loc[mask, "is_hr"]
        )
    return pa

pa = compute_woba_vectorized(pa, woba_weights)

print(f"woba_numerator null after fix: {pa['woba_numerator'].isna().sum():,}")
print(f"woba_numerator > 0: {(pa['woba_numerator'] > 0).sum():,}")
print(pa["woba_numerator"].describe())

print(f"wOBA numerator null: {pa['woba_numerator'].isna().sum():}")

print("\nAggregating by team, pitch type, and count: ")

group_cols = ["pitching_team", "pitch_type", "count", "game_year"]

agg = pa.groupby(group_cols).agg(
    pa_count = ("is_pa_denom", "sum"),
    woba_num = ("woba_numerator", "sum"),
    xwoba_sum = ("estimated_woba_using_speedangle", "sum"),
    xwoba_n = ("estimated_woba_using_speedangle", "count"),
    bb = ("is_bb", "sum"),
    hbp = ("is_hbp", "sum"),
    single = ("is_1b", "sum"),
    double = ("is_2b", "sum"),
    triple = ("is_3b", "sum"),
    home_run = ("is_hr", "sum"),
    k = ("events", lambda x: (x == "strikeout").sum())
).reset_index()

agg["wOBAA"] = (agg["woba_num"] / agg["pa_count"]).round(4)

agg["xwOBAA"] = (agg["xwoba_sum"] / agg["xwoba_n"]).round(4)

min_pa = 20

agg_clean = agg[agg["pa_count"] >= min_pa].copy()

print(f"\nRows before PA filter: {len(agg):}")

print(f"Rows after {min_pa} + PA filter: {len(agg_clean):}")

league_avg = (
    agg_clean.groupby(["pitch_type", "count", "game_year"])
    .agg(
        league_wOBAA = ("wOBAA", "mean"),
        league_xwOBAA = ("xwOBAA", "mean")
    )
    .reset_index()
    .round(4)
)

agg_clean = agg_clean.merge(
    league_avg,
    on = ["pitch_type", "count", "game_year"],
    how="left"
)

agg_clean["wOBAA_minus"] = (
    100 * agg_clean["wOBAA"] / agg_clean["league_wOBAA"]
).round(1)

agg_clean["xwOBAA_minus"] = (
    100 * agg_clean["xwOBAA"] / agg_clean["league_xwOBAA"]
).round(1)

# Pitcher Level

group_cols_pitcher = ["player_name", "game_year", "pitch_type", "count"]

pitcher_agg = pa.groupby(group_cols_pitcher).agg(
    pa_count = ("is_pa_denom", "sum"),
    woba_num = ("woba_numerator", "sum"),
    woba_sum = ("estimated_woba_using_speedangle", "sum"),
    woba_n = ("estimated_woba_using_speedangle", "count"),
    bb = ("is_bb", "sum"),
    hbp = ("is_hbp", "sum"),
    single = ("is_1b", "sum"),
    double = ("is_2b", "sum"),
    triple = ("is_3b", "sum"),
    home_run = ("is_hr", "sum"),
    k = ("events", lambda x: (x== "strikeout").sum())
).reset_index()

pitcher_agg["wOBAA"] = (pitcher_agg["woba_num"] / pitcher_agg["pa_count"]).round(4)

pitcher_agg["xwOBAA"] = (pitcher_agg["woba_sum"] / pitcher_agg["woba_n"]).round(4)

pitcher_agg_clean = pitcher_agg[pitcher_agg["pa_count"] >= min_pa].copy()

pitcher_league_avg = (
    pitcher_agg_clean.groupby(["pitch_type", "count", "game_year"])
    .agg(
        league_wOBAA = ("wOBAA", "mean"),
        league_xwOBAA = ("xwOBAA", "mean")
    ).reset_index()
    .round(4)
)

pitcher_agg_clean = pitcher_agg_clean.merge(
    pitcher_league_avg,
    on = (["pitch_type", "count", "game_year"]),
    how = "left"
)

pitcher_agg_clean["wOBAA_minus"] = (
    100 * pitcher_agg_clean["wOBAA"] / pitcher_agg_clean["league_wOBAA"]
).round(1)

pitcher_agg_clean["xwOBAA_minus"] = (
    100 * pitcher_agg_clean["xwOBAA"] / pitcher_agg_clean["league_xwOBAA"]
).round(1)


print("\nSample: Best FF in 0-2 count by team: ")

print(agg_clean[
    (agg_clean["pitch_type"] == "FF") &
    (agg_clean["count"] == "0-2")
]
.sort_values("wOBAA")
.head(10)
[["pitching_team", "game_year", "count", "pitch_type",
  "pa_count", "wOBAA", "xwOBAA", "wOBAA_minus"]]
  .to_string(index = False))

# Print Best Pitcher FF in 0-2 Count

print("\nSample Best FF in 0-2 count by pitcher: ")

print(
    pitcher_agg_clean[
        (pitcher_agg_clean["pitch_type"] == "FF") &
        (pitcher_agg_clean["count"] == "0-2")
    ]
    .sort_values("wOBAA")
    .head(10)
    [["player_name", "game_year", "pitch_type", "count", "pitch_type",
      "pa_count", "wOBAA", "xwOBAA", "wOBAA_minus"]]
      .to_string(index=False)
)

agg_clean.to_csv("outputs/woba_by_team_pitch_count.csv", index=False)
print("\nSaved: outputs/woba_by_team_pitch_count.csv")

pitcher_agg_clean.to_csv("outputs/woba_by_pitcher_pitch_count.csv", index = False)
print("\nSaved: outputs/woba_by_pitcher_pitch_count.csv.csv")