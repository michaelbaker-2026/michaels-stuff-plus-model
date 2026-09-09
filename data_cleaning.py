import pandas as pd
import numpy as np
import os

statcast_2022_2025 = pd.read_parquet("~/Downloads/statcast_data.parquet")

print(f"Rows: {len(statcast_2022_2025):} | Games: {len(statcast_2022_2025['game_pk'].unique()):} | Players: {len(statcast_2022_2025['player_name'].unique()):}")


statcast_2022_2025 = statcast_2022_2025[statcast_2022_2025["game_type"] == "R"].copy()

statcast_2022_2025["game_date"] = pd.to_datetime(statcast_2022_2025["game_date"])
statcast_2022_2025 = statcast_2022_2025[(statcast_2022_2025["game_date"] >= "2022-03-01") & (statcast_2022_2025["game_date"] <= "2025-10-2")]

print(f"Post Regular Season + Date Filter: {len(statcast_2022_2025): } rows | {statcast_2022_2025['game_pk'].nunique(): } games")

statcast_2022_2025["inning_topbot_ord"] = statcast_2022_2025["inning_topbot"].map({"Top": 0, "Bot": 1})

statcast = statcast_2022_2025.sort_values(
    ["game_date", "game_pk", "inning", "inning_topbot_ord", "at_bat_number", "pitch_number"]
).reset_index(drop=True)

statcast = statcast.drop(columns=["inning_topbot_ord"])

# Per-Game Completeness Check

game_info = statcast.groupby("game_pk").agg(
    game_date = ("game_date", "first"),
    max_inning = ("inning", "max"),
    innings_found = ("inning", "nunique"),
    has_top = ("inning_topbot", lambda x: "Top" in x.values),
    has_bot = ("inning_topbot", lambda x: "Bot" in x.values),
    total_pitches = ("pitch_number", "count")
).reset_index()

game_info["missing_innings"] = game_info["max_inning"] - game_info["innings_found"]

game_info["suspicious_pitch_count"] = (
    (game_info["total_pitches"] < 100) |
    (game_info["total_pitches"] > 500)
)

# Identify Incomplete Half Innings

half_inning_counts = statcast.groupby(
    ["game_pk", "inning"]
)["inning_topbot"].nunique().reset_index()

half_inning_counts.columns = ["game_pk", "inning", "halves"]
missing_halves = half_inning_counts[half_inning_counts["halves"] < 2]

missing_halves = missing_halves.merge(
    game_info[["game_pk", "max_inning"]], on = "game_pk"
)
missing_halves["is_final_inning"] = (
    missing_halves["inning"] == missing_halves["max_inning"]
)

genuine_incomplete = missing_halves[~missing_halves["is_final_inning"]]["game_pk"].unique()

print(f"Games with mid-game missing half-innings: {len(genuine_incomplete): }")


# Inning-Level Out Check

out_events = {
    "strikeout", "field_out", "grounded_into_double_play",
    "double_play", "triple_play", "force_out", "sac_fly",
    "sac_bunt", "fielders_choice_out", "strikeout_double_play", "truncated_pa", "fielders_choice"
}

double_play_events = {
    "grounded_into_double_play", "double_play", "strikeout_double_play"
}

triple_play_events = {
    "triple_play"
}

pa_endings = statcast[statcast["events"].notna() & (statcast["events"] != "")].copy()

pa_endings["outs_on_play"] = np.where(
    pa_endings["events"] == "triple_play", 3,
    np.where(pa_endings["events"].isin(double_play_events), 2,
    np.where(pa_endings["events"].isin(out_events), 1, 0))
)

inning_outs = pa_endings.groupby(
    ["game_pk", "inning", "inning_topbot"]
)["outs_on_play"].sum().reset_index()

inning_outs.columns = ["game_pk", "inning", "inning_topbot", "outs"]

inning_outs = inning_outs.merge(
    game_info[["game_pk", "max_inning"]], on = "game_pk"
)

inning_outs["is_final_half"] = (
    (inning_outs["inning"] == inning_outs["max_inning"])
)

inning_outs["suspicious_out_count"] = np.where(
    (~inning_outs["is_final_half"]) & 
    (~inning_outs["outs"].isin([3, 4])), True,
    np.where(
        inning_outs["is_final_half"] &
        (inning_outs["inning_topbot"] == "Top") &
        (~inning_outs["outs"].isin([3, 4])), True,
        np.where(
            inning_outs["is_final_half"] &
            (inning_outs["inning_topbot"] == "Bot") &
            ((inning_outs["outs"] < 1) | (inning_outs["outs"] >= 5)), True, 
            False
        )
    )
)

inning_outs["suspicious_out_count"] = (
    (inning_outs["outs"] >= 5) |  # impossible out count
    (
        (inning_outs["outs"] == 0) &
        ~(
            (inning_outs["is_final_half"]) &
            (inning_outs["inning_topbot"] == "Bot")
        )
    )
)

suspicious_inning_games = inning_outs[
    inning_outs["suspicious_out_count"]]["game_pk"].unique()

print(f"Games flagged after fix: {len(suspicious_inning_games):,}")
print(f"\nSuspicious flag breakdown:")
print(
    inning_outs
    .groupby(["suspicious_out_count", "outs"])
    .size()
    .reset_index(name="count")
    .to_string()
)

print(f"Games with suspiciuous inning out counts: {len(suspicious_inning_games): }")
print(f"\n Out Count Distribution:\n{inning_outs['outs'].value_counts().sort_index()}")

# Filtering to 9 Inning Games

nine_inning_games = game_info[
    (game_info["max_inning"] == 9) &
    (~game_info["suspicious_pitch_count"]) &
    (game_info["missing_innings"] == 0) &
    (game_info["has_top"] == True)
]["game_pk"].unique()


nine_inning_games = [
    g for g in nine_inning_games

    if g not in genuine_incomplete

    and g not in suspicious_inning_games
]

print(f"Games that pass all filters: {len(nine_inning_games): }")

statcast_clean = statcast[statcast["game_pk"].isin(nine_inning_games)].copy()

print(f"Pitches before: {len(statcast_2022_2025): } | Pitches after: {len(statcast_clean): }")
print(f"Games before: {statcast_2022_2025['game_pk'].nunique(): } | Games after: {statcast_clean['game_pk'].nunique(): }")
print(f"Seasons: {sorted(statcast_clean['game_year'].unique())}")

statcast_clean["horz_break"] = -statcast_clean["pfx_x"] * 12
statcast_clean["ivb"] = statcast_clean["pfx_z"] * 12

statcast_clean["norm_hb"] = np.where(
    statcast_clean["p_throws"] == "L",
    -statcast_clean["horz_break"],
    statcast_clean["horz_break"]
)

dupes = statcast_clean.duplicated(
    subset = ["game_pk", "at_bat_number", "pitch_number"]).sum()

print(f"Duplicate Pitches: {dupes: }")

os.makedirs("data", exist_ok=True)
statcast_clean.to_parquet("data/statcast_clean.parquet", index=False)
print(f"Final Shape: {statcast_clean.shape}")

outs_by_half = pa_endings[
    pa_endings["game_pk"].isin(nine_inning_games)
].groupby(["game_pk", "inning_topbot"])["outs_on_play"].sum().reset_index()

top_outs = outs_by_half[outs_by_half["inning_topbot"] == "Top"].copy()
bot_outs = outs_by_half[outs_by_half["inning_topbot"] == "Bot"].copy()

print("Top half outs per game (home team pitching, away team batting):")
print(top_outs["outs_on_play"].describe())
print(f"\nValue counts:")
print(top_outs["outs_on_play"].value_counts().sort_index())

print("\nBottom half outs per game (away team pitching, home team batting):")
print(bot_outs["outs_on_play"].describe())
print(f"\nValue counts:")
print(bot_outs["outs_on_play"].value_counts().sort_index())

acceptable_top_half_outs = top_outs[top_outs["outs_on_play"] >= 26]["game_pk"].unique()

acceptable_bot_half_outs = bot_outs[bot_outs["outs_on_play"] >= 24]["game_pk"].unique()

final_clean_games = list(set(acceptable_top_half_outs) & set(acceptable_bot_half_outs))

print(f"Games that pass top half outs filter: {len(acceptable_top_half_outs): }")
print(f"Games that pass bottom half outs filter: {len(acceptable_bot_half_outs): }")
print(f"Games that pass both filters: {len(final_clean_games): }")

statcast_final_clean = statcast_clean[statcast_clean["game_pk"].isin(final_clean_games)].copy()

print(f"Final Clean Shape: {statcast_final_clean.shape}")

os.makedirs("data", exist_ok=True)
statcast_final_clean.to_parquet("data/statcast_final_clean.parquet", index=False)

final_outs_by_half = pa_endings[pa_endings["game_pk"].isin(statcast_final_clean["game_pk"].unique())].groupby(["game_pk", "inning_topbot"])["outs_on_play"].sum().reset_index()

print(final_outs_by_half.groupby("inning_topbot")["outs_on_play"].describe())