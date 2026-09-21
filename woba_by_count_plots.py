import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import os
import warnings

warnings.filterwarnings("ignore")

os.makedirs("outputs/plots/count_woba", exist_ok = True)

woba_by_team_count = pd.read_csv("outputs/woba_by_team_pitch_count.csv")

woba_by_pitcher_count = pd.read_csv("outputs/woba_by_pitcher_pitch_count.csv")

print(f"By Team Rows: {len(woba_by_team_count):}")

print(f"By Pitcher Rows: {len(woba_by_pitcher_count):}")

print(f"Seasons from Team Level: {sorted(woba_by_team_count['game_year'].unique())}, Seasons from Pitcher Level: {sorted(woba_by_pitcher_count['game_year'].unique())}")

print(f"Pitch Types from Team Level: {sorted(woba_by_team_count['pitch_type'].unique())} and from Pitcher Level: {sorted(woba_by_pitcher_count['pitch_type'].unique())}")

count_order = [
    "0-0", "0-1", "0-2", "1-0", "2-0", "3-0",
    "1-1", "1-2", "2-1", "2-2", "3-1", "3-2"
]

top_pitches = (
    woba_by_team_count.groupby("pitch_type")["pa_count"]
    .sum()
    .sort_values(ascending = False)
    .head(8)
    .index.tolist()
)

# Team Level HeatMap wOBAA by Count and Pitch Type (2022-2025 Seasons Averaged)

heat_map = (
    woba_by_team_count[woba_by_team_count["pitch_type"].isin(top_pitches)]
    .groupby(["count", "pitch_type"])
    .agg(
        wOBAA = ("wOBAA", "mean"),
        xwOBAA = ("xwOBAA", "mean"),
        pa = ("pa_count", "sum")
    ).reset_index()
)

heat_map = heat_map[heat_map["count"].isin(count_order)]

fig, axes = plt.subplots(1, 2, figsize = (18, 7))

fig.suptitle("Team Level wOBAA and xwOBAA by Count and Pitch Type (2022-2025)",
             fontsize = 14, fontweight = "bold")

for ax, metric, label in zip(
    axes,
    ["wOBAA", "xwOBAA"],
    ["wOBA Against", "xwOBA AGainst"]
):
    pivot = heat_map.pivot(
        index = "pitch_type",
        columns = "count",
        values= metric
    ).reindex(columns = count_order)

    sns.heatmap(
        pivot,
        annot = True, fmt = ".3f",
        cmap = "RdYlGn_r",
        center = pivot.stack().mean(),
        linewidths = 0.5,
        ax = ax,
        cbar_kws= {"label": f"{label} (lower = better)"}
    )

    ax.set_title(f"{label} by Count and Pitch Type")
    ax.set_xlabel("Count")
    ax.set_ylabel("Pitch Type")
    ax.tick_params(axis = "x", rotation = 45)
    ax.tick_params(axis = "y", rotation = 0)

plt.tight_layout()

plt.savefig("outputs/plots/count_woba/pitch_type_team_level.png",
            dpi = 150, bbox_inches = "tight")

plt.close()

print("Saved: pitch_type_team_level.png")

# Pitcher-Pitch Level Combos (3-2)

full_count = woba_by_pitcher_count[
    (woba_by_pitcher_count["count"] == "3-2") &
    (woba_by_pitcher_count["pa_count"] >= 30) &
    (woba_by_pitcher_count["pitch_type"].isin(top_pitches))
].copy()

fig, axes = plt.subplots(1, 2, figsize = (18, 8))

fig.suptitle("Best and Worst Team-Pitch Combos in Full Counts",
             fontsize = 14, fontweight = "bold")

for ax, ascending, title, color in zip(
    axes,
    [True, False],
    ["Best", "Worst"],
    ["#4DAF4A", "#E41A1C"]
):
    data = (
        full_count
        .sort_values("wOBAA", ascending=ascending)
        .head(15)
    )

    data["label"] = data["player_name"] + " " + data["pitch_type"] + " " + data["game_year"].astype(str)

    bars = ax.barh(data["label"], data["wOBAA"], color = color, alpha = 0.8)

    for bar, val in zip(bars, data["wOBAA"]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va = "center", fontsize = 6)

    league_mean = full_count["wOBAA"].mean()

    ax.axvline(league_mean, color = "black", linestyle = "--",
               linewidth = 2, label = f"League Avg: {league_mean: .3f}")

    ax.set_xlabel("wOBAA")
    ax.set_title(title)
    ax.legend(fontsize = 9)
    ax.tick_params(axis = "y", labelsize = 9)
    ax.invert_yaxis()

plt.tight_layout()

plt.savefig("outputs/plots/count_woba/best_worst_full_count_pitches.png",
            dpi = 150, bbox_inches = "tight")

plt.close()

print("Saved: best_worst_full_count_pitches.png")

# xwOBAA in 3-2 Counts

fig, axes = plt.subplots(1, 2, figsize = (18, 8))

plt.suptitle("Expected Best/Worst Full Count Pitcher-Pitch Combos",
             fontsize = 14, fontweight = "bold")

for ax, ascending, title, color in zip(
    axes,
    [True, False],
    ["Best", "Worst"],
    ["#4DAF4A", "#E41A1C"]
):
    x_data = (
        full_count
        .sort_values("xwOBAA", ascending= ascending)
        .head(15)
    )

    x_data["label"] = x_data["player_name"] + " " + x_data["pitch_type"] + " " + x_data["game_year"].astype(str)

    bars = ax.barh(x_data["label"], x_data["xwOBAA"], color = color, alpha = 0.8)

    for bar, val in zip(bars, x_data["xwOBAA"]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va = "center", fontsize = 6)

    x_league_mean = full_count["xwOBAA"].mean()

    ax.axvline(x_league_mean, linestyle = "--", linewidth = 2, color = "black",
               label = f"League Avg: {league_mean: .3f}")

    ax.set_xlabel = ("xwOBAA")
    ax.set_title(title)
    ax.legend(fontsize = 9)
    ax.tick_params(axis = "y", labelsize = 9)
    ax.invert_yaxis()

plt.tight_layout()

plt.savefig("outputs/plots/count_woba/best_worst_xwoba_full_counts.png",
            dpi = 150, bbox_inches = "tight")

plt.close()

print("Saved: best_worst_xwoba_full_counts.png")

