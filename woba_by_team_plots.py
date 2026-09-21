import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import os
import warnings

warnings.filterwarnings("ignore")

os.makedirs("outputs/plots", exist_ok= True)

woba_pitcher = pd.read_csv("outputs/woba_pitcher.csv")

woba_pitch_type = pd.read_csv("outputs/woba_pitch_type.csv")

weight_df = pd.read_csv("outputs/woba_weights_by_season.csv", index_col=0)

woba_weights = {}
for year, row in weight_df.iterrows():
    woba_weights[int(year)] = row.to_dict()

print("woba_weights keys:", list(woba_weights.keys()))
print("Sample weights:", woba_weights[2022])

statcast = pd.read_parquet("data/statcast_final_clean.parquet")


if "game_year" not in statcast.columns:
    statcast["game_year"] = pd.to_datetime(statcast["game_date"]).dt.year


statcast = statcast[statcast["game_type"] == "R"].copy()


statcast["pitching_team"] = np.where(
    statcast["inning_topbot"] == "Top",
    statcast["home_team"],
    statcast["away_team"]
)

pa_statcast = statcast[
    statcast["events"].notna() & (statcast["events"] != "")
].copy()

# Check pa_full has game_year
print(f"game_year in pa_full: {'game_year' in pa_statcast.columns}")
print(pa_statcast["game_year"].value_counts())

print(f"pitching_team in pa_full: {'pitching_team' in pa_statcast.columns}")

# Standardize Athletics team code
pa_statcast["pitching_team"] = pa_statcast["pitching_team"].replace("ATH", "OAK")

# Also update in statcast_full for consistency
statcast["pitching_team"] = statcast["pitching_team"].replace("ATH", "OAK")


print("Computing Team-Level wOBAA: ")

def compute_team_woba(df, weights_by_year, level_cols):

    results = []

    for keys, group in df.groupby(level_cols):
        if not isinstance(keys, tuple):
            keys = (keys, )

        year = keys[level_cols.index("game_year")]

        w = weights_by_year.get(year, {})

        if not w:
            continue

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
            (group["events"].isin([
                "strikeout", "strikeout_double_play", "grounded_into_double_play",
                "field_out", "double_play", "force_out", "fielders_choice_out", "other_out"
            ])).sum() +
            (group["events"] == "sac_fly").sum()
        )

        woba = round(woba_num / woba_denom, 4) if woba_denom > 0 else np.nan

        xwoba = group["estimated_woba_using_speedangle"].mean()

        row = dict(zip(level_cols, keys))

        row.update({
            "woba_denom": int(woba_denom),
            "wOBAA": woba,
            "xwOBAA": round(xwoba, 4) if not np.isnan(xwoba) else np.nan,
            "HR": (group["events"] == "home_run").sum(),
            "K": (group["events"] == "strikeout").sum(),
            "BB": (group["events"] == "walk").sum(),
        })

        results.append(row)

    return pd.DataFrame(results)

# Team Level

woba_team = compute_team_woba(
    pa_statcast,
    woba_weights,
    ["pitching_team", "game_year"]
)

print(woba_team.columns.tolist())
print(woba_team["game_year"].value_counts())

# Team + pitch type level 

# pa_statcast_with_pitch = pa_statcast.merge(
#    statcast[["game_pk", "at_bat_number", "pitch_number", "pitch_type"]]
#    .drop_duplicates(),
#    on = ["game_pk", "at_bat_number", "pitch_number"],
#    how = "left"
# )

woba_team_pitch = compute_team_woba(
    pa_statcast[pa_statcast["pitch_type"].notna()],
    woba_weights,
    ["pitching_team", "pitch_type", "game_year"]
)

# Add wOBAA- and xwOBAA- (normalized to 100, lower = better)

league_woba_yr = (
    woba_team.groupby("game_year")["wOBAA"]
    .mean().rename("league_wOBAA").reset_index()
)

league_xwoba_yr = (
    woba_team.groupby("game_year")["xwOBAA"]
    .mean().rename("league_xwOBAA").reset_index()
)

woba_team = woba_team.merge(league_woba_yr, on = "game_year", how = "left")
woba_team = woba_team.merge(league_xwoba_yr, on = "game_year", how = "left")

woba_team["wOBAA_minus"] = (
    100 * woba_team["wOBAA"] / woba_team["league_wOBAA"]
).round(1)

woba_team["xwOBAA_minus"] = (
    100 * woba_team["xwOBAA"] / woba_team["league_xwOBAA"]
).round(1)

print(f"Teams: {woba_team['pitching_team'].nunique()}")
print(f"Team-pitch type combos: {len(woba_team_pitch):}")


# Season-aggregated team stats (across all years)

woba_team_all = (
    woba_team.groupby("pitching_team")
    .agg(
        wOBAA = ("wOBAA", "mean"),
        xwOBAA = ("xwOBAA", "mean"),
        wOBAA_minus = ("wOBAA_minus", "mean"),
        xwOBAA_minus = ("xwOBAA_minus", "mean"),
        seasons = ("game_year", "nunique")
    )
    .reset_index()
    .sort_values("wOBAA")
)

# Team wOBAA and xwOBAA ranked bar charts

fig, axes = plt.subplots(1, 2, figsize = (18, 10))

fig.suptitle("Team wOBAA and xwOBAA Against (2022-2025 Avg)",
             fontsize = 14, fontweight = "bold")

for ax, (col, label, color) in zip(axes, [
    ("wOBAA", "wOBAA Against", "#E41A1C"),
    ("xwOBAA", "xwOBAA Against", "#377EB8")
]): 
    sorted_df = woba_team_all.sort_values(col, ascending= True)
    colors = [
        "#4DAF4A" if v < woba_team_all[col].mean() else "#E41A1C"
        for v in sorted_df[col]
    ]

    bars = ax.barh(
        sorted_df["pitching_team"],
        sorted_df[col],
        color = colors, alpha = 0.8
    )

    league_mean = woba_team_all[col].mean()

    ax.axvline(league_mean, color = "black", linestyle = "--",
               linewidth = 1.5, label = f"League Avg: {league_mean:.3f}")

    for bar, val in zip(bars, sorted_df[col]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va = "center", fontsize = 8)

    ax.set_xlabel(label)
    ax.set_title(f"{label} by Team\n(green = better than league avg)")
    ax.legend(fontsize = 9)

plt.tight_layout()

plt.savefig("outputs/plots/team_woba_ranked.png",
            dpi = 150, bbox_inches = "tight")

plt.close()

print("Saved: team_woba_ranked.png")

# Team wOBAA- and xwOBAA- Ranked

fig, axes = plt.subplots(1, 2, figsize = (18, 10))

fig.suptitle("Team wOBAA- and xwOBAA- (100 = League Avg, Lower = Better)",
             fontsize = 14, fontweight = "bold")

for ax, (col, label, colors) in zip(axes, [
    ("wOBAA_minus", "wOBAA-", "#E41A1C"),
    ("xwOBAA_minus", "xwOBAA-", "#377EB8")
]): 
    sorted_df = woba_team_all.sort_values(col, ascending=True)

    colors = [
        "#4DAF4A" if v < 100 else "#E41A1C"
        for v in sorted_df[col]
    ]

    bars = ax.barh(
        sorted_df["pitching_team"],
        sorted_df[col],
        color = colors, alpha = 0.8
    )

    ax.axvline(100, color = "black", linestyle = "--",
               linewidth = 1.5, label = "League Avg (100)")

    for bar, val in zip(bars, sorted_df[col]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}", va = "center", fontsize = 8)

    ax.set_xlabel(label)

    ax.set_title(f"{label} by Team\n(green = better than avg)")

    ax.legend(fontsize = 9)

plt.tight_layout()

plt.savefig("outputs/plots/wobaa_minus_by_team_ranked.png",
            dpi = 150, bbox_inches = "tight")

plt.close()

print("Saved: wobaa_minus_by_team_ranked.png")


# wOBAA vs. xwOBAA by Team (Team Regression Candidates)

fig, ax = plt.subplots(figsize = (12, 10))

x = woba_team_all["xwOBAA"]
y = woba_team_all["wOBAA"]

ax.scatter(
    x,
    y,
    s = 80, alpha = 0.8, color = "#888888", zorder = 3
)

for _, row in woba_team_all.iterrows():
    ax.annotate(
        row["pitching_team"],
        (row["xwOBAA"], row["wOBAA"]),
        fontsize = 8, ha = "center", va = "bottom",
        xytext= (0,6), textcoords="offset points"
    )

x_pad = (x.max() - x.min()) * 0.01
y_pad = (y.max() - y.min()) * 0.01

ax.set_xlim(x.min() - x_pad, x.max() + x_pad)
ax.set_ylim(y.min() - y_pad, y.max() + y_pad)

lims = [
    min((x.min(), y.min()) - max(x_pad, y_pad)),
    max((x.max(), y.max()) + max(x_pad, y_pad))
]

ax.plot(lims, lims, "k--", linewidth = 1.5, label = "wOBAA = xwOBAA")

# Regression
slope, intercept, r, p, _ = stats.linregress(x, y)

x_line = np.linspace(x.min() - x_pad, x.max() + x_pad, 100)


ax.plot(x_line, slope * x_line + intercept,
        color = "#E41A1C", linewidth = 2, 
        label = f"Regression (r = {r:.3f})")

# Quadrant Labels

mid_x = woba_team_all["xwOBAA"].mean()

mid_y = woba_team_all["wOBAA"].mean()

ax.axhline(y.mean(), color = "gray", linestyle = ":", linewidth = 0.8, alpha = 0.5)

ax.axvline(x.mean(), color = "gray", linestyle = ":", linewidth = 0.8, alpha = 0.5)

ax.text(x.min() - x_pad * 0.1, y.max() + y_pad * 0.1,
        "Overperforming Contact Quality",
        fontsize=8, color="green", alpha=0.8)

ax.text(x.max() - x_pad * 0.1, y.min() - y_pad * 0.1,
        "Underperforming Contact Quality",
        fontsize=8, color="red", alpha=0.8)

ax.set_xlabel("xwOBAA (Contact Quality Allowed)")
ax.set_ylabel("wOBAA (Actual Outcomes)")

ax.set_title("Team wOBAA vs. xwOBAA - Regression Analysis\n"
             "Above Line = Outperforming Contact Quality | "
             "Below Line = Underperforming Contact Quality")

ax.legend(fontsize = 10)

plt.tight_layout()

plt.savefig("outputs/plots/team_woba_regression_scatter.png",
            dpi = 150, bbox_inches = "tight")

plt.close()

print("Saved: team_woba_regression_scatter.png")

# Best Pitch Type per Team (HeatMap)

# Pivot to team x pitch type matrix for xwOBAA

top_pitch_types = (
    woba_team_pitch.groupby("pitch_type")["woba_denom"]
    .sum()
    .sort_values(ascending = False)
    .head(8)
    .index.tolist()
)

heatmap_data = (woba_team_pitch[woba_team_pitch["pitch_type"].isin(top_pitch_types)]
                .groupby(["pitching_team", "pitch_type"])["xwOBAA"]
                .mean()
                .reset_index()
                .pivot(index = "pitching_team", columns = "pitch_type", values = "xwOBAA")
)

fig, ax = plt.subplots(figsize = (14, 12))

sns.heatmap(
    heatmap_data,
    annot = True, fmt = ".3f",
    cmap =  "RdYlGn_r", # red indicates high xwOBAA, green = low xwOBAA
    center = heatmap_data.stack().mean(),
    linewidths = 0.5,
    ax = ax,
    cbar_kws = {"label": "xwOBAA"}
)

ax.set_title("Team xwOBAA by Pitch Type (2022-2025 avg)\n"
             "Green = better than avg | "
             "Red = worse than avg", fontsize = 12, fontweight = "bold")

ax.set_xlabel("Pitch Type")

ax.set_ylabel("Team")

ax.tick_params(axis = "x", rotation = 45)

ax.tick_params(axis = "y", rotation = 0)

plt.tight_layout()

plt.savefig("outputs/plots/team_xwoba_by_pitch_type_heatmap.png",
            dpi = 150, bbox_inches = "tight")

plt.close()

print("Saved: team_xwoba_by_pitch_type_heatmap.png")


# Year-by-Year Team wOBAA Trends

top_teams = woba_team_all.nsmallest(15, "wOBAA")["pitching_team"].tolist()

bottom_teams = woba_team_all.nlargest(15, "wOBAA")["pitching_team"].tolist()

fig, axes = plt.subplots(1, 2, figsize = (18, 7))
fig.suptitle("Team wOBAA Trends by Season",
             fontsize = 14, fontweight = "bold")

for ax, teams, title in zip(
    axes, 
    [top_teams, bottom_teams],
    ["Best 15 Teams (Lowest wOBAA)", "Worst 15 Teams (highest wOBAA)"]
): 
    team_trend = woba_team[woba_team["pitching_team"].isin(teams)]

    for team in teams:
        td = team_trend[team_trend["pitching_team"] == team].sort_values("game_year")
        ax.plot(td["game_year"], td["wOBAA"],
                marker = "o", linewidth = 2, label = team)

    league_trend = woba_team.groupby("game_year")["wOBAA"].mean().reset_index()

    ax.plot(league_trend["game_year"], league_trend["wOBAA"],
            "k--", linewidth = 2, label = "Label Avg", alpha = 0.7)

    ax.set_xlabel("Season")
    ax.set_ylabel("wOBAA")
    ax.set_title(title)

    ax.legend(fontsize = 8, loc = "upper right")

    ax.set_xticks(sorted(woba_team["game_year"].unique()))

plt.tight_layout()

plt.savefig("outputs/plots/team_woba_trends.png",
            dpi = 150, bbox_inches = "tight")

plt.close()

print("Saved: team_woba_trends.png")

# Save All

woba_team.to_csv("outputs/woba_team.csv", index = False)

woba_team_pitch.to_csv("outputs/woba_team-pitch.csv", index = False)

woba_team_all.to_csv("outputs/woba_team_summary.csv", index = False)

print("\nAll team wOBA plots saved to outputs/plots/")
print("Team data saved to outputs/")


