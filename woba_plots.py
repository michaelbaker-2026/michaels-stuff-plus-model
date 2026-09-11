import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import os
import warnings

warnings.filterwarnings("ignore")

os.makedirs("outputs/plots", exist_ok=True)

# Load Data
woba_pitcher = pd.read_csv("outputs/woba_pitcher.csv")
woba_pitch_type = pd.read_csv("outputs/woba_pitch_type.csv")

# xwOBAA

if "xwOBAA_minus" not in woba_pitcher.columns:
    league_xwoba = (
        woba_pitcher.groupby("game_year")["xwOBAA"]
        .mean()
        .rename("league_xwOBAA")
        .reset_index()
    )

    woba_pitcher = woba_pitcher.merge(league_xwoba, on = "game_year", how = "left")

    woba_pitcher["xwOBAA_minus"] = (
        100 * woba_pitcher["xwOBAA"] / woba_pitcher["league_xwOBAA"]
    ).round(1)



COLORS = {
    "woba": "#E41A1C",
    "xwoba": "#377EB8",
    "minus": "#4DAF4A",
    "neutral": "#888888"
}

# Plot 1- Pitcher Level Distributions of wOBAA, xwOBAA, wOBAA-, xwOBAA-

fig, axes = plt.subplots(2, 2, figsize = (14, 10))

fig.suptitle("Pitcher-Level wOBAA and xwOBAA Distributions",
             fontsize = 14, fontweight = "bold")

axes = axes.flatten()

metrics = [
    ("wOBAA", "wOBAA (lower = better)", COLORS["woba"], False),
    ("xwOBAA", "xwOBAA", COLORS["xwoba"], False),
    ("wOBAA_minus", "wOBAA-", COLORS["minus"], False),
    ("xwOBAA_minus", "xwOBAA-", COLORS["neutral"], False)
]

for ax, (col, label, color, is_minus) in zip(axes, metrics):
    data = woba_pitcher[col].dropna()
    data = data[np.abs(stats.zscore(data)) < 3]

    ax.hist(data, bins = 40, color = color, alpha = 0.7, edgecolor = "white")

    ref_val = 100 if is_minus else data.mean()

    ax.axvline(ref_val, color = "black", linestyle = "--",
               linewidth = 1.5, label = f"{'League Avg (100)' if is_minus else f'Mean: {ref_val:.3f}'}")
    ax.axvline(data.mean(), color = color, linestyle = "--", linewidth = 1.5,
               label = f"Mean: {data.mean():.3f}")

    ax.set_xlabel(label)
    ax.set_ylabel("Count")
    ax.set_title(f"{col} Distribution")
    ax.legend(fontsize = 9)

plt.tight_layout()

plt.savefig("outputs/plots/woba_pitcher_distributions.png",
            dpi = 150, bbox_inches = "tight")

plt.close()

# wOBAA and xwOBAA by Season

fig, axes = plt.subplots(1, 2, figsize = (14, 5))

fig.suptitle("wOBAA and xwOBAA by Season", fontsize = 14, fontweight = "bold")

for ax, (col, label, color) in zip(axes, [
    ("wOBAA", "wOBAA", COLORS["woba"]),
    ("xwOBAA", "xwOBAA", COLORS["xwoba"])
]):
    season_data = [
        woba_pitcher[woba_pitcher["game_year"] == yr][col].dropna()
        for yr in sorted(woba_pitcher["game_year"].unique())
    ]

    years = sorted(woba_pitcher["game_year"].unique())

    bp = ax.boxplot(season_data, patch_artist = True, notch = False)
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticklabels(years)
    ax.set_xlabel("Season")

    ax.set_ylabel(label)

    ax.set_title(f"{label} by Season")

plt.tight_layout()

plt.savefig("outputs/plots/woba_by_season.png",
            dpi = 150, bbox_inches = "tight")

plt.close()

print("Saved: woba_by_season.png")

# wOBAA vs xwOBAA scatterplot (Regression candidates)

fig, ax = plt.subplots(figsize = (10, 8))

scatter_data = woba_pitcher.dropna(subset = ["wOBAA", "xwOBAA"])

scatter_data = scatter_data[
    (np.abs(stats.zscore(scatter_data["wOBAA"])) < 3) &
    (np.abs(stats.zscore(scatter_data["xwOBAA"])) < 3)
]

ax.scatter(
    scatter_data["xwOBAA"],
    scatter_data["wOBAA"],
    alpha = 0.4, s = 20, color = COLORS["neutral"]
)

# Perfect agreement line

lims = [
    min(scatter_data["xwOBAA"].min(), scatter_data["wOBAA"].min()),
    max(scatter_data["xwOBAA"].max(), scatter_data["xwOBAA"].max())
]

ax.plot(lims, lims, "k--", linewidth = 1.5, label = "wOBAA = xwOBAA")

# Regression Line

slope, intercept, r, p, _ = stats.linregress(
    scatter_data["xwOBAA"], scatter_data["wOBAA"]
)

x_line = np.linspace(scatter_data["xwOBAA"].min(),
                     scatter_data["xwOBAA"].max(),
                     100)

ax.plot(x_line, slope * x_line + intercept,
        color = COLORS["xwoba"], linewidth = 2,
        label = f"Regression (r = {r:.3f})")

# Label extreme Outliers

gap = scatter_data["wOBAA"] - scatter_data["xwOBAA"]

outliers = scatter_data[np.abs(gap) > gap.std() * 2.5]

for _, row in outliers.iterrows():
    ax.annotate(
        f"{row['player_name'].split(',')[0]} {int(row['game_year'])}",
        (row["xwOBAA"], row["wOBAA"]),
        fontsize = 7, alpha = 0.7,
        xytext= (5, 5), textcoords="offset points"
    )

ax.set_xlabel("xwOBAA (contact qulity)")
ax.set_ylabel("wOBAA (actual outcomes)")
ax.set_title("wOBAA vs. xwOBAA - Regression Candidates \n"
             "Above Line = underperforming contact quality |"
             "Below Line = overperforming contact quality")
ax.legend(fontsize = 10)

plt.tight_layout()

plt.savefig("outputs/plots/woba_vs_xwoba_scatter.png",
            dpi = 150, bbox_inches = "tight")

print("Saved: woba_vs_xwoba_scatter.png")

# Plot wOBA and xwOBAA by Pitch Type

fig, axes = plt.subplots(1, 2, figsize = (16, 6))

fig.suptitle("wOBAA and xwOBAA by Pitch Type",
             fontsize = 14, fontweight = "bold")

for ax, (col, label, color) in zip (axes, [
    ("wOBAA", "wOBAA", COLORS["woba"]),
    ("xwOBAA", "xwOBAA", COLORS["xwoba"])
]):
    pitch_avg = (
        woba_pitch_type.groupby("pitch_type")[col]
        .agg(["mean", "count"])
        .reset_index()
        .query("count >= 100")
        .sort_values("mean", ascending=True)
    )

    bars = ax.barh(pitch_avg["pitch_type"], pitch_avg["mean"],
                   color = color, alpha = 0.7)

    # League Average Line

    league_mean = woba_pitcher[col].mean()

    ax.axvline(league_mean, color = "black", linestyle = "--",
               linewidth = 1.5, label = f"League Avg: {league_mean:.3f}")

    for bar, val in zip(bars, pitch_avg["mean"]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va = "center", fontsize = 9)

    ax.set_xlabel(label)
    ax.set_title(f"{label} by Pitch Type")
    ax.legend(fontsize = 9)

plt.tight_layout()

plt.savefig("outputs/plots/wobaa_by_pitch_type.png",
            dpi = 150, bbox_inches = "tight")

plt.close()

print("Saved: wobaa_by_pitch_type.png")

# wOBAA- Distribution 

fig, axes = plt.subplots(1, 2, figsize = (14, 5))

fig.suptitle("wOBAA- and xwOBAA- Distributions",
             fontsize = 14, fontweight = "bold")

for ax, (col, label, color) in zip(axes, [
    ("wOBAA_minus", "wOBAA-", COLORS["woba"]),
    ("xwOBAA_minus", "xwOBAA-", COLORS["xwoba"])
]):
    data = woba_pitcher[col].dropna()
    data = data[np.abs(stats.zscore(data)) < 3]

    ax.hist(data, bins = 40, color = color, alpha = 0.7, edgecolor = "white")

    ax.axvline(100, color = "black", linestyle = "--", linewidth = 2, label = "League Avg: 100")

    ax.axvline(data.mean(), color = color, linestyle = "--",
               linewidth = 2, label = f"Mean: {data.mean():.1f}")

    ax.set_xlabel(label)
    ax.set_ylabel("Count")
    ax.set_title(f"{col} Distribution")

plt.tight_layout()

plt.savefig("outputs/plots/woba_minus_distributions.png",
            dpi = 150, bbox_inches = "tight")

plt.close()

print("Saved: woba_minus_distributions.png")

print("\nAll plots saved to outputs/plots")
