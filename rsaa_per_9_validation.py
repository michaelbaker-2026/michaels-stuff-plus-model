import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import pybaseball
from pybaseball import pitching_stats_bref
import os
import warnings
warnings.filterwarnings("ignore")



os.makedirs("outputs/validation", exist_ok=True)

statcast = pd.read_parquet("~/Downloads/statcast_data.parquet")

rv_by_pitcher = pd.read_csv("outputs/runs_value_with_ip.csv")

stuff_summary = pd.read_csv("outputs/stuff_summary_on_run_value.csv")

if "game_year" not in statcast.columns:
    statcast["game_year"] = pd.to_datetime(statcast["game_date"]).dt.year

statcast = statcast[statcast["game_type"] == "R"].copy()

print(f"FUll Dataset: {len(statcast):} rows | {statcast['game_pk'].nunique():} games")

pa = statcast[statcast["events"].notna() & (statcast["events"] != "")]

pa["runs_on_play"] = (pa["post_bat_score"] - pa["bat_score"]).clip(lower = 0)

out_events = {
    "strikeout", "field_out", "grounded_into_double_play",
    "double_play", "triple_play", "force_out", "sac_fly",
    "sac_bunt", "fielders_choice_out", "strikeout_double_play", "truncated_pa", "fielders_choice"
}

double_play_events = {
    "grounded_into_double_play", "strikeout_double_play", "double_play"
}

pa["outs_on_play"] = np.where(
    pa["events"] == "triple_play", 3,
    np.where(pa["events"].isin(double_play_events), 2,
    np.where(pa["events"].isin(out_events), 1, 0))
)

pitching_stats = (
    pa.groupby(["pitcher", "player_name", "game_year"])
    .agg(
        total_outs = ("outs_on_play", "sum"),
        runs = ("runs_on_play", "sum"),
        hr = ("events", lambda x: (x == "home_run").sum()),
        bb = ("events", lambda x: (x == "walk").sum()),
        ibb = ("events", lambda x: (x == "intent_walk").sum()),
        hbp = ("events", lambda x: (x == "hit_by_pitch").sum()),
        k = ("events", lambda x: (x == "strikeout").sum()),
        bf = ("events", "count"),
        singles = ("events", lambda x: (x == "single").sum()),
        doubles = ("events", lambda x: (x == "double").sum()),
        triples = ("events", lambda x: (x == "triple").sum())
    )
    .reset_index()
)

pitching_stats["ip"] = pitching_stats["total_outs"] / 3

pitching_stats["RA9"] = (
    pitching_stats["runs"] / pitching_stats["ip"] * 9
).round(2)

total_era = (pa["runs_on_play"].sum() / (pa.groupby(
    ["game_pk", "inning", "inning_topbot"])["outs_on_play"]
    .sum().sum() / 3)) * 9

total_fip_num = (
    13 * (pa["events"] == "home_run").sum() +
    3 * (pa["events"].isin(["walk", "hit_by_pitch"])).sum() -
    2 * (pa["events"] == "strikeout").sum()
)

total_ip = pa.groupby(
    ["game_pk", "inning", "inning_topbot"])["outs_on_play"].sum().sum() / 3

fip_constant = total_era - (total_fip_num/total_ip)

print(f"FIP Constant: {fip_constant:.3f}")

pitching_stats["FIP"] = (
    (13 * pitching_stats["hr"] +
     3 * (pitching_stats["bb"] - pitching_stats["ibb"] + pitching_stats["hbp"]) -
     2 * pitching_stats["k"]) / pitching_stats["ip"] + fip_constant
).round(2)

league_avg_hr_rate = pitching_stats["hr"].sum() / pitching_stats["bf"].sum()

pitching_stats["xHR"] = pitching_stats["bf"] * league_avg_hr_rate

pitching_stats["xFIP"] = (
    (13 * pitching_stats["xHR"] +
     3 * (pitching_stats["bb"] - pitching_stats["ibb"] + pitching_stats["hbp"]) -
     2 * pitching_stats["k"]) / pitching_stats["ip"] + fip_constant
).round(2)

pitching_stats["K%"] = (pitching_stats["k"]/pitching_stats["bf"]* 100).round(1)
pitching_stats["BB%"] = (pitching_stats["bb"] / pitching_stats["bf"] * 100).round(1)
pitching_stats["HR/9"] = (pitching_stats["hr"] / pitching_stats["ip"] * 9).round(2)

print(f"\nSample Pitcher Stats: ")
print(pitching_stats[pitching_stats["ip"] >= 100]
      .sort_values("RA9")
      .head(10)
      [["player_name", "game_year", "ip", "RA9", "FIP", "xFIP", "K%", "BB%"]]
      .to_string(index = False)
    )





print("\nMerging Datasets")

validation_df = rv_by_pitcher.merge(
    pitching_stats[[
        "pitcher", "game_year", "RA9", "FIP", "xFIP",
        "K%", "BB%", "HR/9"
    ]],
    on = ["pitcher", "game_year"],
    how = "inner"
)

validation_df = validation_df[validation_df["ip_decimal"] >= 30].copy()

print(f"After 30+ IP Filter: {len(validation_df):}")


metrics = ["runs_saved_above_avg_per_9", "runs_saved_per_100", "RA9", "FIP", "xFIP", "K%", "HR/9"]

print("\n RSAA/9 Correlations w Established Metrics: ")

for metric in ["RA9", "FIP", "xFIP", "K%", "BB%"]: 
    mask = validation_df[["runs_saved_above_avg_per_9", metric]].notna().all(axis = 1)

    r, p = stats.pearsonr(
        validation_df.loc[mask, "runs_saved_above_avg_per_9"],
        validation_df.loc[mask, metric]
    )

    direction = "lower better" if metric != "K%" else "higher better"
    print(f"RSAA/9 {metric:6s} ({direction}): r = {r:+.3f} | p = {p:.3e}")



stuff_pitcher = (
    stuff_summary.groupby(["pitcher", "game_year"])
    .apply(lambda x: pd.Series({
        "weight_stuff_plus": np.average(
            x["xgb_stuff_plus"], weights = x["num_pitches"]
        ),
        "total_pitches": x["num_pitches"].sum()
    }))
    .reset_index()
)

stuff_validation = stuff_pitcher.merge(
    pitching_stats[[
        "pitcher", "game_year", "RA9", "FIP", "xFIP"]],
    on = ["pitcher", "game_year"],
    how = "inner"
)

stuff_pitcher = stuff_pitcher[stuff_pitcher["total_pitches"] >= 300]

print("\nwStuff+ Correlations: ")

for metric in ["RA9", "FIP", "xFIP"]:
        mask = stuff_validation[["weight_stuff_plus", metric]].notna().all(axis = 1)

        r, p = stats.pearsonr(
            stuff_validation.loc[mask, "weight_stuff_plus"],
            stuff_validation.loc[mask, metric]
        )

        direction = "lower better" if metric != "K%" else "higher better"

        print(f"wStuff+ vs. {metric:.5s} ({direction}): r = {r:+.3f} | p = {p:.3e}")


corr_matrix = validation_df[metrics].corr().round(3)

print("Correlation Matrix: ")
print(corr_matrix.to_string())




fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("RSAA/9 and Stuff+ Validation",
             fontsize=14, fontweight="bold")

# Flatten to 1D array for easy indexing
axes = axes.flatten()

# Row 1 — RSAA/9 vs ERA, FIP, xFIP (axes 0, 1, 2)
for ax, metric in zip(axes[0:3], ["RA9", "FIP", "xFIP"]):
    x = validation_df["runs_saved_above_avg_per_9"]
    y = validation_df[metric]
    mask = (
        (np.abs(stats.zscore(x)) < 3) &
        (np.abs(stats.zscore(y)) < 3)
    )
    x_c, y_c = x[mask], y[mask]

    ax.scatter(x_c, y_c, alpha=0.3, s=15, color="#888888")
    slope, intercept, r, p, _ = stats.linregress(x_c, y_c)
    x_line = np.linspace(x_c.min(), x_c.max(), 100)
    ax.plot(x_line, slope * x_line + intercept,
            color="#E41A1C", linewidth=2,
            label=f"r = {r:.3f} | p = {p:.2e}")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("RSAA per 9 IP")
    ax.set_ylabel(metric)
    ax.set_title(f"RSAA/9 vs {metric}")
    ax.legend(fontsize=9)

# Row 2 — Stuff+ vs ERA, FIP, xFIP (axes 3, 4, 5)
for ax, metric in zip(axes[3:6], ["RA9", "FIP", "xFIP"]):
    x = stuff_validation["weight_stuff_plus"]
    y = stuff_validation[metric]
    mask = (
        (np.abs(stats.zscore(x)) < 3) &
        (np.abs(stats.zscore(y)) < 3)
    )
    x_c, y_c = x[mask], y[mask]

    ax.scatter(x_c, y_c, alpha=0.3, s=15, color="#377EB8")
    slope, intercept, r, p, _ = stats.linregress(x_c, y_c)
    x_line = np.linspace(x_c.min(), x_c.max(), 100)
    ax.plot(x_line, slope * x_line + intercept,
            color="#E41A1C", linewidth=2,
            label=f"r = {r:.3f} | p = {p:.2e}")
    ax.set_xlabel("Weighted Stuff+ (XGBoost)")
    ax.set_ylabel(metric)
    ax.set_title(f"Stuff+ vs {metric}")
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("outputs/validation/rsaa_stuff_validation.png",
            dpi=150, bbox_inches="tight")
plt.close()

# Need to Investigate YoY Stability, Something is Off

print("\nChecking Year-Over-Year Stability:")

rsaa_wide = rv_by_pitcher.pivot_table(
    index = "pitcher",
    columns = "game_year",
    values = "runs_saved_above_avg_per_9"
).reset_index()

fig, axes = plt.subplots(1, 3, figsize = (16, 5))

fig.suptitle ("RSAA/9 Year-Over-Year Stability",
              fontsize = 14, fontweight = "bold")

year_pairs = [(2022, 2023), (2023, 2024), (2024, 2025)]

for ax, (yr1, yr2) in zip(axes, year_pairs):
    if yr1 not in rsaa_wide.columns or yr2 not in rsaa_wide.columns:
        continue

    pair_df = rsaa_wide[[yr1, yr2]].dropna()
    pair_df = pair_df[
        (np.abs(stats.zscore(pair_df[yr1])) < 3) &
        (np.abs(stats.zscore(pair_df[yr2])) < 3)
    ]

    ax.scatter(pair_df[yr1], pair_df[yr2], alpha = 0.4, s = 15, color = "#377ED8")
    slope, intercept, r, p , se = stats.linregress(pair_df[yr1], pair_df[yr2])

    x_line = np.linspace(pair_df[yr1].min(), pair_df[yr1].max(), 100)

    ax.plot(x_line, slope * x_line + intercept,
            color = "#E41A1C", linewidth = 2, label = f"r = {r:.3f} | n = {len(pair_df)}")

    ax.axhline(0, color = "black", linestyle = "--", linewidth = 0.8, alpha = 0.5)
    ax.axvline(0, color = "black", linestyle = "--", linewidth = 0.8, alpha = 0.5)

    ax.set_xlabel(f"RSAA/9 {yr1}")
    ax.set_ylabel(f"RSAA/9 {yr2}")
    ax.set_title(f"{yr1} --> {yr2}")
    ax.legend(fontsize = 9)

plt.tight_layout()
plt.savefig("outputs/validation/rsaa_yoy_stability.png",
            dpi = 150, bbox_inches = "tight")

plt.close()
print("Saved YoY")

print("All validation plots saved to outputs/validation/")

