import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib as plt
import os
import warnings

warnings.filterwarnings("ignore")

# Data

statcast = pd.read_parquet("data/statcast_with_rv.parquet")

if "game_year" not in statcast.columns:
    statcast["game_year"] = pd.to_datetime(statcast["game_date"]).dt.year

if "horz_break" not in statcast.columns:
    statcast["horz_break"] = -statcast["pfx_x"] * 12
    statcast["ivb"] = statcast["pfz_x"] * 12

statcast["norm_hb"] = np.where(
    statcast["p_throws"] == "L",
    -statcast["norm_hb"],
    statcast["norm_hb"]
)

statcast["velo_pct"] = (
    statcast.groupby("pitcher")["release_speed"].rank(pct=True)
)

fb_velo = (
    statcast.groupby("pitcher")["release_speed"]
    .quantile(0.90)
    .rename("fb_velo_baseline")
)

statcast = statcast.join(fb_velo, on = "pitcher")

statcast["velo_diff"] = statcast["fb_velo_baseline"] - statcast["release_speed"]

mean_ivb = (
    statcast.groupby("pitcher")["ivb"]
    .mean()
    .rename("mean_ivb")
)

statcast = statcast.join(mean_ivb, on = "pitcher")

statcast["ivb_diff"] = statcast["ivb"] - statcast["mean_ivb"]

# Spin Efficiency

statcast["movement_magnitude"] = np.sqrt(
    statcast["horz_break"]**2 + statcast["ivb"]**2
)

statcast["pitch_type"] = statcast["pitch_type"].fillna("UN")

le = LabelEncoder()

statcast["pitch_type_encoded"] = le.fit_transform(statcast["pitch_type"])

pitch_type_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

print(f"Pitch Type Encoding: {pitch_type_mapping}")

print("Feature Engineering Complete")

# Features in the models

FEATURES = [
    "release_speed",
    "norm_hb",
    "ivb",
    "ivb_diff",
    "horz_break",
    "movement_magnitude",
    "velo_pct",
    "velo_diff",
    "release_spin_rate",
    "release_pos_x",
    "release_pos_z",
    "effective_speed",
    "release_extension",
    "pitch_type_encoded",
    "ax",
    "ay",
    "az",
    "vx0",
    "vy0",
    "vz0",
    "plate_x",
    "plate_z"
]

TARGET = "run_value"

# Filter to model-ready rows

model_data = statcast[
    statcast[FEATURES + [TARGET]].notna().all(axis = 1)
].copy()

print(f"Model-Ready Rows: {len(model_data):}")
print(f"Dropped rows: {len(statcast) - len(model_data):}")
print(f"\nPitch-Type Distribution: ")
print(model_data["pitch_type"].value_counts())

# Train/Test Split

train_data = model_data[model_data["game_year"] < 2025]
test_data = model_data[model_data["game_year"] == 2025]

X_train = train_data[FEATURES]
y_train = train_data[TARGET]
X_test = test_data[FEATURES]
y_test = test_data[TARGET]

print(f"Train: {len(X_train):,} pitches ({sorted(train_data['game_year'].unique())})")
print(f"Test: {len(X_test):,} pitches ({sorted(test_data['game_year'].unique())})")

# Train XGBoost

xgb_model = XGBRegressor(
    n_estimators = 1000,
    max_depth = 5,
    learning_rate = 0.05,
    subsample = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 50,
    random_state = 42, 
    n_jobs = -1,
    early_stopping_rounds = 30,
    eval_metric = "rmse"
)

xgb_model.fit(
    X_train, y_train,
    eval_set = [(X_test, y_test)],
    verbose = 50
)

# Train LightGBM

lgb_model = LGBMRegressor(
    n_estimators = 500,
    max_depth= 5,
    learning_rate= 0.05,
    subsample = 0.8,
    colsample_bytree= 0.8,
    min_child_samples= 50,
    num_leaves= 31,
    random_state=42,
    n_jobs=-1,
    verbose = -1 
)

lgb_model.fit(
    X_train, y_train,
    eval_set= [(X_test, y_test)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=30, verbose=True),
        lgb.log_evaluation(period=50)
    ]
)

xgb_preds = xgb_model.predict(X_test)
lgb_preds = lgb_model.predict(X_test)

xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_preds))

xgb_corr = np.corrcoef(xgb_preds, y_test)[0, 1]
lgb_corr = np.corrcoef(lgb_preds, y_test)[0, 1]

print(f"XGBoost - RMSE: {xgb_rmse:.4f} | Correlation: {xgb_corr:.4f}")
print(f"LightGBM - RMSE: {lgb_rmse:.4f} | Correlation: {lgb_corr:.3f}")

xgb_importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": xgb_model.feature_importances_}).sort_values("importance", ascending=False)

lgb_importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": lgb_model.feature_importances_}).sort_values("importance", ascending=False)

print(f"\nXGBoost Feature Importance: ")
print(xgb_importance.to_string(index = False))

print(f"\nLightGBM Feature Importance: ")
print(lgb_importance.to_string(index = False))

# Generate Stuff+ Scores

model_data["xgb_predicted_rv"] = xgb_model.predict(model_data[FEATURES])
model_data["lgb_predicted_rv"] = lgb_model.predict(model_data[FEATURES])


xgb_league_stats = (
    model_data.groupby(["pitch_type", "game_year"])["xgb_predicted_rv"]
    .agg(league_mean = "mean", league_std = "std")
    .reset_index()
    .rename(columns = {
        "league_mean": "xgb_league_mean",
        "league_std": "xgb_league_std"
    })
)

lgb_league_stats = (
    model_data.groupby(["pitch_type", "game_year"])["lgb_predicted_rv"]
    .agg(league_mean = "mean", league_std = "std")
    .reset_index()
    .rename(columns = {
        "league_mean": "lgb_league_mean",
        "league_std": "lgb_league_std"
    })
)

model_data = model_data.merge(xgb_league_stats, on = ["pitch_type", "game_year"], how = "left")

model_data = model_data.merge(lgb_league_stats, on = ["pitch_type", "game_year"], how = "left")

model_data["xgb_Stuff+"] = (
    100 + (
        -10 * model_data["xgb_predicted_rv"] - model_data["xgb_league_mean"])
        /model_data["xgb_league_std"]).round(1)

model_data["lgb_Stuff+"] = (
    100 + (
        -10 * model_data["lgb_predicted_rv"] - model_data["lgb_league_mean"])
        /model_data["lgb_league_std"]).round(1)


print(f"\nXGBoost Stuff+ Distribution: ")
print(model_data["xgb_Stuff+"].describe())

print(f"\nLightGBM Stuff+ Distribution: ")
print(model_data["lgb_Stuff+"].describe())

stuff_summary = (
    model_data.groupby(["pitcher", "player_name", "pitch_type", "game_year"])
    .agg(
        num_pitches = ("pitch_type", "count"),
        xgb_stuff_plus = ("xgb_Stuff+", "mean"),
        lgb_stuff_plus = ("lgb_Stuff+", "mean"),
        avg_release_speed = ("release_speed", "mean"),
        avg_hb = ("horz_break", "mean"),
        avg_ivb = ("ivb", "mean"),
        avg_spin = ("release_spin_rate", "mean"),
        avg_rv = ("run_value", "mean"),
        avg_spin_efficiency = ("movement_magnitude", "mean"),
        avg_ax = ("ax", "mean"),
        avg_az = ("az", "mean"),
        avg_ay = ("ay", "mean"),
        avg_vx0 = ("vx0", "mean"),
        avg_vy0 = ("vy0", "mean"),
        avg_vz0 = ("vz0", "mean"),
        xgb_pred_rv = ("xgb_predicted_rv", "mean"),
        lgb_pred_rv = ("lgb_predicted_rv", "mean")
    )
    .reset_index()
)

stuff_summary = stuff_summary[stuff_summary["num_pitches"] >= 400]

print("\nTop 20 Pitches by XGBoost Stuff+: ")
print(
    stuff_summary
    .sort_values("xgb_stuff_plus", ascending=False)
    .head(20)
    [["player_name", "pitch_type", "game_year",
      "num_pitches", "xgb_stuff_plus", "lgb_stuff_plus",
      "avg_release_speed", "avg_ivb", "avg_hb", 
      "xgb_pred_rv", "lgb_pred_rv"]].to_string(index=False)
)

print("\nBottom 10 Pitches by XGBoost Stuff+: ")
print(
    stuff_summary
    .sort_values("xgb_stuff_plus", ascending=True)
    .head(10)
    [["player_name", "pitch_type", "game_year",
      "num_pitches", "xgb_stuff_plus", "lgb_stuff_plus",
      "avg_release_speed", "avg_ivb", "avg_hb"]].to_string(index=False)
)



print("\nTop 20 Pitches by LightGBM Stuff+: ")
print(
    stuff_summary
    .sort_values("lgb_stuff_plus", ascending=False)
    .head(20)
    [["player_name", "pitch_type", "game_year",
      "num_pitches", "xgb_stuff_plus", "lgb_stuff_plus",
      "avg_release_speed", "avg_ivb", "avg_hb",
      "xgb_pred_rv", "lgb_pred_rv"]].to_string(index=False)
)

print("\nBottom 10 Pitches by LightGBM Stuff+: ")
print(
    stuff_summary
    .sort_values("lgb_stuff_plus", ascending=True)
    .head(10)
    [["player_name", "pitch_type", "game_year",
      "num_pitches", "xgb_stuff_plus", "lgb_stuff_plus",
      "avg_release_speed", "avg_ivb", "avg_hb"]].to_string(index=False)
)

rv_by_pitcher_pitch = (
    model_data.groupby(["pitcher", "player_name", "pitch_type", "game_year"])
    .agg(
        total_rv = ("run_value", "sum"),
        total_pitches = ("run_value", "count"),
        avg_rv = ("run_value", "mean"),
        rv_per_100 = ("run_value", lambda x: x.mean() * 100)       
    ).reset_index()
    .sort_values(["pitcher", "game_year", "total_rv"])
)

rv_by_pitcher = (
    model_data.groupby(["pitcher", "player_name", "game_year"])
    .agg(
        total_rv = ("run_value", "sum"),
        total_pitches = ("run_value", "count"),
        avg_rv_per_pitch = ("run_value", "mean"),
        rv_per_100 = ("run_value", lambda x: x.mean() * 100)
    )
    .reset_index()
    .sort_values("total_rv")
)

print("Best Run Suppressors (lowest = Best)")
print(
    rv_by_pitcher
    .head(20)
    [["player_name", "game_year",
      "total_pitches", "total_rv", "avg_rv_per_pitch",
        "rv_per_100"]].round(3).to_string(index = False)
)

print("Worst Run Suppressors")
print(
    rv_by_pitcher
    .tail(10)
    .sort_values("total_rv", ascending=False)
    [["player_name", "game_year",
      "total_pitches", "total_rv", "avg_rv_per_pitch",
        "rv_per_100"]].round(3).to_string(index = False)
)

print("Best Run Suppressing Pitches")
print(
    rv_by_pitcher_pitch
    .sort_values("total_rv")
    .head(20)
    [["player_name", "pitch_type", "game_year",
      "total_pitches", "total_rv", "avg_rv",
        "rv_per_100"]].round(3).to_string(index = False)
)

print("Worst Run Suppressing Pitches")
print(
    rv_by_pitcher_pitch
    .sort_values("total_rv", ascending=True)
    .tail(20)
    [["player_name", "pitch_type", "game_year",
      "total_pitches", "total_rv", "avg_rv",
        "rv_per_100"]].round(3).to_string(index = False)
)

os.makedirs("outputs", exist_ok=True)

stuff_summary.to_parquet("outputs/stuff_summary_on_run_value.parquet", index=False)
stuff_summary.to_csv("outputs/stuff_summary_on_run_value.csv", index= False)

model_data[[
    "pitcher", "player_name", "game_date", "game_pk", "game_year",
    "pitch_type", "run_value", "xgb_predicted_rv", "lgb_predicted_rv",
    "xgb_Stuff+", "lgb_Stuff+"] + FEATURES].to_parquet("outputs/pitch_level_stuff_plus_on_rv.parquet", index = False)

# Innings Normalizer Runs Saved Above Average

out_events = {
    "strikeout", "field_out", "grounded_into_double_play",
    "double_play", "triple_play", "force_out", "sac_fly",
    "sac_bunt", "fielders_choice_out", "strikeout_double_play", "truncated_pa", "fielders_choice"
}

double_play_events = {
    "strikeout_double_play", "ground_into_double_play", "double_play"
}

triple_play_events = {"triple_play"}

pa_endings = model_data[
    model_data["events"].notna() & (model_data["events"] != "") 
].copy()

pa_endings["outs_on_play"] = np.where(
    pa_endings["events"] == "triple_play", 3,
    np.where(pa_endings["events"].isin(double_play_events), 2,
    np.where(pa_endings["events"].isin(out_events), 1, 0))
)

pitcher_outs = (
    pa_endings.groupby(["pitcher", "player_name", "game_pk", "game_year",
                        "inning", "inning_topbot"])
    ["outs_on_play"].sum()
    .reset_index()
)

pitcher_total_outs = (
    pa_endings.groupby(["pitcher", "player_name", "game_year"])

    ["outs_on_play"].sum()
    .reset_index()
    .rename(columns = {"outs_on_play": "total_outs"})
)

pitcher_total_outs["ip_decimal"] = pitcher_total_outs["total_outs"] / 3

pitcher_total_outs["ip_standard"] = (
    pitcher_total_outs["total_outs"] // 3
).astype(str) + "." + (
    pitcher_total_outs["total_outs"] % 3
).astype(str)

print("Sample IP: ")
print(pitcher_total_outs.head(10).to_string(index = False))

# Join IP to pitcher-level RV

rv_by_pitcher = rv_by_pitcher.merge(
    pitcher_total_outs[["pitcher", "game_year", "total_outs", "ip_decimal", "ip_standard"]],
    on = ["pitcher", "game_year"],
    how = "left"
)

# RAA/9

league_avg_rv_per_pitch = model_data["run_value"].mean()

print(f"\nLeague Avg RV per Pitch: {league_avg_rv_per_pitch:.6f}")

rv_by_pitcher["expected_rv"] = (
    rv_by_pitcher["total_pitches"] * league_avg_rv_per_pitch
)

rv_by_pitcher["runs_saved_above_avg"] = (
    rv_by_pitcher["expected_rv"] - rv_by_pitcher["total_rv"]
)

rv_by_pitcher["runs_saved_above_avg_per_9"] = (
    rv_by_pitcher["runs_saved_above_avg"] / rv_by_pitcher["ip_decimal"] * 9
 ).round(3)

rv_by_pitcher["runs_saved_per_100"] = (
    rv_by_pitcher["runs_saved_above_avg"] / rv_by_pitcher["total_pitches"] * 100
).round(3)

qualified_run_suppressors = rv_by_pitcher[rv_by_pitcher["total_pitches"] >= 400]

print(f"\nBest Run Suppressors - Runs Saved per 9 IP")
print(qualified_run_suppressors
      .sort_values("runs_saved_above_avg_per_9", ascending=False)
      .head(20)
      [["player_name", "game_year", "ip_standard", "total_pitches",
        "total_rv", "runs_saved_above_avg", "runs_saved_above_avg_per_9",
        "runs_saved_per_100"]]
        .round(3)
        .to_string(index=False))

print(f"\nWorst Run Suppressors - Runs Saved per 9 IP")
print(qualified_run_suppressors
      .sort_values("runs_saved_above_avg_per_9", ascending=True)
      .head(20)
      [["player_name", "game_year", "ip_standard", "total_pitches",
        "total_rv", "runs_saved_above_avg", "runs_saved_above_avg_per_9",
        "runs_saved_per_100"]]
        .round(3)
        .to_string(index = False)
      )

qualified_sp = rv_by_pitcher[rv_by_pitcher["ip_decimal"] >= 100]
qualified_rp = rv_by_pitcher[
    (rv_by_pitcher["ip_decimal"] >= 30) &
    (rv_by_pitcher["ip_decimal"] < 100)
]

print(f"\nQualified Starters (100+ IP) Best RSAA/9: ")
print(qualified_sp
      .sort_values("runs_saved_above_avg_per_9", ascending=False)
      .head(20)
      [["player_name", "game_year", "ip_standard", "total_pitches",
        "total_rv", "runs_saved_above_avg", "runs_saved_above_avg_per_9",
         "runs_saved_per_100"]]
         .round(3)
         .to_string(index = False)
    )

print(f"\nQualified Relievers (30-100 IP) Best RSAA/9: ")
print(qualified_rp
      .sort_values("runs_saved_above_avg_per_9", ascending=False)
      .head(20)
      [["player_name", "game_year", "ip_standard", "total_pitches",
        "total_rv", "runs_saved_above_avg", "runs_saved_above_avg_per_9",
        "runs_saved_per_100"]]
        .round(3)
        .to_string(index = False)
    )

print(
    rv_by_pitcher[rv_by_pitcher["player_name"].str.contains("Francis, Bowden")]
    [["player_name", "game_year", "ip_standard", "total_pitches",
      "total_rv", "runs_saved_above_avg_per_9"]]
    .sort_values("game_year")
    .to_string(index=False)
)

rv_by_pitcher.to_csv("outputs/runs_value_with_ip.csv", index = False)
rv_by_pitcher_pitch.to_csv("outputs/run_value_by_ptcher_pitch_type.csv", index=False)


print("\nOutputs saved to outputs")