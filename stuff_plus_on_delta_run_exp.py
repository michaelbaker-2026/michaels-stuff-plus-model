from platform import release

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import warnings

warnings.filterwarnings("ignore")


statcast = pd.read_parquet('data/statcast_final_clean.parquet')

print(f"Rows: {len(statcast):,} | Games: {len(statcast['game_pk'].unique()):,} | Pitchers: {len(statcast['pitcher'].unique()):,}")

# Feature Engineering

if "horz_break" not in statcast.columns:
    statcast["horz_break"] = -statcast["pfx_x"] * 12

    statcast["ivb"] = statcast["pfx_z"] * 12

    statcast["norm_hb"] = np.where(
        statcast["p_throws"] == "L",
        -statcast["horz_break"],
        statcast["horz_break"]
    )

statcast["velo_pct"] = statcast.groupby("pitcher")["release_speed"].rank(pct=True)

statcast["spin_rate_efficiency"] = (np.sqrt(statcast["horz_break"]**2 + statcast["ivb"]**2)/
                                    (statcast["release_spin_rate"]/100))

fb_velo = (
    statcast.groupby("pitcher")["release_speed"]
    .quantile(0.90)
    .rename("fb_velo_baseline")
)

statcast = statcast.join(fb_velo, on = "pitcher")

statcast["velo_diff"] = statcast["fb_velo_baseline"] - statcast["release_speed"]

mean_IVB = statcast.groupby("pitcher")["ivb"].mean().rename("mean_IVB")

statcast = statcast.join(mean_IVB, on = "pitcher")

statcast["ivb_diff"] = statcast["ivb"] - statcast["mean_IVB"]

statcast["pitch_type"] = statcast["pitch_type"].fillna("UN")

le = LabelEncoder()
statcast["pitch_type_encoded"] = le.fit_transform(statcast["pitch_type"])

pitch_type_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Pitch type encoding:")
print(pitch_type_mapping)

statcast["throws_enc"] = (statcast["p_throws"] == "R").astype(int)

print("Feature Engineering Complete")


FEATURES = [
    "release_speed",
    "norm_hb",
    "ivb",
    "horz_break",
    "release_spin_rate",
    "velo_pct",
    "velo_diff",
    "ivb_diff",
    "release_pos_x",
    "release_pos_z",
    "effective_speed",
    "pitch_type_encoded",
    "spin_rate_efficiency",
    "release_extension",
    "ax",
    "ay",
    "az",
    "vx0",
    "vz0",
    "vy0"
]

TARGET = "delta_run_exp"

missing = [f for f in FEATURES + [TARGET] if f not in statcast.columns]
print(f"Missing columns: {missing}")

model_data = statcast[statcast[FEATURES + [TARGET]].notna().all(axis = 1)].copy()

print(f"Model-Ready Rows:  {len(model_data): }")
print(f"Dropped Rows: {len(statcast) - len(model_data): }")
print(f"\nPitch-Type Distribution:")
print(model_data["pitch_type"].value_counts())

# Train/Test Split

training_data = model_data[model_data["game_year"] < 2025]
testing_data = model_data[model_data["game_year"] == 2025]

X_train = training_data[FEATURES]
y_train = training_data[TARGET]
X_test = testing_data[FEATURES]
y_test = testing_data[TARGET]

print(f"Train: {len(X_train):,} pitches ({sorted(training_data['game_year'].unique())})")
print(f"Test: {len(X_test):,} pitches ({sorted(testing_data['game_year'].unique())})")

# Train XGBoost Model

model = XGBRegressor(
    n_estimators = 2000,
    max_depth = 6,
    learning_rate = 0.01,
    subsample = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 50,
    random_state = 42,
    n_jobs = -1,
    early_stopping_rounds = 50,
    eval_metric = "rmse"
)

model.fit(
    X_train, y_train,

    eval_set = [(X_test, y_test)],

    verbose = 50
)


light_model = LGBMRegressor(
    n_estimators = 500,
    max_depth = 5,
    learning_rate = 0.05,
    subsample = 0.8,
    colsample_bytree= 0.8,
    random_state= 42,
    n_jobs= -1,
    verbose = -1
)

light_model.fit(
    X_train, y_train,

    eval_set = [(X_test, y_test)],
    callbacks = [
        lgb.early_stopping(stopping_rounds = 20, verbose = True),
        lgb.log_evaluation(period = 50)
    ]
)

# Model Evaluation (LightGBM)

preds = light_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
corr = np.corrcoef(preds, y_test)[0, 1]

print(f"Test RMSE: {rmse:.4f}")
print(f"Correlation: {corr:.3f}")

# LightGM Feature Importance

lgb_importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": light_model.feature_importances_}).sort_values("importance", ascending= False)

print(f"\nFeature Importance:")
print(lgb_importance.to_string(index = False))

# Model Evaluation (XGBoost)

xgb_preds = model.predict(X_test)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
xgb_corr = np.corrcoef(xgb_preds, y_test)[0, 1]

print(f"Test RMSE (XGBoost): {xgb_rmse:.4f}")
print(f"Correlation: {xgb_corr:.3f}")

# XGBoost Feature Importance

xgb_importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_ }).sort_values("importance", ascending=False)

# Generate Stuff+ Scores

print("\nGenerating Stuff+ Scores: ")

model_data["xgb_predicted_run_value"] = model.predict(model_data[FEATURES])

model_data["lgb_predicted_run_value"] = light_model.predict(model_data[FEATURES])

# Normalize Per Pitch Per Season, 100 = League Average

league_stats = (model_data.groupby(["pitch_type", "game_year"])["xgb_predicted_run_value"]
              .agg(league_mean = "mean", league_std = "std")
              .reset_index()
              .rename(columns = {
                  "league_mean": "xgb_league_mean",
                  "league_std": "xgb_league_std"
              })
)

light_league_stats = (model_data.groupby(["pitch_type", "game_year"])["lgb_predicted_run_value"]
                    .agg(league_mean = "mean", league_std = "std")
                    .rename(columns = {
                        "league_mean": "lgb_league_mean",
                        "league_std": "lgb_league_std"
                    })
)

model_data = model_data.merge(league_stats, on = ["pitch_type", "game_year"], how = "left")

model_data = model_data.merge(light_league_stats, on = ["pitch_type", "game_year"], how = "left")

model_data["xgb_Stuff+"] = (100 + (-10 * (model_data["xgb_predicted_run_value"] - model_data["xgb_league_mean"])
                                          /model_data["xgb_league_std"])).round(1)

model_data["lgb_Stuff+"] = (100 + (-10 *
                                   (model_data["lgb_predicted_run_value"] - model_data["lgb_league_mean"])
                                   /model_data["lgb_league_std"])).round(1)

print("\nStuff+ Distribution (XGBoost):")
print(model_data["xgb_Stuff+"].describe())

print("\nStuff+ Distribution (LightGBM):")
print(model_data["lgb_Stuff+"].describe())

# Aggregate to Pitcher-Picth Type Level

stuff_summary = (
    model_data.groupby(["pitcher", "player_name", "pitch_type", "game_year"])
    .agg(
        num_pitches = ("pitch_type", "count"),
        xgb_stuff_plus = ("xgb_Stuff+", "mean"),
        lgb_stuff_plus = ("lgb_Stuff+", "mean"),
        avg_velo = ("release_speed", "mean"),
        avg_ivb = ("ivb", "mean"),
        avg_horz_break = ("horz_break", "mean"),
        avg_spin = ("release_spin_rate", "mean"),
        avg_xgb_rv = ("xgb_predicted_run_value", "mean"),
        avg_lgb_rv = ("lgb_predicted_run_value", "mean")
    ).reset_index()
)

stuff_summary = stuff_summary[stuff_summary["num_pitches"] >= 100]

print(f"\n Top 20 Pitches by XGBoost Stuff+: ")
print(stuff_summary.sort_values("xgb_stuff_plus", ascending= False)
      .head(20)[["player_name", "pitch_type", "game_year", "num_pitches",
                 "xgb_stuff_plus", "lgb_stuff_plus", "avg_velo", "avg_ivb", "avg_horz_break"]]
                 .to_string(index = False))

print(f"\nStuff+ Distribution (XGBoost): ")
print(stuff_summary["xgb_stuff_plus"].describe())

print(f"Top 20 Pitches by LightGBM Stuff+: ")
print(stuff_summary.sort_values("lgb_stuff_plus", ascending=False)
      .head(20)[["player_name", "pitch_type", "game_year", "num_pitches",
                 "lgb_stuff_plus", "xgb_stuff_plus", "avg_velo", "avg_ivb", "avg_horz_break"]]
                 .to_string(index = False))

print(f"\nStuff+ Distribution (LightGBM): ")
print(stuff_summary["lgb_stuff_plus"].describe())



# Save Outputs

os.makedirs("outputs", exist_ok = True)
stuff_summary.to_parquet("outputs/stuff_plus_summary.parquet", index = False)
stuff_summary.to_csv("outputs/stuff_plus_on_delta_run_exp_summary.csv", index = False)

model_data[[
    "pitcher", "player_name", "game_date", "game_pk", "game_year",
    "pitch_type", "xgb_Stuff+", "lgb_Stuff+", "xgb_predicted_run_value",
    "lgb_predicted_run_value"
] + FEATURES].to_parquet("outputs/pitch_level_stuff_plus.parquet", index = False)

print("\nOutputs saved to outputs")