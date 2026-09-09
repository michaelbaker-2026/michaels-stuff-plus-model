import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from scipy.stats import randint, uniform
import warnings
import os
warnings.filterwarnings("ignore")

model_data = pd.read_parquet("data/statcast_final_clean.parquet")

df = pd.read_parquet("data/statcast_final_clean.parquet")

year_cols = [col for col in model_data.columns if "year" in col.lower() or "date" in col.lower()]
print(year_cols)

if "pitch_type_encoded" not in model_data.columns:
    from sklearn.preprocessing import LabelEncoder
    model_data["pitch_type"] = model_data["pitch_type"].fillna("UN")
    le = LabelEncoder()
    model_data["pitch_type_encoded"] = le.fit_transform(model_data["pitch_type"])

if "norm_hb" not in model_data.columns:
    model_data["horz_break"] = -model_data["pfx_x"] * 12
    model_data["ivb"] = model_data["pfz_x"] * 12
    model_data["norm_hb"] = np.where(
        model_data["p_throws"] == "L",
        -model_data["horz_break"],
        model_data["horz_break"]
    )

model_data["velo_pct"] = model_data.groupby("pitcher")["release_speed"].rank(pct = True)

fb_velo = model_data.groupby("pitcher")["release_speed"].quantile(0.90).rename("fb_velo_baseline")

model_data = model_data.join(fb_velo, on = "pitcher")

model_data["velo_diff"] = model_data["fb_velo_baseline"] - model_data["release_speed"]

mean_ivb = model_data.groupby("pitcher")["ivb"].mean().rename("mean_ivb")

model_data = model_data.join(mean_ivb, on = "pitcher")

model_data["ivb_diff"] = model_data["ivb"] - model_data["mean_ivb"]

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
    "release_extension",
    "ax",
    "ay",
    "az"
]


TARGET = "delta_run_exp"

model_data = model_data[model_data[FEATURES + [TARGET]].notna().all(axis = 1)].copy()

print("Sampling Data for Cross-Validation: ")

cv_sample = (
    model_data
    .groupby("game_year", group_keys=False)
    .apply(lambda x: x.sample(frac = 0.20, random_state = 42))
    .reset_index(drop= True)
)

if "game_year" not in cv_sample.columns:
    cv_sample["game_year"] = pd.to_datetime(cv_sample["game_date"]).dt.year

X_cv = cv_sample[FEATURES]
y_cv = cv_sample[TARGET]

print(f"CV sample size: {len(X_cv):}")
print(f"Seasons: {sorted(cv_sample['game_year'].unique())}")

xgb_param_grid = [
    {
        "n_estimators": 1000,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 30,
    },

    {
        "n_estimators": 1000,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 50,
    },

    {
        "n_estimators": 2000,
        "max_depth": 6,
        "learning_rate": 0.01,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_weight": 50,
    },

    {
        "n_estimators": 1000,
        "max_depth": 4,
        "learning_rate": 0.01,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 20,
    },

    {
        "n_estimators": 500,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 100,
    },
]

lgb_param_grid = [
    {
        "n_estimators": 1000,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 50,
        "num_leaves": 31,
    },

    {
        "n_estimators": 1000,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 100,
        "num_leaves": 63,
    },

    {
        "n_estimators": 2000,
        "max_depth": 6,
        "learning_rate": 0.01,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_samples": 100,
        "num_leaves": 127,
    },

    {
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_samples": 200,
        "num_leaves": 15,
    },

    {
        "n_estimators": 1000,
        "max_depth": 4,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 50,
        "num_leaves": 31,
    },
]

# 5-Fold CV

kf = KFold(n_splits = 5, shuffle = True, random_state = 42)

def cv_score(model, X, y, kf):
    """Return mean and std RMSE across folds"""

    scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)

        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        scores.append(rmse)
        print(f"Fold {fold + 1}: RMSE = {rmse:.4f}")

    return np.mean(scores), np.std(scores)

# XGBoost Grid Search

xgb_results = []

for i, params in enumerate(xgb_param_grid):
    print("\nConfig {i + 1}/{len(xgb_param_grid)}: {params}")

    model = XGBRegressor(
        **params,
        random_state = 42,
        n_jobs = -1,
        eval_metric = "rmse"
    )

    mean_rmse, std_rmse = cv_score(model, X_cv, y_cv, kf)

    xgb_results.append({**params, "mean_rmse": mean_rmse, "std_rmse": std_rmse})

    print(f" -> Mean RMSE: {mean_rmse:.4f} +- {std_rmse:.4f}")

xgb_results_data_frame = pd.DataFrame(xgb_results).sort_values("mean_rmse")

print("\nXGBoost Results (Best First)")
print(xgb_results_data_frame.to_string(index = False))

best_xgb_params = xgb_results_data_frame.iloc[0].drop(["mean_rmse", "std_rmse"]).to_dict()

best_xgb_params = {k: int(v) if k in ["n_estiimators", "max_depth", "min_child_weight"]
                   else v for k, v in best_xgb_params.items()}

print(f"\nBest XGBoost Parameters: {best_xgb_params}")

# LightGBM Parameters Search

lgb_results = []

for i, params in enumerate(lgb_param_grid):
    print(f"\nConfig {i + 1}/{len(lgb_param_grid)}: {params}")

    model_lgb = LGBMRegressor(
        **params,
        random_state= 42,
        n_jobs= -1,
        verbose = -1
    )

    mean_rmse, std_rmse = cv_score(model_lgb, X_cv, y_cv, kf)
    lgb_results.append({**params, "mean_rmse": mean_rmse, "std_rmse": std_rmse})
    print(f" -> Mean RMSE: {mean_rmse:.4f} +- {std_rmse:.4f}")

lgb_results_data_frame = pd.DataFrame(lgb_results).sort_values("mean_rmse")

print("\n LightGBM Results (Ascending): ")
print(lgb_results_data_frame.to_string(index=False))

best_lgb_params = lgb_results_data_frame.iloc[0].drop(["mean_rmse", "std_rmse"]).to_dict()

best_lgb_params = {k: int(v) if k in ["n_estimators", "max_depth", "min_child_samples", "num_leaves"]
                   else v for k, v in best_lgb_params.items()}

print(f"\nBest LightGBM params: {best_lgb_params}")

os.makedirs("outputs", exist_ok=True)
xgb_results_data_frame.to_csv("outputs/xgb_cv_results.csv", index=False)
lgb_results_data_frame.to_csv("outputs/lgb_cv_results.csv", index = False)

print("\nCV results saved to outputs/")
print(f"\nBest XGBoost RMSE: {xgb_results_data_frame['mean_rmse'].min():.4f}")
print(f"\nBest LightGBM RMSE: {lgb_results_data_frame['mean_rmse'].min():.4f}")


