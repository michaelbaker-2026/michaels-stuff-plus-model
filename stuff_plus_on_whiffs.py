from platform import release

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.metrics import log_loss, roc_auc_score, auc, confusion_matrix, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import os 
import warnings

warnings.filterwarnings("ignore")


statcast = pd.read_parquet("data/statcast_final_clean.parquet")

statcast["is_swing"] = statcast["description"].isin([
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
    "foul_bunt",
    "missed_bunt"
]).astype(int)

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

statcast["is_whiff"] = statcast["description"].isin(["swinging_strike", "swinging_strike_blocked"]).astype(int)

swing_data = statcast[statcast["is_swing"] == 1].copy()

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

TARGET = "is_whiff"

missing = [f for f in FEATURES + [TARGET] if f not in swing_data.columns]
print(f"Missing columns: {missing}")


model_data = swing_data[swing_data[FEATURES + [TARGET]].notna().all(axis = 1)].copy()

print(f"Model-Ready Rows:  {len(model_data): }")
print(f"Dropped Rows: {len(statcast) - len(model_data): }")
print(f"\nPitch-Type Distribution:")
print(model_data["pitch_type"].value_counts())

training_data = model_data[model_data["game_year"] < 2025]
testing_data = model_data[model_data["game_year"] == 2025]

X_train = training_data[FEATURES]
y_train = training_data[TARGET]
X_test = testing_data[FEATURES]
y_test = testing_data[TARGET]


# Train XGBoost Model

model = XGBClassifier(
    n_estimators = 1000,
    max_depth = 5,
    learning_rate = 0.05,
    subsample = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 50,
    random_state = 42,
    n_jobs = -1,
    early_stopping_rounds = 20,
    eval_metric = "logloss",
    use_label_encoder = False
)

model.fit(
    X_train, y_train,

    eval_set = [(X_test, y_test)],

    verbose = 50
)


light_model = LGBMClassifier(
    n_estimators = 1000,
    max_depth = 5,
    learning_rate = 0.05,
    subsample = 0.8,
    colsample_bytree= 0.8,
    min_child_samples = 50,
    num_leaves = 31,
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

xgb_probs = model.predict_proba(X_test)[:, 1]
lgb_probs = light_model.predict_proba(X_test)[:, 1]

xgb_auc = roc_auc_score(y_test, xgb_probs)
lgb_auc = roc_auc_score(y_test, lgb_probs)

xgb_logloss = log_loss(y_test, xgb_probs)
lgb_logloss = log_loss(y_test, lgb_probs)


print(f"XGBoost - AUC: {xgb_auc:.4f} | LogLoss: {xgb_logloss:.4f}")
print(f"LightGBM - AUC: {lgb_auc:.4f} | LogLoss: {lgb_logloss:.4f}")

# Feature Importance

xgb_importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_}).sort_values("importance", ascending=False)

lgb_importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": light_model.feature_importances_}).sort_values("importance", ascending=False)


print(f"\nFeature Importance:")
print(xgb_importance.to_string(index = False))

print(f"\nFeature Importance:")
print(lgb_importance.to_string(index = False))



# Predicted Whiff Prob per Pitch

swing_data["xgb_whiff_prob"] = model.predict_proba(swing_data[FEATURES])[:, 1]
swing_data["lgb_whiff_prob"] = model.predict_proba(swing_data[FEATURES])[:, 1]

xgb_league_stats = (
    swing_data.groupby(["pitch_type", "game_year"])["xgb_whiff_prob"]
    .agg(league_mean = "mean", league_std = "std")
    .reset_index()
    .rename(columns = {"league_mean": "xgb_league_mean", "league_std": "xgb_league_std"})
)

lgb_league_stats = (
    swing_data.groupby(["pitch_type", "game_year"])["lgb_whiff_prob"]
    .agg(league_mean = "mean", league_std = "std")
    .reset_index()
    .rename(columns = {"league_mean": "lgb_league_mean", "league_std": "lgb_league_std"})
)

swing_data = swing_data.merge(xgb_league_stats, on = ["pitch_type", "game_year"], how = "left")
swing_data = swing_data.merge(lgb_league_stats, on = ["pitch_type", "game_year"], how = "left")

swing_data["xgb_Stuff+"] = (
    100 + (10 * (swing_data["xgb_whiff_prob"] - swing_data["lgb_league_mean"])
           /swing_data["xgb_league_std"])
).round(1)

swing_data["lgb_Stuff+"] = (
    100 +(10 * (swing_data["xgb_whiff_prob"] - swing_data["lgb_league_mean"])/
          swing_data["lgb_league_std"])
).round(1)

print("\nStuff+ Distribution (XGBoost):")
print(swing_data["xgb_Stuff+"].describe())

print("\nStuff+ Distribution (LightGBM):")
print(swing_data["lgb_Stuff+"].describe())

# Aggregate to Pitcher-Picth Type Level

stuff_summary = (
    swing_data.groupby(["pitcher", "player_name", "pitch_type", "game_year"])
    .agg(
        num_pitches = ("pitch_type", "count"),
        xgb_stuff_plus = ("xgb_Stuff+", "mean"),
        lgb_stuff_plus = ("lgb_Stuff+", "mean"),
        avg_velo = ("release_speed", "mean"),
        avg_ivb = ("ivb", "mean"),
        avg_horz_break = ("horz_break", "mean"),
        avg_spin = ("release_spin_rate", "mean")
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
stuff_summary.to_csv("outputs/stuff_plus_on_whiffs_summary.csv", index = False)

swing_data[[
    "pitcher", "player_name", "game_date", "game_pk", "game_year",
    "pitch_type", "xgb_Stuff+", "lgb_Stuff+"
] + FEATURES].to_parquet("outputs/pitch_level_stuff_plus.parquet", index = False)

print("\nOutputs saved to outputs")

os.makedirs("outputs/plots", exist_ok=True)

# ── Color palette ─────────────────────────────────────────────────────────────
COLORS = {
    "xgb":    "#E41A1C",
    "lgb":    "#377EB8",
    "neutral": "#888888",
    "green":  "#4DAF4A"
}

# Stuff+ Distribution

fig, axes = plt.subplots(1, 2, figsize = (14, 5))

fig.suptitle("Stuff+ Distribution - All Pitches", fontsize = 14, fontweight = "bold")

for ax, col, label, color in zip(
    axes,
    ["xgb_stuff_plus", "lgb_stuff_plus"],
    ["XGBoost", "LightLGM"],
    [COLORS["xgb"], COLORS["lgb"]]
): 
    data = stuff_summary[col].dropna()
    ax.hist(data, bins = 50, color = color, alpha = 0.7, edgecolor = "white")
    ax.axvline(100, color = "black", linestyle = "--", linewidth = 1.5, label = "League Avg (100)")
    ax.axvline(data.mean(), color = color, linestyle = "--", linewidth = 2,
               label = f"Mean: {data.mean():.1f}")
    ax.set_xlabel("Stuff+")
    ax.set_ylabel("Count")
    ax.set_title(f"{label} Stuff+")
    ax.legend()

plt.tight_layout()
plt.savefig("outputs/plots/stuff_plus_distribution.png", dpi = 150, bbox_inches = "tight")
plt.close()
print("Saved: stuff_plus_distribution.png")

fig, axes = plt.subplots(1, 2, figsize = (16, 6))

fig.suptitle("Stuff+ by Pitch Type", fontsize = 14, fontweight = "bold")

for ax, col, label, color in zip(
    axes,
    ["xgb_stuff_plus", "lgb_stuff_plus"],
    ["XGBoost", "LightGBM"],
    [COLORS["xgb"], COLORS["lgb"]]
): 
    pitch_avg = (
        stuff_summary.groupby("pitch_type")[col]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    bars = ax.barh(pitch_avg["pitch_type"], pitch_avg[col], color = color, alpha = 0.7)
    ax.axvline(100, color = "black", linestyle = "--", linewidth = 1.5)
    ax.set_xlabel("Avg Stuff+")
    ax.set_title(f"{label}")
    ax.invert_yaxis()

    for bar, val in zip(bars, pitch_avg[col]):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}", va = "center", fontsize = 9)

    plt.tight_layout()
    plt.savefig("outputs/plots/stuff_plus_by_pitch_type.png", dpi = 150, bbox_inches = "tight")
    plt.close()

    print("Saved: stuff_plus_by_pitch_type.png")

fig, ax = plt.subplots(figsize = (8, 7))

ax.scatter(
    stuff_summary["xgb_stuff_plus"],
    stuff_summary["lgb_stuff_plus"],
    alpha = 0.3, s = 20, color = COLORS["neutral"]
)

lims = [
    min(stuff_summary["xgb_stuff_plus"].min(), stuff_summary["lgb_stuff_plus"].min()),
    max(stuff_summary["xgb_stuff_plus"].max(), stuff_summary["lgb_stuff_plus"].max())
]

ax.plot(lims, lims, "k--", linewidth = 1.5, label = "Perfect Alignment")

corr = stuff_summary[["xgb_stuff_plus", "lgb_stuff_plus"]].corr().iloc[0, 1]
ax.set_xlabel("XGBoost Stuff+")
ax.set_ylabel("LightGBM Stuff+")
ax.set_title(f"XGBoost vs. LightGBM Stuff+\nCorrelation: {corr:.3f}")
ax.legend()

plt.tight_layout()
plt.savefig("outputs/plots/xgb_vs_lgb_correlation.png", dpi = 150, bbox_inches = "tight")
plt.close()
print("Saved: xgb_vs_lgb_correlation.png")

fig, ax = plt.subplots(figsize = (8, 7))

for probs, label, color in [
    (xgb_probs, f"XGBoost (AUC = {xgb_auc:.4f})", COLORS["xgb"]),
    (lgb_probs, f"LightGBM (AUC = {lgb_auc:.4f})", COLORS["lgb"])
]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    ax.plot(fpr, tpr, color = color, linewidth = 2, label = label)

ax.plot([0, 1], [0, 1], "k--", linewidth = 1, label = "Random Classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve -- Whiff Prediction")
ax.legend(loc = "lower right")

plt.tight_layout()
plt.savefig("outputs/plots/roc_curves.png", dpi = 150, bbox_inches = "tight")
plt.close()

print("Saved: roc_curves.png")

fig, ax = plt.subplots(figsize = (8, 7))

for probs, label, color in [
    (xgb_probs, "XGBoost", COLORS["xgb"]),
    (lgb_probs, "LightGBM", COLORS["lgb"])
]: 
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins = 20)
    ax.plot(prob_pred, prob_true, marker = "o", linewidth = 2, color = color, label = label)

ax.plot([0, 1], [0, 1], "k--", linewidth = 1, label = "Perfectly Calibrated")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives (Actual Whiff Rate)")
ax.set_title("Calibration Curve -- Whiff Prediction")
ax.legend()

plt.tight_layout()
plt.savefig("outputs/plots/calibration_curves.png", dpi = 150, bbox_inches = "tight")
plt.close()
print("Saved: calibration_curves.png")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

fig.suptitle("Feature Importance", fontsize=14, fontweight="bold")

for ax, importance_df, label, color in zip(
    axes,
    [xgb_importance, lgb_importance],
    ["XGBoost", "LightGBM"],
    [COLORS["xgb"], COLORS["lgb"]]
):
    imp = importance_df.sort_values("importance", ascending=True)
    ax.barh(imp["feature"], imp["importance"], color=color, alpha=0.7)
    ax.set_xlabel("Importance")
    ax.set_title(f"{label} Feature Importance")

plt.tight_layout()
plt.savefig("outputs/plots/feature_importance.png")
plt.close()

print("SAved: feature_importance.png")

fig, ax = plt.subplots(figsize = (12, 6))

pitch_types_to_show = (
    swing_data.groupby("pitch_type")["xgb_whiff_prob"]
    .count()
    .sort_values(ascending=False)
    .head(8)
    .index
    .to_list()
)

plot_data = swing_data[swing_data["pitch_type"].isin(pitch_types_to_show)]

sns.violinplot(
    data = plot_data,
    x = "pitch_type",
    y = "xgb_whiff_prob",
    palette="Set2",
    ax=ax,
    order = pitch_types_to_show
)

ax.axhline(
    swing_data["xgb_whiff_prob"].mean(),
    color = "black", linestyle = "--", linewidth = 1.5,
    label = f"League Avg: {swing_data['xgb_whiff_prob'].mean():.3f}"
)

ax.set_xlabel("Pitch Type")
ax.set_ylabel("Predicted Whiff Probability")
ax.set_title("Predicted Whiff Probability by Pitch Type (XGBoost)")
ax.legend()

plt.tight_layout()
plt.savefig("outputs/plots/whiff_prob_by_pitch_type.png")
plt.close()

print("Saved: whiff_prob_by_pitch_type.png")

fig, ax = plt.subplots(figsize = (10, 6))

actual_vs_pred = (
    swing_data.groupby("pitch_type")
    .agg(
        actual_whiff_rate = ("is_whiff", "mean"),
        xgb_predicted_rate = ("xgb_whiff_prob", "mean"),
        lgb_predicted_rate = ("lgb_whiff_prob", "mean"),
        n_pitches = ("is_whiff", "count")
    )
    .reset_index()
    .query("n_pitches > = 1000")
    .sort_values("actual_whiff_rate", ascending=False)
)

x = np.arange(len(actual_vs_pred))

width = 0.25

ax.bar(x - width, actual_vs_pred["actual_whiff_rate"],
       width, label = "Actual", color = COLORS["green"], alpha = 0.8)
ax.bar(x, actual_vs_pred["xgb_predicted_rate"], width, 
       label = "XGBoost", color = COLORS["xgb"], alpha = 0.8)
ax.bar(x + width, actual_vs_pred["lgb_predicted_rate"], 
       width, label = "LightGBM", color = COLORS["lgb"], alpha = 0.8)

ax.set_xticks(x)
ax.set_xticklabels(actual_vs_pred["pitch_type"], rotation = 45, ha = "right")
ax.set_ylabel("Whiff Rate")
ax.set_title("Actual vs. Predicted Whiff Rate by Pitch")
ax.legend()

plt.tight_layout()
plt.savefig("outputs/plots/actual_vs_pred_whiff_rate.png", dpi = 150, bbox_inches = "tight")
plt.close()

print("Saved: actual_vs_pred_whiff_rate.png")

print("\nAll plots saved to outputs/plots/")