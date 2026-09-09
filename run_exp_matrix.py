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

run_cols = [col for col in statcast.columns if "score" in col.lower() or "play" in col.lower()]
print(run_cols)

if "game_year" not in statcast.columns:
    statcast["game_year"] = pd.to_datetime(statcast["game_date"]).dt.year

# Build the Base States

print("\nBuilding base states: ")

statcast["base_state"] = (
    np.where(statcast["on_1b"].notna(), "1", "-") +
    np.where(statcast["on_2b"].notna(), "1", "-") +
    np.where(statcast["on_3b"].notna(), "1", "-")
)

statcast["state"] = statcast["base_state"] + "_" + statcast["outs_when_up"].astype(str)

print(f"Unique States: {statcast['state'].unique()}")

print(statcast["state"].value_counts().sort_index())

# Compute Runs Scored from each PA to end of inning

print("\nComputing runs to end of inning: ")

pa_data = statcast[statcast["events"].notna() & (statcast["events"] != "")].copy()

pa_data["runs_on_play"] = pa_data["post_bat_score"] - pa_data["bat_score"].fillna(0)

pa_data = pa_data.sort_values(
    ["game_pk", "inning", "inning_topbot", "at_bat_number"]
)

pa_data["runs_to_end"] = (
    pa_data.groupby(["game_pk", "inning", "inning_topbot"])["runs_on_play"]
    .transform(lambda x: x[::-1].cumsum()[::-1])
)

print(f"PA-Ending pitches: {len(pa_data):}")
print(f"\nRuns to end of inning distribution: ")
print(pa_data["runs_to_end"].describe())

#Build the RE24 Matrix

re24 = (
    pa_data.groupby(["base_state", "outs_when_up"])["runs_to_end"]
    .mean()
    .reset_index()
    .rename(columns = {"runs_to_end": "run_expectancy"})
    .sort_values(["outs_when_up", "base_state"])
)

re24["state"] = re24["base_state"] + "_" + re24["outs_when_up"].astype(str)

print("\nRE24 Matrix")
print(re24.to_string(index = False))

re24_pivot = re24.pivot(
    index="base_state",
    columns="outs_when_up",
    values = "run_expectancy"
).round(3)

re24_pivot.columns = ["0 Outs", "1 Out", "2 Outs"]

print("\n RE24 Pivot Table")
print(re24_pivot.to_string())

print("\n Computing run value per pitch")

statcast = statcast.merge(
    re24[["state", "run_expectancy"]].rename(
        columns={"run_expectancy": "re_start"}
    ),
    on = "state",
    how = "left"
)


# print("Sample states in statcast:")
# print(statcast["state"].value_counts().head(10))

# Check state format in re24
# print("\nStates in re24:")
# print(re24["state"].tolist())

# Check for NaN in the components
# print(f"\nNull base_state: {statcast['base_state'].isna().sum():,}")
# print(f"Null outs_when_up: {statcast['outs_when_up'].isna().sum():,}")
# print(f"Null state: {statcast['state'].isna().sum():,}")

# print(f"outs_when_up dtype: {statcast['outs_when_up'].dtype}")
# print(f"Sample outs values: {statcast['outs_when_up'].unique()}")

statcast = statcast.sort_values(
    ["game_pk", "inning", "inning_topbot", "at_bat_number", "pitch_number"]
).reset_index(drop = True)

statcast["is_last_pitch_in_half"] = ~statcast.duplicated(
    subset= ["game_pk", "inning", "inning_topbot"],
    keep = "last"
)

statcast["next_base_state"] = (
    statcast.groupby(["game_pk", "inning", "inning_topbot"])["base_state"].shift(-1)
)

statcast["next_outs"] = (
    statcast.groupby(["game_pk", "inning", "inning_topbot"])["outs_when_up"].shift(-1)
)

statcast["next_state"] = np.where(
    statcast["is_last_pitch_in_half"],
    "INNING_END",
    statcast["next_base_state"] + "_" + statcast["next_outs"].astype("Int64").astype(str)
)

re24_end = re24[["state", "run_expectancy"]].rename(
    columns = {"state": "next_state",
               "run_expectancy": "re_end"}
)

inning_end_row = pd.DataFrame(
    [{"next_state": "INNING_END",
     "re_end": 0}])

re24_end = pd.concat([re24_end, inning_end_row], ignore_index=True)

if "re_end" in statcast.columns:
    statcast = statcast.drop(columns = ["re_end"])

statcast = statcast.merge(re24_end, on = "next_state", how="left")

# statcast = statcast.merge(
#    re24[["state", "run_expectancy"]].rename(
#        columns = {"state": "next_state",
#                   "run_expectancy": "re_end"}
#    ),
#    on = "next_state",
#    how = "left")

# print(f"re_start null: {statcast['re_start'].isna().sum():,}")
# print(f"re_start non-null: {statcast['re_start'].notna().sum():,}")

# print(f"\nre_end null: {statcast['re_end'].isna().sum():,}")
# print(f"re_end non-null: {statcast['re_end'].notna().sum():,}")

# Check next_state values
# print(f"\nSample next_state values:")
# print(statcast["next_state"].value_counts().head(10))
# print(f"\nNull next_state: {statcast['next_state'].isna().sum():,}")

statcast["runs_on_play"] = (
    statcast["post_bat_score"] - statcast["bat_score"]
).fillna(0)

statcast["run_value"] = (
    statcast["re_end"] - statcast["re_start"] + statcast["runs_on_play"]
)

print("\nRun Value Distribution: ")
print(statcast["run_value"].describe())
print(f"\nNull run values: {statcast['run_value'].isna().sum():}")

# Sanity Checks

print("\nSanity Checks")



rv_by_pitch = (
    (statcast.groupby("pitch_type")["run_value"]
    .agg(["mean", "count"])
    .round(4)
    .sort_values("mean"))
)

print("\nAvg Run Value by pitch type: ")
print(rv_by_pitch.to_string())

rv_by_event = (
    statcast[statcast["events"].notna()]
    .groupby("events")["run_value"]
    .mean()
    .sort_values(ascending=False)
    .head(15)
    .round(4)
)

print("\nAvg run value by event (15): ")
print(rv_by_event.to_string())

os.makedirs("data", exist_ok = True)
os.makedirs("outputs", exist_ok=True)

re24.to_csv("outputs/re_24_matrix.csv", index=False)
re24_pivot.to_csv("outputs/re24_pivot.csv")

statcast.to_parquet("data/statcast_with_rv.parquet", index = False)

print("\nSaved:")
print("  outputs/re24_matrix.csv")
print("  outputs/re24_pivot.csv")
print("  data/statcast_with_rv.parquet")

null_rv = statcast[statcast["run_value"].isna()]

print(f"Null re_restart: {null_rv['re_start'].isna().sum():}")
print(f"Null re_end: {null_rv['re_end'].isna().sum():}")

print(f"\nNull by inning:")
print(null_rv["inning"].value_counts().sort_index().head(10))

print(f"\nNull by event:")
print(null_rv["events"].value_counts().head(10))

print(f"\nNull next state sample: ")
print(null_rv["next_state"].value_counts().head(10))

