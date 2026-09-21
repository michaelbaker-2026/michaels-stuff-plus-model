# MLB Stuff+ Model & wOBAA Pipeline
A pitch quality and situational analytics pipeline built on MLB Statcast data (2022–2025) using Python, XGBoost, LightGBM, and DuckDB.

## Overview

- **RE24 Run Expectancy Matrix** — built from scratch using ~8,700 complete 9-inning games
- **Stuff+** — pitch quality metric normalized to 100 (league average), trained on per-pitch run value
- **RSAA/9** — Runs Saved Above Average per 9 innings, a pitcher-level run suppression metric
- **wOBAA / xwOBAA** — weighted on-base average against and expected version, computed with season-specific weights derived from the RE24 matrix, aggregated at pitcher, pitch type, team, and count level

## Methodology

### Stuff+ Model
1. Clean Statcast pitch-by-pitch data (2022–2025 regular season, complete 9-inning games only)
2. Build RE24 matrix from scratch using base state transitions
3. Compute run value per pitch: `RV = RE(end state) − RE(start state) + runs scored`
4. Train XGBoost and LightGBM regressors on run value using pitch physical characteristics
5. Normalize predictions to 100-point Stuff+ scale (z-score per pitch type per season)

### wOBAA / xwOBAA Pipeline
1. Derive season-specific wOBA weights from RE24 matrix (not fixed FanGraphs weights)
2. Map plate appearance outcomes to wOBA components (BB, HBP, 1B, 2B, 3B, HR)
3. Compute wOBAA (actual) and xwOBAA (expected, using `estimated_woba_using_speedangle`)
4. Normalize to wOBAA− and xwOBAA− (100 = league average, lower = better for pitchers)
5. Aggregate at pitcher level, pitch type level, team level, and team+pitch+count level

## Features Used (Stuff+ Model)

- Velocity, induced vertical break, horizontal break
- Spin rate, release position, extension
- Pitcher-relative velocity percentile and differential
- IVB differential from pitcher mean
- Acceleration components (ax, ay, az)

## Key Results

### Stuff+
- XGBoost and LightGBM Stuff+ rankings are highly correlated (r > 0.95)
- RSAA/9 correlates strongly with ERA and FIP
- Year-over-year Stuff+ stability: r = 0.49–0.62 (vs r = 0.11–0.19 for RSAA/9), confirming pitch quality is a repeatable skill signal

### wOBAA / xwOBAA
- Season-specific RE24 weights (e.g. 2022: BB=0.583, 1B=0.828, HR=2.519)
- All pitch type correlations with Stuff+ are negative — higher Stuff+ predicts lower wOBAA
- Sinker (SI) and four-seam (FF) show strongest Stuff+/wOBAA correlation (r = −0.46)
- LAD 2025 allowed the lowest wOBAA on FF in 0-2 counts (0.0695, wOBAA− = 36.0)
- Full count (3-2) analysis identifies highest-leverage pitch type effectiveness by team

## Project Structure

```
stuff_plus_model/
├── data_cleaning.py           # Game completeness checks, filtering to 9-inning games
├── run_expectancy.py          # RE24 matrix construction and run value computation
├── stuff_plus_rv.py           # Stuff+ model training on run value
├── stuff_plus_on_whiffs.py    # Alternative Stuff+ trained on whiff probability (AUC=0.76)
├── cv_model_stuff_plus.py     # Cross-validation and hyperparameter search
├── validation.py              # ERA/FIP/xFIP validation and YoY stability
├── woba.py                    # wOBAA/xwOBAA pipeline with season-specific RE24 weights
├── woba_by_count_by_team.py   # wOBAA/xwOBAA by team, pitch type, and count
├── woba_team_plots.py         # Team-level wOBAA visualizations
├── woba_count_plots.py        # Count-based wOBAA heatmaps and comparisons
├── sql_practice.py            # DuckDB SQL practice on baseball data
├── python_practice.py         # Pandas interview prep
├── outputs/
│   ├── re24_matrix.csv
│   ├── re24_pivot.csv
│   ├── stuff_plus_rv_summary.csv
│   ├── woba_pitcher.csv
│   ├── woba_pitch_type.csv
│   ├── woba_weights_by_season.csv
│   ├── woba_team.csv
│   ├── woba_by_team_pitch_count.csv
│   ├── plots/
│   │   └── count_woba/        # Count-based wOBAA visualizations
│   └── validation/            # ERA/FIP correlation and YoY stability plots
└── requirements.txt
```

## Requirements

```
pandas
numpy
pybaseball
scikit-learn
xgboost
lightgbm
matplotlib
seaborn
pyarrow
duckdb
scipy
```

## Data

Statcast pitch-by-pitch data pulled via `pybaseball`. Raw parquet files not included due to size (~3GB). See `data_cleaning.py` for the full cleaning pipeline.

Key cleaning decisions:
- Regular season only (`game_type == "R"`)
- Complete 9-inning games only (walk-off detection, out count validation)
- Deduplication on `[game_pk, at_bat_number, pitch_number]`
- Athletics standardized: `"ATH"` → `"OAK"`
