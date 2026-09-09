# MLB Stuff+ Model

A pitch quality model trained on MLB Statcast data (2022-2025) using XGBoost and LightGBM.

## Overview
- **Run Expectancy Matrix (RE24)** — built from ~8,700 complete 9-inning MLB games
- **Stuff+** — pitch quality metric normalized to 100 (league average), trained on per-pitch run value
- **RSAA/9** — Runs Saved Above Average per 9 innings, a pitcher-level run suppression metric

## Methodology
1. Clean Statcast pitch-by-pitch data (2022-2025 regular season, complete 9-inning games only)
2. Build RE24 matrix from scratch using base state transitions
3. Compute run value per pitch: `RV = RE(end state) - RE(start state) + runs scored`
4. Train XGBoost and LightGBM regressors on run value using pitch physical characteristics
5. Normalize predictions to 100-point Stuff+ scale via z-score per pitch type per season

## Features Used
- Velocity, induced vertical break, horizontal break
- Spin rate, release position, extension
- Pitcher-relative velocity percentile and differential
- Acceleration components (ax, ay, az)

## Key Results
- XGBoost and LightGBM produce highly correlated Stuff+ rankings (r > 0.95)
- RSAA/9 correlates strongly with ERA and FIP
- Year-over-year stability confirms metric captures true pitcher skill

## Project Structure
stuff_plus_model/
├── data_cleaning.py # Game completeness checks, filtering to 9-inning games
├── run_expectancy.py # RE24 matrix construction and run value computation
├── stuff_plus_rv.py # Stuff+ model training on run value
├── stuff_plus_on_whiffs.py # Alternative Stuff+ model trained on whiff probability
├── cv_model_stuff_plus.py # Cross-validation and hyperparameter search
├── validation.py # ERA/FIP/xFIP validation and visualizations
├── outputs/
│ ├── re24_matrix.csv
│ ├── re24_pivot.csv
│ ├── stuff_plus_rv_summary.csv
│ └── validation/
└── requirements.txt
## Requirements
## Data
Statcast pitch-by-pitch data pulled via `pybaseball`. 
Raw parquet files not included due to size (~3GB). 
See `data_cleaning.py` for the full cleaning pipeline.
