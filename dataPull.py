import pandas as pd
import os

statcast_2022_2025 = pd.read_parquet("~/Downloads/statcast_data.parquet")

print(f"Rows: {len(statcast_2022_2025):}")
print(f"Columns: {len(statcast_2022_2025.columns):}")
print(f"\nDate Range: {statcast_2022_2025['game_date'].min()} to {statcast_2022_2025['game_date'].max()}")
