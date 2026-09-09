bad_game = inning_outs[inning_outs["outs"] > 3]["game_pk"].iloc[0]
print(f"\nBad game: {bad_game}")
print(pa_endings[pa_endings["game_pk"] == bad_game][
    ["game_pk", "inning", "inning_topbot", "at_bat_number", 
     "pitch_number", "events", "outs_on_play"]
].to_string())

# Check for duplicate game data
print("Checking for duplicate game data...")

# How many times does each game_pk appear
game_pk_counts = statcast.groupby("game_pk").size().reset_index(name="row_count")
print(f"Total unique game_pks: {statcast['game_pk'].nunique(): }")
print(f"Total rows: {len(statcast): }")
print(f"Expected rows if no dupes: {game_pk_counts['row_count'].sum(): }")

# Check distribution of row counts per game
print(f"\nRow count distribution per game_pk:")
print(game_pk_counts["row_count"].describe())

# Find games that appear to be duplicated
# A normal 9-inning game has 250-350 pitches
# If a game has 500-700+ rows it's likely duplicated
suspicious_games = game_pk_counts[game_pk_counts["row_count"] > 500]
print(f"\nGames with >500 rows (likely duplicated): {len(suspicious_games): }")
print(suspicious_games.sort_values("row_count", ascending=False).head(10))

# Check if duplicates are exact row duplicates
exact_dupes = statcast.duplicated().sum()
print(f"\nExact duplicate rows: {exact_dupes: }")

# Check duplicates on pitch-level keys
pitch_dupes = statcast.duplicated(
    subset=["game_pk", "at_bat_number", "pitch_number"]
).sum()

print(f"Duplicate pitch keys (game_pk + at_bat + pitch_number): {pitch_dupes: }")

# Remove exact duplicates first
statcast = statcast.drop_duplicates().reset_index(drop=True)
print(f"After removing exact dupes: {len(statcast):,} rows")

# Then remove pitch-level key duplicates keeping first occurrence
statcast = statcast.drop_duplicates(
    subset=["game_pk", "at_bat_number", "pitch_number"],
    keep="first"
).reset_index(drop=True)
print(f"After removing pitch key dupes: {len(statcast):,} rows")

# Verify
print(f"Unique game_pks after cleaning: {statcast['game_pk'].nunique():,}")
print(f"Avg pitches per game: {len(statcast) / statcast['game_pk'].nunique():.1f}")

# Resave the clean parquet
statcast.to_parquet("~/Downloads/statcast_data.parquet", index=False)
print("Resaved clean parquet")


# Suspicious inning out counts checks

# Look at the 4-out innings in detail
four_out_innings = inning_outs[inning_outs["outs"] == 4]
print(f"4-out innings: {len(four_out_innings):,}")

# Check what events are in these innings
bad_games = four_out_innings["game_pk"].unique()

sample_game = bad_games[0]
sample_inning = four_out_innings[
    four_out_innings["game_pk"] == sample_game
]["inning"].iloc[0]
sample_half = four_out_innings[
    (four_out_innings["game_pk"] == sample_game) &
    (four_out_innings["inning"] == sample_inning)
]["inning_topbot"].iloc[0]

print(f"\nSample 4-out half inning: game {sample_game}, inning {sample_inning} {sample_half}")
pa_endings[
    (pa_endings["game_pk"] == sample_game) &
    (pa_endings["inning"] == sample_inning) &
    (pa_endings["inning_topbot"] == sample_half)
][["at_bat_number", "events", "outs_on_play", "des"]].to_string()

print(f"Games flagged as suspicious (5+ out innings): {len(suspicious_inning_games):,}")

# How many unique games have 5+ out innings
five_plus_games = inning_outs[inning_outs["outs"] >= 5]["game_pk"].unique()
print(f"Unique games with 5+ out innings: {len(five_plus_games):,}")

# How many unique games have exactly 5 out innings  
five_out_games = inning_outs[inning_outs["outs"] == 5]["game_pk"].unique()
print(f"Unique games with exactly 5 out innings: {len(five_out_games):,}")

# What are the other out values >= 5?
print(f"\nAll out values >= 5:")
print(inning_outs[inning_outs["outs"] >= 5]["outs"].value_counts().sort_index())

# Sample of suspicious innings
print(f"\nSample of 5+ out innings:")
print(inning_outs[inning_outs["outs"] >= 5].head(20).to_string())

# Show ALL unique out values
print("All unique out values:")
print(sorted(inning_outs["outs"].unique()))

# Check what the suspicious flag is actually catching
print("Suspicious flag breakdown:")
print(inning_outs.groupby(["suspicious_out_count", "outs"]).size().reset_index(name="count"))

# What out values are being flagged as suspicious?
print("\nOut values flagged as suspicious:")
print(inning_outs[inning_outs["suspicious_out_count"]]["outs"].value_counts().sort_index())

# How many unique games per out value are flagged?
print("\nUnique games flagged by out value:")
print(
    inning_outs[inning_outs["suspicious_out_count"]]
    .groupby("outs")["game_pk"]
    .nunique()
    .sort_index()
)

# How many rows have is_final_half = True?
print(f"is_final_half True: {inning_outs['is_final_half'].sum():,}")
print(f"is_final_half False: {(~inning_outs['is_final_half']).sum():,}")

# Of the flagged 0-out innings, are any is_final_half = True?
print("\nFlagged 0-out innings breakdown:")
print(
    inning_outs[
        (inning_outs["suspicious_out_count"]) & 
        (inning_outs["outs"] == 0)
    ][["game_pk", "inning", "inning_topbot", "outs", "is_final_half", "max_inning"]]
    .head(10)
    .to_string()
)

# Check for NaN max_inning after merge
print(f"\nNaN max_inning: {inning_outs['max_inning'].isna().sum():,}")