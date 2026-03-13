import pandas as pd
from pathlib import Path
from tqdm import tqdm

TS_DIR   = Path("/Volumes/Extreme SSD/STA 141B Final/SCRIPTS/fire_timeseries")
SAVE_DIR = Path("/Volumes/Extreme SSD/STA 141B Final/testing")
SAVE_DIR.mkdir(exist_ok=True)

# Load metadata once
meta = pd.read_csv(TS_DIR / "metadata.csv")
meta_lookup = meta.set_index("fire_id")   # row lookup by fire_id

all_fires = []

for csv_path in tqdm(sorted(TS_DIR.glob("*.csv")), desc="Combining fires"):
    fire_id = csv_path.stem

    # Skip the metadata file itself
    if fire_id == "metadata":
        continue

    # Skip fires not in metadata 
    if fire_id not in meta_lookup.index:
        continue

    # Load time series
    ts = pd.read_csv(csv_path, parse_dates=["date"])
    ts["fire_id"] = fire_id

    # Broadcast metadata columns across all daily rows
    fire_meta = meta_lookup.loc[fire_id]
    for col in meta.columns:
        if col != "fire_id":
            ts[col] = fire_meta[col]

    all_fires.append(ts)

# Combine into one DataFrame
print("Concatenating all fires...")
full = pd.concat(all_fires, ignore_index=True)

# Temperature conversions (to celsuis) 
full["tmmx_c"] = full["tmmx"] - 273.15
full["tmmn_c"] = full["tmmn"] - 273.15

print(f"\nFinal shape: {full.shape}")
print(f"Total fires: {full['fire_id'].nunique()}")
print(f"Date range:  {full['date'].min()} → {full['date'].max()}")
print(f"Columns:     {full.columns.tolist()}")

# Reorder so fire_id is the first column
cols = ["fire_id"] + [c for c in full.columns if c != "fire_id"]
full = full[cols]

# Save
full.to_parquet(SAVE_DIR / "all_fires_full.parquet", index=False)


