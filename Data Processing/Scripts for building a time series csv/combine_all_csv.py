import pandas as pd
from pathlib import Path
from tqdm import tqdm

TS_DIR   = Path("/Volumes/Extreme SSD/STA 141B Final/SCRIPTS/fire_timeseries")
SAVE_DIR = Path("/Volumes/Extreme SSD/STA 141B Final/testing")
SAVE_DIR.mkdir(exist_ok=True)

# Load metadata once
meta = pd.read_csv(TS_DIR / "metadata.csv")
meta_lookup = meta.set_index("fire_id")

all_fires = []

for csv_path in tqdm(sorted(TS_DIR.glob("*.csv")), desc="Combining fires"):
    fire_id = csv_path.stem

    if fire_id == "metadata":
        continue
    if fire_id not in meta_lookup.index:
        continue

    ts = pd.read_csv(csv_path, parse_dates=["date"])
    ts["fire_id"] = fire_id

    fire_meta = meta_lookup.loc[fire_id]
    for col in meta.columns:
        if col != "fire_id":
            ts[col] = fire_meta[col]

    all_fires.append(ts)

print("Concatenating all fires...")
full = pd.concat(all_fires, ignore_index=True)

# Temperature conversions
full["tmmx_c"] = full["tmmx"] - 273.15
full["tmmn_c"] = full["tmmn"] - 273.15

# Put fire_id first
cols = ["fire_id"] + [c for c in full.columns if c != "fire_id"]
full = full[cols]

print(f"\nFinal shape: {full.shape}")
print(f"Total fires: {full['fire_id'].nunique()}")
print(f"Date range:  {full['date'].min()} → {full['date'].max()}")

# Save
full.to_parquet(SAVE_DIR / "all_fires_full.parquet", index=False)
full.to_csv(SAVE_DIR / "all_fires_full.csv", index=False)
print("Done.")
