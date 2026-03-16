
---

## Dataset

The full dataset is too large for GitHub. Download it here:
[Full Dataset Link](https://drive.google.com/file/d/1NIT3e0gDZkYlZrSu-15pCJPMnhei8_Zi/view?usp=sharing)

Once downloaded you will have:
- `metadata.csv` — one row per fire (7,662 fires), static attributes
- `all_fires_full.parquet` — all fires combined, one row per day (~3M rows)

See the [Data Dictionary](https://github.com/maliyer294/STA-141B---Final-project/blob/main/Data%20Processing/WILDFIRE%20Data%20Dictionary%20(1).pdf) for a full description of every column.

---

## Pipeline Overview

1. **Download raw data** from gridMET and NLCD— scripts in `Scripts for downloading data/`
2. **Build time series** — run `build_fire_timeseries_fast.py` to produce one CSV per fire
3. **Combine into full dataset** using `combine_all_csv.py`, which produces `all_fires_full.parquet`
4. **Analyze + model** — see the Jupyter notebooks

---

