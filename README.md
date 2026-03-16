
# California Wildfire Size Prediction — STA 141B Final Project

Predicting final wildfire size (acres) across California (2001–2024) using 
pre-fire weather conditions, topography, and vegetation. Built on CAL FIRE 
historical perimeters linked to daily GridMET weather, USGS elevation data, 
and NLCD land cover.

---
## Data Sources

| Source | Variables | Resolution |
|---|---|---|
| [CAL FIRE](https://www.fire.ca.gov/) | Fire perimeters, ignition/containment dates, cause codes | Vector polygon |
| [GridMET](https://www.climatologylab.org/gridmet.html) | Temperature, VPD, wind speed, precipitation, relative humidity, fuel moisture, fire danger indices | ~4 km daily |
| [USGS NLCD](https://www.mrlc.gov/) | Land cover / vegetation fractions (15 classes) | 30 m annual |
| [USGS NED](https://www.usgs.gov/programs/national-geospatial-program/national-map) | Elevation, slope | 10 m |

---

## Dataset

The full dataset is too large for GitHub. Download it here:
**[Full Dataset — Google Drive link](https://drive.google.com/file/d/1NIT3e0gDZkYlZrSu-15pCJPMnhei8_Zi/view?usp=sharing)** 

The dataset contains **7,662 California wildfires from 2001–2024**.

| Key Files                                                                                                    | Description                                                  |
| ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------ |
| [all_fires_full.parquet](https://drive.google.com/file/d/1NIT3e0gDZkYlZrSu-15pCJPMnhei8_Zi/view?usp=sharing) | All fires combined — one row per day (~3M rows, recommended) |
| [all_fires_full.csv](https://drive.google.com/file/d/1NIT3e0gDZkYlZrSu-15pCJPMnhei8_Zi/view?usp=sharing)     | Same as above in CSV format (~1.5 GB)                        |
| [metadata.csv](https://drive.google.com/file/d/1MviuW0CT_GHBzK1wInWNer-Jub9Pb9Rb/view?usp=sharing)           | One row per fire — static attributes only (29 columns)       |

See the [Data Dictionary](https://github.com/maliyer294/STA-141B---Final-project/blob/main/Data%20Processing/WILDFIRE%20Data%20Dictionary%20(1).pdf) in `Data Processing/` for a full 
description of every variable.

**Key columns:**
- `fire_id` — unique identifier, links metadata to time series (e.g. `DIXIE_2021_01104`)
- `days_to_ignition` — time axis: negative = pre-fire, 0 = ignition day, positive = active fire
- `gis_acres` — **dependent variable**: final fire size in acres

## Pipeline

To reproduce the dataset from raw data:

### Step 1 — Download raw data

**Automated:**
```bash
python "Data Processing/Scripts for downloading data/download_gridMET.py"
python "Data Processing/Scripts for downloading data/download_DEMS.py"
```

**Manual downloads:**
- [NLCD Land Cover](https://www.mrlc.gov/viewer/) — download annual files for 1995–2024
- [CAL FIRE Perimeters](https://www.fire.ca.gov/what-we-do/fire-resource-assessment-program) — download historical fire perimeters GeoPackage

### Step 2 — Build individual fire time series
```bash
python "Data Processing/Scripts for building a time series csv/build_fire_timeseries_fast.py" --lag 365
```

### Step 3 — Combine into one file
```bash
python "Data Processing/Scripts for building a time series csv/combine_all_csv.py"
```
