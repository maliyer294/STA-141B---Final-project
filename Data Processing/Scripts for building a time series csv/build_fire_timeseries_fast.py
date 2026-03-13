#!/usr/bin/env python3
"""
build_fire_timeseries_fast.py

Why this is faster than my previous attempts
---------------------
1. GridMET — instead of reading from 2 GB national NetCDF files on every fire,
   this script loads a California-clipped slice (~250×270 cells) for one
   variable at a time into a numpy array.

2. NLCD — instead of windowing the 105,000×160,000-pixel national file on
   every fire, this script writes a CA-only GeoTIFF to SCRIPTS/cache/ once
   at startup, then reads from that smaller (~46k×41k) file.

Output (identical to build_fire_timeseries.py)
------
SCRIPTS/fire_timeseries/
    metadata.csv            one row per fire  (topo + vegetation + fire info)
    {fire_id}.csv           one row per day   (weather time series)


"""

import argparse
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge as rio_merge
from rasterio.warp import transform_bounds, transform_geom
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.windows import from_bounds as window_from_bounds
from shapely.geometry import box, mapping
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path("/Volumes/Extreme SSD/STA 141B Final")

PERIMETERS_PATH = BASE / "Fire Perimeters/California_Historic_Fire_Perimeters_586217350401785615.gpkg"
TOPO_DIR        = BASE / "Topography/tiles"
VEG_DIRS        = [
    BASE / "Vegetation/Annual_NLCD_LndCov_1995-2004_CU_C1V1",
    BASE / "Vegetation/Annual_NLCD_LndCov_2005-2014_CU_C1V1",
    BASE / "Vegetation/Annual_NLCD_LndCov_2015-2024_CU_C1V1",
]
GRIDMET_DIR = BASE / "GridMET"
OUT_DIR     = BASE / "SCRIPTS/fire_timeseries"
CACHE_DIR   = BASE / "SCRIPTS/cache"

CRS_4326 = CRS.from_epsg(4326)

# California bounding box in EPSG:4326 (with 0.5° buffer around all fires)
CA_LAT_MIN, CA_LAT_MAX = 32.0,  43.0
CA_LON_MIN, CA_LON_MAX = -125.0, -113.5

# ── GridMET variable registry ──────────────────────────────────────────────────
GRIDMET_VARS = {
    "bi":     ("Burning Index",              "burning_index_g"),
    "erc":    ("Energy Release Component",   "energy_release_component-g"),
    "fm100":  ("Fuel Moisture (100 hours)",  "dead_fuel_moisture_100hr"),
    "fm1000": ("Fuel Moisture (1000 hours)", "dead_fuel_moisture_1000hr"),
    "tmmx":   ("Max Air Temperature",        "air_temperature"),
    "tmmn":   ("Min Air Temperature",        "air_temperature"),
    "rmax":   ("Max Humidity",               "relative_humidity"),
    "rmin":   ("Min Humidity",               "relative_humidity"),
    "pr":     ("Precipitation",              "precipitation_amount"),
    "vpd":    ("Vapor Pressure Deficit",     "mean_vapor_pressure_deficit"),
    "vs":     ("Wind Speed",                 "wind_speed"),
}

NLCD_CLASSES = {
    11: "open_water",   21: "dev_open",      22: "dev_low",    23: "dev_med",
    24: "dev_high",     31: "barren",        41: "forest_decid", 42: "forest_ever",
    43: "forest_mixed", 52: "shrub",         71: "grassland",  81: "pasture",
    82: "crops",        90: "wetland_woody", 95: "wetland_herb",
}


# ── OPTIMIZATION 1: GridMET CA pre-clip ────────────────────────────────────────

class GridMetCASlice:
    """
    Holds one year × one variable of GridMET, clipped to California.
    Loaded as float32 numpy array for fast per-fire extraction.
    """
    def __init__(self, values: np.ndarray, lat: np.ndarray,
                 lon: np.ndarray, days: pd.DatetimeIndex):
        # values shape: (n_days, n_lat, n_lon)  — float32
        self.values = values
        self.lat    = lat    # decreasing (N→S)
        self.lon    = lon    # increasing (W→E)
        self.days   = days

    def extract_series(
        self, minx: float, miny: float, maxx: float, maxy: float,
        start: pd.Timestamp, end: pd.Timestamp, pad: float = 0.05,
    ) -> pd.Series:
        """
        Return a daily pd.Series of spatially-averaged values for the
        bounding box [minx,miny,maxx,maxy] and date range [start, end].
        Returns empty Series if no spatial overlap or no days in range.
        """
        lat_mask  = (self.lat >= miny - pad) & (self.lat <= maxy + pad)
        lon_mask  = (self.lon >= minx - pad) & (self.lon <= maxx + pad)
        day_mask  = (self.days >= start) & (self.days <= end)

        if not lat_mask.any() or not lon_mask.any() or not day_mask.any():
            return pd.Series(dtype="float32")

        lat_idx = np.where(lat_mask)[0]
        lon_idx = np.where(lon_mask)[0]
        day_idx = np.where(day_mask)[0]

        # Slice — all numpy, no disk I/O
        subset = self.values[
            np.ix_(day_idx,
                   np.arange(lat_idx[0], lat_idx[-1] + 1),
                   np.arange(lon_idx[0], lon_idx[-1] + 1))
        ]
        daily_means = subset.mean(axis=(1, 2))  # shape: (n_days,)

        return pd.Series(
            daily_means,
            index=pd.DatetimeIndex(self.days[day_idx]),
        )


def load_ca_gridmet(prefix: str, year: int) -> "GridMetCASlice | None":
    """
    Load the California spatial slice of one GridMET variable/year.
    Returns None if the file doesn't exist.
    """
    folder, var_name = GRIDMET_VARS[prefix]
    nc_path = GRIDMET_DIR / folder / f"{prefix}_{year}.nc"
    if not nc_path.exists():
        return None

    ds = xr.open_dataset(nc_path)
    da = ds[var_name]

    # Subset to CA (lat runs N→S so slice is high→low)
    ca_slice = da.sel(
        lat=slice(CA_LAT_MAX, CA_LAT_MIN),
        lon=slice(CA_LON_MIN, CA_LON_MAX),
    )

    # Load into memory as float32 (halves RAM vs float64)
    values = ca_slice.values.astype("float32")          # (days, lat, lon)
    lat    = ca_slice.lat.values.astype("float32")
    lon    = ca_slice.lon.values.astype("float32")
    days   = pd.DatetimeIndex(ca_slice.day.values)

    ds.close()
    return GridMetCASlice(values, lat, lon, days)


# ── OPTIMIZATION 2: NLCD CA cache ─────────────────────────────────────────────

def build_nlcd_index() -> dict[int, Path]:
    index: dict[int, Path] = {}
    for d in VEG_DIRS:
        for f in d.glob("*.tif"):
            if f.name.startswith("._"):
                continue
            for part in f.stem.split("_"):
                if part.isdigit() and len(part) == 4:
                    index[int(part)] = f
                    break
    return index


def create_nlcd_ca_cache(
    nlcd_index: dict[int, Path],
    cache_dir: Path,
    years_needed: list[int],
) -> dict[int, Path]:
    """
    For each required year, write a California-clipped GeoTIFF to cache_dir
    (skipping years already cached). Returns {year: cached_path}.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached: dict[int, Path] = {}

    # Get CA bounds in NLCD CRS from any available NLCD file
    sample_year  = min(nlcd_index.keys(), key=lambda y: abs(y - 2020))
    sample_path  = nlcd_index[sample_year]
    with rasterio.open(sample_path) as src:
        nlcd_crs = src.crs
        ca_bounds_nlcd = transform_bounds(
            CRS_4326, nlcd_crs,
            CA_LON_MIN, CA_LAT_MIN, CA_LON_MAX, CA_LAT_MAX,
        )

    for year in tqdm(years_needed, desc="Creating CA NLCD cache"):
        out_path = cache_dir / f"nlcd_ca_{year}.tif"
        if out_path.exists():
            cached[year] = out_path
            continue

        # Nearest available NLCD year
        src_year = min(nlcd_index, key=lambda y: abs(y - year))
        src_path = nlcd_index[src_year]

        with rasterio.open(src_path) as src:
            win    = window_from_bounds(*ca_bounds_nlcd, src.transform)
            data   = src.read(1, window=win)
            tf     = src.window_transform(win)
            meta   = src.meta.copy()

        meta.update({
            "height": data.shape[0],
            "width":  data.shape[1],
            "transform": tf,
            "compress": "lzw",
            "predictor": 2,
        })
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(data, 1)

        cached[year] = out_path

    return cached


def extract_vegetation_cached(
    fire_geom_4326, year: int, nlcd_cache: dict[int, Path]
) -> dict:
    """
    Extract NLCD land-cover % from the cached CA-only GeoTIFF.

    Uses a bounding-box window read instead of polygon masking.
    Polygon rasterization at 30m resolution was the bottleneck (1+ s/fire);
    bbox windowing is ~50ms for typical fires and ~400ms for the largest.
    The small accuracy cost (including pixels at the polygon edges) is
    negligible for vegetation characterisation.
    """
    result = {f"nlcd_{name}": 0.0 for name in NLCD_CLASSES.values()}
    if not nlcd_cache:
        return result

    cache_year = min(nlcd_cache, key=lambda y: abs(y - year))
    try:
        with rasterio.open(nlcd_cache[cache_year]) as src:
            nodata_val = src.nodata
            # Project fire bbox → NLCD CRS, then read that rectangular window
            geom_proj  = transform_geom(CRS_4326, src.crs, mapping(fire_geom_4326))
            from shapely.geometry import shape as sh_shape
            bounds_nlcd = sh_shape(geom_proj).bounds
            win        = window_from_bounds(*bounds_nlcd, src.transform)
            data       = src.read(1, window=win)

        valid = data[(data != nodata_val) & (data > 0) & (data < 250)]
        if len(valid) == 0:
            return result
        for code, name in NLCD_CLASSES.items():
            result[f"nlcd_{name}"] = float(np.sum(valid == code) / len(valid))
    except Exception:
        pass
    return result


# ── Topography ─────────────────────────────────────────────────────────────────
# Pre-open all DEM tiles once and keep file handles alive for the run.
# Use bounding-box window reads (no polygon rasterization) — same fix as NLCD.

class TopoCache:
    """Holds open rasterio file handles for all DEM tiles."""
    def __init__(self, topo_dir: Path):
        self.tiles: list[tuple[rasterio.DatasetReader, tuple]] = []
        for f in sorted(topo_dir.glob("*.tif")):
            src   = rasterio.open(f)
            b4326 = transform_bounds(src.crs, CRS_4326, *src.bounds)
            self.tiles.append((src, b4326))

    def close(self):
        for src, _ in self.tiles:
            src.close()


def build_topo_index() -> TopoCache:
    return TopoCache(TOPO_DIR)


def extract_topo(fire_geom_4326, topo_cache: TopoCache) -> dict:
    """
    Compute elevation mean/std and slope mean within the fire bounding box.
    Reads each intersecting DEM tile independently (no merging) using a
    bbox window — avoids both polygon rasterisation and in-memory tile merges.
    """
    empty    = dict(elev_mean=np.nan, elev_std=np.nan, slope_mean=np.nan)
    fire_box = box(*fire_geom_4326.bounds)

    all_elev:  list[np.ndarray] = []
    all_slope: list[np.ndarray] = []

    for src, b4326 in topo_cache.tiles:
        if not fire_box.intersects(box(*b4326)):
            continue
        try:
            # Transform fire bbox → tile CRS, then do a rectangular window read
            bounds_tile = transform_bounds(CRS_4326, src.crs, *fire_geom_4326.bounds)
            win  = window_from_bounds(*bounds_tile, src.transform)
            if win.width < 1 or win.height < 1:
                continue
            data = src.read(1, window=win).astype(float)
            nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = np.nan

            valid = data[np.isfinite(data)]
            if len(valid) == 0:
                continue
            all_elev.append(valid.ravel())

            dy, dx = np.gradient(data)
            slope  = np.sqrt(dx**2 + dy**2)
            all_slope.append(slope[np.isfinite(slope)].ravel())
        except Exception:
            continue

    if not all_elev:
        return empty

    combined_elev  = np.concatenate(all_elev)
    combined_slope = np.concatenate(all_slope) if all_slope else np.array([np.nan])
    return dict(
        elev_mean=float(np.nanmean(combined_elev)),
        elev_std=float(np.nanstd(combined_elev)),
        slope_mean=float(np.nanmean(combined_slope)),
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_fire_id(fire_name: str, year: int, idx: int) -> str:
    clean = re.sub(r"[^A-Za-z0-9]+", "_", str(fire_name)).strip("_").upper()
    return f"{clean}_{year}_{idx:05d}"


# ── Main ───────────────────────────────────────────────────────────────────────

def main(lag_days: int = 90, min_acres: float = 1.0):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load fires ─────────────────────────────────────────────────────────────
    print("Loading fire perimeters …")
    gdf = gpd.read_file(PERIMETERS_PATH)
    gdf = gdf[
        (gdf["STATE"] == "CA")
        & (gdf["YEAR_"] >= 2001) & (gdf["YEAR_"] <= 2024)
        & gdf["ALARM_DATE"].notna()
        & (gdf["GIS_ACRES"] >= min_acres)
    ].copy()

    gdf["ALARM_DATE"]    = pd.to_datetime(gdf["ALARM_DATE"], utc=True).dt.tz_localize(None)
    gdf["CONT_DATE"]     = pd.to_datetime(gdf["CONT_DATE"],  utc=True, errors="coerce").dt.tz_localize(None)
    gdf["CONT_DATE"]     = gdf["CONT_DATE"].fillna(gdf["ALARM_DATE"] + pd.Timedelta(days=1))
    gdf["duration_days"] = (gdf["CONT_DATE"] - gdf["ALARM_DATE"]).dt.days.clip(lower=1)
    gdf                  = gdf.to_crs(CRS_4326).reset_index(drop=True)
    print(f"  {len(gdf)} fires to process")

    # ── Build static indexes ───────────────────────────────────────────────────
    print("Building topography index (pre-opening tiles) …")
    topo_cache = build_topo_index()

    print("Building NLCD index …")
    nlcd_index   = build_nlcd_index()
    years_needed = sorted({int(r["YEAR_"]) for _, r in gdf.iterrows()})

    print(f"Creating CA-clipped NLCD cache (one-time, ~30 s) …")
    nlcd_cache = create_nlcd_ca_cache(nlcd_index, CACHE_DIR, years_needed)

    # ── STEP 1: Extract GridMET time series (variable-by-variable) ─────────────
    # per_fire_series[fire_id][prefix] = pd.Series indexed by date
    print("\nExtracting GridMET time series …")
    per_fire_series: dict[str, dict[str, pd.Series]] = defaultdict(dict)

    # Pre-build fire metadata 
    fire_meta = {}
    for idx, row in gdf.iterrows():
        fire_id = make_fire_id(row["FIRE_NAME"], int(row["YEAR_"]), idx)
        fire_meta[fire_id] = {
            "row":          row,
            "alarm":        row["ALARM_DATE"],
            "cont":         row["CONT_DATE"],
            "year":         int(row["YEAR_"]),
            "bounds":       row.geometry.bounds,  # (minx,miny,maxx,maxy)
            "window_start": row["ALARM_DATE"] - pd.Timedelta(days=lag_days),
        }

    for prefix in tqdm(GRIDMET_VARS, desc="  Variables", position=0):
        prev_slice: "GridMetCASlice | None" = None
        prev_year: int = -1

        for year in tqdm(years_needed, desc=f"    {prefix}", position=1, leave=False):
            curr_slice = load_ca_gridmet(prefix, year)

            # Fires whose alarm year == this year
            year_fires = {
                fid: m for fid, m in fire_meta.items() if m["year"] == year
            }

            for fire_id, m in year_fires.items():
                start = m["window_start"]
                end   = m["cont"]
                minx, miny, maxx, maxy = m["bounds"]
                parts: list[pd.Series] = []

                # Pre-fire lag may extend into the previous year
                if start.year < year and prev_slice is not None:
                    s = prev_slice.extract_series(
                        minx, miny, maxx, maxy,
                        start, pd.Timestamp(f"{year - 1}-12-31"),
                    )
                    if not s.empty:
                        parts.append(s)

                # Main portion in the alarm year
                if curr_slice is not None:
                    s = curr_slice.extract_series(
                        minx, miny, maxx, maxy,
                        max(start, pd.Timestamp(f"{year}-01-01")), end,
                    )
                    if not s.empty:
                        parts.append(s)

                    # Fire duration may spill into the next year
                    if end.year > year:
                        next_slice = load_ca_gridmet(prefix, year + 1)
                        if next_slice is not None:
                            s = next_slice.extract_series(
                                minx, miny, maxx, maxy,
                                pd.Timestamp(f"{year + 1}-01-01"), end,
                            )
                            if not s.empty:
                                parts.append(s)

                series = pd.concat(parts).sort_index() if parts else pd.Series(dtype="float32")
                per_fire_series[fire_id][prefix] = series

            prev_slice = curr_slice
            prev_year  = year

    # ── STEP 2: Write per-fire CSVs + collect metadata ─────────────────────────
    print("\nWriting per-fire CSVs …")
    metadata_rows = []

    for fire_id, m in tqdm(fire_meta.items(), desc="  Writing CSVs"):
        row   = m["row"]
        alarm = m["alarm"]
        cont  = m["cont"]
        year  = m["year"]
        geom  = row.geometry

        # Assemble weather DataFrame on a complete daily date range
        full_index = pd.date_range(m["window_start"].date(), cont.date(), freq="D")
        weather_df = pd.DataFrame(index=full_index)
        weather_df.index.name = "date"

        for prefix, series in per_fire_series[fire_id].items():
            if not series.empty:
                weather_df[prefix] = series.reindex(full_index)

        weather_df = weather_df.reset_index()
        weather_df["days_to_ignition"] = (
            weather_df["date"] - alarm.normalize()
        ).dt.days
        weather_df["phase"] = "pre_fire"
        weather_df.loc[weather_df["days_to_ignition"] >= 0, "phase"] = "active"
        weather_df.loc[weather_df["date"] > cont.normalize(), "phase"] = "contained"

        col_order = ["date", "days_to_ignition", "phase"] + list(GRIDMET_VARS.keys())
        weather_df = weather_df[[c for c in col_order if c in weather_df.columns]]
        weather_df.to_csv(OUT_DIR / f"{fire_id}.csv", index=False)

        # Static metadata
        meta = {
            "fire_id":       fire_id,
            "fire_name":     row["FIRE_NAME"],
            "year":          year,
            "alarm_date":    alarm.date(),
            "cont_date":     cont.date(),
            "duration_days": int(row["duration_days"]),
            "gis_acres":     float(row["GIS_ACRES"]),
            "cause":         row["CAUSE"],
            "lon_centroid":  float(geom.centroid.x),
            "lat_centroid":  float(geom.centroid.y),
            "lag_days":      lag_days,
        }
        meta.update(extract_topo(geom, topo_cache))
        meta.update(extract_vegetation_cached(geom, year, nlcd_cache))
        metadata_rows.append(meta)

    meta_df = pd.DataFrame(metadata_rows)
    meta_df.to_csv(OUT_DIR / "metadata.csv", index=False)
    topo_cache.close()

    print(f"\nDone.")
    print(f"  {len(meta_df)} fire time-series → {OUT_DIR}/")
    print(f"  metadata.csv with topo + vegetation → {OUT_DIR}/metadata.csv")
    print(f"  NLCD cache reusable for future runs → {CACHE_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lag",       type=int,   default=90,  help="Pre-fire lag window in days (default: 90)")
    parser.add_argument("--min-acres", type=float, default=1.0, help="Minimum fire size in acres (default: 1)")
    args = parser.parse_args()
    main(lag_days=args.lag, min_acres=args.min_acres)
