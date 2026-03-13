#!/usr/bin/env python3
"""
California DEM Downloader — Tiled Approach (Memory-Safe)
=========================================================

Downloads a Digital Elevation Model for California in small tiles
to avoid running out of memory, then mosaics them into a single file.

Also derives slope and aspect rasters from the final DEM.

Output directory:
    /Volumes/Extreme SSD/STA 141B Final/Topography/
"""

import os
import sys
import time
import traceback

try:
    import py3dep
    import numpy as np
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.crs import CRS
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("\nInstall everything with:")
    print("    pip3 install py3dep rioxarray rasterio numpy geopandas pyproj shapely")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR    = "/Volumes/Extreme SSD/STA 141B Final/Topography"
RESOLUTION    = 30           # meters — 30m
TARGET_CRS    = "EPSG:3310"  

# California is split into a 3x5 grid of tiles (15 tiles total).
# Format: (xmin, ymin, xmax, ymax) in WGS84
CA_TILES = [
    # Row 1 — Southern CA
    (-124.55, 32.45, -120.85, 34.45),
    (-120.85, 32.45, -117.15, 34.45),
    (-117.15, 32.45, -114.10, 34.45),
    # Row 2 — Central CA
    (-124.55, 34.45, -120.85, 36.45),
    (-120.85, 34.45, -117.15, 36.45),
    (-117.15, 34.45, -114.10, 36.45),
    # Row 3 — Central-North CA
    (-124.55, 36.45, -120.85, 38.45),
    (-120.85, 36.45, -117.15, 38.45),
    (-117.15, 36.45, -114.10, 38.45),
    # Row 4 — Northern CA
    (-124.55, 38.45, -120.85, 40.25),
    (-120.85, 38.45, -117.15, 40.25),
    (-117.15, 38.45, -114.10, 40.25),
    # Row 5 — Far Northern CA
    (-124.55, 40.25, -120.85, 42.05),
    (-120.85, 40.25, -117.15, 42.05),
    (-117.15, 40.25, -114.10, 42.05),
]

# Retry settings 
MAX_RETRIES   = 5
RETRY_WAIT    = 15   # seconds between retries


# =============================================================================
# STEP 1: DOWNLOAD TILES
# =============================================================================

def download_tiles(tiles_dir: str) -> list:
    """Download each tile and save to disk. Returns list of saved tile paths."""
    os.makedirs(tiles_dir, exist_ok=True)
    saved_paths = []
    total = len(CA_TILES)

    for i, bbox in enumerate(CA_TILES, 1):
        tile_name = f"tile_{i:02d}_{bbox[0]:.2f}_{bbox[1]:.2f}.tif"
        tile_path = os.path.join(tiles_dir, tile_name)

        if os.path.exists(tile_path) and os.path.getsize(tile_path) > 1024 * 1024:
            size_mb = os.path.getsize(tile_path) / (1024 * 1024)
            print(f"  [{i:>2}/{total}] Tile {i} already exists ({size_mb:.0f} MB) — skipping")
            saved_paths.append(tile_path)
            continue

        print(f"  [{i:>2}/{total}] Downloading tile {i}: {bbox} ...")

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                dem = py3dep.get_dem(bbox, resolution=RESOLUTION)
                dem.rio.to_raster(
                    tile_path,
                    driver="GTiff",
                    compress="lzw",
                    tiled=True,
                    blockxsize=256,
                    blockysize=256,
                )
                size_mb = os.path.getsize(tile_path) / (1024 * 1024)
                print(f"         Saved ({size_mb:.0f} MB)")
                saved_paths.append(tile_path)

                del dem
                break

            except Exception as e:
                if attempt < MAX_RETRIES:
                    print(f"         Attempt {attempt} failed: {e}")
                    print(f"         Retrying in {RETRY_WAIT}s ...")
                    time.sleep(RETRY_WAIT)
                else:
                    print(f"         FAILED after {MAX_RETRIES} attempts: {e}")
                    traceback.print_exc()

        time.sleep(1)  # add a pause between tiles

    return saved_paths


# =============================================================================
# STEP 2: MOSAIC TILES
# =============================================================================

def mosaic_tiles(tile_paths: list, output_path: str):
    """Merge all tiles into a single statewide GeoTIFF."""
    print(f"\n  Mosaicking {len(tile_paths)} tiles ...")

    datasets = [rasterio.open(p) for p in tile_paths]
    mosaic, out_transform = merge(datasets)
    out_meta = datasets[0].meta.copy()

    for ds in datasets:
        ds.close()

    out_meta.update({
        "driver":    "GTiff",
        "height":    mosaic.shape[1],
        "width":     mosaic.shape[2],
        "transform": out_transform,
        "compress":  "lzw",
        "tiled":     True,
        "blockxsize": 256,
        "blockysize": 256,
    })

    with rasterio.open(output_path, "w", **out_meta) as dst:
        dst.write(mosaic)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Mosaic saved: {output_path} ({size_mb:.0f} MB)")


# =============================================================================
# STEP 3: REPROJECT TO EPSG:3310
# =============================================================================

def reproject_raster(src_path: str, dst_path: str, resampling=Resampling.bilinear):
    """Reproject a GeoTIFF to TARGET_CRS (EPSG:3310)."""
    print(f"\n  Reprojecting to {TARGET_CRS} ...")
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, TARGET_CRS, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(
            crs=TARGET_CRS,
            transform=transform,
            width=width,
            height=height,
            compress="lzw",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )
        with rasterio.open(dst_path, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=TARGET_CRS,
                resampling=resampling,
            )
    size_mb = os.path.getsize(dst_path) / (1024 * 1024)
    print(f"  Reprojected: {dst_path} ({size_mb:.0f} MB)")


# =============================================================================
# STEP 4: DERIVE SLOPE AND ASPECT
# =============================================================================

def derive_slope_aspect(dem_path: str, slope_path: str, aspect_path: str):
    """Compute slope (degrees) and aspect (degrees) from a projected DEM."""
    print("\n  Computing slope and aspect ...")
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        nodata = src.nodata
        transform = src.transform
        profile = src.profile.copy()
        profile.update(dtype="float32", compress="lzw")

        cell_x = abs(transform.a)
        cell_y = abs(transform.e)

        if nodata is not None:
            dem[dem == nodata] = np.nan

        dz_dy, dz_dx = np.gradient(dem, cell_y, cell_x)

        # Slope in degrees
        slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))

        # Aspect in degrees (0=North, clockwise)
        aspect = np.degrees(np.arctan2(-dz_dx, dz_dy))
        aspect = (aspect + 360) % 360

        with rasterio.open(slope_path, "w", **profile) as dst:
            dst.write(slope.astype(np.float32), 1)
        size_mb = os.path.getsize(slope_path) / (1024 * 1024)
        print(f"  Slope saved:  {slope_path} ({size_mb:.0f} MB)")

        with rasterio.open(aspect_path, "w", **profile) as dst:
            dst.write(aspect.astype(np.float32), 1)
        size_mb = os.path.getsize(aspect_path) / (1024 * 1024)
        print(f"  Aspect saved: {aspect_path} ({size_mb:.0f} MB)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  California DEM Downloader — Tiled / Memory-Safe")
    print("=" * 60)
    print(f"  Resolution : {RESOLUTION} m")
    print(f"  Tiles      : {len(CA_TILES)} (3×5 grid over California)")
    print(f"  Target CRS : {TARGET_CRS}")
    print(f"  Output dir : {OUTPUT_DIR}")
    print()

    # Verify SSD
    if not os.path.isdir("/Volumes/Extreme SSD"):
        print("ERROR: Extreme SSD not found. Make sure it's connected.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tiles_dir     = os.path.join(OUTPUT_DIR, "tiles")
    mosaic_raw    = os.path.join(OUTPUT_DIR, f"california_dem_{RESOLUTION}m_wgs84.tif")
    mosaic_proj   = os.path.join(OUTPUT_DIR, f"california_dem_{RESOLUTION}m_epsg3310.tif")
    slope_path    = os.path.join(OUTPUT_DIR, f"california_slope_{RESOLUTION}m_epsg3310.tif")
    aspect_path   = os.path.join(OUTPUT_DIR, f"california_aspect_{RESOLUTION}m_epsg3310.tif")

    start = time.time()

    # Step 1: Download tiles
    print("Step 1 of 4: Downloading tiles ...")
    tile_paths = download_tiles(tiles_dir)

    if not tile_paths:
        print("ERROR: No tiles were downloaded successfully.")
        sys.exit(1)

    print(f"\n  {len(tile_paths)}/{len(CA_TILES)} tiles downloaded successfully.")

    # Step 2: Mosaic
    if os.path.exists(mosaic_raw):
        print(f"\nStep 2 of 4: Mosaic already exists — skipping.")
    else:
        print("\nStep 2 of 4: Mosaicking tiles into statewide raster ...")
        mosaic_tiles(tile_paths, mosaic_raw)

    # Step 3: Reproject
    if os.path.exists(mosaic_proj):
        print(f"\nStep 3 of 4: Projected DEM already exists — skipping.")
    else:
        print(f"\nStep 3 of 4: Reprojecting to EPSG:3310 ...")
        reproject_raster(mosaic_raw, mosaic_proj)

    # Step 4: Slope and aspect
    if os.path.exists(slope_path) and os.path.exists(aspect_path):
        print(f"\nStep 4 of 4: Slope/aspect already exist — skipping.")
    else:
        print(f"\nStep 4 of 4: Deriving slope and aspect ...")
        derive_slope_aspect(mosaic_proj, slope_path, aspect_path)

    # Summary
    elapsed = time.time() - start
    print()
    print("=" * 60)
    print("  COMPLETE")
    print("=" * 60)
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print()
    for path, label in [
        (mosaic_proj, "Elevation (m) — use in QGIS for zonal stats"),
        (slope_path,  "Slope (degrees) — model covariate"),
        (aspect_path, "Aspect (degrees) — model covariate"),
    ]:
        if os.path.exists(path):
            mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {mb:>6.0f} MB  {os.path.basename(path)}")
            print(f"           {label}")
    print()
    print("  Tiles folder (can delete after confirming mosaic looks good):")
    print(f"    {tiles_dir}")
    print()


if __name__ == "__main__":
    main()