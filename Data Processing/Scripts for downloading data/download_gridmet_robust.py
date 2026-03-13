#!/usr/bin/env python3
"""
gridMET Robust Batch Downloader
================================
Downloads gridMET NetCDF climate files for 2001–2024 to:
    /Volumes/Extreme SSD/STA 141B Final/GridMET/
       /Volumes/Extreme SSD/STA 141B Final/GridMET/download_log.txt
"""

import os
import sys
import time
import subprocess
import logging
from datetime import datetime, timedelta

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    print("Missing 'requests' library. Install it with:")
    print("    pip3 install requests")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Base output path to external SSD
BASE_DIR = "/Volumes/Extreme SSD/STA 141B Final/GridMET"

# Years to download
YEARS = list(range(2001, 2025))  # 2001 through 2024

# Mapping: folder name on disk -> gridMET variable abbreviation
FOLDER_TO_VARIABLE = {
    "Burning Index":              "bi",
    "Energy Release Component":   "erc",
    "Fuel Moisture (100 hours)":  "fm100",
    "Fuel Moisture (1000 hours)": "fm1000",
    "Max Air Temperature":        "tmmx",
    "Max Humidity":               "rmax",
    "Min Air Temperature":        "tmmn",
    "Min Humidity":               "rmin",
    "Precipitation":              "pr",
    "Vapor Pressure Deficit":     "vpd",
    "Wind Speed":                 "vs",
}

# gridMET base URL
BASE_URL = "http://www.northwestknowledge.net/metdata/data"

# Retry settings
MAX_RETRIES = 10           # Total retry attempts per file
INITIAL_BACKOFF = 5        # Seconds to wait after first failure
MAX_BACKOFF = 300           # Max wait between retries (5 minutes)
REQUEST_TIMEOUT = 120       # Seconds before a request times out
CHUNK_SIZE = 1024 * 1024   # 1 MB download chunks

# Minimum valid file size in bytes (gridMET .nc files are typically 100+ MB)
MIN_VALID_SIZE = 10 * 1024 * 1024  # 10 MB — anything smaller is likely corrupt


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging():
    """Set up logging to both console and a log file on the SSD."""
    log_path = os.path.join(BASE_DIR, "download_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a"),
        ],
    )
    return logging.getLogger(__name__)


# =============================================================================
# HTTP SESSION WITH BUILT-IN RETRIES
# =============================================================================

def create_session() -> requests.Session:
    """
    Create an HTTP session with automatic retries. 
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# =============================================================================
# DOWNLOAD LOGIC
# =============================================================================

def download_file(session, url, output_path, log):
    """
    Download a single file with robust retry logic and exponential backoff.
    """
    backoff = INITIAL_BACKOFF

    for attempt in range(1, MAX_RETRIES + 1):
        tmp_path = output_path + ".partial"

        try:
            # Start the download
            response = session.get(url, stream=True, timeout=REQUEST_TIMEOUT)

            if response.status_code == 404:
                log.warning(f"    File not found (404) — skipping")
                return False

            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
                    downloaded += len(chunk)

            # Validate download
            actual_size = os.path.getsize(tmp_path)

            if total_size > 0 and actual_size < total_size:
                log.warning(
                    f"    Incomplete: got {actual_size:,} of {total_size:,} bytes "
                    f"(attempt {attempt}/{MAX_RETRIES})"
                )
                os.remove(tmp_path)
                raise IOError("Incomplete download")

            if actual_size < MIN_VALID_SIZE:
                log.warning(
                    f"    Suspiciously small: {actual_size:,} bytes "
                    f"(attempt {attempt}/{MAX_RETRIES})"
                )
                os.remove(tmp_path)
                raise IOError("File too small — likely corrupt")

            # Success — rename partial file to final name
            os.rename(tmp_path, output_path)
            return True

        except (requests.exceptions.RequestException, IOError, OSError) as e:
            # Clean up partial file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

            if attempt < MAX_RETRIES:
                log.warning(
                    f"    Attempt {attempt}/{MAX_RETRIES} failed: {e}"
                )
                log.info(f"    Retrying in {backoff}s ...")
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)
            else:
                log.error(
                    f"    FAILED after {MAX_RETRIES} attempts: {e}"
                )
                return False

    return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Verify the SSD is mounted
    if not os.path.isdir(BASE_DIR):
        print(f"ERROR: Cannot find {BASE_DIR}")
        print("Make sure your Extreme SSD is connected and mounted.")
        sys.exit(1)

    # Set up logging
    log = setup_logging()

    log.info("=" * 60)
    log.info("  gridMET Robust Batch Downloader")
    log.info("=" * 60)
    log.info(f"  Output   : {BASE_DIR}")
    log.info(f"  Years    : {YEARS[0]}–{YEARS[-1]} ({len(YEARS)} years)")
    log.info(f"  Variables: {len(FOLDER_TO_VARIABLE)}")
    total_files = len(FOLDER_TO_VARIABLE) * len(YEARS)
    log.info(f"  Total    : {total_files} files")
    log.info("")

    # Create HTTP session
    session = create_session()

    # Track progress
    succeeded = 0
    skipped = 0
    failed = []
    start_time = datetime.now()
    count = 0

    for folder_name, variable in FOLDER_TO_VARIABLE.items():
        folder_path = os.path.join(BASE_DIR, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        log.info(f"{'─' * 55}")
        log.info(f"  {folder_name} ({variable})")
        log.info(f"{'─' * 55}")

        for year in YEARS:
            count += 1
            filename = f"{variable}_{year}.nc"
            output_path = os.path.join(folder_path, filename)
            url = f"{BASE_URL}/{filename}"

            # Skip if already downloaded and valid
            if os.path.exists(output_path) and os.path.getsize(output_path) >= MIN_VALID_SIZE:
                log.info(f"  [{count:>3}/{total_files}] {filename} — already exists, skipping")
                skipped += 1
                continue

            # Remove any invalid/partial previous download
            if os.path.exists(output_path):
                os.remove(output_path)

            log.info(f"  [{count:>3}/{total_files}] Downloading {filename} ...")
            success = download_file(session, url, output_path, log)

            if success:
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                log.info(f"    Done ({size_mb:.0f} MB)")
                succeeded += 1
            else:
                failed.append(f"{folder_name}/{filename}")

            # Brief pause between downloads 
            time.sleep(0.5)

    # Summary
    elapsed = datetime.now() - start_time
    log.info("")
    log.info("=" * 60)
    log.info("  DOWNLOAD COMPLETE")
    log.info("=" * 60)
    log.info(f"  Time elapsed : {str(elapsed).split('.')[0]}")
    log.info(f"  Succeeded    : {succeeded}")
    log.info(f"  Skipped      : {skipped} (already on disk)")
    log.info(f"  Failed       : {len(failed)}")

    if failed:
        log.info("")
        log.info("  Failed files:")
        for f in failed:
            log.info(f"    - {f}")
        log.info("")
        log.info("  Tip: Re-run the script to retry failed downloads.")
        log.info("  Successfully downloaded files will be skipped.")

    log.info("")
    log.info(f"  Log saved to: {os.path.join(BASE_DIR, 'download_log.txt')}")
    log.info("")


if __name__ == "__main__":
    main()
