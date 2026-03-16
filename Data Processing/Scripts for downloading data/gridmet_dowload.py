import requests
import requests_cache
import time
import os

# Cache session to avoid re-downloading files (covered in Lectures 9, 10, 12)
session = requests_cache.CachedSession('/Volumes/Extreme SSD/STA 141B Final/gridmet_cache')

# User-Agent header 
headers = {
    'User-Agent': 'sta141b-student@ucdavis.edu'
}

# Configuration
BASE_URL = "http://www.northwestknowledge.net/metdata/data"
BASE_DIR = "/Volumes/Extreme SSD/STA 141B Final/GridMET_new"  
YEARS = list(range(2001, 2025))

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

def download_file(variable, year, folder_path):
    """Download a single gridMET NetCDF file for a given variable and year."""
    filename = f"{variable}_{year}.nc"
    output_path = os.path.join(folder_path, filename)
    url = f"{BASE_URL}/{filename}"

    # skip if already downloaded
    if os.path.exists(output_path):
        print(f"  {filename} — already exists, skipping")
        return

    time.sleep(1)  # respect rate limit
    response = session.get(url, headers=headers)

    # check for errors 
    response.raise_for_status()

    # write file to disk
    with open(output_path, 'wb') as f:
        f.write(response.content)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  {filename} — done ({size_mb:.0f} MB)")

# loop to download
for folder_name, variable in FOLDER_TO_VARIABLE.items():
    folder_path = os.path.join(BASE_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    print(f"\n--- {folder_name} ({variable}) ---")

    for year in YEARS:
        download_file(variable, year, folder_path)