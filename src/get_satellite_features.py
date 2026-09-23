import argparse
import yaml
import geopandas as gpd
import pandas as pd
import planetary_computer
from pystac_client import Client
from odc.stac import stac_load
import rioxarray as rxr
import xarray as xr
import rasterio
from rasterstats import zonal_stats
from tqdm import tqdm
import numpy as np
import os
import sys

# 🔹 Load Configurations
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

# 🔹 Define Constants
BUFFER_RADIUS = config["buffer_radius"]
START_DATE = config["start_date"]
END_DATE = config["end_date"]
CLOUD_THRESHOLD = config["cloud_threshold"]
OUTPUT_DIR = "data/raw/"
INPUT_FILE = config["input_data"]
bounds = config["bounds"]
SENT_BANDS = config["sentinel_bands"]
LANDSAT_BANDS = config["landsat_bands"]
resolution = config["resolution"]

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🔹 Connect to Planetary Computer
stac_client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    
def calculate_lst(ds):
    """
    Converts Landsat Band 10 brightness temperature to Land Surface Temperature (LST).
    """
    # Convert Band 10 (Brightness Temperature) to Kelvin
    bt = ds["lwir11"] * 0.00341802 + 149.0  # Landsat thermal scaling factors

    # Estimate Land Surface Temperature (LST) using emissivity correction
    emissivity = 0.97  # Approximate emissivity for urban surfaces
    lst = bt / (1 + (10.8 * bt / 14380) * np.log(emissivity))

    return lst

def scale_landsat_bands(ds):
    bands = ["green", "red", "nir08", "swir16"]
    scale = 0.0000275
    offset = -0.2
    for band in bands:
        ds[band] = ds[band] * scale + offset
    return ds

def extract_buffered_mean(data, locations, collection="sentinel"):
    """
    Extracts the mean value of the data within a buffer around each location.
    """
    locations["geometry"] = locations.geometry.buffer(BUFFER_RADIUS)

    for n, indices in data.items():
        if "time" in indices.dims:
            for i in tqdm(range(len(indices.time)), desc="Extracting features"):
                raster_array = indices.isel(time=i).values
                affine_transform = indices.rio.transform()  # Get the affine transform
                # 🔹 Convert to GeoJSON format for rasterstats
                buffered_zones = [geom.__geo_interface__ for geom in locations.geometry]
                # 🔹 Compute zonal statistics (mean NDVI per buffer)
                mean_values = zonal_stats(buffered_zones, raster_array,
                                            affine=affine_transform, stats=["mean"], nodata=np.nan)
                locations[f"{n}_{indices.time.values[i]}_{collection}"] = [x["mean"] for x in mean_values]
        else:
            raster_array = indices.values
            affine_transform = indices.rio.transform()
            buffered_zones = [geom.__geo_interface__ for geom in locations.geometry]
            mean_values = zonal_stats(buffered_zones, raster_array,
                                        affine=affine_transform, stats=["mean"], nodata=np.nan)
            locations[n] = [x["mean"] for x in mean_values]
    return locations

def get_landsat_features(locations):
    files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("landsat")]
    # Load all files
    data = []
    timestamps = []
    for f in files:
        data.append(rxr.open_rasterio(f"{OUTPUT_DIR}/{f}"))
        timestamps.append(pd.to_datetime(f.split("_")[1].split(".")[0]))
    # Merge all data
    time_var = xr.Variable("time", timestamps)
    ds = xr.concat(data, dim=time_var)
    ds = ds.to_dataset('band')
    ds = ds.rename({(i+1):band for i, band in enumerate(LANDSAT_BANDS)})

    # Project locations and data
    proj_crs = "EPSG:32618"
    ds = ds.rio.reproject(proj_crs)

    # Calculate indices
    ds = scale_landsat_bands(ds)

    ndvi = (ds["nir08"] - ds["red"]) / (ds["nir08"] + ds["red"])
    ndwi = (ds["green"] - ds["nir08"]) / (ds["green"] + ds["nir08"])
    mndwi = (ds["green"] - ds["swir16"]) / (ds["green"] + ds["swir16"])
    ndbi = (ds["swir16"] - ds["nir08"]) / (ds["swir16"] + ds["nir08"])
    lst = calculate_lst(ds)

    # Extract features
    indices_dict = {"ndvi": ndvi, "ndwi": ndwi, "ndbi": ndbi, "lst": lst}
    df = extract_buffered_mean(indices_dict, locations, collection="landsat")
    return df

def get_sentinel_features(locations):
    files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("sentinel")]

    # Load all files
    data = []
    timestamps = []
    for f in files:
        data.append(rxr.open_rasterio(f"{OUTPUT_DIR}/{f}"))
        timestamps.append(pd.to_datetime(f.split("_")[1].split(".")[0]))
    # Merge all data
    time_var = xr.Variable("time", timestamps)
    ds = xr.concat(data, dim=time_var)
    ds = ds.to_dataset('band')
    ds = ds.rename({(i+1):band for i, band in enumerate(SENT_BANDS)})

    
    # Project data
    proj_crs = "EPSG:32618"
    ds = ds.rio.reproject(proj_crs)

    # Calculate indices
    coastal = ds["B01"]
    rededge = ds["B06"]
    ndvi = (ds["B08"] - ds["B04"]) / (ds["B08"] + ds["B04"])
    ndwi = (ds["B03"] - ds["B08"]) / (ds["B03"] + ds["B08"])
    mndwi = (ds["B03"] - ds["B11"]) / (ds["B03"] + ds["B11"])
    ndbi = (ds["B11"] - ds["B08"]) / (ds["B11"] + ds["B08"])

    # Extract features
    indices_dict = {"coastal": coastal, "rededge": rededge, "ndvi": ndvi, "ndwi": ndwi, "ndbi": ndbi}
    df = extract_buffered_mean(indices_dict, locations, collection="sentinel")
    return df

def get_satellite_features(locations):
    """
    Extracts features from both Sentinel-2 and Landsat-8 data.
    """
    proj_crs = "EPSG:32618"
    locations = locations.to_crs(proj_crs)

    # Load Sentinel-2 Data
    sentinel_df = get_sentinel_features(locations)
    # Load Landsat-8 Data
    landsat_df = get_landsat_features(locations)
    landsat_cols = [col for col in landsat_df.columns if "landsat" in col]
    landsat_df = landsat_df[landsat_cols]

    # Merge DataFrames
    df = pd.concat([sentinel_df, landsat_df], axis=1)

    return df

    



