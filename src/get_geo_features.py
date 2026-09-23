# Data Science
import numpy as np
import pandas as pd

# Multi-dimensional arrays and datasets
import xarray as xr

# Geospatial raster data handling
import rioxarray as rxr

# Geospatial data analysis
import geopandas as gpd

# Geospatial operations
import rasterio
from rasterio import windows  
from rasterio import features  
from rasterio import warp
from rasterio.warp import transform_bounds 
from rasterio.windows import from_bounds 
import fiona
fiona.drvsupport.supported_drivers['libkml'] = 'rw' 
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'

# Others
import os
from tqdm import tqdm
import joblib
tqdm.pandas()

def compute_building_density(uhi_point, buildings, radius):
    # Convert to projected CRS for accurate area calculations (UTM for NYC)
    buildings = buildings.to_crs("EPSG:32618")  # Adjust CRS as needed
    uhi_point = gpd.GeoDataFrame({'geometry': [uhi_point.geometry]}, crs="EPSG:4326").to_crs("EPSG:32618")
    # print(buildings.crs, uhi_point.crs)
    buffer = uhi_point.buffer(radius)  # Create circular area
    
    # Clip buildings to buffer area
    clipped_buildings = buildings.intersection(buffer.geometry.iloc[0])
    # the number of buildings that are not empty in the buffer area
    n = clipped_buildings[clipped_buildings.is_empty == False].shape[0]
    
    # Compute only the clipped building area
    building_area = clipped_buildings.area.sum()
    total_area = buffer.area.iloc[0]  # Total circular area
    density = building_area / total_area

    min_distance = buildings.distance(uhi_point.geometry.iloc[0]).min()
    return density, n, min_distance

def get_building_features(data):
    file = "data/raw/Building_Footprint.kml"
    gdf_list = []
    for layer in fiona.listlayers(file) :    
        gdf = gpd.read_file(file, driver='LIBKML', layer=layer)
        gdf_list.append(gdf)

    buildings = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))

    data[['building_density_100', 'n_buildings_100', 'dist_nearest_building_100']] = data.progress_apply(lambda x: compute_building_density(x, buildings, 100), axis=1, result_type="expand")
    return data

def map_weather_to_uhi(uhi_point, weather):
    bronx_loc = (40.87248, -73.89352)
    manhattan_loc = (40.76754, -73.96449)
    bronx_loc = gpd.GeoSeries(gpd.points_from_xy([bronx_loc[1]], [bronx_loc[0]]),
                                crs="EPSG:4326").to_crs("EPSG:32618")
    manhattan_loc = gpd.GeoSeries(gpd.points_from_xy([manhattan_loc[1]], [manhattan_loc[0]]),
                                crs="EPSG:4326").to_crs("EPSG:32618")
    uhi_point = gpd.GeoDataFrame({'geometry': [uhi_point.geometry]}, crs="EPSG:4326").to_crs("EPSG:32618")
    # find the closest location
    loc = uhi_point.geometry.iloc[0]
    bronx_dist = bronx_loc.distance(loc)
    # print(bronx_dist)
    manhattan_dist = manhattan_loc.distance(loc)
    # print(manhattan_dist)
    if bronx_dist.iloc[0] < manhattan_dist.iloc[0]:
        weather_data = weather['Bronx']
    else:
        weather_data = weather['Manhattan']
    
    # get the three hour averages
    weather_data['Date / Time'] = pd.to_datetime(weather_data['Date / Time'])
    return weather_data.resample('3h', on='Date / Time').mean().values.flatten().tolist()

def get_weather_features(data):
    weather = pd.read_excel("data/raw/NY_Mesonet_Weather.xlsx", sheet_name=None)
    weather_features = data.progress_apply(lambda x: map_weather_to_uhi(x, weather), axis=1, result_type="expand")
    weather_cols = ['temp', 'humidity', 'wind speed', 'wind direction','solar flux']
    weather_cols = [col+'_'+str(i) for i in range(5) for col in weather_cols]
    data[weather_cols] = weather_features
    return data