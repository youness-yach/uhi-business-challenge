"""
features.py — Compute spectral indices and derived features from raw extracted data.

DATA UNITS CONTRACT (what each extractor outputs):
  Sentinel-2:  B01–B12 are raw DN values (uint16 range, ~0–10000)
  Landsat vis:  red, green, blue, nir08 are surface reflectance (0 to ~0.5)
  Landsat thermal:  lwir11 is brightness temperature in °C (NOT land surface temp)
  DEM:  elevation in meters above sea level
  Buildings:  building_density_100m in buildings/m²

This file computes derived features from those raw values.
It does NOT re-scale or re-convert anything — that's done once in extract.py.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Sentinel-2 spectral indices
# ---------------------------------------------------------------------------

def sentinel_features(df):
    """
    Compute 19 spectral indices from Sentinel-2 band columns.

    Input bands are raw DN values (~0–10000 range).
    All indices are dimensionless ratios except:
      - LAI: leaf area index (from empirical NDVI relationship)
      - LSE: land surface emissivity (0.986–0.990 range)
      - Albedo: broadband albedo (0–1 range, uses DN/10000 scaling)

    Requires columns: B02, B03, B04, B08, B11, B12 at minimum.
    """
    B02, B03, B04 = df["B02"], df["B03"], df["B04"]
    B08, B11, B12 = df["B08"], df["B11"], df["B12"]

    # vegetation indices
    df["NDVI"]  = (B08 - B04) / (B08 + B04)
    df["EVI"]   = 2.5 * (B08 - B04) / (B08 + 6*B04 - 7.5*B02 + 10000)
    df["SAVI"]  = (B08 - B04) / (B08 + B04 + 0.5) * 1.5
    df["GNDVI"] = (B08 - B03) / (B08 + B03)
    df["LAI"]   = 0.57 * np.exp(2.33 * df["NDVI"])

    # water indices
    df["NDWI"]  = (B03 - B08) / (B03 + B08)
    df["MNDWI"] = (B03 - B11) / (B03 + B11)
    df["NDMI"]  = (B08 - B11) / (B08 + B11)

    # built-up / urban indices
    df["NDBI"]  = (B11 - B08) / (B11 + B08)
    df["ISA"]   = (df["NDBI"] - df["NDVI"]) / (df["NDBI"] + df["NDVI"])
    df["BU"]    = df["NDBI"] - df["NDVI"]
    df["NDISI"] = (B11 - (B02+B08+B12)/3) / (B11 + (B02+B08+B12)/3)

    # soil / bare area indices
    df["NDBaI"] = (B11 - B12) / (B11 + B12)
    df["BSI"]   = ((B11+B04) - (B08+B02)) / ((B11+B04) + (B08+B02))
    df["DBSI"]  = (B11 - B03) / (B11 + B03) - df["NDVI"]

    # physical surface properties
    Pv          = np.clip(((df["NDVI"] - 0.2) / (0.5 - 0.2))**2, 0, 1)
    df["LSE"]   = 0.004 * Pv + 0.986
    df["Albedo"] = (
        0.356*(B02/10000) + 0.130*(B04/10000) + 0.373*(B08/10000)
        + 0.085*(B11/10000) + 0.072*(B12/10000) - 0.0018
    )

    # band ratios
    df["SWIR1_NIR"] = B11 / (B08 + 1e-10)
    df["SWIR2_NIR"] = B12 / (B08 + 1e-10)

    return df


# ---------------------------------------------------------------------------
# Landsat-derived features
# ---------------------------------------------------------------------------

def landsat_features(df):
    """
    Compute LST and NDVI from Landsat columns.

    Input columns:
      nir08, red: surface reflectance (0 to ~0.5 range)
      lwir11: brightness temperature in °C (from extract_landsat)

    LST calculation:
      Brightness temperature (BT) ≠ Land Surface Temperature (LST).
      BT is what the sensor measures assuming a perfect blackbody (emissivity=1).
      Real surfaces have emissivity < 1, so actual LST is higher than BT.

      The mono-window method corrects BT using NDVI-derived emissivity:
        LST = BT / (1 + (λ × BT / ρ) × ln(ε))
      where:
        λ = 10.895 μm (Landsat 8 Band 10 center wavelength)
        ρ = h×c/σ = 14388 μm·K (Planck constant × speed of light / Boltzmann)
        ε = emissivity estimated from NDVI

      This is the standard single-channel method from Artis & Carnahan (1982).

    IMPORTANT: lwir11 arrives as °C from extract.py. The mono-window formula
    needs Kelvin, so we convert to K, apply correction, convert back to °C.
    This is NOT a double-conversion — it's BT(°C) → BT(K) → LST(K) → LST(°C).
    """
    # NDVI from Landsat bands (separate from Sentinel NDVI)
    df["NDVI_Landsat"] = (df["nir08"] - df["red"]) / (df["nir08"] + df["red"])

    # emissivity from NDVI using Sobrino et al. (2004) thresholds:
    #   NDVI < 0.2  → bare soil/water     → ε = 0.97
    #   NDVI > 0.5  → full vegetation     → ε = 0.99
    #   0.2–0.5     → mixed (linear blend) → ε = 0.004 × Pv + 0.986
    pv = np.clip(((df["NDVI_Landsat"] - 0.2) / (0.5 - 0.2))**2, 0, 1)
    emissivity = np.where(
        df["NDVI_Landsat"] < 0.2, 0.97,
        np.where(df["NDVI_Landsat"] > 0.5, 0.99, 0.004 * pv + 0.986)
    )
    df["emissivity"] = emissivity

    # LST via mono-window method
    # lwir11 is brightness temp in °C → convert to Kelvin for the formula
    BT_kelvin = df["lwir11"] + 273.15

    wavelength = 10.895    # μm — Landsat 8 TIRS Band 10 center wavelength
    rho = 14388.0          # μm·K — h*c/σ (Planck radiation constant)

    # LST(K) = BT(K) / (1 + (λ × BT(K) / ρ) × ln(ε))
    df["LST"] = BT_kelvin / (1 + (wavelength * BT_kelvin / rho) * np.log(emissivity)) - 273.15

    return df


# ---------------------------------------------------------------------------
# DEM features
# ---------------------------------------------------------------------------

def dem_features(df):
    """
    Elevation is already extracted as meters above sea level.
    Placeholder for future: slope, aspect, TPI (need full raster grid).
    """
    return df


# ---------------------------------------------------------------------------
# Convenience: compute all at once
# ---------------------------------------------------------------------------

def compute_all(sentinel_df=None, landsat_df=None, dem_df=None):
    """Apply all feature functions to whatever DataFrames are provided."""
    out = {}
    if sentinel_df is not None:
        out["sentinel"] = sentinel_features(sentinel_df.copy())
    if landsat_df is not None:
        out["landsat"] = landsat_features(landsat_df.copy())
    if dem_df is not None:
        out["dem"] = dem_features(dem_df.copy())
    return out
