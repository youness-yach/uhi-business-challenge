"""
Pre-built location configs for the UHI challenge.

Usage:
    from uhi_pipe.config import LOCATIONS, CHILE, BRAZIL, SIERRA_LEONE
"""

CHILE = {
    "label": "Chile",
    "points": "https://raw.githubusercontent.com/chase-kusterer/bc-II/main/Data/sample_chile_uhi_data.csv",
    "bbox": (-33.88, -71.05, -33.23, -70.04),
    "time_window": "2024-01-15/2024-02-15",
}

BRAZIL = {
    "label": "Brazil",
    "points": "https://raw.githubusercontent.com/chase-kusterer/bc-II/main/Data/Sample_Brazil_uhi_data.csv",
    "bbox": (-23.015, -43.52, -22.82, -43.17),  # south, west, north, east
    "time_window": "2023-01-01/2023-03-15",
}
# Note: original notebook had lower_left=(-23.015, -43.17), upper_right=(-22.82, -43.52)
# That means west=-43.52 (more negative = further west), east=-43.17. This is correct.

SIERRA_LEONE = {
    "label": "Sierra Leone",
    "points": "https://raw.githubusercontent.com/chase-kusterer/bc-II/main/Data/Validation_Dataset.csv",
    "bbox": (8.36, -13.32, 8.52, -13.10),
    "time_window": "2023-01-15/2023-02-28",
}

# all 3 as a list — ready to pass to build_multi_location()
LOCATIONS = [CHILE, BRAZIL, SIERRA_LEONE]

# training only (Chile + Brazil)
TRAIN_LOCATIONS = [CHILE, BRAZIL]

# validation only
VALIDATION_LOCATIONS = [SIERRA_LEONE]
