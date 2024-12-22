# config.py
import os
from pathlib import Path

# Directory paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, NOTEBOOKS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data processing configurations
DATE_FORMAT = "%Y-%m-%d"
RANDOM_SEED = 42

# Model configurations
CLUSTERING_CONFIG = {
    "n_clusters": 4,
    "random_state": RANDOM_SEED
}

FORECAST_CONFIG = {
    "periods": 30,
    "seasonal_periods": 7
}

# Visualization configurations
VIZ_CONFIG = {
    "style": "seaborn",
    "figsize": (12, 8)
}
