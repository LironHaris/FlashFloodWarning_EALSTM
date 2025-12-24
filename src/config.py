import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw' / 'Caravan_extension_Israel_Ver4' / 'Caravan_extension_Israel_Ver4'

ATTRIBUTES_DIR = RAW_DATA_DIR / 'attributes' / 'il'
TIMESERIES_DIR = RAW_DATA_DIR / 'timeseries' / 'csv' / 'il'

ATTRIBUTES_FILES = [
    'attributes_caravan_il.csv',
    'attributes_hydroatlas_il.csv',
    'attributes_other_il.csv'
]

PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
DATE_COL = 'date'
TARGET_COL = 'streamflow'
PRECIP_COL = 'total_precipitation_sum'
DYNAMIC_FEATURES = [
    'total_precipitation_sum',
    'temperature_2m_mean', 'temperature_2m_min', 'temperature_2m_max',
    'volumetric_soil_water_layer_1_mean', 'volumetric_soil_water_layer_1_min', 'volumetric_soil_water_layer_1_max',
    'surface_net_solar_radiation_mean',
    'u_component_of_wind_10m_mean', 'v_component_of_wind_10m_mean'
]