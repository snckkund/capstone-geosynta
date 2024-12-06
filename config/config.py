"""
Configuration settings for the climate analysis project.
"""

# Google Earth Engine Configuration
GEE_CONFIG = {
    'PROJECT_ID': 'ee-geosynta',
    'DATE_RANGE': {
        'START': '2015-01-01',
        'END': '2023-01-01'
    },
    'REGION': 'Madhya Pradesh'
}

# Dataset Configuration
DATASET_CONFIG = {
    'LST': 'MODIS/061/MOD11A2',
    'PRECIPITATION': 'UCSB-CHG/CHIRPS/PENTAD',
    'HUMIDITY': 'NASA/GLDAS/V021/NOAH/G025/T3H',
    'SOLAR_RADIATION': 'NASA/GLDAS/V021/NOAH/G025/T3H',
    'SOIL_MOISTURE': 'NASA/GLDAS/V021/NOAH/G025/T3H',
    'WIND_SPEED': 'NASA/GLDAS/V021/NOAH/G025/T3H',
    'EVAPOTRANSPIRATION': 'MODIS/061/MOD16A2',
    'NDVI': 'MODIS/061/MOD13Q1',
    'CLOUD_COVER': 'COPERNICUS/S2_CLOUD_PROBABILITY',
    'CO': 'COPERNICUS/S5P/NRTI/L3_CO',
    'LANDSAT': 'LANDSAT/LC08/C02/T1_L2'
}

# Model Configuration
MODEL_CONFIG = {
    'LSTM': {
        'UNITS': 100,
        'ACTIVATION': 'relu',
        'OPTIMIZER': 'adam',
        'LOSS': 'mse',
        'METRICS': ['mse'],
        'BATCH_SIZE': 16,
        'EPOCHS': 10,
        'VALIDATION_SPLIT': 0.2,
        'LOOK_BACK': 10
    },
    'CNN': {
        'FILTERS': [32, 64, 128],
        'KERNEL_SIZE': 3,
        'POOL_SIZE': 2,
        'DENSE_UNITS': [64, 32],
        'ACTIVATION': 'relu',
        'FINAL_ACTIVATION': 'linear'
    },
    'ARIMA': {
        'ORDER': (1, 1, 1),
        'SEASONAL_ORDER': (1, 1, 1, 12)
    }
}

# Visualization Configuration
VIZ_CONFIG = {
    'FIGURE_SIZE': (12, 8),
    'STYLE': 'seaborn',
    'DPI': 300,
    'CMAP': 'viridis',
    'SAVE_FORMAT': 'png'
}

# Logging Configuration
LOG_CONFIG = {
    'LEVEL': 'INFO',
    'FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'FILENAME': 'climate_analysis.log'
}
