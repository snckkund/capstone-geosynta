# capstone-geosynta
# Climate Change Prediction System

A comprehensive machine learning system for analyzing and predicting climate patterns in Madhya Pradesh, India, using advanced geospatial data processing and LSTM-based deep learning techniques.

## Overview

This project leverages multiple satellite data sources and machine learning to provide accurate climate predictions and analysis. It combines data from various sources including MODIS, CHIRPS, GLDAS, and COPERNICUS to create a robust climate prediction model.

## Features

- Advanced data retrieval using Google Earth Engine (GEE)
- Multi-source climate data integration
- LSTM-based prediction model
- Comprehensive climate indices calculation
- Geospatial data processing
- Dynamic sequence generation
- Flexible configuration system

## Prerequisites

- Python 3.12
- Google Earth Engine account
- Sufficient computational resources for ML training

## Dependencies

```python
ee>=0.9.0
tensorflow>=2.13.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
rasterio>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/snckkund/capstone-geosynta.git
cd capstone-geosynta
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure Google Earth Engine:
```bash
earthengine authenticate
```

## Data Sources

- **MODIS**: Land Surface Temperature (LST) and Normalized Difference Vegetation Index (NDVI)
- **CHIRPS**: Precipitation data
- **GLDAS**: Humidity, Solar Radiation, Soil Moisture, Wind Speed
- **COPERNICUS**: Cloud Cover

## Project Structure

```
capstone-geosynta/
├── main.py              # Main execution script
├── models/
│   ├── lstm_model.py    # LSTM model implementation
│   ├── cnn_model.py     # CNN model implementation
├── utils/
│   ├── gee_utils.py     # Google Earth Engine utilities
│   ├── data_utils.py    # Data processing utilities
│   └── climate_indices.py # Climate indices calculations
├── data/                # Data storage directory
│   ├── raw/            # Raw data from various sources
│   └── processed/      # Processed and prepared datasets
├── notebooks/          # Jupyter notebooks for analysis
├── visualization/      # Visualization scripts and outputs
├── output/            # Model outputs and predictions
├── config/
│   └── config.yaml      # Configuration parameters
├── tests/               # Unit tests
└── README.md
```

## Usage

1. Set up Google Earth Engine authentication:
```bash
Run the Cell 2 in the CapstoneGroup180.ipynb and setup the GEE authentication
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the main analysis script:
```bash
python main.py
```

The script will:
- Initialize Google Earth Engine and fetch climate data
- Process and prepare data for machine learning
- Train LSTM and CNN models
- Calculate climate indices (SPEI, PDSI, SPI, THI)
- Generate visualizations in the `output` directory
- Save model outputs in the `output` directory
- Log progress in `climate_analysis.log`

4. View results:
- Training history plots: `output/training_history.png`
- Climate indices plots: `output/climate_indices.png`
- Correlation heatmaps: `output/correlation_heatmap.png`
- Model predictions: `output/predictions{feature}.png`

## Configuration

The project uses a comprehensive configuration system defined in `config/config.py`:

### GEE Configuration
```python
GEE_CONFIG = {
    'PROJECT_ID': 'ee-geosynta',
    'DATE_RANGE': {
        'START': '2015-01-01',
        'END': '2023-01-01'
    },
    'REGION': 'Madhya Pradesh'
}
```

### Model Configuration
```python
MODEL_CONFIG = {
    'LSTM': {
        'UNITS': 100,
        'ACTIVATION': 'relu',
        'OPTIMIZER': 'adam',
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
    }
}
```

### Visualization Configuration
```python
VIZ_CONFIG = {
    'FIGURE_SIZE': (12, 8),
    'STYLE': 'seaborn',
    'DPI': 300,
    'CMAP': 'viridis',
    'SAVE_FORMAT': 'png'
}
```

## Model Architecture

The system uses an LSTM-based architecture with:
- Dynamic input shape handling
- Dropout layers for regularization
- Early stopping mechanism
- MinMax scaling for feature normalization

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- Sushant Tripathi - *Initial work* - [Sushant2520](https://github.com/Sushant2520)
- Atin Kumar Srivastava - *Initial work* - [atin20721](https://github.com/atin20721)
- Shivangi - *Initial work* - [shivangichaudhary](https://github.com/shivangichaudhary)
- SN Chandra Kanta Kund - *Initial work* - [saiaryansahoo](https://github.com/saiaryansahoo)
- Sai Aryan Sahoo - *Initial work* - [snckkund](https://github.com/snckkund)

## Acknowledgments

- Google Earth Engine for providing satellite data access
- NASA GLDAS for climate data
- CHIRPS for precipitation data