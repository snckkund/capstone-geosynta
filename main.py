"""
Main script for climate analysis and prediction.
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import os
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
tf.get_logger().setLevel('ERROR')

# Local imports
from utils.gee_utils import (initialize_gee, get_region_of_interest, 
                           fetch_gee_data, convert_gee_data_to_numpy)
from utils.data_utils import (scale_data, prepare_data_for_ml)
from utils.climate_indices import (calculate_spei, calculate_pdsi, calculate_spi,
                                 calculate_thi, calculate_heat_index, 
                                 calculate_drought_index, calculate_relative_humidity)
from models.lstm_model import build_lstm_model, train_model, predict_future
from visualization.plot_utils import (plot_training_history, plot_climate_indices,
                                    create_correlation_heatmap)
from config.config import MODEL_CONFIG, LOG_CONFIG

class ProgressCallback(tf.keras.callbacks.Callback):
    """Callback for tracking training progress."""
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of each epoch."""
        logging.info(f"\nStarting epoch {epoch + 1}")
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        if logs is None:
            logs = {}
        logging.info(f"Epoch {epoch + 1} completed - "
                    f"loss: {logs.get('loss', 'N/A'):.4f}, "
                    f"val_loss: {logs.get('val_loss', 'N/A')}")

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=LOG_CONFIG['LEVEL'],
        format=LOG_CONFIG['FORMAT'],
        handlers=[
            logging.FileHandler(LOG_CONFIG['FILENAME']),
            logging.StreamHandler()  # Also print to console
        ]
    )

def main():
    """Main execution function."""
    try:
        # Setup logging
        setup_logging()
        logging.info("Starting climate analysis")
        
        # Initialize GEE
        logging.info("Initializing Google Earth Engine...")
        initialize_gee()
        
        # Get region of interest
        logging.info("Getting region of interest...")
        region = get_region_of_interest()
        
        # Fetch climate data
        logging.info("Fetching climate data from Google Earth Engine...")
        gee_datasets = fetch_gee_data(region)
        
        # Convert GEE data to numpy arrays
        logging.info("Converting GEE data to numpy arrays...")
        numpy_datasets = convert_gee_data_to_numpy(gee_datasets, region)
        
        # Handle humidity data separately
        logging.info("Processing humidity data...")
        if 'Humidity' in numpy_datasets:
            humidity_data = numpy_datasets.pop('Humidity')
            logging.info("Calculating relative humidity...")
            relative_humidity = calculate_relative_humidity(
                humidity_data['temperature_2m'],
                humidity_data['surface_pressure'],
                humidity_data['specific_humidity']
            )
            numpy_datasets['Relative_Humidity'] = relative_humidity
        
        # Process and scale data
        logging.info("Scaling data...")
        scaled_data, scalers = scale_data(numpy_datasets)
        
        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)
        
        # Prepare data for ML
        logging.info("Preparing data for machine learning...")
        X, y = prepare_data_for_ml(scaled_data, MODEL_CONFIG['LSTM']['LOOK_BACK'])
        
        if len(X) == 0:
            raise ValueError("No sequences could be created. Check your data length and look_back parameter.")
        
        # Split data
        logging.info("Splitting data into train and validation sets...")
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Build and train LSTM model
        logging.info("Building LSTM model...")
        n_features = X.shape[-1]  # Get number of features from data
        lstm_model = build_lstm_model((MODEL_CONFIG['LSTM']['LOOK_BACK'], n_features))
        
        logging.info("Training LSTM model...")
        progress_callback = ProgressCallback()
        history = train_model(lstm_model, X_train, y_train, X_val, y_val, 
                            callbacks=[progress_callback])
        
        # Plot training history
        logging.info("Plotting training history...")
        plot_training_history(history.history, 'output/training_history.png')
        
        # Generate predictions
        logging.info("Generating predictions...")
        predictions = lstm_model.predict(X_val, verbose=1)
        
        # Calculate climate indices
        logging.info("Calculating climate indices...")
        indices = {
            'SPEI': calculate_spei(scaled_data['Precipitation'], 
                                 scaled_data['Evapotranspiration']),
            'PDSI': calculate_pdsi(scaled_data['Precipitation'],
                                 scaled_data['LST']),
            'SPI': calculate_spi(scaled_data['Precipitation']),
            'THI': calculate_thi(scaled_data['LST'],
                               scaled_data['Relative_Humidity']),
            'Heat_Index': calculate_heat_index(scaled_data['LST'],
                                             scaled_data['Relative_Humidity']),
            'Drought_Index': calculate_drought_index(
                scaled_data['Precipitation'],
                scaled_data['Evapotranspiration'],
                scaled_data['SoilMoisture']
            )
        }
        
        # Generate dates for plotting
        logging.info("Generating visualization data...")
        dates = pd.date_range(start='2010-01-01', periods=len(scaled_data['LST']), 
                            freq='M')
        
        # Plot climate indices
        logging.info("Plotting climate indices...")
        plot_climate_indices(indices, dates, 'output/climate_indices.png')
        
        # Create correlation heatmap
        logging.info("Creating correlation heatmap...")
        df = pd.DataFrame(scaled_data)
        create_correlation_heatmap(df, 'output/correlation_heatmap.png')
        
        logging.info("Climate analysis completed successfully")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
