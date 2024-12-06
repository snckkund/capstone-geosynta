"""
Utility functions for data preparation and processing.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Tuple, List
import logging

def calculate_relative_humidity(T: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Calculate relative humidity using specific humidity, pressure, and temperature.
    
    Args:
        T: Temperature array
        p: Pressure array
        q: Specific humidity array
        
    Returns:
        Relative humidity array
    """
    try:
        T0 = 273.16  # Reference temperature
        numerator = 0.263 * p * q
        denominator = np.exp(17.67 * (T - T0) / (T - 29.65))
        return numerator / denominator
    except Exception as e:
        logging.error(f"Failed to calculate relative humidity: {str(e)}")
        raise

def scale_data(data: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, MinMaxScaler]]:
    """
    Scale the data using MinMaxScaler.
    
    Args:
        data: Dictionary of numpy arrays to scale
        
    Returns:
        Tuple of (scaled data dictionary, dictionary of scalers)
    """
    try:
        scaled_data = {}
        scalers = {}
        
        for key, values in data.items():
            if isinstance(values, dict):
                # Skip multi-band data (like humidity components)
                continue
                
            # Reshape for scaling if needed
            original_shape = values.shape
            if len(original_shape) == 1:
                values = values.reshape(-1, 1)
            
            # Create and fit scaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_values = scaler.fit_transform(values)
            
            # Reshape back if needed
            if len(original_shape) == 1:
                scaled_values = scaled_values.ravel()
            
            scaled_data[key] = scaled_values
            scalers[key] = scaler
        
        logging.info(f"Successfully scaled {len(scaled_data)} datasets")
        return scaled_data, scalers
        
    except Exception as e:
        logging.error(f"Error scaling data: {str(e)}")
        raise

def align_data_lengths(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Align all arrays to the same length by truncating to the shortest one.
    
    Args:
        data: Dictionary of numpy arrays
        
    Returns:
        Dictionary of aligned numpy arrays
    """
    try:
        # Find the shortest length
        lengths = [arr.shape[0] for arr in data.values()]
        min_length = min(lengths)
        
        # Truncate all arrays to the shortest length
        aligned_data = {
            key: values[:min_length] for key, values in data.items()
        }
        
        if min_length < max(lengths):
            logging.warning(f"Arrays truncated from max length {max(lengths)} to {min_length}")
        
        return aligned_data
        
    except Exception as e:
        logging.error(f"Error aligning data lengths: {str(e)}")
        raise

def prepare_data_for_ml(data: Dict[str, np.ndarray], 
                       look_back: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for machine learning by creating sequences.
    
    Args:
        data: Dictionary of scaled numpy arrays
        look_back: Number of time steps to look back
        
    Returns:
        Tuple of (X, y) arrays for training
    """
    try:
        # First align all arrays to the same length
        aligned_data = align_data_lengths(data)
        
        # Stack arrays along feature dimension
        combined_data = np.stack(list(aligned_data.values()), axis=1)
        
        X, y = [], []
        for i in range(len(combined_data) - look_back):
            # Create sequence of look_back time steps
            sequence = combined_data[i:(i + look_back)]
            target = combined_data[i + look_back]
            X.append(sequence)
            y.append(target)
        
        # Convert to numpy arrays with proper shapes
        X = np.array(X)  # Shape: (samples, look_back, features)
        y = np.array(y)  # Shape: (samples, features)
        
        logging.info(f"Created sequences with shapes - X: {X.shape}, y: {y.shape}")
        return X, y
        
    except Exception as e:
        logging.error(f"Error preparing data for ML: {str(e)}")
        raise

def create_time_series_dataset(data: np.ndarray, dates: List[str], 
                             feature_names: List[str]) -> pd.DataFrame:
    """
    Create a pandas DataFrame with time series data.
    
    Args:
        data: Array of climate data
        dates: List of date strings
        feature_names: List of feature names
        
    Returns:
        DataFrame with time series data
    """
    try:
        df = pd.DataFrame(data, columns=feature_names)
        df['date'] = pd.to_datetime(dates)
        df.set_index('date', inplace=True)
        return df
    except Exception as e:
        logging.error(f"Failed to create time series dataset: {str(e)}")
        raise
