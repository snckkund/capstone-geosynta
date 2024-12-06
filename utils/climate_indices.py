"""
Functions for calculating various climate indices.
"""
import numpy as np
from typing import Dict, Union
import logging

def calculate_spei(precip: np.ndarray, evapotranspiration: np.ndarray) -> np.ndarray:
    """
    Calculate Standardized Precipitation-Evapotranspiration Index.
    
    Args:
        precip: Precipitation array
        evapotranspiration: Evapotranspiration array
        
    Returns:
        SPEI array
    """
    try:
        difference = precip - evapotranspiration
        return (difference - np.mean(difference)) / np.std(difference)
    except Exception as e:
        logging.error(f"Failed to calculate SPEI: {str(e)}")
        raise

def calculate_pdsi(precip: np.ndarray, temp: np.ndarray) -> np.ndarray:
    """
    Calculate Palmer Drought Severity Index.
    
    Args:
        precip: Precipitation array
        temp: Temperature array
        
    Returns:
        PDSI array
    """
    try:
        water_balance = precip - temp  # Simplified water balance model
        z = water_balance - np.mean(water_balance)
        return z + water_balance
    except Exception as e:
        logging.error(f"Failed to calculate PDSI: {str(e)}")
        raise

def calculate_spi(precip: np.ndarray) -> np.ndarray:
    """
    Calculate Standardized Precipitation Index.
    
    Args:
        precip: Precipitation array
        
    Returns:
        SPI array
    """
    try:
        return (precip - np.mean(precip)) / np.std(precip)
    except Exception as e:
        logging.error(f"Failed to calculate SPI: {str(e)}")
        raise

def calculate_thi(temp: np.ndarray, humidity: np.ndarray) -> np.ndarray:
    """
    Calculate Temperature-Humidity Index.
    
    Args:
        temp: Temperature array
        humidity: Humidity array
        
    Returns:
        THI array
    """
    try:
        return temp - 0.55 * (1 - humidity) * (temp - 14.5)
    except Exception as e:
        logging.error(f"Failed to calculate THI: {str(e)}")
        raise

def calculate_cei(extreme_data: np.ndarray, total_area: int) -> float:
    """
    Calculate Climate Extremes Index.
    
    Args:
        extreme_data: Binary array indicating extreme conditions
        total_area: Total area being analyzed
        
    Returns:
        CEI value
    """
    try:
        extreme_area = np.count_nonzero(extreme_data)
        return (extreme_area / total_area) * 100
    except Exception as e:
        logging.error(f"Failed to calculate CEI: {str(e)}")
        raise

def calculate_ccii(*arrays: np.ndarray) -> np.ndarray:
    """
    Calculate Composite Climate Change Intensity Index.
    
    Args:
        *arrays: Variable number of climate factor arrays
        
    Returns:
        CCII array
    """
    try:
        factors = np.stack(arrays, axis=-1)
        weights = np.ones(factors.shape[-1]) / factors.shape[-1]
        return np.dot(factors, weights)
    except Exception as e:
        logging.error(f"Failed to calculate CCII: {str(e)}")
        raise

def calculate_heat_index(temp: np.ndarray, humidity: np.ndarray) -> np.ndarray:
    """
    Calculate Heat Index.
    
    Args:
        temp: Temperature array (in Celsius)
        humidity: Relative humidity array (in percentage)
        
    Returns:
        Heat Index array
    """
    try:
        # Convert Celsius to Fahrenheit for the standard heat index formula
        temp_f = (temp * 9/5) + 32
        
        # Simplified Rothfusz regression
        hi = 0.5 * (temp_f + 61.0 + ((temp_f - 68.0) * 1.2) + (humidity * 0.094))
        
        # Convert back to Celsius
        return (hi - 32) * 5/9
    except Exception as e:
        logging.error(f"Failed to calculate Heat Index: {str(e)}")
        raise

def calculate_relative_humidity(T: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Calculate relative humidity using specific humidity, pressure, and temperature.
    
    Args:
        T: Temperature array
        p: Pressure array
        q: Specific humidity array
        
    Returns:
        Relative humidity array (0-100)
    """
    try:
        T0 = 273.16  # Reference temperature
        numerator = 0.263 * p * q
        denominator = np.exp(17.67 * (T - T0) / (T - 29.65))
        rh = (numerator / denominator).clip(0, 100)
        return rh
        
    except Exception as e:
        logging.error(f"Failed to calculate relative humidity: {str(e)}")
        raise

def calculate_drought_index(precip: np.ndarray, evaporation: np.ndarray, 
                          soil_moisture: np.ndarray) -> np.ndarray:
    """
    Calculate Custom Drought Index.
    
    Args:
        precip: Precipitation array
        evaporation: Evaporation array
        soil_moisture: Soil moisture array
        
    Returns:
        Drought Index array
    """
    try:
        # Normalize each component
        p_norm = (precip - np.mean(precip)) / np.std(precip)
        e_norm = (evaporation - np.mean(evaporation)) / np.std(evaporation)
        sm_norm = (soil_moisture - np.mean(soil_moisture)) / np.std(soil_moisture)
        
        # Weighted combination
        weights = np.array([0.4, 0.3, 0.3])  # Adjustable weights
        return np.stack([p_norm, -e_norm, sm_norm], axis=-1) @ weights
    except Exception as e:
        logging.error(f"Failed to calculate Drought Index: {str(e)}")
        raise
