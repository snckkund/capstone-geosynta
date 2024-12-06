"""
CNN model implementation for spatial climate pattern prediction.
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import numpy as np
import logging
import sys
sys.path.append('..')
from config.config import MODEL_CONFIG

def build_cnn_model(input_shape: tuple) -> Sequential:
    """
    Build and compile CNN model for climate pattern prediction.
    
    Args:
        input_shape: Shape of input data (height, width, channels)
        
    Returns:
        Compiled CNN model
    """
    try:
        model = Sequential()
        
        # Add convolutional layers
        for filters in MODEL_CONFIG['CNN']['FILTERS']:
            model.add(Conv2D(
                filters=filters,
                kernel_size=MODEL_CONFIG['CNN']['KERNEL_SIZE'],
                activation=MODEL_CONFIG['CNN']['ACTIVATION'],
                input_shape=input_shape if model.layers == [] else None,
                padding='same'
            ))
            model.add(MaxPooling2D(
                pool_size=MODEL_CONFIG['CNN']['POOL_SIZE']
            ))
        
        # Flatten the output
        model.add(Flatten())
        
        # Add dense layers
        for units in MODEL_CONFIG['CNN']['DENSE_UNITS']:
            model.add(Dense(
                units=units,
                activation=MODEL_CONFIG['CNN']['ACTIVATION']
            ))
        
        # Output layer
        model.add(Dense(
            units=1,  # Predict single value
            activation=MODEL_CONFIG['CNN']['FINAL_ACTIVATION']
        ))
        
        # Compile model
        model.compile(
            optimizer=MODEL_CONFIG['LSTM']['OPTIMIZER'],
            loss=MODEL_CONFIG['LSTM']['LOSS'],
            metrics=MODEL_CONFIG['LSTM']['METRICS']
        )
        
        logging.info("Successfully built CNN model")
        return model
    except Exception as e:
        logging.error(f"Failed to build CNN model: {str(e)}")
        raise

def prepare_spatial_data(data: np.ndarray, window_size: int) -> tuple:
    """
    Prepare spatial windows for CNN input.
    
    Args:
        data: 2D array of climate data
        window_size: Size of spatial windows
        
    Returns:
        X: Input windows
        y: Center pixel values
    """
    try:
        height, width = data.shape
        X, y = [], []
        
        for i in range(window_size, height - window_size):
            for j in range(window_size, width - window_size):
                window = data[i-window_size:i+window_size+1,
                            j-window_size:j+window_size+1]
                X.append(window)
                y.append(data[i, j])
        
        X = np.array(X)[..., np.newaxis]  # Add channel dimension
        y = np.array(y)
        
        return X, y
    except Exception as e:
        logging.error(f"Failed to prepare spatial data: {str(e)}")
        raise

def train_cnn_model(model: Sequential, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> tf.keras.callbacks.History:
    """
    Train the CNN model.
    
    Args:
        model: Compiled CNN model
        X_train: Training windows
        y_train: Training targets
        X_val: Validation windows
        y_val: Validation targets
        
    Returns:
        Training history
    """
    try:
        history = model.fit(
            X_train, y_train,
            epochs=MODEL_CONFIG['LSTM']['EPOCHS'],
            batch_size=MODEL_CONFIG['LSTM']['BATCH_SIZE'],
            validation_data=(X_val, y_val)
        )
        
        logging.info("Successfully trained CNN model")
        return history
    except Exception as e:
        logging.error(f"Failed to train CNN model: {str(e)}")
        raise
