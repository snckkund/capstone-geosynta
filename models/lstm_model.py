"""
LSTM model implementation for climate prediction.
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import logging
from typing import List, Tuple, Optional

def build_lstm_model(input_shape: Tuple[int, int], 
                    units: int = 50,
                    dropout: float = 0.2) -> Sequential:
    """
    Build LSTM model for climate prediction.
    
    Args:
        input_shape: Shape of input data (timesteps, features)
        units: Number of LSTM units
        dropout: Dropout rate
        
    Returns:
        Compiled LSTM model
    """
    try:
        model = Sequential([
            LSTM(units=units, 
                 input_shape=input_shape,
                 return_sequences=False),
            Dropout(dropout),
            Dense(input_shape[1])  # Output same number of features as input
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse'
        )
        
        logging.info(f"Successfully built LSTM model with input shape {input_shape}")
        return model
        
    except Exception as e:
        logging.error(f"Failed to build LSTM model: {str(e)}")
        raise

def train_model(model: Sequential,
                X_train: np.ndarray,
                y_train: np.ndarray,
                X_val: np.ndarray,
                y_val: np.ndarray,
                epochs: int = 10,
                batch_size: int = 32,
                callbacks: Optional[List] = None) -> tf.keras.callbacks.History:
    """
    Train LSTM model.
    
    Args:
        model: LSTM model to train
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        epochs: Number of epochs
        batch_size: Batch size
        callbacks: List of callbacks
        
    Returns:
        Training history
    """
    try:
        # Add early stopping if not in callbacks
        if callbacks is None:
            callbacks = []
        
        if not any(isinstance(cb, EarlyStopping) for cb in callbacks):
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
        
    except Exception as e:
        logging.error(f"Failed to train LSTM model: {str(e)}")
        raise

def predict_future(model: Sequential,
                  last_sequence: np.ndarray,
                  n_steps: int,
                  look_back: int) -> np.ndarray:
    """
    Generate future predictions.
    
    Args:
        model: Trained LSTM model
        last_sequence: Last known sequence of data
        n_steps: Number of steps to predict into future
        look_back: Number of lookback steps
        
    Returns:
        Array of predictions
    """
    try:
        predictions = []
        current_sequence = last_sequence[-look_back:].copy()
        
        for _ in range(n_steps):
            # Reshape for prediction
            input_seq = current_sequence.reshape((1, look_back, -1))
            
            # Get prediction
            next_pred = model.predict(input_seq, verbose=0)
            predictions.append(next_pred[0])
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = next_pred[0]
        
        return np.array(predictions)
        
    except Exception as e:
        logging.error(f"Failed to generate predictions: {str(e)}")
        raise
