"""
Visualization utilities for climate data and model results.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
import sys
sys.path.append('..')
from config.config import VIZ_CONFIG

def set_plot_style():
    """Set the default plot style."""
    plt.style.use(VIZ_CONFIG['STYLE'])
    plt.rcParams['figure.figsize'] = VIZ_CONFIG['FIGURE_SIZE']
    plt.rcParams['figure.dpi'] = VIZ_CONFIG['DPI']

def plot_training_history(history: Dict[str, List[float]], 
                        save_path: Optional[str] = None):
    """
    Plot training history metrics.
    
    Args:
        history: Dictionary containing loss and metric values
        save_path: Optional path to save the plot
    """
    try:
        set_plot_style()
        plt.figure()
        
        for metric in history.keys():
            if not metric.startswith('val_'):
                plt.plot(history[metric], label=f'Training {metric}')
                if f'val_{metric}' in history:
                    plt.plot(history[f'val_{metric}'], 
                            label=f'Validation {metric}',
                            linestyle='--')
        
        plt.title('Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    except Exception as e:
        logging.error(f"Failed to plot training history: {str(e)}")
        raise

def plot_predictions(actual: np.ndarray, predicted: np.ndarray, 
                    dates: pd.DatetimeIndex, feature_name: str,
                    save_path: Optional[str] = None):
    """
    Plot actual vs predicted values.
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        dates: Array of dates
        feature_name: Name of the feature being plotted
        save_path: Optional path to save the plot
    """
    try:
        set_plot_style()
        plt.figure()
        
        plt.plot(dates, actual, label='Actual', color='blue')
        plt.plot(dates, predicted, label='Predicted', 
                color='red', linestyle='--')
        
        plt.title(f'Actual vs Predicted {feature_name}')
        plt.xlabel('Date')
        plt.ylabel(feature_name)
        plt.legend()
        plt.grid(True)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    except Exception as e:
        logging.error(f"Failed to plot predictions: {str(e)}")
        raise

def plot_spatial_pattern(data: np.ndarray, title: str, 
                        save_path: Optional[str] = None):
    """
    Plot spatial pattern of climate data.
    
    Args:
        data: 2D array of climate data
        title: Plot title
        save_path: Optional path to save the plot
    """
    try:
        set_plot_style()
        plt.figure()
        
        im = plt.imshow(data, cmap=VIZ_CONFIG['CMAP'])
        plt.colorbar(im)
        
        plt.title(title)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    except Exception as e:
        logging.error(f"Failed to plot spatial pattern: {str(e)}")
        raise

def plot_climate_indices(indices: Dict[str, np.ndarray], 
                        dates: pd.DatetimeIndex,
                        save_path: Optional[str] = None):
    """
    Plot multiple climate indices.
    
    Args:
        indices: Dictionary of climate indices
        dates: Array of dates
        save_path: Optional path to save the plot
    """
    try:
        set_plot_style()
        n_indices = len(indices)
        fig, axes = plt.subplots(n_indices, 1, 
                                figsize=(12, 4*n_indices))
        
        for (name, values), ax in zip(indices.items(), axes):
            ax.plot(dates, values)
            ax.set_title(f'{name} Index')
            ax.set_xlabel('Date')
            ax.set_ylabel('Index Value')
            ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    except Exception as e:
        logging.error(f"Failed to plot climate indices: {str(e)}")
        raise

def create_correlation_heatmap(data: pd.DataFrame, 
                             save_path: Optional[str] = None):
    """
    Create correlation heatmap of climate variables.
    
    Args:
        data: DataFrame containing climate variables
        save_path: Optional path to save the plot
    """
    try:
        set_plot_style()
        plt.figure(figsize=(10, 8))
        
        correlation = data.corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', 
                   center=0, vmin=-1, vmax=1)
        
        plt.title('Correlation between Climate Variables')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    except Exception as e:
        logging.error(f"Failed to create correlation heatmap: {str(e)}")
        raise
