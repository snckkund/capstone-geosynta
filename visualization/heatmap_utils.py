import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_comparison_heatmap(original_data, predicted_data, feature_names):
    """
    Generate three side-by-side heatmaps comparing original values, predicted values, and their differences.
    
    Parameters:
    -----------
    original_data : numpy.ndarray
        Array of original feature values with shape (n_samples, n_features)
    predicted_data : numpy.ndarray
        Array of predicted feature values with shape (n_samples, n_features)
    feature_names : list
        List of feature names for labeling the heatmap
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the heatmaps
    """
    # Ensure data is in the right shape (transpose if needed)
    original_data = np.asarray(original_data)
    predicted_data = np.asarray(predicted_data)
    
    if original_data.shape[0] > original_data.shape[1]:
        original_data = original_data.T
        predicted_data = predicted_data.T
    
    # Calculate differences
    diff_data = predicted_data - original_data
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot original data
    sns.heatmap(original_data, ax=ax1, cmap='viridis', 
                yticklabels=feature_names, xticklabels=False)
    ax1.set_title('Original Values')
    ax1.set_xlabel('Time Steps')
    
    # Plot predicted data
    sns.heatmap(predicted_data, ax=ax2, cmap='viridis',
                yticklabels=feature_names, xticklabels=False)
    ax2.set_title('Predicted Values')
    ax2.set_xlabel('Time Steps')
    
    # Plot differences with diverging colormap centered at zero
    sns.heatmap(diff_data, ax=ax3, cmap='RdBu_r', center=0,
                yticklabels=feature_names, xticklabels=False)
    ax3.set_title('Differences (Predicted - Original)')
    ax3.set_xlabel('Time Steps')
    
    plt.tight_layout()
    return fig

def plot_feature_heatmaps(data_array, feature_names, time_labels=None):
    """
    Create detailed heatmaps for each feature showing temporal patterns.
    
    Parameters:
    -----------
    data_array : numpy.ndarray
        Array of feature values with shape (n_features, n_timesteps) or (n_timesteps, n_features)
    feature_names : list
        List of feature names for labeling
    time_labels : list, optional
        List of time labels for x-axis
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the heatmap
    """
    # Ensure data is in the right shape (transpose if needed)
    data_array = np.asarray(data_array)
    
    if data_array.shape[0] > data_array.shape[1]:
        data_array = data_array.T
        
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Create heatmap
    sns.heatmap(data_array, ax=ax, cmap='viridis',
                yticklabels=feature_names,
                xticklabels=time_labels if time_labels is not None else False)
    
    ax.set_title('Feature Values Over Time')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Features')
    
    plt.tight_layout()
    return fig
