"""
Plotting functions for battery RUL prediction visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Default plot configuration
PLOT_CONFIG = {
    'figure_size': (12, 8),
    'scatter_alpha': 0.8,
    'trend_line_alpha': 0.6,
    'grid_alpha': 0.3,
    'show_trend_lines': True
}

def plot_rul_predictions(actual_rul, predicted_rul, cycle_numbers_test, max_cycles_est, model_name=None):
    """
    Create and save RUL prediction plot
    
    Parameters:
    -----------
    actual_rul : array-like
        Actual RUL values
    predicted_rul : array-like
        Predicted RUL values
    cycle_numbers_test : array-like
        Cycle numbers for test data
    max_cycles_est : int
        Maximum cycles estimate
    model_name : str, optional
        Name of the model being plotted
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    print("Creating RUL predictions plot")
    
    # Create a new figure with specified size
    plt.figure(figsize=PLOT_CONFIG['figure_size'])
    
    # Plot actual and predicted RUL with larger markers and more opacity
    plt.scatter(cycle_numbers_test, actual_rul, c='blue', label='Actual RUL', 
               alpha=0.8, s=100, marker='o')
    plt.scatter(cycle_numbers_test, predicted_rul, c='red', label='Predicted RUL', 
               alpha=0.8, s=100, marker='^')
    
    if PLOT_CONFIG['show_trend_lines']:
        # Add trend lines with higher visibility
        z_actual = np.polyfit(cycle_numbers_test, actual_rul, 1)
        p_actual = np.poly1d(z_actual)
        plt.plot(cycle_numbers_test, p_actual(cycle_numbers_test), "b--", 
                alpha=0.6, linewidth=2, label='Actual Trend')
        
        z_pred = np.polyfit(cycle_numbers_test, predicted_rul, 1)
        p_pred = np.poly1d(z_pred)
        plt.plot(cycle_numbers_test, p_pred(cycle_numbers_test), "r--", 
                alpha=0.6, linewidth=2, label='Predicted Trend')
    
    # Set axis limits with padding
    plt.xlim(-5, max_cycles_est + 5)
    plt.ylim(-5, max_cycles_est + 5)
    
    # Add labels and title with larger font sizes
    plt.xlabel('Cycle Number', fontsize=12)
    plt.ylabel('Remaining Useful Life (cycles)', fontsize=12)
    title = 'RUL Predictions vs Actual Values'
    if model_name:
        title += f'\nModel: {model_name}'
    title += f', Max Life: {max_cycles_est} cycles'
    plt.title(title, fontsize=14, pad=20)
    
    # Add legend with better placement
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Add grid with better visibility
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Calculate statistics
    rmse = np.sqrt(mean_squared_error(actual_rul, predicted_rul))
    mae = mean_absolute_error(actual_rul, predicted_rul)
    r2 = r2_score(actual_rul, predicted_rul)
    
    # Add statistics text box with better formatting
    stats_text = (f'Model: {model_name if model_name else "Unknown"}\n'
                 f'RMSE: {rmse:.3f}\n'
                 f'MAE: {mae:.3f}\n'
                 f'R²: {r2:.3f}\n'
                 f'Cycle Range: {np.min(cycle_numbers_test):.0f} - {np.max(cycle_numbers_test):.0f}')
    
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8),
             verticalalignment='top',
             fontsize=10)
    
    # Adjust layout to prevent text cutoff
    plt.tight_layout()
    
    return plt.gcf()  # Return the figure object

def save_plot(fig, filename, dpi=300):
    """
    Save a matplotlib figure to file
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to save
    filename : str
        The filename to save to
    dpi : int, optional
        The resolution in dots per inch
    """
    fig.savefig(filename, bbox_inches='tight', dpi=dpi)
    plt.close(fig) 