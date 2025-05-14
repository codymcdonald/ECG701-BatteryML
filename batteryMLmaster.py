import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from itertools import product

from src.visualization.plotting import plot_rul_predictions, save_plot

# ============= Configuration Options =============

# Data Configuration
EXCEL_DATA_PATH = r'C:/Users/cdmcd/PycharmProjects/batteryMLmaster/data/B0026.xlsx'  # Path to your Excel data file
PROCESS_ALL_EXCEL_FILES = False  # Set to True to process all Excel files in the directory
EXCEL_DIRECTORY = r'C:/Users/cdmcd/PycharmProjects/batteryMLmaster/data'  # Directory containing Excel files

# Battery Configuration
MAX_CYCLES = 25  # Maximum life cycles for the battery
COMPARE_MODELS = True
PARAMETER_SWEEP = True  # Whether to perform parameter sweep for each model

# Model Selection (uncomment one)
SELECTED_MODEL = 'RandomForest'  # Default model
# SELECTED_MODEL = 'AdaBoost'
# SELECTED_MODEL = 'Linear'
# SELECTED_MODEL = 'Ridge'
# SELECTED_MODEL = 'Lasso'
# SELECTED_MODEL = 'GradientBoosting'
# SELECTED_MODEL = 'SVR'

# Parameter Sweep Configurations
PARAMETER_SWEEP_RANGES = {
    'RandomForest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'GradientBoosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 5, 7]
    },
    'AdaBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.1, 0.5, 1.0]
    },
    'Ridge': {
        'alpha': [0.1, 1.0, 10.0]
    },
    'Lasso': {
        'alpha': [0.1, 1.0, 10.0]
    },
    'SVR': {
        'kernel': ['rbf', 'linear', 'poly'],
        'C': [0.1, 1.0, 10.0],
        'epsilon': [0.1, 0.2, 0.3],
        'gamma': ['scale', 'auto']
    }
}

# Model Parameters
MODEL_PARAMS = {
    'RandomForest': {
        'n_estimators': 100,     # Number of trees
        'max_depth': None,       # Maximum depth of trees (None for unlimited)
        'min_samples_split': 2,  # Minimum samples required to split
        'min_samples_leaf': 1,   # Minimum samples required at leaf node
        'random_state': 42
    },
    'GradientBoosting': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'random_state': 42
    },
    'AdaBoost': {
        'n_estimators': 50,
        'learning_rate': 1.0,
        'random_state': 42
    },
    'Linear': {
        # Linear regression has no parameters to tune
    },
    'Ridge': {
        'alpha': 1.0,  # Regularization strength
        'random_state': 42
    },
    'Lasso': {
        'alpha': 1.0,  # Regularization strength
        'random_state': 42
    },
    'SVR': {
        'kernel': 'rbf',     # Kernel type ('rbf', 'linear', 'poly')
        'C': 1.0,           # Regularization parameter
        'epsilon': 0.1,     # Epsilon in the epsilon-SVR model
        'gamma': 'scale'    # Kernel coefficient
    }
}

# Data Processing Options
TRAIN_TEST_SPLIT_RATIO = 0.2  # 20% for testing
RANDOM_SEED = 42             # Random seed for reproducibility

# Feature Engineering Options
FEATURE_ENGINEERING = {
    'use_cycle_number': True,
    'use_current_stats': True,  # Mean, std, max, min of current
    'use_voltage_stats': True, # Mean, std, max, min of voltage
    'use_temp_stats': True,    # Mean, std, max, min of temperature
}

# Visualization Options
PLOT_CONFIG = {
    'figure_size': (12, 8),
    'scatter_alpha': 0.6,
    'grid_alpha': 0.3,
    'show_trend_lines': True,
    'trend_line_alpha': 0.3
}

# ============= Helper Functions =============

def get_model(model_name):
    """Return the selected model with configured parameters"""
    if model_name == 'RandomForest':
        return RandomForestRegressor(**MODEL_PARAMS['RandomForest'])
    elif model_name == 'GradientBoosting':
        # Only use n_estimators and random_state for GradientBoosting
        params = {
            'n_estimators': MODEL_PARAMS['GradientBoosting']['n_estimators'],
            'random_state': MODEL_PARAMS['GradientBoosting']['random_state']
        }
        return GradientBoostingRegressor(**params)
    elif model_name == 'AdaBoost':
        return AdaBoostRegressor(**MODEL_PARAMS['AdaBoost'])
    elif model_name == 'Linear':
        return LinearRegression()
    elif model_name == 'Ridge':
        return Ridge(**MODEL_PARAMS['Ridge'])
    elif model_name == 'Lasso':
        return Lasso(**MODEL_PARAMS['Lasso'])
    elif model_name == 'SVR':
        return SVR(**MODEL_PARAMS['SVR'])
    else:
        print(f"Warning: Unknown model '{model_name}', using RandomForest")
        return RandomForestRegressor(**MODEL_PARAMS['RandomForest'])

def load_excel_data(file_path):
    """
    Load data from Excel file
    Args:
        file_path (str): Path to the Excel file
    Returns:
        dict: Data dictionary with the same structure as pickle data
    """
    print(f"Loading Excel data from {file_path}")
    try:
        df = pd.read_excel(file_path)
        print("Successfully loaded Excel data")
        print(f"Available columns: {list(df.columns)}")
        
        # Convert DataFrame to the expected dictionary structure
        cycle_data = []
        for cycle_num in df['Cycle_Index'].unique():
            cycle_df = df[df['Cycle_Index'] == cycle_num]
            cycle_dict = {
                'cycle_number': cycle_num,
                'current_in_A': cycle_df['Current(A)'].values,
                'voltage_in_V': cycle_df['Voltage(V)'].values,
                'temperature_in_C': cycle_df['Surface_Temp(degC)'].values
            }
            cycle_data.append(cycle_dict)
        
        data = {
            'cell_id': 'NASA_cell',  # Default cell ID
            'cycle_data': cycle_data,
            'nominal_capacity_in_Ah': df['Capacity'].mean() if 'Capacity' in df.columns else None
        }
        
        print(f"Processed {len(cycle_data)} cycles")
        return data
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None

def extract_features(df):
    """
    Extract features from cycle data
    Args:
        df (pd.DataFrame): DataFrame containing cycle data
    Returns:
        tuple: (features DataFrame, RUL values)
    """
    print(f"Extracting features from {len(df)} cycles")
    
    # Group by cycle number
    cycle_groups = df.groupby('Cycle')
    
    features_list = []
    rul_list = []
    
    for cycle_num, cycle_df in cycle_groups:
        try:
            # Extract features
            cycle_features = {
                'cycle_number': cycle_num,
                'mean_current': cycle_df['Current_measured'].mean(),
                'std_current': cycle_df['Current_measured'].std(),
                'mean_voltage': cycle_df['Voltage_measured'].mean(),
                'std_voltage': cycle_df['Voltage_measured'].std(),
                'mean_temperature': cycle_df['Temperature_measured'].mean(),
                'std_temperature': cycle_df['Temperature_measured'].std()
            }
            
            features_list.append(cycle_features)
            rul_list.append(cycle_num)  # We'll calculate RUL later
            
        except Exception as e:
            print(f"Warning: Skipping cycle {cycle_num} due to error: {e}")
            continue
    
    if not features_list:
        print("Error: No valid features extracted from the data")
        return pd.DataFrame(), pd.Series()
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    rul_series = pd.Series(rul_list, name='cycle_number')
    
    return features_df, rul_series

def calculate_rul(rul_series, max_cycles):
    """
    Calculate Remaining Useful Life for each cycle
    Args:
        rul_series (pd.Series): Series containing cycle numbers
        max_cycles (int): Maximum expected cycles
    Returns:
        pd.Series: RUL values
    """
    print(f"Calculating RUL values (max cycles: {max_cycles})")
    rul = max_cycles - rul_series
    rul = rul.clip(lower=0)  # Ensure RUL doesn't go below 0
    return rul

def train_rul_model(features, rul, config):
    """Train a model to predict RUL"""
    # Print feature statistics before scaling
    print("\nFeature statistics before scaling:")
    print(features.describe())
    print("\nRUL statistics before scaling:")
    print(rul.describe())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    X_scaled = pd.DataFrame(X_scaled, columns=features.columns)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, rul,
        test_size=config.get('training', {}).get('test_size', 0.2),
        random_state=config.get('training', {}).get('random_state', 42)
    )
    
    # Get and train model
    model = get_model(config)
    print(f"Training {type(model).__name__}...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"R2 Score: {r2:.3f}")
    
    # Print feature importances if available
    if hasattr(model, 'feature_importances_'):
        importances = pd.DataFrame({
            'feature': features.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nFeature Importances:")
        print(importances)
    
    return model, scaler, (X_test, y_test, y_pred)

def compare_models(features, rul):
    """Compare different models' performance on RUL prediction"""
    models = ['RandomForest', 'GradientBoosting', 'AdaBoost', 'Linear', 'Ridge', 'Lasso', 'SVR']
    results = []
    
    # Scale features once for all models
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    X_scaled = pd.DataFrame(X_scaled, columns=features.columns)
    
    # Split data once for all models
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, rul,
        test_size=TRAIN_TEST_SPLIT_RATIO,
        random_state=RANDOM_SEED
    )
    
    print("\nModel Comparison:")
    print("-" * 60)
    print(f"{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R2 Score':<10}")
    print("-" * 60)
    
    for model_name in models:
        # Get and train model
        model = get_model(model_name)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"{model_name:<20} {rmse:<10.3f} {mae:<10.3f} {r2:<10.3f}")
        
        results.append({
            'model_name': model_name,
            'model': model,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred
        })
        
        # Create and save individual model plots
        fig = plot_rul_predictions(y_test, y_pred, X_test.index, MAX_CYCLES, model_name)
        save_plot(fig, f'plots/rul_predictions_{model_name}.png')
    
    # Find best model based on RMSE
    best_model = min(results, key=lambda x: x['rmse'])
    print("\nBest Model:")
    print(f"Model: {best_model['model_name']}")
    print(f"RMSE: {best_model['rmse']:.3f}")
    print(f"MAE: {best_model['mae']:.3f}")
    print(f"R2 Score: {best_model['r2']:.3f}")
    
    # Create and save best model predictions with timestamp
    fig = plot_rul_predictions(y_test, best_model['predictions'], X_test.index, MAX_CYCLES, best_model['model_name'])
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    save_plot(fig, f'plots/rul_predictions_best_{best_model["model_name"]}_{timestamp}.png')
    
    return best_model['model'], scaler, (X_test, y_test, best_model['predictions'])

def get_valid_params(model_name, params):
    """Filter out invalid parameters for a given model type"""
    if model_name == 'RandomForest':
        valid_keys = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'random_state']
    elif model_name == 'GradientBoosting':
        valid_keys = ['n_estimators', 'random_state']
    elif model_name == 'AdaBoost':
        valid_keys = ['n_estimators', 'learning_rate', 'random_state']
    elif model_name == 'Linear':
        valid_keys = []
    elif model_name == 'Ridge':
        valid_keys = ['alpha', 'random_state']
    elif model_name == 'Lasso':
        valid_keys = ['alpha', 'random_state']
    elif model_name == 'SVR':
        valid_keys = ['C', 'epsilon', 'gamma', 'kernel']
    else:
        valid_keys = []
    
    # Only include parameters that are both in valid_keys and in params
    valid_params = {}
    for key in valid_keys:
        if key in params:
            valid_params[key] = params[key]
    
    return valid_params

def perform_parameter_sweep(model_name, X_train, X_test, y_train, y_test):
    """
    Perform parameter sweep for a given model
    Returns the best model configuration and its metrics
    """
    if model_name not in PARAMETER_SWEEP_RANGES or model_name == 'Linear':
        # For Linear Regression or if no sweep range defined, use default parameters
        model = get_model(model_name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {
            'Model': model_name,
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred),
            'Parameters': 'default'
        }
        return metrics, model

    # Get parameter ranges for this model
    param_ranges = PARAMETER_SWEEP_RANGES[model_name]
    
    # Generate all parameter combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    
    best_metrics = None
    best_model = None
    best_rmse = float('inf')
    
    print(f"\nPerforming parameter sweep for {model_name}...")
    total_combinations = np.prod([len(vals) for vals in param_values])
    print(f"Testing {total_combinations} parameter combinations")
    
    # Try each parameter combination
    for i, values in enumerate(product(*param_values), 1):
        params = dict(zip(param_names, values))
        
        # Add random_state only for models that support it
        if model_name != 'SVR':
            params['random_state'] = RANDOM_SEED
        
        try:
            if model_name == 'RandomForest':
                model = RandomForestRegressor(**params)
            elif model_name == 'GradientBoosting':
                model = GradientBoostingRegressor(**params)
            elif model_name == 'AdaBoost':
                model = AdaBoostRegressor(**params)
            elif model_name == 'Ridge':
                model = Ridge(**params)
            elif model_name == 'Lasso':
                model = Lasso(**params)
            elif model_name == 'SVR':
                # For SVR, only use valid parameters
                svr_params = {
                    'kernel': params['kernel'],
                    'C': params['C'],
                    'epsilon': params['epsilon']
                }
                # Only add gamma if using rbf or poly kernel
                if params['kernel'] in ['rbf', 'poly']:
                    svr_params['gamma'] = params['gamma']
                model = SVR(**svr_params)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_metrics = {
                    'Model': model_name,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'Parameters': str(params)
                }
                best_model = model
            
            if i % 10 == 0:
                print(f"Processed {i}/{total_combinations} combinations")
                
        except Exception as e:
            print(f"Warning: Parameter combination {params} failed with error: {e}")
            continue
    
    if best_model is None:
        print(f"Warning: No valid parameter combinations found for {model_name}. Using default parameters.")
        model = get_model(model_name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        best_metrics = {
            'Model': model_name,
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred),
            'Parameters': 'default (all combinations failed)'
        }
        best_model = model
    
    return best_metrics, best_model

def process_excel_file(file_path):
    """
    Process a single Excel file and perform RUL prediction
    Args:
        file_path (str): Path to the Excel file
    """
    print(f"\nProcessing file: {file_path}")
    
    # Get the base name of the Excel file (without extension)
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Load data
    print("\nLoading data...")
    try:
        df = pd.read_excel(file_path)
        print(f"Successfully loaded data with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Extract features and calculate RUL
    print("\nExtracting features...")
    features_df, cycle_numbers = extract_features(df)
    if features_df.empty:
        print("Error: No features could be extracted from the data")
        return
    
    print(f"Extracted features shape: {features_df.shape}")
    print(f"Cycle numbers shape: {cycle_numbers.shape}")
        
    rul = calculate_rul(cycle_numbers, MAX_CYCLES)
    print(f"RUL shape: {rul.shape}")
    
    # Check data integrity
    if features_df.shape[0] == 0 or rul.shape[0] == 0:
        print("Error: No valid features or RUL values extracted.")
        return
    
    if features_df.shape[0] != rul.shape[0]:
        print(f"Error: Feature shape {features_df.shape} does not match RUL shape {rul.shape}")
        return
    
    print("\nFeature statistics:")
    print(features_df.describe())
    
    # Create main results directory with timestamp
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    main_results_dir = os.path.join('plots', f'batch_{timestamp}')
    os.makedirs(main_results_dir, exist_ok=True)
    
    # Create subdirectory for this specific file
    file_results_dir = os.path.join(main_results_dir, file_name)
    os.makedirs(file_results_dir, exist_ok=True)
    
    if COMPARE_MODELS:
        # Create comparison directory for this file
        comparison_dir = os.path.join(file_results_dir, 'comparison')
        os.makedirs(comparison_dir, exist_ok=True)
        print(f"\nCreated comparison directory: {comparison_dir}")
        
        print("\nRunning model comparison...")
        models = ['RandomForest', 'GradientBoosting', 'AdaBoost', 'Linear', 'Ridge', 'Lasso', 'SVR']
        results = []
        
        # Scale features once for all models
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features_df)
        X_scaled = pd.DataFrame(X_scaled, columns=features_df.columns)
        
        # Split data once for all models
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, rul,
            test_size=TRAIN_TEST_SPLIT_RATIO,
            random_state=RANDOM_SEED
        )
        
        for model_name in models:
            if PARAMETER_SWEEP and model_name != 'Linear':
                metrics, model = perform_parameter_sweep(model_name, X_train, X_test, y_train, y_test)
                results.append(metrics)
            else:
                # Use default parameters
                model = get_model(model_name)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                metrics = {
                    'Model': model_name,
                    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'MAE': mean_absolute_error(y_test, y_pred),
                    'R2': r2_score(y_test, y_pred),
                    'Parameters': 'default'
                }
                results.append(metrics)
            
            # Create and save plot for this model in the comparison directory
            fig = plot_rul_predictions(y_test, model.predict(X_test), X_test.index, MAX_CYCLES, model_name)
            plot_filename = os.path.join(comparison_dir, f'{file_name}_rul_predictions_{model_name}.png')
            save_plot(fig, plot_filename)
            print(f"Saved plot for {model_name} to: {plot_filename}")
        
        # Convert results to DataFrame and save
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('RMSE')  # Sort by RMSE (best first)
        
        # Save results to CSV in the comparison directory
        results_path = os.path.join(comparison_dir, f'{file_name}_model_comparison_results.csv')
        results_df.to_csv(results_path, index=False)
        
        print("\nBest Model Results:")
        best_model = results_df.iloc[0]
        print(f"Model: {best_model['Model']}")
        print(f"RMSE: {best_model['RMSE']:.3f}")
        print(f"MAE: {best_model['MAE']:.3f}")
        print(f"R2 Score: {best_model['R2']:.3f}")
        print(f"Parameters: {best_model['Parameters']}")
        print(f"\nComparison results saved to: {comparison_dir}")
        
    else:
        # Use only the selected model
        print(f"\nTraining {SELECTED_MODEL} model...")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features_df)
        X_scaled = pd.DataFrame(X_scaled, columns=features_df.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, rul,
            test_size=TRAIN_TEST_SPLIT_RATIO,
            random_state=RANDOM_SEED
        )
        
        # Get and train the selected model
        model = get_model(SELECTED_MODEL)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nModel Performance:")
        print(f"RMSE: {rmse:.3f}")
        print(f"MAE: {mae:.3f}")
        print(f"R2 Score: {r2:.3f}")
        
        # Create and save the plot in the results directory
        fig = plot_rul_predictions(y_test, y_pred, X_test.index, MAX_CYCLES, SELECTED_MODEL)
        plot_filename = os.path.join(file_results_dir, f'{file_name}_rul_predictions_{SELECTED_MODEL}.png')
        save_plot(fig, plot_filename)
        print(f"\nPrediction plot saved as: {plot_filename}")
    
    print(f"\nRUL prediction process completed for {file_path}")

def main():
    """
    Main function to run RUL prediction
    """
    print(f"Maximum cycles: {MAX_CYCLES}")
    print(f"Selected model: {SELECTED_MODEL}")
    print(f"Compare models: {COMPARE_MODELS}")
    print(f"Parameter sweep: {PARAMETER_SWEEP}")
    print("Feature Engineering:")
    for feature, enabled in FEATURE_ENGINEERING.items():
        print(f"  {feature}: {'Enabled' if enabled else 'Disabled'}")
    
    if PROCESS_ALL_EXCEL_FILES:
        # Get all Excel files in the directory
        excel_files = [f for f in os.listdir(EXCEL_DIRECTORY) if f.endswith('.xlsx')]
        if not excel_files:
            print(f"No Excel files found in {EXCEL_DIRECTORY}")
            return
        
        print(f"\nFound {len(excel_files)} Excel files to process:")
        for i, file in enumerate(excel_files, 1):
            print(f"{i}. {file}")
        
        # Process each Excel file
        for excel_file in excel_files:
            file_path = os.path.join(EXCEL_DIRECTORY, excel_file)
            process_excel_file(file_path)
    else:
        # Process single file
        process_excel_file(EXCEL_DATA_PATH)
    
    print("\nAll files processed successfully!")

if __name__ == "__main__":
    main() 