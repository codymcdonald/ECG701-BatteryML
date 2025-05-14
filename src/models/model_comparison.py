"""
Model comparison utilities for testing different models and parameters.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from itertools import product
from src.config.config import MODEL_PARAMS, TRAIN_TEST_SPLIT_RATIO, RANDOM_SEED
from src.models.model_factory import get_model

def generate_parameter_grids():
    """
    Generate parameter grids for different models
    Returns:
        dict: Dictionary of parameter grids for each model
    """
    param_grids = {
        'RandomForest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
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
            'C': [0.1, 1.0, 10.0],
            'epsilon': [0.1, 0.2],
            'gamma': ['scale', 'auto']
        }
    }
    return param_grids

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate a model's performance using multiple metrics
    Args:
        model: Trained model
        X_train, X_test: Training and test features
        y_train, y_test: Training and test targets
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred),
        'CV_Score': np.mean(cross_val_score(model, X_train, y_train, cv=5))
    }
    
    return metrics

def compare_models(X_train, X_test, y_train, y_test, feature_names):
    """
    Compare different models and their parameters
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test targets
        feature_names: List of feature names
    Returns:
        pd.DataFrame: DataFrame containing comparison results
    """
    param_grids = generate_parameter_grids()
    results = []
    
    for model_name, param_grid in param_grids.items():
        print(f"\nTesting {model_name}...")
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        for params in param_combinations:
            # Create parameter dictionary
            param_dict = dict(zip(param_names, params))
            
            # Get and configure model
            model = get_model()
            model.set_params(**param_dict)
            
            # Train and evaluate model
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
            
            # Get feature importances if available
            feature_importances = {}
            if hasattr(model, 'feature_importances_'):
                for name, importance in zip(feature_names, model.feature_importances_):
                    feature_importances[f'importance_{name}'] = importance
            
            # Store results
            result = {
                'Model': model_name,
                **param_dict,
                **metrics,
                **feature_importances
            }
            results.append(result)
            
            print(f"Parameters: {param_dict}")
            print(f"RMSE: {metrics['RMSE']:.3f}, R2: {metrics['R2']:.3f}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by RMSE (best performing first)
    results_df = results_df.sort_values('RMSE')
    
    # Save results to CSV
    results_df.to_csv('plots/model_comparison_results.csv', index=False)
    print("\nResults saved to 'plots/model_comparison_results.csv'")
    
    return results_df

def plot_model_comparison(results_df):
    """
    Create comparison plots for different models
    Args:
        results_df: DataFrame containing model comparison results
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create plots directory if it doesn't exist
    from pathlib import Path
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Plot RMSE comparison
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Model', y='RMSE', data=results_df)
    plt.title('RMSE Comparison Across Models')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/model_rmse_comparison.png')
    plt.close()
    
    # Plot R2 comparison
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Model', y='R2', data=results_df)
    plt.title('R2 Score Comparison Across Models')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/model_r2_comparison.png')
    plt.close()
    
    # Plot CV Score comparison
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Model', y='CV_Score', data=results_df)
    plt.title('Cross-Validation Score Comparison Across Models')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/model_cv_comparison.png')
    plt.close() 