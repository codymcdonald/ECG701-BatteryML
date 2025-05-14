"""
Model factory for creating and configuring ML models.
"""
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from src.config.config import MODEL_PARAMS, SELECTED_MODEL

def get_model():
    """Return the selected model with configured parameters"""
    if SELECTED_MODEL == 'RandomForest':
        return RandomForestRegressor(**MODEL_PARAMS['RandomForest'])
    elif SELECTED_MODEL == 'GradientBoosting':
        return GradientBoostingRegressor(**MODEL_PARAMS['GradientBoosting'])
    elif SELECTED_MODEL == 'AdaBoost':
        return AdaBoostRegressor(**MODEL_PARAMS['AdaBoost'])
    elif SELECTED_MODEL == 'Linear':
        return LinearRegression()
    elif SELECTED_MODEL == 'Ridge':
        return Ridge(**MODEL_PARAMS['Ridge'])
    elif SELECTED_MODEL == 'Lasso':
        return Lasso(**MODEL_PARAMS['Lasso'])
    elif SELECTED_MODEL == 'SVR':
        return SVR(**MODEL_PARAMS['SVR'])
    else:
        print(f"Warning: Unknown model '{SELECTED_MODEL}', using RandomForest")
        return RandomForestRegressor(**MODEL_PARAMS['RandomForest']) 