"""
Modeling Module

This module handles the training and evaluation of machine learning models:
1. Model training
2. Hyperparameter tuning
3. Model evaluation
4. Prediction generation
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


def prepare_training_data(features_df, target_col='champion', test_size=0.2, random_state=42):
    """
    Prepare data for training by splitting into features and target
    
    Args:
        features_df (pandas.DataFrame): DataFrame with features
        target_col (str): Name of the target column
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    if features_df is None:
        return None, None, None, None, None
    
    # Make a copy to avoid modifying the original
    df = features_df.copy()
    
    # For historical data, we need to create a 'champion' column
    # In a real implementation, this would be based on the final standings
    # For this example, we'll assume the team in position 1 is the champion
    if target_col not in df.columns and 'position' in df.columns:
        df[target_col] = (df['position'] == 1).astype(int)
    
    # Select features (exclude the target and any non-feature columns)
    non_feature_cols = [target_col, 'team_name', 'team', 'position']
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    
    # Split data into features and target
    X = df[feature_cols]
    y = df[target_col]
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, feature_cols


def train_random_forest(X_train, y_train, param_grid=None):
    """
    Train a Random Forest classifier with hyperparameter tuning
    
    Args:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target
        param_grid (dict, optional): Hyperparameter grid for GridSearchCV
    
    Returns:
        tuple: (trained_model, best_params)
    """
    if X_train is None or y_train is None:
        return None, None
    
    # Default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
    
    # Create the model
    rf = RandomForestClassifier(random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"Random Forest best parameters: {best_params}")
    
    return best_model, best_params


def train_xgboost(X_train, y_train, param_grid=None):
    """
    Train an XGBoost classifier with hyperparameter tuning
    
    Args:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target
        param_grid (dict, optional): Hyperparameter grid for GridSearchCV
    
    Returns:
        tuple: (trained_model, best_params)
    """
    if X_train is None or y_train is None:
        return None, None
    
    # Default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
    
    # Create the model
    xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"XGBoost best parameters: {best_params}")
    
    return best_model, best_params


def train_logistic_regression(X_train, y_train, param_grid=None):
    """
    Train a Logistic Regression classifier with hyperparameter tuning
    
    Args:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target
        param_grid (dict, optional): Hyperparameter grid for GridSearchCV
    
    Returns:
        tuple: (trained_model, best_params)
    """
    if X_train is None or y_train is None:
        return None, None
    
    # Default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter': [100, 200, 300]
        }
    
    # Create the model
    lr = LogisticRegression(random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=lr,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"Logistic Regression best parameters: {best_params}")
    
    return best_model, best_params


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a trained model on the test set
    
    Args:
        model: Trained model
        X_test (pandas.DataFrame): Test features
        y_test (pandas.Series): Test target
        model_name (str): Name of the model for reporting
    
    Returns:
        dict: Dictionary of evaluation metrics
    """
    if model is None or X_test is None or y_test is None:
        return None
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Calculate ROC AUC if possible
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    except:
        roc_auc = None
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print evaluation results
    print(f"\nEvaluation results for {model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    # Return metrics as a dictionary
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }
    
    return metrics


def get_feature_importance(model, feature_names, model_name):
    """
    Get feature importance from a trained model
    
    Args:
        model: Trained model
        feature_names (list): List of feature names
        model_name (str): Name of the model
    
    Returns:
        pandas.DataFrame: DataFrame with feature importances
    """
    if model is None or feature_names is None:
        return None
    
    # Get feature importances (different methods depending on model type)
    if model_name == 'Random Forest' or model_name == 'XGBoost':
        importances = model.feature_importances_
    elif model_name == 'Logistic Regression':
        importances = np.abs(model.coef_[0])
    else:
        print(f"Feature importance not available for {model_name}")
        return None
    
    # Create a DataFrame with feature names and importances
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    
    return feature_importance_df


def predict_champion(model, current_season_data, team_names, feature_cols):
    """
    Predict the champion for the current season
    
    Args:
        model: Trained model
        current_season_data (pandas.DataFrame): Data for the current season
        team_names (list): List of team names
        feature_cols (list): List of feature columns used by the model
    
    Returns:
        pandas.DataFrame: DataFrame with prediction probabilities for each team
    """
    if model is None or current_season_data is None:
        return None
    
    # Make sure we have all the required features
    missing_features = [col for col in feature_cols if col not in current_season_data.columns]
    if missing_features:
        print(f"Missing features in current season data: {missing_features}")
        return None
    
    # Select only the features used by the model
    X = current_season_data[feature_cols]
    
    # Make predictions
    probabilities = model.predict_proba(X)[:, 1]
    
    # Create a DataFrame with team names and probabilities
    predictions_df = pd.DataFrame({
        'team': team_names,
        'champion_probability': probabilities
    })
    
    # Sort by probability in descending order
    predictions_df = predictions_df.sort_values('champion_probability', ascending=False)
    
    return predictions_df


def save_model(model, model_name, season=None):
    """
    Save a trained model to disk
    
    Args:
        model: Trained model
        model_name (str): Name of the model
        season (str, optional): Season the model was trained on
    
    Returns:
        str: Path to the saved model
    """
    if model is None:
        return None
    
    # Create models directory if it doesn't exist
    os.makedirs("../models", exist_ok=True)
    
    # Create a filename
    season_str = season or "current"
    filename = f"../models/{model_name.lower().replace(' ', '_')}_{season_str}.joblib"
    
    # Save the model
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")
    
    return filename


def load_model(model_name, season=None):
    """
    Load a trained model from disk
    
    Args:
        model_name (str): Name of the model
        season (str, optional): Season the model was trained on
    
    Returns:
        object: Loaded model
    """
    # Create a filename
    season_str = season or "current"
    filename = f"../models/{model_name.lower().replace(' ', '_')}_{season_str}.joblib"
    
    # Check if the file exists
    if not os.path.exists(filename):
        print(f"Model file {filename} not found")
        return None
    
    # Load the model
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    
    return model


def train_and_evaluate_models(features_df, current_season_data=None, season=None):
    """
    Train and evaluate multiple models
    
    Args:
        features_df (pandas.DataFrame): DataFrame with features for training
        current_season_data (pandas.DataFrame, optional): Data for the current season
        season (str, optional): Season identifier
    
    Returns:
        tuple: (models_dict, metrics_dict, predictions_dict)
    """
    # Prepare training data
    X_train, X_test, y_train, y_test, feature_cols = prepare_training_data(features_df)
    
    if X_train is None:
        print("Failed to prepare training data")
        return None, None, None
    
    # Dictionary to store trained models
    models = {}
    
    # Dictionary to store evaluation metrics
    metrics = {}
    
    # Dictionary to store predictions
    predictions = {}
    
    # Train Random Forest
    print("\nTraining Random Forest...")
    rf_model, rf_params = train_random_forest(X_train, y_train)
    models['Random Forest'] = rf_model
    
    # Evaluate Random Forest
    rf_metrics = evaluate_model(rf_model, X_test, y_test, 'Random Forest')
    metrics['Random Forest'] = rf_metrics
    
    # Get feature importance for Random Forest
    rf_importance = get_feature_importance(rf_model, feature_cols, 'Random Forest')
    
    # Train XGBoost
    print("\nTraining XGBoost...")
    xgb_model, xgb_params = train_xgboost(X_train, y_train)
    models['XGBoost'] = xgb_model
    
    # Evaluate XGBoost
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, 'XGBoost')
    metrics['XGBoost'] = xgb_metrics
    
    # Get feature importance for XGBoost
    xgb_importance = get_feature_importance(xgb_model, feature_cols, 'XGBoost')
    
    # Train Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_model, lr_params = train_logistic_regression(X_train, y_train)
    models['Logistic Regression'] = lr_model
    
    # Evaluate Logistic Regression
    lr_metrics = evaluate_model(lr_model, X_test, y_test, 'Logistic Regression')
    metrics['Logistic Regression'] = lr_metrics
    
    # Get feature importance for Logistic Regression
    lr_importance = get_feature_importance(lr_model, feature_cols, 'Logistic Regression')
    
    # Save models
    for model_name, model in models.items():
        save_model(model, model_name, season)
    
    # Make predictions for the current season if data is provided
    if current_season_data is not None:
        team_names = current_season_data['team_name'].tolist() if 'team_name' in current_season_data.columns else current_season_data['team'].tolist()
        
        for model_name, model in models.items():
            print(f"\nMaking predictions with {model_name}...")
            pred_df = predict_champion(model, current_season_data, team_names, feature_cols)
            predictions[model_name] = pred_df
    
    return models, metrics, predictions


if __name__ == "__main__":
    # Load processed data
    try:
        features_df = pd.read_csv("../data/processed/features_current.csv")
        print("Loaded features data")
    except FileNotFoundError:
        print("Features data not found. Please run data_processing.py first.")
        features_df = None
    
    # Train and evaluate models
    if features_df is not None:
        models, metrics, predictions = train_and_evaluate_models(features_df)
        
        # Print the best model based on F1 score
        if metrics is not None:
            best_model = max(metrics.items(), key=lambda x: x[1]['f1'])
            print(f"\nBest model based on F1 score: {best_model[0]} with F1 = {best_model[1]['f1']:.4f}")
        
        # Print predictions if available
        if predictions is not None:
            for model_name, pred_df in predictions.items():
                print(f"\nPredictions from {model_name}:")
                print(pred_df.head(5))  # Show top 5 teams