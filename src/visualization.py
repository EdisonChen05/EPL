"""
Visualization Module

This module handles the creation of visualizations for the EPL prediction system:
1. Data exploration visualizations
2. Model evaluation visualizations
3. Prediction visualizations
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go


def create_output_dir():
    """Create the visualizations directory if it doesn't exist"""
    os.makedirs("../visualizations", exist_ok=True)


def plot_team_standings(standings_df, season=None, save=True):
    """
    Plot the current standings of teams
    
    Args:
        standings_df (pandas.DataFrame): DataFrame with team standings
        season (str, optional): Season identifier
        save (bool): Whether to save the plot to disk
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if standings_df is None or 'team_name' not in standings_df.columns or 'points' not in standings_df.columns:
        print("Invalid standings data for visualization")
        return None
    
    # Sort by points in descending order
    df = standings_df.sort_values('points', ascending=False).reset_index(drop=True)
    
    # Create the figure
    plt.figure(figsize=(12, 8))
    
    # Create a bar plot
    ax = sns.barplot(x='points', y='team_name', data=df, palette='viridis')
    
    # Add labels and title
    plt.xlabel('Points')
    plt.ylabel('Team')
    season_str = f" ({season})" if season else ""
    plt.title(f'Premier League Standings{season_str}')
    
    # Add point values as text
    for i, v in enumerate(df['points']):
        ax.text(v + 0.5, i, str(v), va='center')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if requested
    if save:
        create_output_dir()
        season_str = season or "current"
        plt.savefig(f"../visualizations/standings_{season_str}.png", dpi=300, bbox_inches='tight')
        print(f"Saved standings plot to ../visualizations/standings_{season_str}.png")
    
    return plt.gcf()


def plot_correlation_matrix(features_df, save=True):
    """
    Plot a correlation matrix of features
    
    Args:
        features_df (pandas.DataFrame): DataFrame with features
        save (bool): Whether to save the plot to disk
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if features_df is None:
        print("Invalid features data for visualization")
        return None
    
    # Select only numeric columns
    numeric_df = features_df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create the figure
    plt.figure(figsize=(14, 12))
    
    # Create a heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    
    # Add title
    plt.title('Feature Correlation Matrix')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if requested
    if save:
        create_output_dir()
        plt.savefig("../visualizations/correlation_matrix.png", dpi=300, bbox_inches='tight')
        print("Saved correlation matrix plot to ../visualizations/correlation_matrix.png")
    
    return plt.gcf()


def plot_feature_importance(importance_df, model_name, save=True):
    """
    Plot feature importance from a model
    
    Args:
        importance_df (pandas.DataFrame): DataFrame with feature importances
        model_name (str): Name of the model
        save (bool): Whether to save the plot to disk
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if importance_df is None or 'feature' not in importance_df.columns or 'importance' not in importance_df.columns:
        print("Invalid feature importance data for visualization")
        return None
    
    # Sort by importance
    df = importance_df.sort_values('importance', ascending=False).head(15)  # Top 15 features
    
    # Create the figure
    plt.figure(figsize=(12, 8))
    
    # Create a bar plot
    ax = sns.barplot(x='importance', y='feature', data=df, palette='viridis')
    
    # Add labels and title
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Feature Importance ({model_name})')
    
    # Add importance values as text
    for i, v in enumerate(df['importance']):
        ax.text(v + 0.01, i, f"{v:.4f}", va='center')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if requested
    if save:
        create_output_dir()
        plt.savefig(f"../visualizations/feature_importance_{model_name.lower().replace(' ', '_')}.png", 
                   dpi=300, bbox_inches='tight')
        print(f"Saved feature importance plot to ../visualizations/feature_importance_{model_name.lower().replace(' ', '_')}.png")
    
    return plt.gcf()


def plot_confusion_matrix(cm, model_name, save=True):
    """
    Plot a confusion matrix
    
    Args:
        cm (numpy.ndarray): Confusion matrix
        model_name (str): Name of the model
        save (bool): Whether to save the plot to disk
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if cm is None:
        print("Invalid confusion matrix data for visualization")
        return None
    
    # Create the figure
    plt.figure(figsize=(8, 6))
    
    # Create a heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', 
                xticklabels=['Not Champion', 'Champion'],
                yticklabels=['Not Champion', 'Champion'])
    
    # Add labels and title
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix ({model_name})')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if requested
    if save:
        create_output_dir()
        plt.savefig(f"../visualizations/confusion_matrix_{model_name.lower().replace(' ', '_')}.png", 
                   dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix plot to ../visualizations/confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    
    return plt.gcf()


def plot_roc_curve(y_test, y_pred_proba, model_name, save=True):
    """
    Plot a ROC curve
    
    Args:
        y_test (numpy.ndarray): True labels
        y_pred_proba (numpy.ndarray): Predicted probabilities
        model_name (str): Name of the model
        save (bool): Whether to save the plot to disk
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if y_test is None or y_pred_proba is None:
        print("Invalid data for ROC curve visualization")
        return None
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create the figure
    plt.figure(figsize=(8, 6))
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Add labels and title
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({model_name})')
    plt.legend(loc="lower right")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if requested
    if save:
        create_output_dir()
        plt.savefig(f"../visualizations/roc_curve_{model_name.lower().replace(' ', '_')}.png", 
                   dpi=300, bbox_inches='tight')
        print(f"Saved ROC curve plot to ../visualizations/roc_curve_{model_name.lower().replace(' ', '_')}.png")
    
    return plt.gcf()


def plot_model_comparison(metrics_dict, save=True):
    """
    Plot a comparison of model performance metrics
    
    Args:
        metrics_dict (dict): Dictionary of model metrics
        save (bool): Whether to save the plot to disk
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if metrics_dict is None or len(metrics_dict) == 0:
        print("Invalid metrics data for visualization")
        return None
    
    # Extract metrics for each model
    models = []
    accuracy = []
    precision = []
    recall = []
    f1 = []
    
    for model_name, metrics in metrics_dict.items():
        models.append(model_name)
        accuracy.append(metrics['accuracy'])
        precision.append(metrics['precision'])
        recall.append(metrics['recall'])
        f1.append(metrics['f1'])
    
    # Create a DataFrame for plotting
    df = pd.DataFrame({
        'Model': models,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })
    
    # Melt the DataFrame for easier plotting
    df_melted = pd.melt(df, id_vars=['Model'], var_name='Metric', value_name='Value')
    
    # Create the figure
    plt.figure(figsize=(12, 8))
    
    # Create a grouped bar plot
    ax = sns.barplot(x='Model', y='Value', hue='Metric', data=df_melted, palette='viridis')
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.legend(title='Metric')
    
    # Add values as text
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{p.get_height():.3f}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if requested
    if save:
        create_output_dir()
        plt.savefig("../visualizations/model_comparison.png", dpi=300, bbox_inches='tight')
        print("Saved model comparison plot to ../visualizations/model_comparison.png")
    
    return plt.gcf()


def plot_prediction_probabilities(predictions_df, model_name, save=True):
    """
    Plot prediction probabilities for teams
    
    Args:
        predictions_df (pandas.DataFrame): DataFrame with prediction probabilities
        model_name (str): Name of the model
        save (bool): Whether to save the plot to disk
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if predictions_df is None or 'team' not in predictions_df.columns or 'champion_probability' not in predictions_df.columns:
        print("Invalid prediction data for visualization")
        return None
    
    # Sort by probability in descending order
    df = predictions_df.sort_values('champion_probability', ascending=False)
    
    # Create the figure
    plt.figure(figsize=(12, 8))
    
    # Create a bar plot
    ax = sns.barplot(x='champion_probability', y='team', data=df, palette='viridis')
    
    # Add labels and title
    plt.xlabel('Probability of Winning EPL')
    plt.ylabel('Team')
    plt.title(f'EPL Champion Prediction Probabilities ({model_name})')
    
    # Add probability values as text
    for i, v in enumerate(df['champion_probability']):
        ax.text(v + 0.01, i, f"{v:.4f}", va='center')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if requested
    if save:
        create_output_dir()
        plt.savefig(f"../visualizations/prediction_probabilities_{model_name.lower().replace(' ', '_')}.png", 
                   dpi=300, bbox_inches='tight')
        print(f"Saved prediction probabilities plot to ../visualizations/prediction_probabilities_{model_name.lower().replace(' ', '_')}.png")
    
    return plt.gcf()


def plot_interactive_predictions(predictions_dict, save=True):
    """
    Create an interactive plot of prediction probabilities using Plotly
    
    Args:
        predictions_dict (dict): Dictionary of model predictions
        save (bool): Whether to save the plot to disk
    
    Returns:
        plotly.graph_objects.Figure: The created figure
    """
    if predictions_dict is None or len(predictions_dict) == 0:
        print("Invalid prediction data for visualization")
        return None
    
    # Create a DataFrame with predictions from all models
    dfs = []
    for model_name, pred_df in predictions_dict.items():
        df = pred_df.copy()
        df['model'] = model_name
        dfs.append(df)
    
    combined_df = pd.concat(dfs)
    
    # Create the figure
    fig = px.bar(combined_df, x='champion_probability', y='team', color='model',
                barmode='group', orientation='h',
                title='EPL Champion Prediction Probabilities by Model',
                labels={'champion_probability': 'Probability of Winning EPL', 'team': 'Team', 'model': 'Model'},
                height=800)
    
    # Update layout
    fig.update_layout(
        xaxis_title='Probability of Winning EPL',
        yaxis_title='Team',
        legend_title='Model',
        font=dict(size=12)
    )
    
    # Save the figure if requested
    if save:
        create_output_dir()
        fig.write_html("../visualizations/interactive_predictions.html")
        print("Saved interactive predictions plot to ../visualizations/interactive_predictions.html")
    
    return fig


def plot_team_metrics(features_df, team_col='team_name', metrics=None, save=True):
    """
    Plot selected metrics for top teams
    
    Args:
        features_df (pandas.DataFrame): DataFrame with team features
        team_col (str): Column name for team names
        metrics (list, optional): List of metrics to plot
        save (bool): Whether to save the plot to disk
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if features_df is None or team_col not in features_df.columns:
        print("Invalid features data for visualization")
        return None
    
    # Default metrics if none provided
    if metrics is None:
        # Try to find common metrics
        possible_metrics = ['xG', 'xGA', 'xG_diff', 'possession', 'points_per_game']
        metrics = [m for m in possible_metrics if m in features_df.columns]
        
        if not metrics:
            print("No valid metrics found in the data")
            return None
    
    # Filter to only include metrics that exist in the data
    metrics = [m for m in metrics if m in features_df.columns]
    
    if not metrics:
        print("None of the specified metrics found in the data")
        return None
    
    # Sort by points or another relevant metric if available
    if 'points' in features_df.columns:
        df = features_df.sort_values('points', ascending=False).head(6)  # Top 6 teams
    elif 'position' in features_df.columns:
        df = features_df.sort_values('position').head(6)  # Top 6 teams
    else:
        df = features_df.head(6)  # Just take the first 6
    
    # Create a figure with subplots
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
    
    # If only one metric, axes is not a list
    if len(metrics) == 1:
        axes = [axes]
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        sns.barplot(x=team_col, y=metric, data=df, ax=ax, palette='viridis')
        ax.set_title(f'{metric} by Team')
        ax.set_xlabel('Team')
        ax.set_ylabel(metric)
        
        # Rotate x-axis labels if needed
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add values as text
        for j, v in enumerate(df[metric]):
            ax.text(j, v + 0.01 * max(df[metric]), f"{v:.2f}", ha='center')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if requested
    if save:
        create_output_dir()
        plt.savefig("../visualizations/team_metrics.png", dpi=300, bbox_inches='tight')
        print("Saved team metrics plot to ../visualizations/team_metrics.png")
    
    return plt.gcf()


def create_visualizations(features_df, metrics_dict=None, predictions_dict=None, importance_dict=None):
    """
    Create all visualizations
    
    Args:
        features_df (pandas.DataFrame): DataFrame with features
        metrics_dict (dict, optional): Dictionary of model metrics
        predictions_dict (dict, optional): Dictionary of model predictions
        importance_dict (dict, optional): Dictionary of feature importances
    """
    create_output_dir()
    
    # Plot team standings
    if features_df is not None and 'team_name' in features_df.columns and 'points' in features_df.columns:
        plot_team_standings(features_df)
    
    # Plot correlation matrix
    if features_df is not None:
        plot_correlation_matrix(features_df)
    
    # Plot team metrics
    if features_df is not None:
        plot_team_metrics(features_df)
    
    # Plot model comparison
    if metrics_dict is not None:
        plot_model_comparison(metrics_dict)
    
    # Plot feature importance for each model
    if importance_dict is not None:
        for model_name, importance_df in importance_dict.items():
            plot_feature_importance(importance_df, model_name)
    
    # Plot confusion matrix for each model
    if metrics_dict is not None:
        for model_name, metrics in metrics_dict.items():
            if 'confusion_matrix' in metrics:
                plot_confusion_matrix(metrics['confusion_matrix'], model_name)
    
    # Plot prediction probabilities for each model
    if predictions_dict is not None:
        for model_name, pred_df in predictions_dict.items():
            plot_prediction_probabilities(pred_df, model_name)
        
        # Create interactive plot with all models
        plot_interactive_predictions(predictions_dict)


if __name__ == "__main__":
    # Load processed data
    try:
        features_df = pd.read_csv("../data/processed/features_current.csv")
        print("Loaded features data")
    except FileNotFoundError:
        print("Features data not found. Please run data_processing.py first.")
        features_df = None
    
    # Create basic visualizations
    if features_df is not None:
        create_visualizations(features_df)