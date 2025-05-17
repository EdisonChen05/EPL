"""
Streamlit App

This module creates a Streamlit web application for the EPL prediction system.
It allows users to view predictions, visualizations, and model results interactively.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
from src.data_collection import FootballDataAPI, UnderstatScraper, FBrefScraper
from src.data_processing import clean_standings_data, clean_understat_data, clean_fbref_data, engineer_features
from src.modeling import predict_champion, load_model
from src.visualization import (
    plot_team_standings, plot_correlation_matrix, plot_feature_importance,
    plot_confusion_matrix, plot_model_comparison, plot_prediction_probabilities,
    plot_interactive_predictions, plot_team_metrics
)


def load_data():
    """Load processed data and models"""
    data = {}
    
    # Load processed data
    try:
        data['features'] = pd.read_csv("data/processed/features_current.csv")
        data['normalized'] = pd.read_csv("data/processed/normalized_features_current.csv")
    except FileNotFoundError:
        st.warning("Processed data not found. Some features may not be available.")
    
    # Load models
    models = {}
    model_files = [f for f in os.listdir("models") if f.endswith(".joblib")]
    
    for model_file in model_files:
        model_name = model_file.split("_")[0]
        try:
            models[model_name] = joblib.load(f"models/{model_file}")
        except:
            st.warning(f"Failed to load model: {model_file}")
    
    data['models'] = models
    
    return data


def make_predictions(data):
    """Make predictions using loaded models"""
    predictions = {}
    
    if 'features' not in data or 'models' not in data:
        return predictions
    
    features_df = data['features']
    models = data['models']
    
    # Get team names
    team_names = features_df['team_name'].tolist() if 'team_name' in features_df.columns else features_df['team'].tolist()
    
    # Get feature columns (excluding non-feature columns)
    non_feature_cols = ['team_name', 'team', 'position', 'champion']
    feature_cols = [col for col in features_df.columns if col not in non_feature_cols]
    
    # Make predictions with each model
    for model_name, model in models.items():
        try:
            pred_df = predict_champion(model, features_df, team_names, feature_cols)
            predictions[model_name] = pred_df
        except Exception as e:
            st.warning(f"Failed to make predictions with {model_name}: {e}")
    
    return predictions


def main():
    """Main function for the Streamlit app"""
    st.set_page_config(
        page_title="EPL Champion Prediction System",
        page_icon="⚽",
        layout="wide"
    )
    
    st.title("⚽ EPL Champion Prediction System")
    st.markdown("""
    This application uses advanced football metrics and machine learning to predict 
    the next English Premier League champion.
    """)
    
    # Load data and models
    data = load_data()
    
    # Make predictions
    predictions = make_predictions(data)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Predictions", "Team Analysis", "Model Insights", "About"])
    
    # Tab 1: Predictions
    with tab1:
        st.header("Championship Predictions")
        
        if predictions:
            # Create a DataFrame with predictions from all models
            dfs = []
            for model_name, pred_df in predictions.items():
                df = pred_df.copy()
                df['model'] = model_name
                dfs.append(df)
            
            combined_df = pd.concat(dfs)
            
            # Create an interactive plot
            fig = px.bar(combined_df, x='champion_probability', y='team', color='model',
                        barmode='group', orientation='h',
                        title='EPL Champion Prediction Probabilities by Model',
                        labels={'champion_probability': 'Probability of Winning EPL', 'team': 'Team', 'model': 'Model'},
                        height=600)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show top 3 teams for each model
            st.subheader("Top 3 Predicted Champions by Model")
            
            cols = st.columns(len(predictions))
            for i, (model_name, pred_df) in enumerate(predictions.items()):
                with cols[i]:
                    st.markdown(f"**{model_name.title()}**")
                    top3 = pred_df.head(3)
                    for j, row in top3.iterrows():
                        st.markdown(f"{j+1}. {row['team']} ({row['champion_probability']:.4f})")
        else:
            st.info("No predictions available. Please run the model training first.")
    
    # Tab 2: Team Analysis
    with tab2:
        st.header("Team Analysis")
        
        if 'features' in data:
            features_df = data['features']
            
            # Team selection
            teams = features_df['team_name'].tolist() if 'team_name' in features_df.columns else features_df['team'].tolist()
            selected_team = st.selectbox("Select a team to analyze:", teams)
            
            # Display team stats
            st.subheader(f"{selected_team} Statistics")
            
            team_data = features_df[features_df['team_name'] == selected_team] if 'team_name' in features_df.columns else features_df[features_df['team'] == selected_team]
            
            # Create columns for stats
            cols = st.columns(4)
            
            # Basic stats
            if 'position' in team_data.columns:
                cols[0].metric("Position", int(team_data['position'].values[0]))
            
            if 'points' in team_data.columns:
                cols[1].metric("Points", int(team_data['points'].values[0]))
            
            if 'played' in team_data.columns and 'won' in team_data.columns:
                cols[2].metric("Win %", f"{(team_data['won'].values[0] / team_data['played'].values[0] * 100):.1f}%")
            
            if 'goals_for' in team_data.columns and 'goals_against' in team_data.columns:
                cols[3].metric("Goal Difference", int(team_data['goals_for'].values[0] - team_data['goals_against'].values[0]))
            
            # Advanced stats
            st.subheader("Advanced Metrics")
            
            # Select metrics to display
            all_metrics = [col for col in team_data.columns if col not in ['team_name', 'team', 'position', 'champion']]
            default_metrics = ['xG', 'xGA', 'xG_diff', 'possession', 'points_per_game']
            default_metrics = [m for m in default_metrics if m in all_metrics]
            
            selected_metrics = st.multiselect(
                "Select metrics to display:",
                all_metrics,
                default=default_metrics
            )
            
            if selected_metrics:
                # Create a radar chart for the selected metrics
                team_metrics = team_data[selected_metrics].values[0]
                
                # Normalize metrics for radar chart
                max_values = features_df[selected_metrics].max()
                normalized_metrics = team_metrics / max_values
                
                # Create radar chart
                fig = px.line_polar(
                    r=normalized_metrics,
                    theta=selected_metrics,
                    line_close=True,
                    range_r=[0, 1],
                    title=f"{selected_team} Performance Profile"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show actual values
                st.subheader("Metric Values")
                metric_df = pd.DataFrame({
                    'Metric': selected_metrics,
                    'Value': team_metrics,
                    'League Max': max_values.values,
                    'League Avg': features_df[selected_metrics].mean().values
                })
                
                st.dataframe(metric_df, hide_index=True)
            
            # Team comparison
            st.subheader("Compare with Other Teams")
            
            compare_teams = st.multiselect(
                "Select teams to compare with:",
                [t for t in teams if t != selected_team],
                default=teams[:3] if selected_team not in teams[:3] else [teams[3]]
            )
            
            if compare_teams:
                compare_teams = [selected_team] + compare_teams
                compare_df = features_df[features_df['team_name'].isin(compare_teams)] if 'team_name' in features_df.columns else features_df[features_df['team'].isin(compare_teams)]
                
                # Select metrics for comparison
                if not selected_metrics:
                    selected_metrics = default_metrics
                
                # Create comparison chart
                for metric in selected_metrics:
                    if metric in compare_df.columns:
                        fig = px.bar(
                            compare_df,
                            x='team_name' if 'team_name' in compare_df.columns else 'team',
                            y=metric,
                            title=f"{metric} Comparison",
                            color='team_name' if 'team_name' in compare_df.columns else 'team'
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No team data available. Please run the data processing first.")
    
    # Tab 3: Model Insights
    with tab3:
        st.header("Model Insights")
        
        if 'features' in data and 'models' in data:
            # Feature correlation
            st.subheader("Feature Correlation")
            
            features_df = data['features']
            
            # Select only numeric columns
            numeric_df = features_df.select_dtypes(include=[np.number])
            
            # Calculate correlation matrix
            corr_matrix = numeric_df.corr()
            
            # Create a heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                        square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
            plt.title('Feature Correlation Matrix')
            st.pyplot(fig)
            
            # Feature importance
            st.subheader("Feature Importance")
            
            # Model selection for feature importance
            model_names = list(data['models'].keys())
            if model_names:
                selected_model = st.selectbox("Select a model:", model_names)
                
                try:
                    model = data['models'][selected_model]
                    
                    # Get feature names
                    non_feature_cols = ['team_name', 'team', 'position', 'champion']
                    feature_cols = [col for col in features_df.columns if col not in non_feature_cols]
                    
                    # Get feature importances
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                    elif hasattr(model, 'coef_'):
                        importances = np.abs(model.coef_[0])
                    else:
                        st.warning("Feature importance not available for this model")
                        importances = None
                    
                    if importances is not None:
                        # Create a DataFrame with feature names and importances
                        importance_df = pd.DataFrame({
                            'feature': feature_cols[:len(importances)],
                            'importance': importances
                        })
                        
                        # Sort by importance
                        importance_df = importance_df.sort_values('importance', ascending=False)
                        
                        # Plot feature importance
                        fig = px.bar(
                            importance_df.head(15),
                            x='importance',
                            y='feature',
                            orientation='h',
                            title=f"Top 15 Feature Importances ({selected_model})"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying feature importance: {e}")
            else:
                st.info("No models available. Please run the model training first.")
        else:
            st.info("No model data available. Please run the model training first.")
    
    # Tab 4: About
    with tab4:
        st.header("About the EPL Champion Prediction System")
        
        st.markdown("""
        ### Project Overview
        
        This project uses advanced football metrics and machine learning to forecast the next English Premier League champion.
        
        ### Data Sources
        
        - **Football-Data.org API**: Official match results, standings, and basic team statistics
        - **Understat**: Expected goals (xG) and other advanced metrics
        - **FBref**: Detailed team and player performance statistics
        
        ### Features Used
        
        The model uses a variety of features including:
        
        - Team statistics (points, wins, draws, losses)
        - Expected Goals (xG) and Expected Goals Against (xGA)
        - Possession percentage
        - Shots per 90 minutes
        - Pass completion rate
        - Team form over last 5 matches
        - xG differential
        - Strength of schedule
        
        ### Models
        
        The system uses multiple machine learning models:
        
        - **Random Forest Classifier**: Ensemble learning method that builds multiple decision trees
        - **XGBoost**: Gradient boosting algorithm known for its performance and speed
        - **Logistic Regression**: Simple but interpretable model for binary classification
        
        ### How to Use
        
        1. View the current predictions in the "Predictions" tab
        2. Analyze individual team performance in the "Team Analysis" tab
        3. Explore model insights and feature importance in the "Model Insights" tab
        
        ### Project Repository
        
        The code for this project is available on GitHub: [EPL Prediction System](https://github.com/yourusername/epl_prediction_system)
        """)


if __name__ == "__main__":
    main()