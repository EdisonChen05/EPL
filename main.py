"""
Main Module

This is the main entry point for the EPL prediction system.
It orchestrates the data collection, processing, modeling, and visualization steps.
"""

import os
import pandas as pd
import argparse
from src.data_collection import collect_and_save_data
from src.data_processing import process_data
from src.modeling import train_and_evaluate_models, load_model, predict_champion
from src.visualization import create_visualizations, plot_prediction_probabilities


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='EPL Champion Prediction System')
    
    parser.add_argument('--collect-data', action='store_true',
                        help='Collect data from APIs and web scraping')
    
    parser.add_argument('--process-data', action='store_true',
                        help='Process and clean the collected data')
    
    parser.add_argument('--train-models', action='store_true',
                        help='Train and evaluate prediction models')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualizations')
    
    parser.add_argument('--predict', action='store_true',
                        help='Make predictions for the current season')
    
    parser.add_argument('--season', type=str, default=None,
                        help='Season to analyze (format: YYYY-YYYY, e.g., 2023-2024)')
    
    parser.add_argument('--all', action='store_true',
                        help='Run all steps (collect, process, train, visualize, predict)')
    
    return parser.parse_args()


def main():
    """Main function to run the EPL prediction system"""
    args = parse_arguments()
    
    # If --all is specified, run all steps
    if args.all:
        args.collect_data = True
        args.process_data = True
        args.train_models = True
        args.visualize = True
        args.predict = True
    
    # Step 1: Collect data
    if args.collect_data:
        print("\n=== Step 1: Collecting Data ===")
        collect_and_save_data(args.season)
    
    # Step 2: Process data
    if args.process_data:
        print("\n=== Step 2: Processing Data ===")
        features_df, normalized_df, scaler = process_data(args.season)
    else:
        # Load processed data if not processing
        season_str = args.season or "current"
        try:
            features_df = pd.read_csv(f"data/processed/features_{season_str}.csv")
            normalized_df = pd.read_csv(f"data/processed/normalized_features_{season_str}.csv")
            print(f"Loaded processed data for {season_str} season")
        except FileNotFoundError:
            print(f"Processed data for {season_str} season not found. Run with --process-data first.")
            features_df = None
            normalized_df = None
    
    # Step 3: Train and evaluate models
    models = {}
    metrics = {}
    predictions = {}
    if args.train_models:
        if features_df is not None:
            print("\n=== Step 3: Training and Evaluating Models ===")
            models, metrics, predictions = train_and_evaluate_models(features_df, features_df, args.season)
        else:
            print("Cannot train models: processed data not available")
    
    # Step 4: Make predictions
    if args.predict and not args.train_models:  # Skip if already predicted during training
        if features_df is not None:
            print("\n=== Step 4: Making Predictions ===")
            
            # Load models if not trained in this run
            if not models:
                model_names = ['random_forest', 'xgboost', 'logistic_regression']
                for model_name in model_names:
                    model = load_model(model_name, args.season)
                    if model is not None:
                        models[model_name] = model
            
            # Make predictions with each model
            for model_name, model in models.items():
                # Get team names
                team_names = features_df['team_name'].tolist() if 'team_name' in features_df.columns else features_df['team'].tolist()
                
                # Get feature columns (excluding non-feature columns)
                non_feature_cols = ['team_name', 'team', 'position', 'champion']
                feature_cols = [col for col in features_df.columns if col not in non_feature_cols]
                
                # Make predictions
                pred_df = predict_champion(model, features_df, team_names, feature_cols)
                predictions[model_name] = pred_df
                
                # Print predictions
                print(f"\nPredictions from {model_name}:")
                print(pred_df.head(5))  # Show top 5 teams
        else:
            print("Cannot make predictions: processed data not available")
    
    # Step 5: Create visualizations
    if args.visualize:
        if features_df is not None:
            print("\n=== Step 5: Creating Visualizations ===")
            create_visualizations(features_df, metrics, predictions)
        else:
            print("Cannot create visualizations: processed data not available")
    
    print("\nEPL Champion Prediction System completed successfully!")


if __name__ == "__main__":
    main()