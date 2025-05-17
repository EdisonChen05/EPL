"""
Data Processing Module

This module handles the cleaning and preprocessing of EPL data:
1. Data cleaning (handling missing values, outliers)
2. Feature engineering
3. Data transformation (normalization, encoding)
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_raw_data(season=None):
    """
    Load raw data from CSV files
    
    Args:
        season (str, optional): Season in format YYYY-YYYY (e.g., "2023-2024")
                               If None, loads current season
    
    Returns:
        tuple: (standings_df, matches_df, understat_df, fbref_df)
    """
    season_str = season or "current"
    data_dir = "../data/raw"
    
    # Load standings data
    try:
        standings_df = pd.read_csv(f"{data_dir}/standings_{season_str}.csv")
    except FileNotFoundError:
        print(f"Standings data for {season_str} not found.")
        standings_df = None
    
    # Load matches data
    try:
        matches_df = pd.read_csv(f"{data_dir}/matches_{season_str}.csv")
    except FileNotFoundError:
        print(f"Matches data for {season_str} not found.")
        matches_df = None
    
    # Load Understat data
    try:
        understat_df = pd.read_csv(f"{data_dir}/understat_stats_{season_str}.csv")
    except FileNotFoundError:
        print(f"Understat data for {season_str} not found.")
        understat_df = None
    
    # Load FBref data
    try:
        fbref_df = pd.read_csv(f"{data_dir}/fbref_stats_{season_str}.csv")
    except FileNotFoundError:
        print(f"FBref data for {season_str} not found.")
        fbref_df = None
    
    return standings_df, matches_df, understat_df, fbref_df


def clean_standings_data(standings_df):
    """
    Clean the standings data
    
    Args:
        standings_df (pandas.DataFrame): Raw standings data
    
    Returns:
        pandas.DataFrame: Cleaned standings data
    """
    if standings_df is None:
        return None
    
    # Make a copy to avoid modifying the original
    df = standings_df.copy()
    
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        print(f"Missing values in standings data: {df.isnull().sum()}")
        # Fill missing values or drop rows as appropriate
        df = df.dropna()
    
    # Ensure numeric columns are of the right type
    numeric_cols = ['position', 'played', 'won', 'drawn', 'lost', 
                    'points', 'goals_for', 'goals_against', 'goal_difference']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Standardize team names (if needed)
    # This would depend on how team names are formatted in different data sources
    
    return df


def clean_matches_data(matches_df):
    """
    Clean the matches data
    
    Args:
        matches_df (pandas.DataFrame): Raw matches data
    
    Returns:
        pandas.DataFrame: Cleaned matches data
    """
    if matches_df is None:
        return None
    
    # Make a copy to avoid modifying the original
    df = matches_df.copy()
    
    # Convert date to datetime
    if 'match_date' in df.columns:
        df['match_date'] = pd.to_datetime(df['match_date'])
    
    # Handle missing scores for matches that haven't been played yet
    if 'home_score' in df.columns and 'away_score' in df.columns:
        # Keep track of which matches have been played
        df['played'] = (~df['home_score'].isna()) & (~df['away_score'].isna())
    
    # Ensure numeric columns are of the right type
    numeric_cols = ['matchday', 'home_score', 'away_score']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def clean_understat_data(understat_df):
    """
    Clean the Understat data
    
    Args:
        understat_df (pandas.DataFrame): Raw Understat data
    
    Returns:
        pandas.DataFrame: Cleaned Understat data
    """
    if understat_df is None:
        return None
    
    # Make a copy to avoid modifying the original
    df = understat_df.copy()
    
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        print(f"Missing values in Understat data: {df.isnull().sum()}")
        # Fill missing values with appropriate methods
        # For numeric columns, we might use mean or median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
    
    # Standardize team names (if needed)
    # This would depend on how team names are formatted in different data sources
    
    return df


def clean_fbref_data(fbref_df):
    """
    Clean the FBref data
    
    Args:
        fbref_df (pandas.DataFrame): Raw FBref data
    
    Returns:
        pandas.DataFrame: Cleaned FBref data
    """
    if fbref_df is None:
        return None
    
    # Make a copy to avoid modifying the original
    df = fbref_df.copy()
    
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        print(f"Missing values in FBref data: {df.isnull().sum()}")
        # Fill missing values with appropriate methods
        # For numeric columns, we might use mean or median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
    
    # Standardize team names (if needed)
    # This would depend on how team names are formatted in different data sources
    
    return df


def engineer_features(standings_df, matches_df, understat_df, fbref_df):
    """
    Engineer features from the cleaned data
    
    Args:
        standings_df (pandas.DataFrame): Cleaned standings data
        matches_df (pandas.DataFrame): Cleaned matches data
        understat_df (pandas.DataFrame): Cleaned Understat data
        fbref_df (pandas.DataFrame): Cleaned FBref data
    
    Returns:
        pandas.DataFrame: DataFrame with engineered features
    """
    # Start with the standings data as our base
    if standings_df is None:
        print("Cannot engineer features: standings data is missing")
        return None
    
    # Make a copy to avoid modifying the original
    features_df = standings_df.copy()
    
    # Add win percentage
    if 'played' in features_df.columns and 'won' in features_df.columns:
        features_df['win_pct'] = features_df['won'] / features_df['played']
    
    # Add points per game
    if 'played' in features_df.columns and 'points' in features_df.columns:
        features_df['points_per_game'] = features_df['points'] / features_df['played']
    
    # Add goal-related metrics
    if 'goals_for' in features_df.columns and 'goals_against' in features_df.columns and 'played' in features_df.columns:
        features_df['goals_for_per_game'] = features_df['goals_for'] / features_df['played']
        features_df['goals_against_per_game'] = features_df['goals_against'] / features_df['played']
        features_df['goal_difference_per_game'] = features_df['goal_difference'] / features_df['played']
    
    # Merge with Understat data if available
    if understat_df is not None:
        # Ensure team names are consistent before merging
        # This might require some preprocessing
        features_df = pd.merge(features_df, understat_df, 
                              left_on='team_name', right_on='team', 
                              how='left')
        
        # Add xG-related features
        if 'xG' in features_df.columns and 'xGA' in features_df.columns:
            features_df['xG_per_game'] = features_df['xG'] / features_df['played']
            features_df['xGA_per_game'] = features_df['xGA'] / features_df['played']
            
            # Add over/under performance metrics
            if 'goals_for' in features_df.columns and 'goals_against' in features_df.columns:
                features_df['goals_vs_xG'] = features_df['goals_for'] - features_df['xG']
                features_df['goals_against_vs_xGA'] = features_df['goals_against'] - features_df['xGA']
    
    # Merge with FBref data if available
    if fbref_df is not None:
        # Ensure team names are consistent before merging
        # This might require some preprocessing
        features_df = pd.merge(features_df, fbref_df, 
                              left_on='team_name', right_on='team', 
                              how='left')
    
    # Calculate form over last 5 matches if matches data is available
    if matches_df is not None and 'played' in matches_df.columns:
        # This would be more complex in a real implementation
        # We would need to sort matches by date and calculate rolling metrics
        pass
    
    # Drop duplicate columns that might have been created during merges
    if 'team_x' in features_df.columns and 'team_y' in features_df.columns:
        features_df = features_df.drop(columns=['team_x', 'team_y'])
    
    # Drop any columns that are no longer needed
    # This would depend on the specific requirements
    
    return features_df


def normalize_features(features_df):
    """
    Normalize features to have mean 0 and standard deviation 1
    
    Args:
        features_df (pandas.DataFrame): DataFrame with engineered features
    
    Returns:
        pandas.DataFrame: DataFrame with normalized features
    """
    if features_df is None:
        return None
    
    # Make a copy to avoid modifying the original
    df = features_df.copy()
    
    # Select only numeric columns for normalization
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove any columns we don't want to normalize
    # For example, we might want to keep 'position' as is
    if 'position' in numeric_cols:
        numeric_cols.remove('position')
    
    # Create a scaler
    scaler = StandardScaler()
    
    # Fit and transform the numeric columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df, scaler


def process_data(season=None):
    """
    Process data for the given season
    
    Args:
        season (str, optional): Season in format YYYY-YYYY (e.g., "2023-2024")
                               If None, processes current season
    
    Returns:
        tuple: (features_df, normalized_df, scaler)
    """
    # Create processed data directory if it doesn't exist
    os.makedirs("../data/processed", exist_ok=True)
    
    # Load raw data
    standings_df, matches_df, understat_df, fbref_df = load_raw_data(season)
    
    # Clean data
    cleaned_standings = clean_standings_data(standings_df)
    cleaned_matches = clean_matches_data(matches_df)
    cleaned_understat = clean_understat_data(understat_df)
    cleaned_fbref = clean_fbref_data(fbref_df)
    
    # Engineer features
    features_df = engineer_features(cleaned_standings, cleaned_matches, 
                                   cleaned_understat, cleaned_fbref)
    
    # Normalize features
    normalized_df, scaler = normalize_features(features_df)
    
    # Save processed data
    season_str = season or "current"
    if features_df is not None:
        features_df.to_csv(f"../data/processed/features_{season_str}.csv", index=False)
        print(f"Saved features to ../data/processed/features_{season_str}.csv")
    
    if normalized_df is not None:
        normalized_df.to_csv(f"../data/processed/normalized_features_{season_str}.csv", index=False)
        print(f"Saved normalized features to ../data/processed/normalized_features_{season_str}.csv")
    
    return features_df, normalized_df, scaler


if __name__ == "__main__":
    # Process data for the current season
    process_data()
    
    # Optionally process data for previous seasons
    # process_data("2022-2023")
    # process_data("2021-2022")