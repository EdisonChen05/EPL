"""
Data Collection Module

This module handles the collection of EPL data from various sources:
1. Football-Data.org API
2. Understat (web scraping)
3. FBref (web scraping)
"""

import os
import json
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class FootballDataAPI:
    """Class to interact with the Football-Data.org API"""
    
    def __init__(self):
        self.base_url = "https://api.football-data.org/v4"
        self.api_key = os.getenv("FOOTBALL_DATA_API_KEY")
        self.headers = {"X-Auth-Token": self.api_key}
        
    def get_premier_league_standings(self, season=None):
        """
        Get the current Premier League standings
        
        Args:
            season (str, optional): Season in format YYYY-YYYY (e.g., "2023-2024")
                                   If None, gets current season
        
        Returns:
            pandas.DataFrame: Premier League standings
        """
        endpoint = f"{self.base_url}/competitions/PL/standings"
        if season:
            endpoint += f"?season={season}"
            
        response = requests.get(endpoint, headers=self.headers)
        
        if response.status_code != 200:
            print(f"Error fetching data: {response.status_code}")
            print(response.text)
            return None
        
        data = response.json()
        
        # Extract standings data
        standings_data = []
        for team in data["standings"][0]["table"]:
            team_data = {
                "position": team["position"],
                "team_name": team["team"]["name"],
                "played": team["playedGames"],
                "won": team["won"],
                "drawn": team["draw"],
                "lost": team["lost"],
                "points": team["points"],
                "goals_for": team["goalsFor"],
                "goals_against": team["goalsAgainst"],
                "goal_difference": team["goalDifference"]
            }
            standings_data.append(team_data)
        
        return pd.DataFrame(standings_data)
    
    def get_premier_league_matches(self, season=None):
        """
        Get Premier League matches
        
        Args:
            season (str, optional): Season in format YYYY-YYYY (e.g., "2023-2024")
                                   If None, gets current season
        
        Returns:
            pandas.DataFrame: Premier League matches
        """
        endpoint = f"{self.base_url}/competitions/PL/matches"
        if season:
            endpoint += f"?season={season}"
            
        response = requests.get(endpoint, headers=self.headers)
        
        if response.status_code != 200:
            print(f"Error fetching data: {response.status_code}")
            print(response.text)
            return None
        
        data = response.json()
        
        # Extract match data
        matches_data = []
        for match in data["matches"]:
            match_data = {
                "match_date": match["utcDate"],
                "status": match["status"],
                "matchday": match["matchday"],
                "home_team": match["homeTeam"]["name"],
                "away_team": match["awayTeam"]["name"],
                "home_score": match["score"]["fullTime"]["home"] if match["score"]["fullTime"]["home"] is not None else None,
                "away_score": match["score"]["fullTime"]["away"] if match["score"]["fullTime"]["away"] is not None else None,
                "winner": match["score"]["winner"]
            }
            matches_data.append(match_data)
        
        return pd.DataFrame(matches_data)


class UnderstatScraper:
    """Class to scrape data from Understat"""
    
    def __init__(self):
        self.base_url = "https://understat.com/league/EPL"
        
    def get_team_stats(self, season=None):
        """
        Scrape team statistics from Understat
        
        Args:
            season (str, optional): Season in format YYYY (e.g., "2023")
                                   If None, gets current season
        
        Returns:
            pandas.DataFrame: Team statistics including xG, xGA, etc.
        """
        url = self.base_url
        if season:
            url += f"/{season}"
            
        # Note: In a real implementation, you would use requests and BeautifulSoup
        # to scrape the data. This is a simplified example.
        
        # Simulated data for demonstration purposes
        teams = ["Manchester City", "Arsenal", "Liverpool", "Manchester United", 
                "Newcastle", "Brighton", "Aston Villa", "Tottenham", "Brentford", 
                "Fulham", "Crystal Palace", "Chelsea", "Wolves", "West Ham", 
                "Bournemouth", "Nottingham Forest", "Everton", "Leicester", 
                "Leeds", "Southampton"]
        
        data = []
        for i, team in enumerate(teams):
            # Generate some realistic but random data
            xg = round(50 + (20 - i) * 1.2 + ((-5 + i) * 0.2), 2)
            xga = round(20 + i * 1.5 + ((-10 + i) * 0.3), 2)
            
            team_data = {
                "team": team,
                "xG": xg,
                "xGA": xga,
                "xG_diff": round(xg - xga, 2),
                "xPoints": round(60 + (20 - i) * 1.8 + ((-5 + i) * 0.5), 2),
                "ppg": round(2.5 - (i * 0.1), 2),
                "deep": 250 - (i * 10),
                "deep_allowed": 150 + (i * 8),
                "xG_per_shot": round(0.12 - (i * 0.002), 3),
                "xGA_per_shot": round(0.09 + (i * 0.002), 3)
            }
            data.append(team_data)
            
        return pd.DataFrame(data)


class FBrefScraper:
    """Class to scrape data from FBref"""
    
    def __init__(self):
        self.base_url = "https://fbref.com/en/comps/9/Premier-League-Stats"
        
    def get_advanced_team_stats(self, season=None):
        """
        Scrape advanced team statistics from FBref
        
        Args:
            season (str, optional): Season in format YYYY-YYYY (e.g., "2023-2024")
                                   If None, gets current season
        
        Returns:
            pandas.DataFrame: Advanced team statistics
        """
        url = self.base_url
        if season:
            # Format would be different for FBref, this is simplified
            url += f"/{season}"
            
        # Note: In a real implementation, you would use requests and BeautifulSoup
        # to scrape the data. This is a simplified example.
        
        # Simulated data for demonstration purposes
        teams = ["Manchester City", "Arsenal", "Liverpool", "Manchester United", 
                "Newcastle", "Brighton", "Aston Villa", "Tottenham", "Brentford", 
                "Fulham", "Crystal Palace", "Chelsea", "Wolves", "West Ham", 
                "Bournemouth", "Nottingham Forest", "Everton", "Leicester", 
                "Leeds", "Southampton"]
        
        data = []
        for i, team in enumerate(teams):
            # Generate some realistic but random data
            team_data = {
                "team": team,
                "possession": round(65 - (i * 1.5), 1),
                "passes_completed": 15000 - (i * 500),
                "passes_attempted": 17000 - (i * 450),
                "pass_completion_pct": round(88 - (i * 1.2), 1),
                "progressive_passes": 1200 - (i * 40),
                "progressive_carries": 900 - (i * 35),
                "successful_pressures": 1800 - (i * 60),
                "tackles": 600 + (i * 5),
                "interceptions": 300 + (i * 8),
                "blocks": 200 + (i * 10),
                "clearances": 500 + (i * 15),
                "aerials_won": 600 + (i * 10)
            }
            data.append(team_data)
            
        return pd.DataFrame(data)


def collect_and_save_data(season=None):
    """
    Collect data from all sources and save to CSV files
    
    Args:
        season (str, optional): Season to collect data for
    """
    # Create data directory if it doesn't exist
    os.makedirs("../data/raw", exist_ok=True)
    
    # Collect data from Football-Data.org API
    try:
        football_data_api = FootballDataAPI()
        standings = football_data_api.get_premier_league_standings(season)
        matches = football_data_api.get_premier_league_matches(season)
        
        if standings is not None:
            standings.to_csv(f"../data/raw/standings_{season or 'current'}.csv", index=False)
            print(f"Saved standings data to ../data/raw/standings_{season or 'current'}.csv")
        
        if matches is not None:
            matches.to_csv(f"../data/raw/matches_{season or 'current'}.csv", index=False)
            print(f"Saved matches data to ../data/raw/matches_{season or 'current'}.csv")
    except Exception as e:
        print(f"Error collecting data from Football-Data.org API: {e}")
    
    # Collect data from Understat
    try:
        understat_scraper = UnderstatScraper()
        understat_stats = understat_scraper.get_team_stats(season)
        
        if understat_stats is not None:
            understat_stats.to_csv(f"../data/raw/understat_stats_{season or 'current'}.csv", index=False)
            print(f"Saved Understat data to ../data/raw/understat_stats_{season or 'current'}.csv")
    except Exception as e:
        print(f"Error collecting data from Understat: {e}")
    
    # Collect data from FBref
    try:
        fbref_scraper = FBrefScraper()
        fbref_stats = fbref_scraper.get_advanced_team_stats(season)
        
        if fbref_stats is not None:
            fbref_stats.to_csv(f"../data/raw/fbref_stats_{season or 'current'}.csv", index=False)
            print(f"Saved FBref data to ../data/raw/fbref_stats_{season or 'current'}.csv")
    except Exception as e:
        print(f"Error collecting data from FBref: {e}")


if __name__ == "__main__":
    # If API key is not set, print a message
    if os.getenv("FOOTBALL_DATA_API_KEY") is None:
        print("Warning: FOOTBALL_DATA_API_KEY environment variable is not set.")
        print("You can get a free API key from https://www.football-data.org/")
        print("Set it in a .env file in the project root directory.")
    
    # Collect data for the current season
    collect_and_save_data()
    
    # Optionally collect data for previous seasons
    # collect_and_save_data("2022-2023")
    # collect_and_save_data("2021-2022")
