"""
Premier League Data Collection Module
Fetches current season data including standings, fixtures, and team statistics
"""

import requests
import pandas as pd
from datetime import datetime
import json
from typing import Dict, List, Optional

class PremierLeagueDataCollector:
    """Collects and processes Premier League data from various sources"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize data collector
        
        Args:
            api_key: API key for football-data.org (optional, can use free tier)
        """
        self.api_key = api_key or "YOUR_API_KEY_HERE"
        self.base_url = "https://api.football-data.org/v4"
        self.headers = {'X-Auth-Token': self.api_key}
        self.competition_id = 'PL'  # Premier League
        
    def get_current_standings(self) -> pd.DataFrame:
        """Fetch current Premier League standings"""
        try:
            url = f"{self.base_url}/competitions/{self.competition_id}/standings"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                standings_data = []
                
                for team in data['standings'][0]['table']:
                    standings_data.append({
                        'team': team['team']['name'],
                        'position': team['position'],
                        'played': team['playedGames'],
                        'won': team['won'],
                        'draw': team['draw'],
                        'lost': team['lost'],
                        'points': team['points'],
                        'goals_for': team['goalsFor'],
                        'goals_against': team['goalsAgainst'],
                        'goal_difference': team['goalDifference']
                    })
                
                return pd.DataFrame(standings_data)
            else:
                print(f"API Error: {response.status_code}")
                return self._get_mock_standings()
                
        except Exception as e:
            print(f"Error fetching standings: {e}")
            return self._get_mock_standings()
    
    def get_team_matches(self, team_id: int, limit: int = 10) -> pd.DataFrame:
        """Fetch recent matches for a team"""
        try:
            url = f"{self.base_url}/teams/{team_id}/matches"
            params = {'limit': limit, 'status': 'FINISHED'}
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                matches = []
                
                for match in data['matches']:
                    matches.append({
                        'date': match['utcDate'],
                        'home_team': match['homeTeam']['name'],
                        'away_team': match['awayTeam']['name'],
                        'home_score': match['score']['fullTime']['home'],
                        'away_score': match['score']['fullTime']['away']
                    })
                
                return pd.DataFrame(matches)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching matches: {e}")
            return pd.DataFrame()
    
    def calculate_last_n_form(self, team_name: str, n: int = 10) -> Dict:
        """
        Calculate team form based on last N matches
        
        Returns:
            Dict with wins, draws, losses, goals scored/conceded, points
        """
        # This is a simplified version - in production, fetch actual match data
        matches_df = self._get_mock_recent_matches(team_name, n)
        
        form = {
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'goals_for': 0,
            'goals_against': 0,
            'points': 0,
            'form_string': ''
        }
        
        for _, match in matches_df.iterrows():
            if match['team'] == team_name:
                if match['result'] == 'W':
                    form['wins'] += 1
                    form['points'] += 3
                    form['form_string'] += 'W'
                elif match['result'] == 'D':
                    form['draws'] += 1
                    form['points'] += 1
                    form['form_string'] += 'D'
                else:
                    form['losses'] += 1
                    form['form_string'] += 'L'
                
                form['goals_for'] += match['goals_for']
                form['goals_against'] += match['goals_against']
        
        return form
    
    def get_injury_data(self, team_name: str) -> Dict:
        """
        Get injury data for a team
        Note: This requires a different API or web scraping
        """
        # Mock implementation - replace with actual data source
        return self._get_mock_injuries(team_name)
    
    def _get_mock_standings(self) -> pd.DataFrame:
        """Generate mock standings data for development/testing"""
        teams_data = [
            {'team': 'Liverpool', 'position': 1, 'played': 24, 'won': 17, 'draw': 5, 'lost': 2, 'points': 56, 'goals_for': 59, 'goals_against': 24, 'goal_difference': 35},
            {'team': 'Arsenal', 'position': 2, 'played': 24, 'won': 16, 'draw': 6, 'lost': 2, 'points': 54, 'goals_for': 55, 'goals_against': 22, 'goal_difference': 33},
            {'team': 'Manchester City', 'position': 3, 'played': 24, 'won': 15, 'draw': 7, 'lost': 2, 'points': 52, 'goals_for': 54, 'goals_against': 26, 'goal_difference': 28},
            {'team': 'Aston Villa', 'position': 4, 'played': 24, 'won': 14, 'draw': 4, 'lost': 6, 'points': 46, 'goals_for': 49, 'goals_against': 35, 'goal_difference': 14},
            {'team': 'Tottenham', 'position': 5, 'played': 24, 'won': 13, 'draw': 5, 'lost': 6, 'points': 44, 'goals_for': 52, 'goals_against': 38, 'goal_difference': 14},
            {'team': 'Newcastle', 'position': 6, 'played': 24, 'won': 12, 'draw': 6, 'lost': 6, 'points': 42, 'goals_for': 48, 'goals_against': 35, 'goal_difference': 13},
            {'team': 'Manchester United', 'position': 7, 'played': 24, 'won': 11, 'draw': 5, 'lost': 8, 'points': 38, 'goals_for': 38, 'goals_against': 35, 'goal_difference': 3},
            {'team': 'West Ham', 'position': 8, 'played': 24, 'won': 10, 'draw': 7, 'lost': 7, 'points': 37, 'goals_for': 42, 'goals_against': 40, 'goal_difference': 2},
            {'team': 'Brighton', 'position': 9, 'played': 24, 'won': 10, 'draw': 6, 'lost': 8, 'points': 36, 'goals_for': 45, 'goals_against': 43, 'goal_difference': 2},
            {'team': 'Chelsea', 'position': 10, 'played': 24, 'won': 9, 'draw': 7, 'lost': 8, 'points': 34, 'goals_for': 41, 'goals_against': 39, 'goal_difference': 2},
            {'team': 'Wolves', 'position': 11, 'played': 24, 'won': 9, 'draw': 6, 'lost': 9, 'points': 33, 'goals_for': 35, 'goals_against': 38, 'goal_difference': -3},
            {'team': 'Fulham', 'position': 12, 'played': 24, 'won': 8, 'draw': 7, 'lost': 9, 'points': 31, 'goals_for': 36, 'goals_against': 40, 'goal_difference': -4},
            {'team': 'Bournemouth', 'position': 13, 'played': 24, 'won': 8, 'draw': 6, 'lost': 10, 'points': 30, 'goals_for': 37, 'goals_against': 45, 'goal_difference': -8},
            {'team': 'Crystal Palace', 'position': 14, 'played': 24, 'won': 7, 'draw': 8, 'lost': 9, 'points': 29, 'goals_for': 30, 'goals_against': 38, 'goal_difference': -8},
            {'team': 'Brentford', 'position': 15, 'played': 24, 'won': 7, 'draw': 7, 'lost': 10, 'points': 28, 'goals_for': 36, 'goals_against': 42, 'goal_difference': -6},
            {'team': 'Nottingham Forest', 'position': 16, 'played': 24, 'won': 6, 'draw': 8, 'lost': 10, 'points': 26, 'goals_for': 32, 'goals_against': 42, 'goal_difference': -10},
            {'team': 'Everton', 'position': 17, 'played': 24, 'won': 6, 'draw': 7, 'lost': 11, 'points': 25, 'goals_for': 28, 'goals_against': 38, 'goal_difference': -10},
            {'team': 'Luton Town', 'position': 18, 'played': 24, 'won': 5, 'draw': 6, 'lost': 13, 'points': 21, 'goals_for': 32, 'goals_against': 50, 'goal_difference': -18},
            {'team': 'Burnley', 'position': 19, 'played': 24, 'won': 4, 'draw': 5, 'lost': 15, 'points': 17, 'goals_for': 28, 'goals_against': 52, 'goal_difference': -24},
            {'team': 'Sheffield United', 'position': 20, 'played': 24, 'won': 3, 'draw': 4, 'lost': 17, 'points': 13, 'goals_for': 24, 'goals_against': 60, 'goal_difference': -36},
        ]
        return pd.DataFrame(teams_data)
    
    def _get_mock_recent_matches(self, team_name: str, n: int = 10) -> pd.DataFrame:
        """Generate mock recent match results"""
        import random
        random.seed(hash(team_name) % 1000)
        
        matches = []
        results = ['W', 'D', 'L']
        weights = [0.45, 0.25, 0.30]  # Typical distribution
        
        for i in range(n):
            result = random.choices(results, weights=weights)[0]
            if result == 'W':
                gf, ga = random.randint(1, 4), random.randint(0, 2)
            elif result == 'D':
                score = random.randint(0, 2)
                gf, ga = score, score
            else:
                gf, ga = random.randint(0, 2), random.randint(1, 4)
            
            matches.append({
                'team': team_name,
                'result': result,
                'goals_for': gf,
                'goals_against': ga
            })
        
        return pd.DataFrame(matches)
    
    def _get_mock_injuries(self, team_name: str) -> Dict:
        """Generate mock injury data"""
        import random
        random.seed(hash(team_name) % 1000)
        
        num_injuries = random.randint(1, 5)
        return {
            'total_injuries': num_injuries,
            'key_players_out': random.randint(0, min(2, num_injuries)),
            'injury_severity_score': num_injuries * random.uniform(0.5, 1.5)
        }
    
    def get_complete_dataset(self) -> pd.DataFrame:
        """
        Generate complete dataset with all features for ML model
        """
        standings = self.get_current_standings()
        
        # Add form data
        form_data = []
        injury_data = []
        
        for team in standings['team']:
            form = self.calculate_last_n_form(team, 10)
            injuries = self.get_injury_data(team)
            
            form_data.append(form)
            injury_data.append(injuries)
        
        standings['last_10_wins'] = [f['wins'] for f in form_data]
        standings['last_10_draws'] = [f['draws'] for f in form_data]
        standings['last_10_losses'] = [f['losses'] for f in form_data]
        standings['last_10_points'] = [f['points'] for f in form_data]
        standings['last_10_gf'] = [f['goals_for'] for f in form_data]
        standings['last_10_ga'] = [f['goals_against'] for f in form_data]
        standings['form_string'] = [f['form_string'] for f in form_data]
        
        standings['total_injuries'] = [i['total_injuries'] for i in injury_data]
        standings['key_players_out'] = [i['key_players_out'] for i in injury_data]
        standings['injury_severity'] = [i['injury_severity_score'] for i in injury_data]
        
        return standings


if __name__ == "__main__":
    # Test the data collector
    collector = PremierLeagueDataCollector()
    
    print("Fetching Premier League Data...")
    standings = collector.get_complete_dataset()
    
    print("\n=== Current Standings with Form & Injuries ===")
    print(standings.to_string(index=False))
    
    # Save to CSV
    standings.to_csv('current_season_data.csv', index=False)
    print("\nData saved to current_season_data.csv")
