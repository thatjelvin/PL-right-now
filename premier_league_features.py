"""
Feature Engineering Module for Premier League Season Prediction
Creates meaningful features from raw data for ML models
"""

import pandas as pd
import numpy as np
from typing import List, Tuple

class FeatureEngineering:
    """
    Creates features for predicting final season standings
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with current season data
        
        Args:
            data: DataFrame with current standings and statistics
        """
        self.data = data.copy()
        self.total_matches = 38  # Premier League season length
        
    def calculate_points_per_game(self) -> pd.Series:
        """Calculate current points per game rate"""
        return self.data['points'] / self.data['played']
    
    def calculate_goals_per_game(self) -> Tuple[pd.Series, pd.Series]:
        """Calculate goals scored and conceded per game"""
        gpg_for = self.data['goals_for'] / self.data['played']
        gpg_against = self.data['goals_against'] / self.data['played']
        return gpg_for, gpg_against
    
    def calculate_win_rate(self) -> pd.Series:
        """Calculate win percentage"""
        return self.data['won'] / self.data['played']
    
    def calculate_form_weighted_points(self, recent_weight: float = 1.5) -> pd.Series:
        """
        Calculate weighted points rate giving more weight to recent form
        
        Args:
            recent_weight: How much more to weight recent form (last 10)
        """
        overall_ppg = self.data['points'] / self.data['played']
        recent_ppg = self.data['last_10_points'] / 10
        
        # Weighted average favoring recent form
        weighted_ppg = (overall_ppg + recent_weight * recent_ppg) / (1 + recent_weight)
        return weighted_ppg
    
    def calculate_momentum(self) -> pd.Series:
        """
        Calculate team momentum based on recent vs overall performance
        Positive = improving, Negative = declining
        """
        overall_ppg = self.data['points'] / self.data['played']
        recent_ppg = self.data['last_10_points'] / 10
        
        return recent_ppg - overall_ppg
    
    def calculate_injury_impact(self) -> pd.Series:
        """
        Calculate injury impact factor
        Higher values = worse injury situation
        """
        # Normalize injuries as a percentage impact
        injury_impact = (
            self.data['total_injuries'] * 0.1 + 
            self.data['key_players_out'] * 0.3 +
            self.data['injury_severity'] * 0.05
        )
        return injury_impact
    
    def calculate_remaining_matches(self) -> pd.Series:
        """Calculate matches remaining in season"""
        return self.total_matches - self.data['played']
    
    def calculate_attack_strength(self) -> pd.Series:
        """
        Calculate relative attack strength
        Combines overall and recent goal scoring
        """
        overall_attack = self.data['goals_for'] / self.data['played']
        recent_attack = self.data['last_10_gf'] / 10
        
        # Weight recent form more heavily
        attack_strength = (overall_attack * 0.4 + recent_attack * 0.6)
        return attack_strength
    
    def calculate_defense_strength(self) -> pd.Series:
        """
        Calculate relative defense strength (lower is better)
        """
        overall_defense = self.data['goals_against'] / self.data['played']
        recent_defense = self.data['last_10_ga'] / 10
        
        # Weight recent form more heavily (lower is better)
        defense_strength = (overall_defense * 0.4 + recent_defense * 0.6)
        return defense_strength
    
    def calculate_consistency(self) -> pd.Series:
        """
        Estimate consistency based on draws vs decisive results
        """
        total_matches = self.data['played']
        decisive_matches = self.data['won'] + self.data['lost']
        consistency = decisive_matches / total_matches
        return consistency
    
    def create_all_features(self) -> pd.DataFrame:
        """
        Create complete feature set for ML model
        
        Returns:
            DataFrame with all engineered features
        """
        features = self.data.copy()
        
        # Basic rates
        features['points_per_game'] = self.calculate_points_per_game()
        features['win_rate'] = self.calculate_win_rate()
        features['draw_rate'] = features['draw'] / features['played']
        
        # Goals
        gpg_for, gpg_against = self.calculate_goals_per_game()
        features['goals_per_game_for'] = gpg_for
        features['goals_per_game_against'] = gpg_against
        features['goal_difference_per_game'] = features['goal_difference'] / features['played']
        
        # Advanced metrics
        features['weighted_ppg'] = self.calculate_form_weighted_points()
        features['momentum'] = self.calculate_momentum()
        features['attack_strength'] = self.calculate_attack_strength()
        features['defense_strength'] = self.calculate_defense_strength()
        features['consistency'] = self.calculate_consistency()
        
        # Injury impact
        features['injury_impact'] = self.calculate_injury_impact()
        
        # Contextual
        features['matches_remaining'] = self.calculate_remaining_matches()
        features['season_progress'] = features['played'] / self.total_matches
        
        # Form features
        features['recent_win_rate'] = features['last_10_wins'] / 10
        features['recent_goal_diff'] = (features['last_10_gf'] - features['last_10_ga']) / 10
        
        # Position-based features
        features['is_top_4'] = (features['position'] <= 4).astype(int)
        features['is_relegation_zone'] = (features['position'] >= 18).astype(int)
        features['is_top_half'] = (features['position'] <= 10).astype(int)
        
        return features
    
    def get_ml_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract only the features needed for ML model
        
        Args:
            features_df: Full features DataFrame
            
        Returns:
            DataFrame with only ML-relevant features
        """
        ml_feature_columns = [
            'position',
            'played',
            'points',
            'goal_difference',
            'points_per_game',
            'win_rate',
            'draw_rate',
            'goals_per_game_for',
            'goals_per_game_against',
            'weighted_ppg',
            'momentum',
            'attack_strength',
            'defense_strength',
            'consistency',
            'injury_impact',
            'matches_remaining',
            'season_progress',
            'recent_win_rate',
            'recent_goal_diff',
            'last_10_points'
        ]
        
        return features_df[['team'] + ml_feature_columns]
    
    def predict_final_points(self, weighted_ppg: pd.Series, 
                            matches_remaining: pd.Series,
                            current_points: pd.Series) -> pd.Series:
        """
        Simple projection of final points based on current form
        
        Args:
            weighted_ppg: Weighted points per game
            matches_remaining: Games left to play
            current_points: Current point total
            
        Returns:
            Projected final points
        """
        projected_points = weighted_ppg * matches_remaining
        final_points = current_points + projected_points
        return final_points
    
    def project_final_standings_simple(self) -> pd.DataFrame:
        """
        Simple projection of final standings (baseline method)
        """
        features = self.create_all_features()
        
        features['projected_final_points'] = self.predict_final_points(
            features['weighted_ppg'],
            features['matches_remaining'],
            features['points']
        )
        
        # Sort by projected points
        projection = features[['team', 'position', 'points', 
                              'weighted_ppg', 'projected_final_points']].copy()
        projection = projection.sort_values('projected_final_points', 
                                           ascending=False)
        projection['projected_position'] = range(1, len(projection) + 1)
        
        return projection


def create_training_data_from_historical(historical_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create training data from historical seasons
    
    Args:
        historical_df: Historical season data with final outcomes
        
    Returns:
        Tuple of (features, targets)
    """
    # This would process historical data if available
    # For now, returns None to indicate no historical data
    return None, None


if __name__ == "__main__":
    # Test feature engineering
    from premier_league_data import PremierLeagueDataCollector
    
    print("Loading data...")
    collector = PremierLeagueDataCollector()
    data = collector.get_complete_dataset()
    
    print("\nEngineering features...")
    fe = FeatureEngineering(data)
    features = fe.create_all_features()
    
    print("\n=== Feature Summary ===")
    print(features[['team', 'position', 'points_per_game', 'momentum', 
                    'attack_strength', 'defense_strength', 'injury_impact']].to_string(index=False))
    
    print("\n=== Simple Projection (Baseline) ===")
    projection = fe.project_final_standings_simple()
    print(projection.to_string(index=False))
    
    # Save features
    features.to_csv('engineered_features.csv', index=False)
    projection.to_csv('simple_projection.csv', index=False)
    print("\nFeatures saved to engineered_features.csv")
    print("Simple projection saved to simple_projection.csv")
