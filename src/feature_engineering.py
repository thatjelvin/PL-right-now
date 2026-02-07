"""
Feature engineering for Premier League prediction.

This module transforms raw team data into features suitable for ML models.
"""

import numpy as np
from typing import List, Dict, Tuple
from .data_models import Team, LeagueTable


class FeatureExtractor:
    """Extract features from Premier League team data for ML models."""

    FEATURE_NAMES = [
        'current_position',
        'points',
        'points_per_game',
        'goal_difference',
        'goals_per_game',
        'goals_conceded_per_game',
        'win_rate',
        'draw_rate',
        'loss_rate',
        'recent_form_normalized',
        'home_form',
        'away_form',
        'injured_players_count',
        'suspended_players_count',
        'squad_fitness_score',
        'games_remaining',
        'max_possible_points',
        'points_needed_for_top4',
        'points_needed_to_avoid_relegation',
    ]

    def __init__(self, total_games: int = 38):
        """
        Initialize the feature extractor.
        
        Args:
            total_games: Total number of games in the season
        """
        self.total_games = total_games

    def extract_features(self, team: Team, league: LeagueTable) -> np.ndarray:
        """
        Extract all features for a single team.
        
        Args:
            team: Team object with current data
            league: Current league table for context
        
        Returns:
            numpy array of features
        """
        games_remaining = self.total_games - team.played
        max_possible_points = team.points + (games_remaining * 3)
        
        # Calculate points needed for objectives
        top4_threshold = self._estimate_top4_threshold(league)
        relegation_threshold = self._estimate_relegation_threshold(league)
        
        features = [
            team.position,
            team.points,
            team.points_per_game,
            team.goal_difference,
            team.goals_for / max(1, team.played),
            team.goals_against / max(1, team.played),
            team.won / max(1, team.played),
            team.drawn / max(1, team.played),
            team.lost / max(1, team.played),
            team.recent_form_normalized,
            team.home_form,
            team.away_form,
            team.injured_players_count,
            team.suspended_players_count,
            team.squad_fitness_score / 100,  # Normalize to 0-1
            games_remaining,
            max_possible_points,
            max(0, top4_threshold - team.points),
            max(0, relegation_threshold - team.points) * -1,  # Negative = safe
        ]
        
        return np.array(features)

    def _estimate_top4_threshold(self, league: LeagueTable) -> int:
        """Estimate points needed for top 4 finish."""
        if len(league.teams) < 4:
            return 70
        fourth_place = league.teams[3]
        games_remaining = self.total_games - fourth_place.played
        projected = fourth_place.points + (fourth_place.points_per_game * games_remaining)
        return int(projected) + 2  # Add buffer

    def _estimate_relegation_threshold(self, league: LeagueTable) -> int:
        """Estimate points needed to avoid relegation."""
        if len(league.teams) < 18:
            return 35
        eighteenth_place = league.teams[17]
        games_remaining = self.total_games - eighteenth_place.played
        projected = eighteenth_place.points + (eighteenth_place.points_per_game * games_remaining)
        return int(projected) + 3  # Add safety buffer

    def extract_all_features(self, league: LeagueTable) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features for all teams in the league.
        
        Args:
            league: LeagueTable object
        
        Returns:
            Tuple of (feature matrix, team names)
        """
        features = []
        team_names = []
        
        for team in league.teams:
            features.append(self.extract_features(team, league))
            team_names.append(team.name)
        
        return np.array(features), team_names

    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.FEATURE_NAMES.copy()
