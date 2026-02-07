"""
Machine Learning model for Premier League season prediction.

This module contains the ML model that predicts final league positions
based on current standings, form, injuries, and other factors.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings

from .data_models import Team, LeagueTable
from .feature_engineering import FeatureExtractor


class SeasonPredictor:
    """
    Machine Learning model to predict Premier League season outcomes.
    
    Uses ensemble methods to predict final positions, points, and
    probabilities for various outcomes (title, top 4, relegation).
    """

    def __init__(self, total_games: int = 38):
        """
        Initialize the season predictor.
        
        Args:
            total_games: Total number of games in the season
        """
        self.total_games = total_games
        self.feature_extractor = FeatureExtractor(total_games)
        self.scaler = StandardScaler()
        
        # Ensemble of models for robust predictions
        self.points_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        self.position_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self._is_fitted = False

    def _generate_training_data(self, league: LeagueTable) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate training data from current league state.
        
        Uses the current state and projects outcomes based on
        historical patterns and current form.
        
        Args:
            league: Current league table
        
        Returns:
            Tuple of (features, target_points, target_positions)
        """
        features, _ = self.feature_extractor.extract_all_features(league)
        
        # Generate target predictions based on current trajectory
        target_points = []
        target_positions = []
        
        for i, team in enumerate(league.teams):
            games_remaining = self.total_games - team.played
            
            # Project final points with form adjustment
            base_ppg = team.points_per_game
            form_adjustment = (team.recent_form_normalized - 0.5) * 0.3
            fitness_adjustment = (team.squad_fitness_score / 100 - 0.85) * 0.1
            
            adjusted_ppg = base_ppg + form_adjustment + fitness_adjustment
            adjusted_ppg = max(0, min(3, adjusted_ppg))  # Clamp to valid range
            
            projected_points = team.points + (adjusted_ppg * games_remaining)
            
            # Add some variance based on games remaining
            variance = games_remaining * 0.15
            
            target_points.append(projected_points)
            target_positions.append(i + 1)  # Current position as base
        
        return features, np.array(target_points), np.array(target_positions)

    def fit(self, league: LeagueTable) -> 'SeasonPredictor':
        """
        Fit the model on current league data.
        
        Args:
            league: Current league table
        
        Returns:
            self for method chaining
        """
        features, target_points, target_positions = self._generate_training_data(league)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit models
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.points_model.fit(features_scaled, target_points)
            self.position_model.fit(features_scaled, target_positions)
        
        self._is_fitted = True
        return self

    def predict(self, league: LeagueTable) -> List[Dict]:
        """
        Predict season outcomes for all teams.
        
        Args:
            league: Current league table
        
        Returns:
            List of prediction dictionaries for each team
        """
        if not self._is_fitted:
            self.fit(league)
        
        features, team_names = self.feature_extractor.extract_all_features(league)
        features_scaled = self.scaler.transform(features)
        
        # Get base predictions
        predicted_points = self.points_model.predict(features_scaled)
        predicted_positions = self.position_model.predict(features_scaled)
        
        # Build predictions list
        predictions = []
        for i, team in enumerate(league.teams):
            games_remaining = self.total_games - team.played
            
            # Calculate probabilities based on points projections
            probs = self._calculate_outcome_probabilities(
                team, predicted_points[i], games_remaining, league
            )
            
            predictions.append({
                'team': team.name,
                'current_position': team.position,
                'current_points': team.points,
                'predicted_final_points': round(predicted_points[i], 1),
                'predicted_final_position': int(round(predicted_positions[i])),
                'games_remaining': games_remaining,
                'title_probability': probs['title'],
                'top4_probability': probs['top4'],
                'relegation_probability': probs['relegation'],
                'form_rating': round(team.recent_form_normalized * 10, 1),
                'squad_fitness': round(team.squad_fitness_score, 1),
                'injured_players': team.injured_players_count,
            })
        
        # Sort by predicted points for ranking
        predictions.sort(key=lambda x: x['predicted_final_points'], reverse=True)
        
        # Adjust predicted positions based on sorted order
        for i, pred in enumerate(predictions):
            pred['predicted_final_position'] = i + 1
        
        return predictions

    def _calculate_outcome_probabilities(
        self, 
        team: Team, 
        predicted_points: float, 
        games_remaining: int,
        league: LeagueTable
    ) -> Dict[str, float]:
        """
        Calculate probabilities for various outcomes.
        
        Args:
            team: Team object
            predicted_points: Model's point prediction
            games_remaining: Remaining games
            league: Current league table
        
        Returns:
            Dictionary with title, top4, and relegation probabilities
        """
        # Get current standings context
        leader_points = league.teams[0].points if league.teams else 0
        fourth_points = league.teams[3].points if len(league.teams) > 3 else 0
        seventeenth_points = league.teams[16].points if len(league.teams) > 16 else 0
        
        max_possible = team.points + (games_remaining * 3)
        min_possible = team.points
        
        # Title probability
        points_behind_leader = leader_points - team.points
        if team.position == 1:
            title_prob = 0.7 + (0.3 * (1 - games_remaining / self.total_games))
        elif max_possible < leader_points:
            title_prob = 0.0
        else:
            gap_factor = max(0, 1 - (points_behind_leader / (games_remaining * 3 + 1)))
            title_prob = gap_factor * 0.5 * (team.recent_form_normalized + 0.2)
        
        # Top 4 probability
        points_behind_fourth = fourth_points - team.points
        if team.position <= 4:
            top4_prob = 0.75 + (0.25 * team.recent_form_normalized)
        elif max_possible < fourth_points:
            top4_prob = 0.05
        else:
            gap_factor = max(0, 1 - (points_behind_fourth / (games_remaining * 2 + 1)))
            top4_prob = gap_factor * 0.6 + (0.2 * team.recent_form_normalized)
        
        # Relegation probability
        points_above_relegation = team.points - seventeenth_points
        if team.position >= 18:
            rel_prob = 0.6 + (0.3 * (1 - team.recent_form_normalized))
        elif points_above_relegation > games_remaining * 3:
            rel_prob = 0.0
        else:
            safety_factor = min(1, points_above_relegation / (games_remaining + 1))
            rel_prob = max(0, (1 - safety_factor) * 0.4 * (1 - team.recent_form_normalized))
        
        return {
            'title': round(min(1.0, max(0.0, title_prob)), 3),
            'top4': round(min(1.0, max(0.0, top4_prob)), 3),
            'relegation': round(min(1.0, max(0.0, rel_prob)), 3)
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the fitted models.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self._is_fitted:
            return {}
        
        feature_names = self.feature_extractor.get_feature_names()
        
        # Use points model feature importance (Gradient Boosting)
        importances = self.points_model.feature_importances_
        
        return dict(zip(feature_names, importances))

    def predict_remaining_season(self, league: LeagueTable) -> Dict:
        """
        Generate comprehensive season predictions.
        
        Args:
            league: Current league table
        
        Returns:
            Dictionary with detailed season predictions
        """
        predictions = self.predict(league)
        
        # Identify key outcomes
        title_contenders = [p for p in predictions if p['title_probability'] > 0.1]
        top4_race = [p for p in predictions if 0.3 < p['top4_probability'] < 0.9]
        relegation_battle = [p for p in predictions if p['relegation_probability'] > 0.2]
        
        return {
            'season': league.season,
            'current_matchweek': league.matchweek,
            'games_remaining': self.total_games - league.matchweek,
            'team_predictions': predictions,
            'title_contenders': title_contenders,
            'top4_race': top4_race,
            'relegation_battle': relegation_battle,
            'predicted_champion': predictions[0]['team'] if predictions else None,
            'predicted_relegated': [p['team'] for p in predictions[-3:]] if len(predictions) >= 3 else [],
        }
