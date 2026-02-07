"""
Tests for the predictor module.
"""

import pytest
import numpy as np
from src.data_models import Team, LeagueTable, Player, MatchResult
from src.predictor import SeasonPredictor
from src.feature_engineering import FeatureExtractor
from src.data_provider import get_sample_league_table


class TestFeatureExtractor:
    """Tests for the FeatureExtractor class."""

    def test_extract_features_shape(self):
        """Test that features have correct shape."""
        extractor = FeatureExtractor()
        league = get_sample_league_table()
        
        features, team_names = extractor.extract_all_features(league)
        
        assert features.shape[0] == 20  # 20 teams
        assert features.shape[1] == len(extractor.get_feature_names())
        assert len(team_names) == 20

    def test_feature_names(self):
        """Test that feature names are returned correctly."""
        extractor = FeatureExtractor()
        names = extractor.get_feature_names()
        
        assert 'current_position' in names
        assert 'points' in names
        assert 'recent_form_normalized' in names
        assert 'squad_fitness_score' in names


class TestSeasonPredictor:
    """Tests for the SeasonPredictor class."""

    def test_predictor_initialization(self):
        """Test predictor initializes correctly."""
        predictor = SeasonPredictor(total_games=38)
        assert predictor.total_games == 38
        assert predictor._is_fitted is False

    def test_predictor_fit(self):
        """Test predictor fitting."""
        predictor = SeasonPredictor()
        league = get_sample_league_table()
        
        predictor.fit(league)
        
        assert predictor._is_fitted is True

    def test_predictor_predict(self):
        """Test predictor generates valid predictions."""
        predictor = SeasonPredictor()
        league = get_sample_league_table()
        
        predictions = predictor.predict(league)
        
        assert len(predictions) == 20
        for pred in predictions:
            assert 'team' in pred
            assert 'predicted_final_points' in pred
            assert 'predicted_final_position' in pred
            assert 'title_probability' in pred
            assert 'top4_probability' in pred
            assert 'relegation_probability' in pred
            
            # Check probability bounds
            assert 0 <= pred['title_probability'] <= 1
            assert 0 <= pred['top4_probability'] <= 1
            assert 0 <= pred['relegation_probability'] <= 1

    def test_predict_remaining_season(self):
        """Test full season prediction."""
        predictor = SeasonPredictor()
        league = get_sample_league_table()
        
        result = predictor.predict_remaining_season(league)
        
        assert 'season' in result
        assert 'current_matchweek' in result
        assert 'team_predictions' in result
        assert 'predicted_champion' in result
        assert 'predicted_relegated' in result
        
        assert len(result['team_predictions']) == 20
        assert len(result['predicted_relegated']) == 3

    def test_feature_importance(self):
        """Test feature importance retrieval."""
        predictor = SeasonPredictor()
        league = get_sample_league_table()
        
        # Before fitting
        assert predictor.get_feature_importance() == {}
        
        # After fitting
        predictor.fit(league)
        importance = predictor.get_feature_importance()
        
        assert len(importance) > 0
        assert all(isinstance(v, (int, float)) for v in importance.values())

    def test_predictions_ordered_by_points(self):
        """Test that predictions are ordered by predicted points."""
        predictor = SeasonPredictor()
        league = get_sample_league_table()
        
        predictions = predictor.predict(league)
        
        # Check descending order by predicted points
        for i in range(len(predictions) - 1):
            assert predictions[i]['predicted_final_points'] >= predictions[i + 1]['predicted_final_points']

    def test_top_team_has_high_title_probability(self):
        """Test that league leader has high title probability."""
        predictor = SeasonPredictor()
        league = get_sample_league_table()
        
        predictions = predictor.predict(league)
        
        # Find the predicted champion (first in sorted list)
        champion_pred = predictions[0]
        
        # Champion should have reasonable title probability
        assert champion_pred['title_probability'] > 0.3

    def test_bottom_teams_have_relegation_risk(self):
        """Test that bottom teams have relegation probability."""
        predictor = SeasonPredictor()
        league = get_sample_league_table()
        
        predictions = predictor.predict(league)
        
        # Get bottom 3 predictions
        bottom_3 = predictions[-3:]
        
        # At least some should have relegation risk
        has_relegation_risk = any(p['relegation_probability'] > 0.1 for p in bottom_3)
        assert has_relegation_risk


class TestIntegration:
    """Integration tests for the full prediction pipeline."""

    def test_full_pipeline(self):
        """Test the complete prediction pipeline."""
        from src.data_provider import get_sample_league_table
        from src.predictor import SeasonPredictor
        
        # Get data
        league = get_sample_league_table()
        assert league is not None
        assert len(league.teams) == 20
        
        # Create predictor
        predictor = SeasonPredictor()
        
        # Fit and predict
        result = predictor.predict_remaining_season(league)
        
        # Validate result structure
        assert result['predicted_champion'] is not None
        assert len(result['predicted_relegated']) == 3
        assert all(len(p['team']) > 0 for p in result['team_predictions'])
