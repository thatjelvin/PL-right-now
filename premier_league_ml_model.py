"""
Premier League Season Prediction ML Model
Uses multiple algorithms to predict final season standings
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class PremierLeaguePredictor:
    """
    Machine Learning model to predict Premier League final standings
    """
    
    def __init__(self):
        """Initialize prediction models"""
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1)
        }
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features for modeling
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Tuple of (feature array, feature names)
        """
        # Select relevant features for prediction
        feature_cols = [
            'points',
            'goal_difference',
            'points_per_game',
            'win_rate',
            'goals_per_game_for',
            'goals_per_game_against',
            'weighted_ppg',
            'momentum',
            'attack_strength',
            'defense_strength',
            'injury_impact',
            'matches_remaining',
            'recent_win_rate',
            'recent_goal_diff',
            'last_10_points'
        ]
        
        self.feature_columns = feature_cols
        X = df[feature_cols].values
        
        return X, feature_cols
    
    def create_synthetic_training_data(self, current_data: pd.DataFrame, 
                                      n_scenarios: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create synthetic training data by simulating season outcomes
        This is used when historical data is not available
        
        Args:
            current_data: Current season data with features
            n_scenarios: Number of synthetic scenarios to generate
            
        Returns:
            Tuple of (X_train, y_train)
        """
        np.random.seed(42)
        X_train_list = []
        y_train_list = []
        
        for _ in range(n_scenarios):
            # Create variations of current data
            scenario = current_data.copy()
            
            for idx, row in scenario.iterrows():
                # Simulate remaining season with some randomness
                avg_ppg = row['weighted_ppg']
                matches_left = row['matches_remaining']
                current_points = row['points']
                
                # Add realistic variation to PPG
                std_dev = 0.3  # Standard deviation for point variation
                simulated_ppg = np.random.normal(avg_ppg, std_dev)
                simulated_ppg = max(0, min(3, simulated_ppg))  # Clamp between 0-3
                
                # Adjust for injuries
                injury_penalty = row['injury_impact'] * 0.05
                simulated_ppg -= injury_penalty
                
                # Adjust for momentum
                momentum_adjustment = row['momentum'] * 0.2
                simulated_ppg += momentum_adjustment
                
                # Calculate final points
                additional_points = simulated_ppg * matches_left
                final_points = current_points + additional_points
                
                # Prepare features
                X_train_list.append(self.prepare_features(scenario.iloc[[idx]])[0][0])
                y_train_list.append(final_points)
        
        return np.array(X_train_list), np.array(y_train_list)
    
    def train_on_synthetic_data(self, current_data: pd.DataFrame):
        """
        Train models using synthetic data
        
        Args:
            current_data: Current season data with all features
        """
        print("Creating synthetic training data...")
        X_train, y_train = self.create_synthetic_training_data(current_data, n_scenarios=500)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        print("Training models...")
        for name, model in self.models.items():
            model.fit(X_train_scaled, y_train)
            print(f"  ✓ {name} trained")
        
        self.trained = True
    
    def predict_final_points(self, features_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Predict final points using all models
        
        Args:
            features_df: DataFrame with engineered features
            
        Returns:
            Dictionary of predictions from each model
        """
        if not self.trained:
            raise ValueError("Models must be trained before prediction")
        
        X, _ = self.prepare_features(features_df)
        X_scaled = self.scaler.transform(X)
        
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X_scaled)
        
        return predictions
    
    def predict_ensemble(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ensemble prediction from all models
        
        Args:
            features_df: DataFrame with engineered features
            
        Returns:
            DataFrame with team predictions
        """
        predictions = self.predict_final_points(features_df)
        
        # Create ensemble by averaging predictions
        ensemble_pred = np.mean([pred for pred in predictions.values()], axis=0)
        
        results = pd.DataFrame({
            'team': features_df['team'],
            'current_position': features_df['position'],
            'current_points': features_df['points'],
            'matches_remaining': features_df['matches_remaining'],
            'predicted_final_points': ensemble_pred.round(1)
        })
        
        # Add individual model predictions
        for name, pred in predictions.items():
            results[f'{name}_prediction'] = pred.round(1)
        
        # Sort by predicted points
        results = results.sort_values('predicted_final_points', ascending=False)
        results['predicted_final_position'] = range(1, len(results) + 1)
        
        # Calculate position change
        results['position_change'] = results['current_position'] - results['predicted_final_position']
        
        return results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from tree-based models
        
        Returns:
            DataFrame with feature importances
        """
        if not self.trained:
            raise ValueError("Models must be trained before getting feature importance")
        
        # Get importance from Random Forest
        rf_importance = self.models['random_forest'].feature_importances_
        
        # Get importance from Gradient Boosting
        gb_importance = self.models['gradient_boosting'].feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'random_forest_importance': rf_importance,
            'gradient_boosting_importance': gb_importance
        })
        
        # Calculate average importance
        importance_df['average_importance'] = (
            importance_df['random_forest_importance'] + 
            importance_df['gradient_boosting_importance']
        ) / 2
        
        importance_df = importance_df.sort_values('average_importance', ascending=False)
        
        return importance_df
    
    def predict_with_confidence(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict with confidence intervals based on model agreement
        
        Args:
            features_df: DataFrame with engineered features
            
        Returns:
            DataFrame with predictions and confidence metrics
        """
        predictions = self.predict_final_points(features_df)
        
        # Calculate mean and std dev across models
        all_preds = np.array([pred for pred in predictions.values()])
        mean_pred = np.mean(all_preds, axis=0)
        std_pred = np.std(all_preds, axis=0)
        
        results = pd.DataFrame({
            'team': features_df['team'],
            'current_position': features_df['position'],
            'current_points': features_df['points'],
            'predicted_final_points': mean_pred.round(1),
            'prediction_std': std_pred.round(2),
            'confidence_95_low': (mean_pred - 1.96 * std_pred).round(1),
            'confidence_95_high': (mean_pred + 1.96 * std_pred).round(1)
        })
        
        # Sort by predicted points
        results = results.sort_values('predicted_final_points', ascending=False)
        results['predicted_position'] = range(1, len(results) + 1)
        
        # Add confidence category
        results['prediction_confidence'] = pd.cut(
            results['prediction_std'],
            bins=[0, 1, 2, 100],
            labels=['High', 'Medium', 'Low']
        )
        
        return results
    
    def analyze_key_battles(self, predictions_df: pd.DataFrame) -> Dict:
        """
        Analyze key battles (title race, top 4, relegation)
        
        Args:
            predictions_df: DataFrame with predictions
            
        Returns:
            Dictionary with key battle analysis
        """
        battles = {}
        
        # Title race (top 3)
        title_contenders = predictions_df.nsmallest(3, 'predicted_position')
        battles['title_race'] = {
            'contenders': title_contenders[['team', 'predicted_final_points']].to_dict('records'),
            'points_gap': (
                title_contenders.iloc[0]['predicted_final_points'] - 
                title_contenders.iloc[1]['predicted_final_points']
            ).round(1)
        }
        
        # Top 4 race (Champions League)
        top4_battle = predictions_df.nsmallest(6, 'predicted_position')
        battles['top_4_race'] = {
            'teams': top4_battle[['team', 'predicted_position', 'predicted_final_points']].to_dict('records'),
            'tightness': top4_battle.iloc[5]['predicted_final_points'] - top4_battle.iloc[3]['predicted_final_points']
        }
        
        # Relegation battle (bottom 5)
        relegation_battle = predictions_df.nlargest(5, 'predicted_position')
        battles['relegation_battle'] = {
            'teams_at_risk': relegation_battle[['team', 'predicted_position', 'predicted_final_points']].to_dict('records'),
            'safety_margin': (
                relegation_battle.iloc[0]['predicted_final_points'] - 
                relegation_battle.iloc[-1]['predicted_final_points']
            ).round(1)
        }
        
        return battles


if __name__ == "__main__":
    from premier_league_data import PremierLeagueDataCollector
    from premier_league_features import FeatureEngineering
    
    print("=" * 60)
    print("PREMIER LEAGUE SEASON PREDICTION MODEL")
    print("=" * 60)
    
    # Load and prepare data
    print("\n1. Loading current season data...")
    collector = PremierLeagueDataCollector()
    data = collector.get_complete_dataset()
    
    print("2. Engineering features...")
    fe = FeatureEngineering(data)
    features = fe.create_all_features()
    
    # Initialize and train predictor
    print("3. Training ML models...")
    predictor = PremierLeaguePredictor()
    predictor.train_on_synthetic_data(features)
    
    # Make predictions
    print("\n4. Generating predictions...")
    predictions = predictor.predict_with_confidence(features)
    
    print("\n" + "=" * 60)
    print("PREDICTED FINAL STANDINGS")
    print("=" * 60)
    print(predictions[['team', 'current_position', 'predicted_position', 
                       'current_points', 'predicted_final_points', 
                       'confidence_95_low', 'confidence_95_high']].to_string(index=False))
    
    # Feature importance
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)
    importance = predictor.get_feature_importance()
    print(importance.to_string(index=False))
    
    # Key battles
    print("\n" + "=" * 60)
    print("KEY BATTLES ANALYSIS")
    print("=" * 60)
    battles = predictor.analyze_key_battles(predictions)
    
    print("\n🏆 TITLE RACE:")
    for team in battles['title_race']['contenders']:
        print(f"  {team['team']}: {team['predicted_final_points']:.1f} points")
    print(f"  Gap: {battles['title_race']['points_gap']:.1f} points")
    
    print("\n🎯 TOP 4 RACE:")
    for idx, team in enumerate(battles['top_4_race']['teams'][:4], 1):
        print(f"  {idx}. {team['team']}: {team['predicted_final_points']:.1f} points")
    
    print("\n⚠️  RELEGATION BATTLE:")
    for team in battles['relegation_battle']['teams_at_risk']:
        status = "🔴" if team['predicted_position'] >= 18 else "🟡"
        print(f"  {status} {team['team']}: {team['predicted_final_points']:.1f} points (Pos: {team['predicted_position']})")
    
    # Save results
    predictions.to_csv('ml_predictions.csv', index=False)
    importance.to_csv('feature_importance.csv', index=False)
    
    print("\n" + "=" * 60)
    print("Results saved to ml_predictions.csv and feature_importance.csv")
    print("=" * 60)
