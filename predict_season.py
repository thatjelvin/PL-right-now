#!/usr/bin/env python3
"""
Premier League Season Predictor - Main Entry Point

This script provides a command-line interface for predicting
how the Premier League season will end based on current standings,
injuries, player conditions, and recent form.
"""

import argparse
import json
import sys
from typing import Optional

from src.data_provider import get_sample_league_table
from src.predictor import SeasonPredictor


def format_prediction_table(predictions: list) -> str:
    """Format predictions as a readable table."""
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("PREMIER LEAGUE SEASON PREDICTION")
    lines.append("=" * 100)
    lines.append(f"\n{'Pos':<4} {'Team':<20} {'Pts':<5} {'Pred Pts':<9} {'Pred Pos':<9} "
                 f"{'Title %':<9} {'Top 4 %':<9} {'Rel %':<7} {'Form':<6} {'Injuries':<9}")
    lines.append("-" * 100)
    
    for pred in predictions:
        lines.append(
            f"{pred['current_position']:<4} "
            f"{pred['team']:<20} "
            f"{pred['current_points']:<5} "
            f"{pred['predicted_final_points']:<9} "
            f"{pred['predicted_final_position']:<9} "
            f"{pred['title_probability']*100:>6.1f}%  "
            f"{pred['top4_probability']*100:>6.1f}%  "
            f"{pred['relegation_probability']*100:>4.1f}%  "
            f"{pred['form_rating']:<6} "
            f"{pred['injured_players']:<9}"
        )
    
    return "\n".join(lines)


def format_summary(result: dict) -> str:
    """Format the prediction summary."""
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("PREDICTION SUMMARY")
    lines.append("=" * 100)
    lines.append(f"\nSeason: {result['season']}")
    lines.append(f"Current Matchweek: {result['current_matchweek']}")
    lines.append(f"Games Remaining: {result['games_remaining']}")
    
    lines.append(f"\n🏆 Predicted Champion: {result['predicted_champion']}")
    
    if result['title_contenders']:
        lines.append("\n📊 Title Contenders:")
        for team in result['title_contenders']:
            lines.append(f"   • {team['team']}: {team['title_probability']*100:.1f}% chance")
    
    if result['top4_race']:
        lines.append("\n🎯 Top 4 Battle:")
        for team in result['top4_race']:
            lines.append(f"   • {team['team']}: {team['top4_probability']*100:.1f}% chance")
    
    if result['relegation_battle']:
        lines.append("\n⚠️ Relegation Danger:")
        for team in result['relegation_battle']:
            lines.append(f"   • {team['team']}: {team['relegation_probability']*100:.1f}% risk")
    
    if result['predicted_relegated']:
        lines.append(f"\n📉 Predicted Relegated: {', '.join(result['predicted_relegated'])}")
    
    return "\n".join(lines)


def main(output_format: str = 'table', output_file: Optional[str] = None) -> dict:
    """
    Main function to run the Premier League prediction.
    
    Args:
        output_format: 'table' for human-readable, 'json' for machine-readable
        output_file: Optional file path to save results
    
    Returns:
        Prediction results dictionary
    """
    print("Loading Premier League data...")
    league = get_sample_league_table()
    
    print(f"Analyzing {len(league.teams)} teams from matchweek {league.matchweek}...")
    
    # Create and fit the predictor
    predictor = SeasonPredictor(total_games=38)
    predictor.fit(league)
    
    # Get predictions
    print("Running prediction model...")
    result = predictor.predict_remaining_season(league)
    
    # Format output
    if output_format == 'json':
        output = json.dumps(result, indent=2)
        print(output)
    else:
        print(format_prediction_table(result['team_predictions']))
        print(format_summary(result))
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    # Print feature importance
    importance = predictor.get_feature_importance()
    if importance:
        print("\n" + "=" * 100)
        print("MODEL FEATURE IMPORTANCE (Top 10)")
        print("=" * 100)
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        for feature, score in sorted_importance:
            bar = "█" * int(score * 50)
            print(f"  {feature:<35} {score:.4f} {bar}")
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict Premier League season outcomes based on current standings, "
                    "injuries, player conditions, and recent form."
    )
    parser.add_argument(
        "--format", 
        choices=["table", "json"], 
        default="table",
        help="Output format (default: table)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        help="Save results to file (JSON format)"
    )
    
    args = parser.parse_args()
    
    try:
        main(output_format=args.format, output_file=args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
