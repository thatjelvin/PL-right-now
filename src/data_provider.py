"""
Sample data provider for Premier League prediction.

This module provides sample/mock data for demonstration purposes.
In production, this would be replaced with real API data fetching.
"""

from typing import List
from .data_models import Team, Player, MatchResult, LeagueTable
import random


def generate_sample_last_10_matches(team_strength: float) -> List[MatchResult]:
    """
    Generate sample last 10 matches based on team strength.
    
    Args:
        team_strength: 0-1 value indicating team quality
    
    Returns:
        List of 10 MatchResult objects
    """
    opponents = [
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
        "Chelsea", "Crystal Palace", "Everton", "Fulham", "Liverpool",
        "Manchester City", "Manchester United", "Newcastle", "Nottingham Forest",
        "Southampton", "Tottenham", "West Ham", "Wolves", "Ipswich", "Leicester"
    ]
    
    matches = []
    for i in range(10):
        is_home = random.random() > 0.5
        # Base win probability increases with team strength
        win_prob = 0.3 + (team_strength * 0.4) + (0.1 if is_home else 0)
        draw_prob = 0.25
        
        rand = random.random()
        if rand < win_prob:
            goals_scored = random.randint(1, 4)
            goals_conceded = random.randint(0, goals_scored - 1)
        elif rand < win_prob + draw_prob:
            goals = random.randint(0, 3)
            goals_scored = goals
            goals_conceded = goals
        else:
            goals_conceded = random.randint(1, 4)
            goals_scored = random.randint(0, goals_conceded - 1)
        
        matches.append(MatchResult(
            opponent=random.choice(opponents),
            is_home=is_home,
            goals_scored=goals_scored,
            goals_conceded=goals_conceded
        ))
    
    return matches


def generate_sample_squad(team_strength: float, injury_rate: float = 0.15) -> List[Player]:
    """
    Generate a sample squad for a team.
    
    Args:
        team_strength: 0-1 value indicating team quality
        injury_rate: Probability of each player being injured
    
    Returns:
        List of Player objects
    """
    positions = {
        "Goalkeeper": 3,
        "Defender": 8,
        "Midfielder": 8,
        "Forward": 6
    }
    
    squad = []
    for position, count in positions.items():
        for i in range(count):
            is_injured = random.random() < injury_rate
            form = max(1, min(10, 5 + (team_strength * 4) + random.gauss(0, 1.5)))
            
            squad.append(Player(
                name=f"{position}_{i+1}",
                position=position,
                is_injured=is_injured,
                injury_severity=random.randint(3, 8) if is_injured else 0,
                is_suspended=random.random() < 0.05,
                form_rating=round(form, 1)
            ))
    
    return squad


def get_sample_league_table() -> LeagueTable:
    """
    Generate a sample Premier League table with realistic data.
    
    This represents a mid-season snapshot with 22 matchweeks played.
    
    Returns:
        LeagueTable object with 20 teams
    """
    # Current season sample data (approximately matchweek 22)
    teams_data = [
        ("Liverpool", 1, 22, 16, 4, 2, 52, 18, 52),
        ("Arsenal", 2, 22, 14, 5, 3, 48, 22, 47),
        ("Nottingham Forest", 3, 22, 13, 5, 4, 35, 20, 44),
        ("Chelsea", 4, 22, 11, 7, 4, 42, 26, 40),
        ("Newcastle", 5, 22, 11, 6, 5, 38, 24, 39),
        ("Manchester City", 6, 22, 11, 5, 6, 43, 28, 38),
        ("Bournemouth", 7, 22, 10, 7, 5, 35, 26, 37),
        ("Aston Villa", 8, 22, 10, 6, 6, 34, 29, 36),
        ("Fulham", 9, 22, 9, 7, 6, 32, 28, 34),
        ("Brighton", 10, 22, 8, 9, 5, 34, 30, 33),
        ("Brentford", 11, 22, 9, 5, 8, 38, 35, 32),
        ("Manchester United", 12, 22, 8, 5, 9, 29, 30, 29),
        ("West Ham", 13, 22, 7, 6, 9, 28, 35, 27),
        ("Tottenham", 14, 22, 7, 5, 10, 40, 35, 26),
        ("Crystal Palace", 15, 22, 5, 9, 8, 24, 30, 24),
        ("Everton", 16, 22, 5, 8, 9, 20, 28, 23),
        ("Wolves", 17, 22, 6, 5, 11, 30, 42, 23),
        ("Leicester", 18, 22, 5, 5, 12, 26, 45, 20),
        ("Ipswich", 19, 22, 3, 8, 11, 21, 42, 17),
        ("Southampton", 20, 22, 2, 5, 15, 15, 47, 11),
    ]
    
    teams = []
    for i, data in enumerate(teams_data):
        name, pos, played, won, drawn, lost, gf, ga, pts = data
        
        # Calculate team strength based on position (1=strongest)
        team_strength = 1 - (i / 20)
        
        # Generate injury rate based on various factors
        injury_rate = 0.1 + random.random() * 0.15
        
        team = Team(
            name=name,
            position=pos,
            played=played,
            won=won,
            drawn=drawn,
            lost=lost,
            goals_for=gf,
            goals_against=ga,
            points=pts,
            last_10_matches=generate_sample_last_10_matches(team_strength),
            squad=generate_sample_squad(team_strength, injury_rate)
        )
        teams.append(team)
    
    return LeagueTable(
        teams=teams,
        matchweek=22,
        season="2024-25"
    )
