"""
Data models for Premier League prediction.

This module contains data classes representing Premier League teams,
their standings, injuries, and match history.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import date


@dataclass
class Player:
    """Represents a player in the Premier League."""
    name: str
    position: str
    is_injured: bool = False
    injury_severity: int = 0  # 0-10 scale, 0 = not injured
    is_suspended: bool = False
    form_rating: float = 5.0  # 0-10 scale


@dataclass
class MatchResult:
    """Represents a single match result."""
    opponent: str
    is_home: bool
    goals_scored: int
    goals_conceded: int
    date: Optional[str] = None

    @property
    def result(self) -> str:
        """Return W, D, or L based on score."""
        if self.goals_scored > self.goals_conceded:
            return 'W'
        elif self.goals_scored < self.goals_conceded:
            return 'L'
        return 'D'

    @property
    def points(self) -> int:
        """Return points earned from this match."""
        if self.result == 'W':
            return 3
        elif self.result == 'D':
            return 1
        return 0


@dataclass
class Team:
    """Represents a Premier League team with all relevant data."""
    name: str
    position: int
    played: int
    won: int
    drawn: int
    lost: int
    goals_for: int
    goals_against: int
    points: int
    last_10_matches: List[MatchResult] = field(default_factory=list)
    squad: List[Player] = field(default_factory=list)

    @property
    def goal_difference(self) -> int:
        """Calculate goal difference."""
        return self.goals_for - self.goals_against

    @property
    def points_per_game(self) -> float:
        """Calculate average points per game."""
        if self.played == 0:
            return 0.0
        return self.points / self.played

    @property
    def injured_players_count(self) -> int:
        """Count of injured players."""
        return sum(1 for p in self.squad if p.is_injured)

    @property
    def suspended_players_count(self) -> int:
        """Count of suspended players."""
        return sum(1 for p in self.squad if p.is_suspended)

    @property
    def squad_fitness_score(self) -> float:
        """Calculate overall squad fitness (0-100)."""
        if not self.squad:
            return 100.0
        total_impact = sum(p.injury_severity for p in self.squad if p.is_injured)
        max_impact = len(self.squad) * 10
        return max(0, 100 - (total_impact / max_impact * 100))

    @property
    def recent_form(self) -> float:
        """Calculate form based on last 10 matches (0-30 points max)."""
        if not self.last_10_matches:
            return 15.0  # Neutral form if no data
        return sum(m.points for m in self.last_10_matches[-10:])

    @property
    def recent_form_normalized(self) -> float:
        """Normalize recent form to 0-1 scale."""
        return self.recent_form / 30.0

    @property
    def home_form(self) -> float:
        """Calculate home form from last 10 matches."""
        home_matches = [m for m in self.last_10_matches if m.is_home]
        if not home_matches:
            return 0.5
        return sum(m.points for m in home_matches) / (len(home_matches) * 3)

    @property
    def away_form(self) -> float:
        """Calculate away form from last 10 matches."""
        away_matches = [m for m in self.last_10_matches if not m.is_home]
        if not away_matches:
            return 0.5
        return sum(m.points for m in away_matches) / (len(away_matches) * 3)


@dataclass
class LeagueTable:
    """Represents the current Premier League table."""
    teams: List[Team]
    matchweek: int
    season: str

    def get_team(self, name: str) -> Optional[Team]:
        """Get a team by name."""
        for team in self.teams:
            if team.name.lower() == name.lower():
                return team
        return None

    def get_teams_by_position_range(self, start: int, end: int) -> List[Team]:
        """Get teams within a position range."""
        return [t for t in self.teams if start <= t.position <= end]
