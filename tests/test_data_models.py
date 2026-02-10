"""
Tests for data models.
"""

import pytest
from src.data_models import Player, MatchResult, Team, LeagueTable


class TestPlayer:
    """Tests for the Player data class."""

    def test_player_creation(self):
        """Test basic player creation."""
        player = Player(name="Test Player", position="Midfielder")
        assert player.name == "Test Player"
        assert player.position == "Midfielder"
        assert player.is_injured is False
        assert player.form_rating == 5.0

    def test_injured_player(self):
        """Test injured player attributes."""
        player = Player(
            name="Injured Player",
            position="Forward",
            is_injured=True,
            injury_severity=7
        )
        assert player.is_injured is True
        assert player.injury_severity == 7


class TestMatchResult:
    """Tests for the MatchResult data class."""

    def test_win_result(self):
        """Test win detection."""
        match = MatchResult(
            opponent="Arsenal",
            is_home=True,
            goals_scored=2,
            goals_conceded=1
        )
        assert match.result == 'W'
        assert match.points == 3

    def test_draw_result(self):
        """Test draw detection."""
        match = MatchResult(
            opponent="Chelsea",
            is_home=False,
            goals_scored=1,
            goals_conceded=1
        )
        assert match.result == 'D'
        assert match.points == 1

    def test_loss_result(self):
        """Test loss detection."""
        match = MatchResult(
            opponent="Liverpool",
            is_home=True,
            goals_scored=0,
            goals_conceded=3
        )
        assert match.result == 'L'
        assert match.points == 0


class TestTeam:
    """Tests for the Team data class."""

    def test_team_creation(self):
        """Test basic team creation."""
        team = Team(
            name="Test FC",
            position=1,
            played=22,
            won=15,
            drawn=4,
            lost=3,
            goals_for=45,
            goals_against=20,
            points=49
        )
        assert team.name == "Test FC"
        assert team.position == 1
        assert team.points == 49

    def test_goal_difference(self):
        """Test goal difference calculation."""
        team = Team(
            name="Test FC",
            position=1,
            played=10,
            won=5,
            drawn=3,
            lost=2,
            goals_for=20,
            goals_against=12,
            points=18
        )
        assert team.goal_difference == 8

    def test_points_per_game(self):
        """Test points per game calculation."""
        team = Team(
            name="Test FC",
            position=1,
            played=10,
            won=5,
            drawn=3,
            lost=2,
            goals_for=20,
            goals_against=12,
            points=18
        )
        assert team.points_per_game == 1.8

    def test_squad_fitness_with_injuries(self):
        """Test squad fitness calculation with injuries."""
        players = [
            Player(name="Player1", position="GK", is_injured=True, injury_severity=5),
            Player(name="Player2", position="DEF", is_injured=False),
            Player(name="Player3", position="MID", is_injured=True, injury_severity=3),
        ]
        team = Team(
            name="Test FC",
            position=1,
            played=10,
            won=5,
            drawn=3,
            lost=2,
            goals_for=20,
            goals_against=12,
            points=18,
            squad=players
        )
        # Total injury severity = 8, max = 30, fitness = 100 - (8/30 * 100) = 73.33
        assert 70 < team.squad_fitness_score < 77

    def test_recent_form(self):
        """Test recent form calculation."""
        matches = [
            MatchResult("A", True, 2, 0),  # W = 3
            MatchResult("B", False, 1, 1),  # D = 1
            MatchResult("C", True, 0, 1),   # L = 0
        ]
        team = Team(
            name="Test FC",
            position=1,
            played=10,
            won=5,
            drawn=3,
            lost=2,
            goals_for=20,
            goals_against=12,
            points=18,
            last_10_matches=matches
        )
        assert team.recent_form == 4  # 3 + 1 + 0


class TestLeagueTable:
    """Tests for the LeagueTable data class."""

    def test_get_team(self):
        """Test getting team by name."""
        teams = [
            Team("Liverpool", 1, 22, 16, 4, 2, 52, 18, 52),
            Team("Arsenal", 2, 22, 14, 5, 3, 48, 22, 47),
        ]
        league = LeagueTable(teams=teams, matchweek=22, season="2024-25")
        
        assert league.get_team("Liverpool") is not None
        assert league.get_team("Liverpool").position == 1
        assert league.get_team("Unknown") is None

    def test_get_teams_by_position_range(self):
        """Test getting teams by position range."""
        teams = [
            Team("Team1", 1, 22, 16, 4, 2, 52, 18, 52),
            Team("Team2", 2, 22, 14, 5, 3, 48, 22, 47),
            Team("Team3", 3, 22, 13, 5, 4, 35, 20, 44),
            Team("Team4", 4, 22, 11, 7, 4, 42, 26, 40),
        ]
        league = LeagueTable(teams=teams, matchweek=22, season="2024-25")
        
        top4 = league.get_teams_by_position_range(1, 4)
        assert len(top4) == 4
        
        top2 = league.get_teams_by_position_range(1, 2)
        assert len(top2) == 2
