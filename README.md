# Premier League Season Predictor 🏆

A machine learning model to predict how the Premier League season will end based on current table standings, injuries, player conditions, and last 10 matches.

## Features

- **Current Standings Analysis**: Incorporates current league position, points, goal difference
- **Form Analysis**: Analyzes last 10 matches including home/away performance
- **Squad Fitness**: Considers injured and suspended players
- **ML-Based Predictions**: Uses Gradient Boosting and Random Forest ensemble
- **Probability Estimates**: Title, Top 4, and Relegation probabilities for each team

## Installation

```bash
# Clone the repository
git clone https://github.com/thatjelvin/PL-right-now.git
cd PL-right-now

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Prediction

```bash
python predict_season.py
```

### Output as JSON

```bash
python predict_season.py --format json
```

### Save Results to File

```bash
python predict_season.py --output predictions.json
```

## Model Details

### Features Used

The model uses 19 features extracted from team data:

| Feature | Description |
|---------|-------------|
| `current_position` | Team's current league position |
| `points` | Current points total |
| `points_per_game` | Average points per game |
| `goal_difference` | Goals scored minus goals conceded |
| `goals_per_game` | Average goals scored per game |
| `goals_conceded_per_game` | Average goals conceded per game |
| `win_rate` | Percentage of games won |
| `draw_rate` | Percentage of games drawn |
| `loss_rate` | Percentage of games lost |
| `recent_form_normalized` | Form based on last 10 matches (0-1) |
| `home_form` | Home performance in recent matches |
| `away_form` | Away performance in recent matches |
| `injured_players_count` | Number of injured players |
| `suspended_players_count` | Number of suspended players |
| `squad_fitness_score` | Overall squad fitness (0-100) |
| `games_remaining` | Games left in the season |
| `max_possible_points` | Maximum achievable points |
| `points_needed_for_top4` | Points gap to 4th place |
| `points_needed_to_avoid_relegation` | Points gap to safety |

### Models

The predictor uses an ensemble of:
- **Gradient Boosting Regressor**: For points prediction
- **Random Forest Regressor**: For position prediction

### Output

The model provides:
- Predicted final points for each team
- Predicted final position
- Title probability
- Top 4 probability
- Relegation probability
- Title contenders list
- Top 4 battle analysis
- Relegation danger zone

## Project Structure

```
PL-right-now/
├── predict_season.py      # Main CLI script
├── requirements.txt       # Python dependencies
├── src/
│   ├── __init__.py
│   ├── data_models.py     # Data classes (Player, Team, Match, League)
│   ├── data_provider.py   # Sample data generation
│   ├── feature_engineering.py  # Feature extraction
│   └── predictor.py       # ML model implementation
└── tests/
    ├── __init__.py
    ├── test_data_models.py
    └── test_predictor.py
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Example Output

```
====================================================================================================
PREMIER LEAGUE SEASON PREDICTION
====================================================================================================

Pos  Team                 Pts   Pred Pts  Pred Pos  Title %   Top 4 %   Rel %   Form   Injuries 
----------------------------------------------------------------------------------------------------
1    Liverpool            52    91.7      1           87.4%    96.7%   0.0%  8.7    4        
2    Arsenal              47    83.2      2           50.9%    98.3%   0.0%  9.3    9        
3    Nottingham Forest    44    76.1      3           32.1%    89.2%   0.0%  5.7    14       
...

🏆 Predicted Champion: Liverpool

📊 Title Contenders:
   • Liverpool: 87.4% chance
   • Arsenal: 50.9% chance
   ...

⚠️ Relegation Danger:
   • Leicester: 81.0% risk
   • Ipswich: 75.0% risk
   • Southampton: 80.0% risk
```

## Extending the Model

### Using Real Data

To use real data instead of sample data, implement a data provider that:

1. Fetches current standings from a football API
2. Retrieves injury reports
3. Gets recent match results

Example integration:

```python
from src.data_models import LeagueTable, Team, Player, MatchResult
from src.predictor import SeasonPredictor

# Create league table from your data source
league = LeagueTable(
    teams=[...],  # List of Team objects
    matchweek=22,
    season="2024-25"
)

# Run predictions
predictor = SeasonPredictor()
result = predictor.predict_remaining_season(league)
```

## License

MIT License - see [LICENSE](LICENSE) for details.
