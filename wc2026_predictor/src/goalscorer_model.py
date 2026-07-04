"""
src/goalscorer_model.py
=======================
Player-level goal scorer prediction for the WC2026 Knockout Stage.

Predicts:
- Which players score in each match
- How many goals they score
- Returns jersey numbers for the submission CSV

Methodology:
- Each player's scoring probability is proportional to:
    scoring_rate * expected_minutes * role_weight * availability
- Total expected goals (from Poisson model) are allocated among players
- Penalty takers get additional probability for penalty-scenario goals
"""

import logging
from typing import Dict, List, Any, Tuple

import numpy as np

from src import config
from src.data_ingestion import DataIngestion

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# ROLE WEIGHTS
# Multipliers applied to base scoring rate by position/role
# ─────────────────────────────────────────────────────────────
ROLE_WEIGHTS: Dict[str, float] = {
    "FWD": 1.00,   # Forwards: full weight
    "MID": 0.55,   # Midfielders: meaningful but lower
    "DEF": 0.18,   # Defenders: occasional goals
    "GK":  0.00,   # Goalkeepers: essentially zero
}


def _player_goal_weight(player: Dict[str, Any]) -> float:
    """
    Compute a player's relative goal-scoring weight.

    Weight = scoring_rate * avg_minutes/90 * role_weight * availability

    Parameters
    ----------
    player : dict
        Row from player_data.csv as a dict.

    Returns
    -------
    float
        Unnormalised scoring weight (higher = more likely to score).
    """
    rate         = float(player.get("scoring_rate", 0.0))
    minutes      = float(player.get("avg_minutes_per_game", 60.0))
    pos          = str(player.get("position", "MID"))
    role_w       = ROLE_WEIGHTS.get(pos, 0.55)
    availability = float(player.get("availability", 1.0))

    weight = rate * (minutes / 90.0) * role_w * availability
    return max(weight, 0.0)


def _penalty_taker_bonus(player: Dict[str, Any], p_penalty: float) -> float:
    """
    Add penalty shootout scoring bonus for the designated penalty taker.

    In penalty shootouts, the PK taker is almost guaranteed to have at
    least one attempt. We add a small bonus proportional to P(penalties).

    Parameters
    ----------
    player : dict
    p_penalty : float
        Probability that the match goes to penalties.

    Returns
    -------
    float
        Penalty bonus to add to base weight.
    """
    if int(player.get("pk_taker", 0)) == 1:
        return p_penalty * 0.80  # 80% chance PK taker converts
    return 0.0


def predict_scorers(
    team: str,
    expected_goals: float,
    p_penalty: float,
    players_df,
    rng: np.random.Generator,
    n_samples: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Predict goal scorers for a team in a match.

    Parameters
    ----------
    team : str
        Team name.
    expected_goals : float
        Expected goals for this team (from Poisson model).
    p_penalty : float
        Probability this match goes to penalties (for PK bonus).
    players_df : pd.DataFrame
        Player data from DataIngestion.
    rng : np.random.Generator
        Seeded random generator.
    n_samples : int
        Number of Monte Carlo scorer samples for averaging.

    Returns
    -------
    list of dict
        Sorted by scoring probability. Each dict has:
        {player_name, jersey, position, goal_prob, expected_goals}
    """
    team_players = players_df[players_df["team"] == team]
    if team_players.empty:
        logger.warning("No player data found for %s", team)
        return []

    players = team_players.to_dict("records")

    # Compute base weights
    weights = []
    for p in players:
        w = _player_goal_weight(p) + _penalty_taker_bonus(p, p_penalty)
        weights.append(w)

    weights = np.array(weights, dtype=float)
    total_w = weights.sum()

    if total_w == 0:
        # Uniform distribution as fallback
        weights = np.ones(len(players))
        total_w = weights.sum()

    # Normalise to probability distribution
    probs = weights / total_w

    # Determine how many goals to distribute
    # Sample from Poisson for n_samples simulations
    goal_counts = rng.poisson(expected_goals, size=n_samples)

    # Allocate goals to players using multinomial sampling
    scorer_counts = np.zeros(len(players), dtype=float)
    for n_goals in goal_counts:
        if n_goals == 0:
            continue
        # Sample which players score each goal
        scorers = rng.choice(len(players), size=n_goals, p=probs, replace=True)
        for s in scorers:
            scorer_counts[s] += 1

    # Average over samples
    avg_goals_per_player = scorer_counts / n_samples

    # Build result list
    results = []
    for i, p in enumerate(players):
        # goal_prob: probability player scores at least 1 goal
        # = 1 - P(scores 0 goals) = 1 - exp(-expected_goals_for_player)
        exp_goals = float(avg_goals_per_player[i])
        goal_prob = float(np.clip(1.0 - np.exp(-exp_goals), 0.0, 1.0))
        results.append({
            "player_name":    p.get("player_name", f"Player_{i}"),
            "jersey":         int(p.get("jersey", i + 1)),
            "position":       p.get("position", "MID"),
            "goal_prob":      round(goal_prob, 4),
            "expected_goals": round(exp_goals, 4),
            "pk_taker":       int(p.get("pk_taker", 0)),
        })

    # Sort by expected goals descending
    return sorted(results, key=lambda x: -x["expected_goals"])


def predict_match_scorers(
    home_team: str,
    away_team: str,
    xg_home: float,
    xg_away: float,
    home_score: int,
    away_score: int,
    p_penalty: float,
    players_df,
    rng: np.random.Generator,
) -> Tuple[List[int], List[int]]:
    """
    Predict actual goal scorers (jersey numbers) for a specific match result.

    Parameters
    ----------
    home_team, away_team : str
    xg_home, xg_away : float
        Expected goals used for scorer weighting.
    home_score, away_score : int
        The predicted final scoreline to allocate goals to.
    p_penalty : float
        Probability of penalty shootout.
    players_df : pd.DataFrame
    rng : np.random.Generator

    Returns
    -------
    tuple[list[int], list[int]]
        (home_jersey_numbers, away_jersey_numbers) for scorers.
    """
    def _pick_jerseys(team: str, n_goals: int, xg: float) -> List[int]:
        if n_goals == 0:
            return []

        team_players = players_df[players_df["team"] == team]
        if team_players.empty:
            return []

        players = team_players.to_dict("records")
        weights = np.array([_player_goal_weight(p) for p in players], dtype=float)
        total_w = weights.sum()
        if total_w == 0:
            weights = np.ones(len(players))
            total_w = weights.sum()
        probs = weights / total_w

        # Sample n_goals scorers (with replacement — a player can score twice)
        scorer_indices = rng.choice(len(players), size=n_goals, p=probs, replace=True)
        jerseys = [int(players[i]["jersey"]) for i in scorer_indices]
        return sorted(jerseys)

    home_jerseys = _pick_jerseys(home_team, home_score, xg_home)
    away_jerseys = _pick_jerseys(away_team, away_score, xg_away)

    return home_jerseys, away_jerseys


class GoalscorerModel:
    """
    Wrapper model for predicting scorers across all tournament fixtures.

    Attributes
    ----------
    players_df : pd.DataFrame
    rng : np.random.Generator
    """

    def __init__(self, data_ingestion: DataIngestion) -> None:
        self.players_df = data_ingestion.players
        self.rng        = np.random.default_rng(config.RANDOM_SEED)

    def predict_fixture(
        self,
        home_team: str,
        away_team: str,
        xg_home: float,
        xg_away: float,
        home_score: int,
        away_score: int,
        p_penalty: float = 0.0,
    ) -> Tuple[List[int], List[int]]:
        """
        Predict jersey numbers of scorers for a fixture.

        Parameters
        ----------
        home_team, away_team : str
        xg_home, xg_away : float
        home_score, away_score : int
            Most-likely predicted scoreline.
        p_penalty : float

        Returns
        -------
        tuple[list[int], list[int]]
            (home_scorer_jerseys, away_scorer_jerseys)
        """
        return predict_match_scorers(
            home_team, away_team,
            xg_home, xg_away,
            home_score, away_score,
            p_penalty,
            self.players_df,
            self.rng,
        )

    def top_scorers(self, team: str, xg: float, p_penalty: float = 0.0) -> List[Dict]:
        """
        Return the top likely scorers for a team (for the report).

        Parameters
        ----------
        team : str
        xg : float
            Expected goals for this team.
        p_penalty : float

        Returns
        -------
        list of dict
        """
        return predict_scorers(
            team, xg, p_penalty, self.players_df, self.rng, n_samples=5000
        )[:5]
