"""
src/match_model.py
==================
Poisson-based match outcome model for knockout football.

Implements:
- Dixon-Coles-inspired Poisson xG model with low-score correction
- Win / Extra Time / Penalty Shootout probability estimation (corrected KO rules)
- Score distribution sampling with optional xG noise for uncertainty propagation
- Elo-based win probability (used as ensemble cross-check)

Key fix (v2): In knockout football, 100% of 90-min draws go to ET.
The old model applied ET_BASE_PROBABILITY=0.30, discarding 70% of draw probability.
"""

import logging
from typing import Dict, Tuple, Any, Optional

import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize_scalar

from src import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# POISSON EXPECTED GOALS MODEL
# ─────────────────────────────────────────────────────────────

def compute_expected_goals(
    atk_home: float,
    def_home: float,
    atk_away: float,
    def_away: float,
    is_home: bool = False,
    home_is_host: bool = False,
) -> Tuple[float, float]:
    """
    Compute expected goals (xG) for each team using the Dixon-Coles framework.

    The model:
        xG_home = attack_home * defense_away * league_avg * home_factor
        xG_away = attack_away * defense_home * league_avg

    where defense_strength is how many goals the team concedes per game
    relative to average (lower = better defense).

    Parameters
    ----------
    atk_home : float
        Attack strength of the home/first team (goals scored / avg).
    def_home : float
        Defense strength of the home/first team (goals conceded / avg).
    atk_away : float
        Attack strength of the away/second team.
    def_away : float
        Defense strength of the away/second team.
    is_home : bool
        Whether this is a proper home game (not neutral ground).
    home_is_host : bool
        Whether the home team is a co-host nation.

    Returns
    -------
    tuple[float, float]
        (xg_home, xg_away)
    """
    league_avg = config.LEAGUE_AVG_GOALS

    if is_home:
        home_factor = config.HOME_ADVANTAGE_FACTOR
    elif home_is_host:
        home_factor = config.HOST_NATION_BOOST
    else:
        home_factor = config.NEUTRAL_GROUND_FACTOR

    xg_home = atk_home * def_away * league_avg * home_factor
    xg_away = atk_away * def_home * league_avg

    # Clip to sensible football range
    xg_home = float(np.clip(xg_home, 0.20, 4.50))
    xg_away = float(np.clip(xg_away, 0.20, 4.50))

    return xg_home, xg_away


def _dixon_coles_correction(x: int, y: int, mu: float, nu: float, rho: float) -> float:
    """
    Apply Dixon-Coles low-score correlation correction.

    Low-score matches (0-0, 1-0, 0-1, 1-1) are more common than the
    independent Poisson model predicts. This correction adjusts for that.

    Parameters
    ----------
    x, y : int
        Goals scored by home and away team.
    mu, nu : float
        Expected goals for home and away.
    rho : float
        Correlation parameter (negative → low scores more likely).

    Returns
    -------
    float
        Correction multiplier tau(x, y).
    """
    if x == 0 and y == 0:
        return 1 - mu * nu * rho
    elif x == 0 and y == 1:
        return 1 + mu * rho
    elif x == 1 and y == 0:
        return 1 + nu * rho
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1.0


def score_probability_matrix(
    xg_home: float,
    xg_away: float,
    max_goals: int = 8,
    rho: float = None,
) -> np.ndarray:
    """
    Compute the full score probability matrix P[home_goals, away_goals].

    Uses independent Poisson distributions with Dixon-Coles low-score
    correction (rho defaults to config.DIXON_COLES_RHO = -0.12).

    Parameters
    ----------
    xg_home : float
        Expected goals for home team.
    xg_away : float
        Expected goals for away team.
    max_goals : int
        Maximum goals per team to model (truncated beyond this).
    rho : float or None
        Dixon-Coles correlation parameter. Defaults to config value.

    Returns
    -------
    np.ndarray
        Shape (max_goals+1, max_goals+1) probability matrix.
    """
    if rho is None:
        rho = config.DIXON_COLES_RHO

    goals = np.arange(max_goals + 1)
    p_home = poisson.pmf(goals, xg_home)
    p_away = poisson.pmf(goals, xg_away)

    # Outer product: independent Poisson
    matrix = np.outer(p_home, p_away)

    # Apply DC correction for low scores
    for i in range(min(2, max_goals + 1)):
        for j in range(min(2, max_goals + 1)):
            tau = _dixon_coles_correction(i, j, xg_home, xg_away, rho)
            matrix[i, j] *= tau

    # Renormalise
    matrix /= matrix.sum()
    return matrix


def compute_outcome_probabilities(
    score_matrix: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute P(home win), P(draw), P(away win) from the score matrix.

    Parameters
    ----------
    score_matrix : np.ndarray
        Score probability matrix from score_probability_matrix().

    Returns
    -------
    tuple[float, float, float]
        (p_home_win, p_draw, p_away_win) — sums to 1.0.
    """
    p_home_win = float(np.sum(np.tril(score_matrix, -1)))
    p_draw     = float(np.sum(np.diag(score_matrix)))
    p_away_win = float(np.sum(np.triu(score_matrix,  1)))

    # Renormalise due to floating-point
    total = p_home_win + p_draw + p_away_win
    return p_home_win / total, p_draw / total, p_away_win / total


def compute_knockout_probabilities(
    p_home_win: float,
    p_draw: float,
    p_away_win: float,
    pk_rate_home: float,
    pk_rate_away: float,
) -> Dict[str, float]:
    """
    Compute Win / Extra Time / Penalty Shootout probabilities for a KO match.

    CORRECTED LOGIC (v2):
    In knockout rounds, ALL 90-min draws → extra time (tournament rules).
    Of those reaching ET, ~42% go to penalties (historical WC KO base rate).
    ET goals split proportionally to the 90-min dominance ratio.

    Parameters
    ----------
    p_home_win, p_draw, p_away_win : float
        Poisson 90-minute outcome probabilities (sum to 1.0).
    pk_rate_home, pk_rate_away : float
        Historical penalty shootout win rates for each team.

    Returns
    -------
    dict
        Keys: p_home_90, p_draw_90, p_away_90, p_et, p_pk,
              p_home_total, p_away_total
    """
    # 100% of draws go to ET (KO tournament rule)
    p_et = p_draw * config.ET_BASE_PROBABILITY  # = p_draw * 1.0

    # Of ET games, fraction go to penalty shootout
    p_pk = p_et * config.PENALTY_BASE_PROBABILITY
    p_et_decided = p_et * (1.0 - config.PENALTY_BASE_PROBABILITY)

    # ET decided goals: split proportionally to 90-min strength
    # Use p_home_win / (p_home_win + p_away_win) as conditional prob
    if (p_home_win + p_away_win) > 0:
        home_et_share = p_home_win / (p_home_win + p_away_win)
        away_et_share = p_away_win / (p_home_win + p_away_win)
    else:
        home_et_share = 0.50
        away_et_share = 0.50

    p_home_et_win = p_et_decided * home_et_share
    p_away_et_win = p_et_decided * away_et_share

    # Penalty shootout outcome weighted by historical conversion rates
    pk_total = pk_rate_home + pk_rate_away
    if pk_total > 0:
        p_home_pk_win = p_pk * (pk_rate_home / pk_total)
        p_away_pk_win = p_pk * (pk_rate_away / pk_total)
    else:
        p_home_pk_win = p_pk * 0.50
        p_away_pk_win = p_pk * 0.50

    p_home_total = p_home_win + p_home_et_win + p_home_pk_win
    p_away_total = p_away_win + p_away_et_win + p_away_pk_win

    # Normalise to exactly 1.0
    total = p_home_total + p_away_total
    if total > 0:
        p_home_total /= total
        p_away_total /= total

    return {
        "p_home_90":    round(p_home_win, 4),
        "p_draw_90":    round(p_draw, 4),
        "p_away_90":    round(p_away_win, 4),
        "p_et":         round(p_et, 4),
        "p_pk":         round(p_pk, 4),
        "p_home_total": round(p_home_total, 4),
        "p_away_total": round(p_away_total, 4),
    }


def elo_win_probability(elo_a: float, elo_b: float) -> float:
    """
    Compute Win probability for team A using standard Elo formula.

    Parameters
    ----------
    elo_a, elo_b : float
        Elo ratings for teams A and B.

    Returns
    -------
    float
        Probability that team A wins (0-1).
    """
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def fit_dixon_coles_rho(matches: list) -> float:
    """
    Estimate the Dixon-Coles rho parameter from historical match data via MLE.

    Parameters
    ----------
    matches : list of dict
        Each dict must have keys 'home_goals' and 'away_goals' (int).

    Returns
    -------
    float
        Fitted rho value (typically -0.20 to 0.0).
    """
    if not matches:
        return config.DIXON_COLES_RHO

    avg_home = np.mean([m["home_goals"] for m in matches])
    avg_away = np.mean([m["away_goals"] for m in matches])

    def neg_log_likelihood(rho: float) -> float:
        ll = 0.0
        for m in matches:
            hg = int(m["home_goals"])
            ag = int(m["away_goals"])
            tau = _dixon_coles_correction(hg, ag, avg_home, avg_away, rho)
            tau = max(tau, 1e-10)
            ll += np.log(tau)
        return -ll

    result = minimize_scalar(neg_log_likelihood, bounds=(-0.5, 0.0), method="bounded")
    rho = float(result.x)
    logger.info("Fitted Dixon-Coles rho = %.4f (from %d matches)", rho, len(matches))
    return rho


def sample_score(
    xg_home: float,
    xg_away: float,
    rng: np.random.Generator,
    noise_sigma: float = 0.0,
) -> Tuple[int, int]:
    """
    Sample a match scoreline from the Poisson distribution.

    Parameters
    ----------
    xg_home, xg_away : float
        Expected goals.
    rng : np.random.Generator
        NumPy random generator (for reproducibility).
    noise_sigma : float
        If > 0, adds Gaussian noise to xG before sampling (uncertainty propagation).

    Returns
    -------
    tuple[int, int]
        (home_goals, away_goals)
    """
    if noise_sigma > 0:
        xg_home = float(np.clip(xg_home + rng.normal(0, noise_sigma), 0.1, 6.0))
        xg_away = float(np.clip(xg_away + rng.normal(0, noise_sigma), 0.1, 6.0))
    home_goals = rng.poisson(xg_home)
    away_goals = rng.poisson(xg_away)
    return int(home_goals), int(away_goals)


def most_likely_score(score_matrix: np.ndarray) -> Tuple[int, int]:
    """
    Return the most likely scoreline from the probability matrix.

    Parameters
    ----------
    score_matrix : np.ndarray
        Score probability matrix.

    Returns
    -------
    tuple[int, int]
        (home_goals, away_goals) with highest probability.
    """
    idx = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
    return int(idx[0]), int(idx[1])


def top_k_scores(score_matrix: np.ndarray, k: int = 5) -> list:
    """
    Return the top-k most likely scorelines with probabilities.

    Parameters
    ----------
    score_matrix : np.ndarray
        Score probability matrix.
    k : int
        Number of top scores to return.

    Returns
    -------
    list of dict
        [{'home': int, 'away': int, 'prob': float}, ...]
    """
    flat_indices = np.argsort(score_matrix, axis=None)[::-1][:k]
    results = []
    for fi in flat_indices:
        i, j = np.unravel_index(fi, score_matrix.shape)
        results.append({
            "home": int(i),
            "away": int(j),
            "prob": round(float(score_matrix[i, j]), 4),
        })
    return results


class MatchPredictor:
    """
    End-to-end match prediction for a single knockout fixture.

    Attributes
    ----------
    home_team : str
    away_team : str
    features : dict
        Fixture feature dict from feature_engineering.
    xg_home, xg_away : float
    score_matrix : np.ndarray
    outcome_probs : dict
    most_likely_score : tuple[int, int]
    elo_win_prob_home : float
    """

    def __init__(
        self,
        home_team: str,
        away_team: str,
        features: Dict[str, Any],
        home_is_host: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        home_team : str
            Listed-first team name.
        away_team : str
            Listed-second team name.
        features : dict
            Output of compute_fixture_features().
        home_is_host : bool
            True if home team is a co-host.
        """
        self.home_team = home_team
        self.away_team = away_team
        self.features  = features

        # Compute expected goals
        self.xg_home, self.xg_away = compute_expected_goals(
            atk_home=features["atk_a"],
            def_home=features["def_a"],
            atk_away=features["atk_b"],
            def_away=features["def_b"],
            is_home=False,               # WC is neutral ground
            home_is_host=home_is_host,
        )

        # Score probability matrix
        self.score_matrix = score_probability_matrix(self.xg_home, self.xg_away)

        # Outcome probabilities
        p_hw, p_d, p_aw = compute_outcome_probabilities(self.score_matrix)
        self.outcome_probs = compute_knockout_probabilities(
            p_hw, p_d, p_aw,
            pk_rate_home=features.get("pk_rate_a", 0.50),
            pk_rate_away=features.get("pk_rate_b", 0.50),
        )

        # Most likely score
        self.most_likely_score = most_likely_score(self.score_matrix)
        self.top_scores = top_k_scores(self.score_matrix, k=5)

        # Elo-based win probability (ensemble cross-check)
        elo_a = features.get("f_elo_a", 0.5) * (config.ELO_MAX - config.ELO_MIN) + config.ELO_MIN
        elo_b = features.get("f_elo_b", 0.5) * (config.ELO_MAX - config.ELO_MIN) + config.ELO_MIN
        self.elo_win_prob_home = elo_win_probability(elo_a, elo_b)

        logger.debug(
            "%s vs %s | xG: %.2f–%.2f | P(home): %.1f%% | MLS: %d-%d",
            home_team, away_team, self.xg_home, self.xg_away,
            self.outcome_probs["p_home_total"] * 100,
            *self.most_likely_score,
        )

    def simulate_once(self, rng: np.random.Generator, noise_sigma: float = 0.0) -> str:
        """
        Simulate one match outcome for Monte Carlo.

        CORRECTED LOGIC (v2):
        - Draws ALWAYS go to ET in knockout football (no conditional).
        - ET goal probability splits on 90-min dominance ratio only.
        - ET penalty probability uses team-level PK historical rates.

        Parameters
        ----------
        rng : np.random.Generator
            NumPy seeded generator.
        noise_sigma : float
            If > 0, adds xG noise per simulation for uncertainty propagation.

        Returns
        -------
        str
            Name of the winning team.
        """
        home_g, away_g = sample_score(self.xg_home, self.xg_away, rng, noise_sigma)

        if home_g > away_g:
            return self.home_team
        elif away_g > home_g:
            return self.away_team
        else:
            # Draw → ALWAYS goes to Extra Time in KO rounds
            # Determine ET resolution: either goal in ET or penalties
            p_pk_given_et = config.PENALTY_BASE_PROBABILITY

            if rng.uniform() < p_pk_given_et:
                # Penalty shootout
                pk_a = self.features.get("pk_rate_a", 0.50)
                pk_b = self.features.get("pk_rate_b", 0.50)
                pk_total = pk_a + pk_b
                p_home_wins_pk = pk_a / pk_total if pk_total > 0 else 0.50
                return self.home_team if rng.uniform() < p_home_wins_pk else self.away_team
            else:
                # ET goal — split by 90-min dominance ratio (not total probability)
                p_hw = self.outcome_probs["p_home_90"]
                p_aw = self.outcome_probs["p_away_90"]
                denom = p_hw + p_aw
                p_home_et = p_hw / denom if denom > 0 else 0.50
                return self.home_team if rng.uniform() < p_home_et else self.away_team
