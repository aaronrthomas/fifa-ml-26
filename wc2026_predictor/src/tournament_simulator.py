"""
src/tournament_simulator.py
============================
Monte Carlo tournament simulator for WC 2026 Knockout Stage.

Runs N_SIMULATIONS full tournament simulations from QF → Final.
Accumulates probability distributions for every stage.
Results are cached to disk (hash-checked) to avoid redundant re-runs.
"""

import os
import hashlib
import json
import logging
import pickle
from typing import Dict, List, Optional, Any

import numpy as np
from tqdm import tqdm

from src import config
from src.data_ingestion import DataIngestion
from src.feature_engineering import compute_fixture_features, compute_tsi
from src.match_model import MatchPredictor

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# BRACKET RESOLUTION
# ─────────────────────────────────────────────────────────────

def resolve_bracket_teams(
    r16_results: Dict[str, str],
    data_ingestion: DataIngestion,
) -> Dict[str, str]:
    """
    Determine the most likely QF team for each slot based on R16 results
    or model-predicted R16 outcomes.

    If a R16 result is confirmed (from live scrape), use it directly.
    Otherwise, run the match model to predict the most probable winner.

    Parameters
    ----------
    r16_results : dict
        Maps match_id → winning team (empty if not yet played).
    data_ingestion : DataIngestion
        Loaded data object.

    Returns
    -------
    dict
        Maps match_id → resolved winning team name.
    """
    resolved: Dict[str, str] = dict(r16_results)

    for match_id, team_a, team_b, date in config.R16_MATCHES:
        if match_id in resolved:
            logger.info("R16 %s: %s (confirmed)", match_id, resolved[match_id])
            continue

        # Model-predict the winner
        features = compute_fixture_features(team_a, team_b, data_ingestion)
        predictor = MatchPredictor(team_a, team_b, features)
        winner = team_a if predictor.outcome_probs["p_home_total"] >= 0.50 else team_b
        resolved[match_id] = winner
        logger.info(
            "R16 %s: %s (predicted, P=%.1f%%)",
            match_id, winner,
            predictor.outcome_probs["p_home_total"] * 100
            if winner == team_a
            else predictor.outcome_probs["p_away_total"] * 100,
        )

    return resolved


def build_qf_lineup(resolved_r16: Dict[str, str]) -> List[tuple]:
    """
    Build the list of QF fixtures from resolved R16 results.

    Parameters
    ----------
    resolved_r16 : dict
        match_id → winning team.

    Returns
    -------
    list of (qf_id, team_a, team_b, date, venue)
    """
    qf_fixtures = []
    for qf_id, r16_a_id, r16_b_id, date, venue in config.QF_BRACKET:
        team_a = resolved_r16.get(r16_a_id, "TBD_A")
        team_b = resolved_r16.get(r16_b_id, "TBD_B")
        qf_fixtures.append((qf_id, team_a, team_b, date, venue))
        logger.info("QF %s: %s vs %s", qf_id, team_a, team_b)
    return qf_fixtures


# ─────────────────────────────────────────────────────────────
# MATCH PREDICTOR CACHE
# ─────────────────────────────────────────────────────────────

class FixtureCache:
    """
    Caches MatchPredictor objects to avoid redundant feature computations
    during Monte Carlo simulations (which simulate the same possible
    fixtures many times).
    """

    def __init__(self, data_ingestion: DataIngestion) -> None:
        self._data = data_ingestion
        self._cache: Dict[tuple, MatchPredictor] = {}

    def get_predictor(self, team_a: str, team_b: str) -> MatchPredictor:
        """
        Return a cached MatchPredictor for (team_a, team_b).

        Parameters
        ----------
        team_a, team_b : str
            Team names.

        Returns
        -------
        MatchPredictor
        """
        key = (team_a, team_b)
        if key not in self._cache:
            features = compute_fixture_features(team_a, team_b, self._data)
            is_host = team_a in config.HOST_NATIONS
            self._cache[key] = MatchPredictor(team_a, team_b, features, home_is_host=is_host)
        return self._cache[key]


# ─────────────────────────────────────────────────────────────
# SINGLE TOURNAMENT SIMULATION
# ─────────────────────────────────────────────────────────────

def _simulate_stage(
    fixtures: List[tuple],
    fixture_cache: FixtureCache,
    rng: np.random.Generator,
) -> List[str]:
    """
    Simulate one stage (e.g. QF) and return the list of winners.

    Parameters
    ----------
    fixtures : list of (id, team_a, team_b, date, venue)
    fixture_cache : FixtureCache
    rng : np.random.Generator

    Returns
    -------
    list of str
        Winners of each fixture in order.
    """
    winners = []
    for fixture_id, team_a, team_b, *_ in fixtures:
        predictor = fixture_cache.get_predictor(team_a, team_b)
        winner = predictor.simulate_once(rng, noise_sigma=config.XG_NOISE_SIGMA)
        winners.append(winner)
    return winners


def _simulate_full_tournament(
    qf_fixtures: List[tuple],
    fixture_cache: FixtureCache,
    rng: np.random.Generator,
) -> Dict[str, str]:
    """
    Run one complete WC knockout tournament simulation from QF to Final.

    Parameters
    ----------
    qf_fixtures : list of (id, team_a, team_b, date, venue)
    fixture_cache : FixtureCache
    rng : np.random.Generator

    Returns
    -------
    dict
        Keys: qf_winners (list), sf_winners (list), third_place, champion, runner_up
    """
    # Quarter Finals
    qf_winners = _simulate_stage(qf_fixtures, fixture_cache, rng)

    # Build SF fixtures from QF winners
    sf_fixtures = []
    for i, (sf_id, qf_a_id, qf_b_id, date, venue) in enumerate(config.SF_BRACKET):
        # QF1 winner vs QF2 winner, QF3 winner vs QF4 winner
        team_a = qf_winners[2 * i]     if 2 * i < len(qf_winners) else "TBD"
        team_b = qf_winners[2 * i + 1] if 2 * i + 1 < len(qf_winners) else "TBD"
        sf_fixtures.append((sf_id, team_a, team_b, date, venue))

    # Semi Finals
    sf_winners = _simulate_stage(sf_fixtures, fixture_cache, rng)
    sf_losers  = [
        sf_fixtures[i][1] if sf_winners[i] == sf_fixtures[i][2] else sf_fixtures[i][2]
        for i in range(len(sf_fixtures))
    ]

    # Final
    fin_id, fin_sf_a, fin_sf_b, fin_date, fin_venue = config.FINAL
    final_teams = sf_winners[:2] if len(sf_winners) >= 2 else ["TBD", "TBD"]
    final_fix = [(fin_id, final_teams[0], final_teams[1], fin_date, fin_venue)]
    final_winner = _simulate_stage(final_fix, fixture_cache, rng)[0]
    final_loser  = final_teams[1] if final_winner == final_teams[0] else final_teams[0]

    # Third place
    third_teams = sf_losers[:2] if len(sf_losers) >= 2 else ["TBD", "TBD"]
    third_fix = [("3RD", third_teams[0], third_teams[1], config.THIRD_PLACE[3], config.THIRD_PLACE[4])]
    third_winner = _simulate_stage(third_fix, fixture_cache, rng)[0]

    return {
        "qf_winners":    qf_winners,
        "sf_winners":    sf_winners,
        "sf_losers":     sf_losers,
        "champion":      final_winner,
        "runner_up":     final_loser,
        "third_place":   third_winner,
        "all_teams":     [f[1] for f in qf_fixtures] + [f[2] for f in qf_fixtures],
    }


# ─────────────────────────────────────────────────────────────
# MONTE CARLO ENGINE
# ─────────────────────────────────────────────────────────────

class TournamentSimulator:
    """
    Monte Carlo engine for WC 2026 Knockout Stage.

    Runs N_SIMULATIONS full tournament simulations and accumulates
    probability distributions for every team at every stage.

    Attributes
    ----------
    n_sims : int
        Number of simulations run.
    results : dict
        Raw simulation output counts.
    probabilities : dict
        Normalised probability distributions.
    """

    def __init__(
        self,
        qf_fixtures: List[tuple],
        data_ingestion: DataIngestion,
        n_sims: int = config.N_SIMULATIONS,
    ) -> None:
        """
        Parameters
        ----------
        qf_fixtures : list of (id, team_a, team_b, date, venue)
            Quarter-final fixture list.
        data_ingestion : DataIngestion
        n_sims : int
            Number of Monte Carlo simulations.
        """
        self.qf_fixtures   = qf_fixtures
        self.data          = data_ingestion
        self.n_sims        = n_sims
        self.fixture_cache = FixtureCache(data_ingestion)

        # All possible teams in this tournament
        self.all_teams: List[str] = []
        for _, ta, tb, *_ in qf_fixtures:
            if ta not in self.all_teams:
                self.all_teams.append(ta)
            if tb not in self.all_teams:
                self.all_teams.append(tb)

        self.results:       Dict[str, Any] = {}
        self.probabilities: Dict[str, Any] = {}

    def _init_counters(self) -> Dict[str, Any]:
        """Initialise simulation counters."""
        return {
            "qf_winner":     {t: 0 for t in self.all_teams},
            "sf_qualifier":  {t: 0 for t in self.all_teams},
            "finalist":      {t: 0 for t in self.all_teams},
            "champion":      {t: 0 for t in self.all_teams},
            "runner_up":     {t: 0 for t in self.all_teams},
            "third_place":   {t: 0 for t in self.all_teams},
        }

    def run(self) -> None:
        """
        Run all Monte Carlo simulations.

        Pre-builds all possible fixture predictors, then loops N_SIMULATIONS
        times sampling one tournament result each iteration.
        """
        # Check for cached results
        cache_path = self._cache_path()
        if os.path.exists(cache_path):
            logger.info("Loading cached simulation results from %s", cache_path)
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            self.results       = cached["results"]
            self.probabilities = cached["probabilities"]
            logger.info("Cache loaded. Champion probs: %s",
                        {k: f'{v:.1%}' for k, v in
                         sorted(self.probabilities["champion"].items(),
                                key=lambda x: -x[1])[:5]})
            return

        logger.info(
            "Running %s Monte Carlo simulations...", f"{self.n_sims:,}"
        )

        # Pre-warm fixture cache for all QF matchups
        for _, ta, tb, *_ in self.qf_fixtures:
            self.fixture_cache.get_predictor(ta, tb)

        rng     = np.random.default_rng(config.RANDOM_SEED)
        counts  = self._init_counters()

        for _ in tqdm(range(self.n_sims), desc="Simulating", unit="sim"):
            result = _simulate_full_tournament(self.qf_fixtures, self.fixture_cache, rng)

            for team in result.get("qf_winners", []):
                if team in counts["qf_winner"]:
                    counts["qf_winner"][team] += 1

            for team in result.get("sf_winners", []):
                if team in counts["sf_qualifier"]:
                    counts["sf_qualifier"][team] += 1

            champion   = result.get("champion")
            runner_up  = result.get("runner_up")
            third      = result.get("third_place")

            if champion in counts["finalist"]:
                counts["finalist"][champion] += 1
            if runner_up in counts["finalist"]:
                counts["finalist"][runner_up] += 1
            if champion in counts["champion"]:
                counts["champion"][champion] += 1
            if runner_up in counts["runner_up"]:
                counts["runner_up"][runner_up] += 1
            if third and third in counts["third_place"]:
                counts["third_place"][third] += 1

        self.results = counts
        self.probabilities = {
            stage: {
                team: count / self.n_sims
                for team, count in stage_counts.items()
            }
            for stage, stage_counts in counts.items()
        }

        logger.info("Simulation complete.")
        self._log_top_results()
        self._save_cache(cache_path)

    def _log_top_results(self) -> None:
        """Log the top champion probabilities."""
        champ = sorted(
            self.probabilities.get("champion", {}).items(),
            key=lambda x: -x[1]
        )
        logger.info("=== Champion Probabilities ===")
        for team, prob in champ:
            if prob > 0.005:
                logger.info("  %-15s %.1f%%", team, prob * 100)

    def _cache_key(self) -> str:
        """Generate a hash-based cache key from fixtures, sim count, and model params.

        Critical: must include all model parameters that affect simulation outcomes.
        Any change to ET probability, rho, ensemble weights etc. invalidates the cache.
        """
        key_data = json.dumps({
            "fixtures":             [(f[0], f[1], f[2]) for f in self.qf_fixtures],
            "n_sims":               self.n_sims,
            "seed":                 config.RANDOM_SEED,
            # Model parameters — cache is invalid if any of these change
            "et_base_prob":         config.ET_BASE_PROBABILITY,
            "penalty_base_prob":    config.PENALTY_BASE_PROBABILITY,
            "rho":                  config.DIXON_COLES_RHO,
            "xg_noise_sigma":       config.XG_NOISE_SIGMA,
            "ensemble_poisson_w":   config.ENSEMBLE_POISSON_WEIGHT,
            "ensemble_ml_w":        config.ENSEMBLE_ML_WEIGHT,
            "league_avg_goals":     config.LEAGUE_AVG_GOALS,
        }, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()[:12]

    def _cache_path(self) -> str:
        """Return the path to the simulation cache file."""
        os.makedirs(config.SIM_CACHE_DIR, exist_ok=True)
        return os.path.join(config.SIM_CACHE_DIR, f"sim_{self._cache_key()}.pkl")

    def _save_cache(self, path: str) -> None:
        """Persist simulation results to disk."""
        with open(path, "wb") as f:
            pickle.dump({
                "results":       self.results,
                "probabilities": self.probabilities,
                "n_sims":        self.n_sims,
                "fixtures":      self.qf_fixtures,
            }, f)
        logger.info("Simulation results cached to %s", path)

    def get_champion_odds(self) -> List[tuple]:
        """
        Return sorted list of (team, champion_probability) descending.

        Returns
        -------
        list of (str, float)
        """
        return sorted(
            self.probabilities.get("champion", {}).items(),
            key=lambda x: -x[1]
        )

    def get_stage_probs(self, stage: str) -> Dict[str, float]:
        """
        Return probability dict for a given stage.

        Parameters
        ----------
        stage : str
            One of: qf_winner, sf_qualifier, finalist, champion, runner_up.

        Returns
        -------
        dict
            team → probability
        """
        return self.probabilities.get(stage, {})

    def get_upset_probabilities(self) -> List[Dict[str, Any]]:
        """
        Identify upset probabilities — cases where the lower-ranked team
        has a meaningful chance of winning.

        Returns
        -------
        list of dict
            Fixtures sorted by upset potential.
        """
        upsets = []
        for _, team_a, team_b, date, venue in self.qf_fixtures:
            predictor = self.fixture_cache.get_predictor(team_a, team_b)
            p_a = predictor.outcome_probs["p_home_total"]
            p_b = predictor.outcome_probs["p_away_total"]
            # Upset if underdog has >30% chance
            if min(p_a, p_b) > 0.30:
                upset_team = team_a if p_a < p_b else team_b
                upsets.append({
                    "fixture":     f"{team_a} vs {team_b}",
                    "upset_team":  upset_team,
                    "upset_prob":  round(min(p_a, p_b), 3),
                    "date":        date,
                })
        return sorted(upsets, key=lambda x: -x["upset_prob"])
