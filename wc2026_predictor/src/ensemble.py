"""
src/ensemble.py
===============
Ensemble predictor: blends Dixon-Coles Poisson win probability with
the calibrated ML classifier's win probability.

Provides a drop-in replacement interface for MatchPredictor, so the
rest of the pipeline (main.py, output_generator.py) needs no changes.

Blend formula:
    p_home_ensemble = POISSON_WEIGHT * p_poisson + ML_WEIGHT * p_ml

Both components are already calibrated:
 - Poisson model: inherently probabilistic, rho-corrected
 - ML model: wrapped in CalibratedClassifierCV(isotonic) during training
"""

import logging
import os
from typing import Dict, List, Any, Optional

import numpy as np
import joblib
import json

from src import config
from src.match_model import MatchPredictor

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTION FOR ML MODEL
# Maps fixture feature dict → ML feature vector
# Must match FEATURE_COLS order in ml_validation.py
# ─────────────────────────────────────────────────────────────

_ML_FEATURE_COLS = [
    "elo_diff", "form_diff", "squad_val_diff", "xg_diff",
    "gd_diff", "h2h_adv", "host", "tournament_diff",
]


def _features_to_ml_vector(features: Dict[str, Any]) -> np.ndarray:
    """
    Extract the ML feature vector from a fixture feature dict.

    Parameters
    ----------
    features : dict
        Output of compute_fixture_features().

    Returns
    -------
    np.ndarray
        Shape (1, n_features) — ready for model.predict_proba().
    """
    # Map fixture features to ML model input columns
    # elo_diff: raw Elo difference (team_a - team_b)
    elo_diff = features.get("elo_diff", 0.0)

    # form_diff: normalised form score difference
    form_diff = features.get("form_diff", 0.0)

    # squad_val_diff: normalised squad value difference
    squad_val_diff = features.get("squad_val_diff", 0.0)

    # xg_diff: xG differential difference — use tsi feature difference
    xg_diff = features.get("xg_diff_diff", 0.0)

    # gd_diff: goal difference differential — use tsi feature difference
    gd_diff = features.get("f_xg_diff_a", 0.0) - features.get("f_xg_diff_b", 0.0)

    # h2h_adv: H2H win rate for team_a
    h2h_adv = features.get("h2h_a_win_rate", 0.5)

    # host: 1 if team_a is a host nation
    host = float(features.get("f_host_a", 0.0) > 0)

    # tournament_diff: KO win rate difference
    tournament_diff = features.get("tournament_diff", 0.0)

    return np.array([[elo_diff, form_diff, squad_val_diff, xg_diff,
                      gd_diff, h2h_adv, host, tournament_diff]], dtype=float)


class EnsemblePredictor:
    """
    Ensemble match predictor blending Poisson and calibrated ML probabilities.

    This class wraps a MatchPredictor (Poisson model) and optionally blends
    it with the saved best calibrated ML classifier. If the ML model file
    does not exist (e.g., first run before `--validate-only`), it falls back
    gracefully to pure Poisson.

    Exposes the same interface as MatchPredictor for drop-in compatibility.

    Attributes
    ----------
    home_team, away_team : str
    features : dict
    poisson_predictor : MatchPredictor
    outcome_probs : dict  (blended)
    xg_home, xg_away : float
    most_likely_score : tuple
    top_scores : list
    elo_win_prob_home : float
    ml_win_prob_home : float or None
    blend_weight_poisson : float
    blend_weight_ml : float
    """

    _ml_model = None    # class-level cache: loaded once, shared across instances
    _ml_loaded = False  # sentinel to avoid repeated failed loads

    def __init__(
        self,
        home_team: str,
        away_team: str,
        features: Dict[str, Any],
        home_is_host: bool = False,
        poisson_weight: float = config.ENSEMBLE_POISSON_WEIGHT,
        ml_weight: float = config.ENSEMBLE_ML_WEIGHT,
    ) -> None:
        """
        Parameters
        ----------
        home_team : str
        away_team : str
        features : dict
            Output of compute_fixture_features().
        home_is_host : bool
        poisson_weight : float
            Weight for the Poisson model (default from config).
        ml_weight : float
            Weight for the ML model (default from config).
        """
        self.home_team = home_team
        self.away_team = away_team
        self.features  = features
        self.blend_weight_poisson = poisson_weight
        self.blend_weight_ml = ml_weight

        # Build Poisson model
        self.poisson_predictor = MatchPredictor(
            home_team, away_team, features, home_is_host=home_is_host
        )

        # Copy convenience attributes
        self.xg_home          = self.poisson_predictor.xg_home
        self.xg_away          = self.poisson_predictor.xg_away
        self.score_matrix     = self.poisson_predictor.score_matrix
        self.most_likely_score = self.poisson_predictor.most_likely_score
        self.top_scores       = self.poisson_predictor.top_scores
        self.elo_win_prob_home = self.poisson_predictor.elo_win_prob_home

        # Try to get ML probability
        self.ml_win_prob_home = self._get_ml_probability(features)

        # Blend probabilities
        self.outcome_probs = self._blend(
            self.poisson_predictor.outcome_probs,
            self.ml_win_prob_home,
        )

        logger.debug(
            "%s vs %s | Poisson P(home)=%.1f%% | ML P(home)=%.1f%% | "
            "Ensemble P(home)=%.1f%%",
            home_team, away_team,
            self.poisson_predictor.outcome_probs["p_home_total"] * 100,
            (self.ml_win_prob_home or 0.5) * 100,
            self.outcome_probs["p_home_total"] * 100,
        )

    @classmethod
    def _load_ml_model(cls) -> Optional[Any]:
        """
        Load the best calibrated ML model from disk (cached at class level).

        Returns None if the model file does not exist.
        """
        if cls._ml_loaded:
            return cls._ml_model

        model_path = os.path.join(config.MODELS_DIR, "saved", "best_model.joblib")
        cls._ml_loaded = True

        if not os.path.exists(model_path):
            logger.warning(
                "ML model not found at %s. Run with --validate-only first, "
                "or include ML validation step. Falling back to pure Poisson.",
                model_path,
            )
            cls._ml_model = None
            return None

        try:
            cls._ml_model = joblib.load(model_path)
            logger.info("Loaded calibrated ML model from %s", model_path)
        except Exception as exc:
            logger.warning("Failed to load ML model: %s. Falling back to Poisson.", exc)
            cls._ml_model = None

        return cls._ml_model

    def _get_ml_probability(self, features: Dict[str, Any]) -> Optional[float]:
        """
        Get the ML model's win probability for team_a (home).

        Returns
        -------
        float or None
            P(home wins) from the ML model, or None if unavailable.
        """
        model = self._load_ml_model()
        if model is None:
            return None

        try:
            X = _features_to_ml_vector(features)
            proba = model.predict_proba(X)[0]
            # proba[1] = P(outcome=1) = P(team_a wins)
            return float(np.clip(proba[1], 0.05, 0.95))
        except Exception as exc:
            logger.warning("ML prediction failed for %s vs %s: %s",
                           self.home_team, self.away_team, exc)
            return None

    def _blend(
        self,
        poisson_probs: Dict[str, float],
        ml_prob_home: Optional[float],
    ) -> Dict[str, float]:
        """
        Blend Poisson and ML win probabilities.

        If ML probability is unavailable, falls back to pure Poisson.
        ET and PK sub-probabilities retain the Poisson values (the ML model
        only predicts binary win/loss, not match phase).

        Parameters
        ----------
        poisson_probs : dict
            Output of compute_knockout_probabilities().
        ml_prob_home : float or None
            ML model win probability for home team.

        Returns
        -------
        dict
            Blended outcome probability dict (same schema as MatchPredictor).
        """
        if ml_prob_home is None:
            # No ML model available — use pure Poisson
            return dict(poisson_probs)

        w_p = self.blend_weight_poisson
        w_m = self.blend_weight_ml

        # Normalise weights in case they don't sum to 1
        total_w = w_p + w_m
        w_p /= total_w
        w_m /= total_w

        p_home_poisson = poisson_probs["p_home_total"]
        p_away_poisson = poisson_probs["p_away_total"]

        # Blend
        p_home_blended = w_p * p_home_poisson + w_m * ml_prob_home
        p_away_blended = 1.0 - p_home_blended

        # Re-normalise (should already sum to 1 but guard against float drift)
        total = p_home_blended + p_away_blended
        p_home_blended /= total
        p_away_blended /= total

        blended = dict(poisson_probs)
        blended["p_home_total"] = round(p_home_blended, 4)
        blended["p_away_total"] = round(p_away_blended, 4)

        return blended

    def simulate_once(self, rng: np.random.Generator, noise_sigma: float = 0.0) -> str:
        """
        Simulate one match outcome using blended probabilities.

        Uses the Poisson model's simulate_once() (which handles ET/PK paths)
        for match phase modelling, but biases the result toward the ensemble
        probability via rejection sampling.

        Parameters
        ----------
        rng : np.random.Generator
        noise_sigma : float

        Returns
        -------
        str
            Winning team name.
        """
        # For simulation, we use the Poisson model's detailed ET/PK path,
        # but scale the outcome by the ensemble probability using a weighted coin.
        # This maintains the structural validity of the Poisson simulation while
        # incorporating the ML signal.
        p_home_ensemble = self.outcome_probs["p_home_total"]

        # Sample the score path from Poisson (captures ET/PK realism)
        poisson_winner = self.poisson_predictor.simulate_once(rng, noise_sigma)

        # With probability |ensemble - poisson|, override with ensemble decision
        p_home_poisson = self.poisson_predictor.outcome_probs["p_home_total"]
        delta = abs(p_home_ensemble - p_home_poisson)

        if rng.uniform() < delta:
            # Override: use ensemble probability to decide
            return self.home_team if rng.uniform() < p_home_ensemble else self.away_team
        else:
            # Keep Poisson result
            return poisson_winner
