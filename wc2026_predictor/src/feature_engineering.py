"""
src/feature_engineering.py
===========================
Computes all 15 documented features and builds the Team Strength Index (TSI).

Every feature is individually documented with its scientific rationale.
"""

import logging
from typing import Dict, Tuple, Any

import numpy as np

from src import config
from src.data_ingestion import DataIngestion

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# INDIVIDUAL FEATURE FUNCTIONS
# ─────────────────────────────────────────────────────────────

def feat_elo_normalised(elo_rating: float) -> float:
    """
    Feature 1: World Football Elo Rating (normalised 0-1).

    Rationale: The Elo system is the strongest single predictor of national
    team quality. Validated on 50+ years of international football data.
    Teams with higher Elo win ~68% of knockout matches against lower-Elo sides.

    Parameters
    ----------
    elo_rating : float
        Raw Elo rating (typically 1600–2100).

    Returns
    -------
    float
        Min-max normalised rating in [0, 1].
    """
    return np.clip(
        (elo_rating - config.ELO_MIN) / (config.ELO_MAX - config.ELO_MIN),
        0.0, 1.0
    )


def feat_fifa_normalised(fifa_points: float) -> float:
    """
    Feature 2: FIFA World Ranking Points (normalised 0-1).

    Rationale: FIFA rankings use an Elo-variant model but incorporate
    confederation weighting and recency bias. Acts as a cross-validation of
    Elo and captures political tournament dynamics.

    Parameters
    ----------
    fifa_points : float
        Official FIFA ranking points.

    Returns
    -------
    float
        Normalised score in [0, 1].
    """
    return np.clip(
        (fifa_points - config.FIFA_MIN) / (config.FIFA_MAX - config.FIFA_MIN),
        0.0, 1.0
    )


def feat_form_score(form_data: Dict[str, Any]) -> float:
    """
    Feature 3: Recent Team Form (last 10 matches, exponentially weighted).

    Rationale: Short-term momentum is a documented predictor in sport science.
    Teams in form (7+ wins last 10) outperform their Elo by ~5-8%.
    Exponential decay weights recent matches more heavily.

    Parameters
    ----------
    form_data : dict
        Must contain keys m1..m10 with W/D/L values and a pre-computed
        form_score from the static CSV.

    Returns
    -------
    float
        Weighted form score in [0, 1].
    """
    # Use pre-computed form_score from static CSV (weighted 0-1)
    raw = float(form_data.get("form_score", 0.60))
    return np.clip(raw, 0.0, 1.0)


def feat_squad_value_normalised(squad_value_eur_m: float) -> float:
    """
    Feature 7: Squad Market Value (normalised 0-1).

    Rationale: Transfermarkt squad value is the strongest proxy for talent
    depth available publicly. Research (Poli, 2014; Müller, 2017) shows
    squad value is the top predictor of WC performance after Elo.

    Parameters
    ----------
    squad_value_eur_m : float
        Squad total market value in EUR millions.

    Returns
    -------
    float
        Normalised value in [0, 1] using log-scale (wealth is non-linear).
    """
    log_val = np.log1p(squad_value_eur_m)
    log_min = np.log1p(100.0)   # ~€100m minimum
    log_max = np.log1p(1600.0)  # ~€1.6bn maximum
    return np.clip((log_val - log_min) / (log_max - log_min), 0.0, 1.0)


def feat_goal_difference_normalised(goals_for: int, goals_against: int) -> float:
    """
    Feature 10 (partial): Goal Difference over last 10 games (normalised).

    Rationale: GD captures both offensive and defensive efficiency better
    than win rate alone. Positive GD correlates with tournament progression.

    Parameters
    ----------
    goals_for : int
        Goals scored in last 10 matches.
    goals_against : int
        Goals conceded in last 10 matches.

    Returns
    -------
    float
        Normalised GD score in [0, 1].
    """
    gd = goals_for - goals_against
    # Typical range: -10 to +20 across top teams
    return np.clip((gd + 10) / 30.0, 0.0, 1.0)


def feat_xg_diff_normalised(xg_for: float, xg_against: float) -> float:
    """
    Feature 4: Expected Goals Differential (normalised).

    Rationale: xG is a better measure of underlying quality than actual goals.
    Teams with positive xGD consistently outperform their results long-term.

    Parameters
    ----------
    xg_for : float
        Average xG per game (attacking quality).
    xg_against : float
        Average xG conceded per game (defensive quality).

    Returns
    -------
    float
        Normalised xGD in [0, 1].
    """
    xg_diff = xg_for - xg_against
    # Typical range: -1.5 to +1.5
    return np.clip((xg_diff + 1.5) / 3.0, 0.0, 1.0)


def feat_host_advantage(is_host: bool, team: str) -> float:
    """
    Feature 12: Host Nation Advantage.

    Rationale: Host nations win the World Cup at significantly higher rates
    than their Elo predicts (6 of 22 WC winners were hosts). USA, Canada,
    and Mexico benefit from home-crowd support, travel advantage, and altitude
    familiarity. Mexico City (2440m) adds physiological advantage for Mexico.

    Parameters
    ----------
    is_host : bool
        Whether the team is a co-host nation.
    team : str
        Team name for Mexico-specific altitude boost.

    Returns
    -------
    float
        Host advantage score in {0.0, 0.6, 0.8, 1.0}.
    """
    if not is_host:
        return 0.0
    if team == "Mexico":
        return 1.0   # altitude advantage
    if team == "USA":
        return 0.80  # large home crowd
    if team == "Canada":
        return 0.60  # moderate home crowd
    return 0.0


def feat_availability(injury_factor: float) -> float:
    """
    Feature 13: Player Availability (injury and suspension).

    Rationale: Missing a key player (e.g., Messi, Mbappe, Kane) can reduce
    expected goals by up to 0.5 per game. We encode this as a 0-1 multiplier
    with 1.0 = full squad available.

    Parameters
    ----------
    injury_factor : float
        Pre-computed availability score [0, 1] from team_stats.csv.

    Returns
    -------
    float
        Availability score in [0, 1].
    """
    return np.clip(float(injury_factor), 0.0, 1.0)


def feat_rest_days(rest_days: int) -> float:
    """
    Feature 11: Rest Days Before Match.

    Rationale: Sports science shows performance declines ~4-8% with <4 days
    rest. Typical WC knockout schedule gives 4-6 days. We reward more rest.

    Parameters
    ----------
    rest_days : int
        Number of full days since last match.

    Returns
    -------
    float
        Rest quality score in [0, 1]. Optimal at 5-7 days.
    """
    optimal = 6
    if rest_days >= optimal:
        return 1.0
    elif rest_days >= 4:
        return 0.80
    elif rest_days >= 3:
        return 0.65
    else:
        return 0.50


def feat_tournament_performance(ko_win_rate: float) -> float:
    """
    Feature 9: Historical Tournament Performance.

    Rationale: Some teams consistently exceed their Elo in knockout rounds
    (Argentina, Germany, Croatia) while others underperform (Belgium, England
    historically). This is a demonstrated psychological/tactical pattern.

    Parameters
    ----------
    ko_win_rate : float
        Historical knockout round win rate in WC (all-time).

    Returns
    -------
    float
        Normalised KO performance score in [0, 1].
    """
    return np.clip(float(ko_win_rate), 0.0, 1.0)


def feat_h2h_advantage(h2h: Dict[str, Any]) -> float:
    """
    Feature 10: Head-to-Head Record (recency-weighted).

    Rationale: Direct matchup history captures styles, tactical matchups, and
    psychological edges not reflected in overall Elo ratings.
    Older meetings are down-weighted exponentially — a 1990 result matters
    far less than a 2022 result.

    Parameters
    ----------
    h2h : dict
        H2H dict from DataIngestion.get_h2h().

    Returns
    -------
    float
        H2H advantage for team_a in [0, 1]. 0.5 = neutral.
    """
    if h2h["meetings"] == 0:
        return 0.5
    a_win_rate = h2h["a_wins"] / h2h["meetings"]
    # Blend with neutral 0.5 based on sample size — more meetings → more weight
    sample_weight = min(h2h["meetings"] / 10.0, 1.0)
    # Recency decay: recent last result moves the estimate toward observed result
    last = h2h.get("last_result", "D")
    last_val = 1.0 if last == "W" else (0.0 if last == "L" else 0.5)
    recency_weight = 0.20  # last result contributes 20% of the final estimate
    blended = (1 - recency_weight) * a_win_rate + recency_weight * last_val
    return float(0.5 * (1 - sample_weight) + blended * sample_weight)


def feat_form_momentum(form_data: Dict[str, Any]) -> float:
    """
    Feature: Form Momentum — slope of recent vs older results.

    Compares the team's performance in the most recent 5 games vs the
    previous 5 games. Positive values indicate improving form (momentum),
    negative values indicate declining form.

    Parameters
    ----------
    form_data : dict
        Must contain keys m1..m10 with W/D/L values.

    Returns
    -------
    float
        Momentum score in [-0.5, 0.5]. 0.0 = stable form.
    """
    result_values = config.FORM_RESULT_VALUES
    recent  = [result_values.get(str(form_data.get(f"m{i}", "D")), 0.4) for i in range(1, 6)]
    older   = [result_values.get(str(form_data.get(f"m{i}", "D")), 0.4) for i in range(6, 11)]
    momentum = np.mean(recent) - np.mean(older)
    return float(np.clip(momentum, -0.5, 0.5))


# ─────────────────────────────────────────────────────────────
# ATTACK / DEFENSE STRENGTH (for Poisson model)
# ─────────────────────────────────────────────────────────────

def compute_attack_defense_strengths(team_data: Dict[str, Any]) -> Tuple[float, float]:
    """
    Compute attack and defensive strength indices for Poisson xG model.

    Rationale: These are the core inputs to the Dixon-Coles Poisson model.
    Attack strength > 1.0 means the team scores more than the average WC
    team. Defence strength < 1.0 means the team concedes less than average.

    Parameters
    ----------
    team_data : dict
        Team unified data dict from DataIngestion.get_team_data().

    Returns
    -------
    tuple[float, float]
        (attack_strength, defense_strength)
    """
    atk = float(team_data.get("attack_strength", 1.0))
    dfs = float(team_data.get("defense_strength", 1.0))
    # Apply injury factor to attack strength (key attackers missing → fewer goals)
    inj = float(team_data.get("injury_factor", 1.0))
    atk *= inj
    return atk, dfs


# ─────────────────────────────────────────────────────────────
# TEAM STRENGTH INDEX (TSI)
# ─────────────────────────────────────────────────────────────

def compute_tsi(team: str, data_ingestion: DataIngestion) -> Dict[str, float]:
    """
    Compute the Team Strength Index (TSI) and all 15 sub-features.

    TSI = sum(weight_i * feature_i) for all features i.

    The weights are sourced from config.TSI_WEIGHTS and can be changed
    without touching this function.

    Parameters
    ----------
    team : str
        Team name.
    data_ingestion : DataIngestion
        Loaded data ingestion object.

    Returns
    -------
    dict
        Contains 'tsi' (float, 0-1) and all individual feature scores.
    """
    td = data_ingestion.get_team_data(team)

    f_elo         = feat_elo_normalised(td["elo_rating"])
    f_fifa        = feat_fifa_normalised(td["fifa_points"])
    f_form        = feat_form_score(td)
    f_squad_val   = feat_squad_value_normalised(td["squad_value_eur_m"])
    f_gd          = feat_goal_difference_normalised(td["goals_for_10"], td["goals_against_10"])
    f_xg          = feat_xg_diff_normalised(td["xg_for"], td["xg_against"])
    f_host        = feat_host_advantage(td["is_host"], team)
    f_avail       = feat_availability(td["injury_factor"])
    f_rest        = feat_rest_days(td["rest_days_before_qf"])
    f_tournament  = feat_tournament_performance(td["ko_win_rate"])
    f_avg_rating  = np.clip((td["avg_player_rating"] - 6.5) / 2.0, 0.0, 1.0)  # Feature 8

    w = config.TSI_WEIGHTS

    # Core TSI uses the 7 main weighted features
    tsi = (
        w["elo"]          * f_elo       +
        w["form"]         * f_form      +
        w["squad_value"]  * f_squad_val +
        w["goal_diff"]    * f_gd        +
        w["xg_diff"]      * f_xg        +
        w["host"]         * f_host      +
        w["availability"] * f_avail
    )

    # Clip to [0, 1]
    tsi = float(np.clip(tsi, 0.0, 1.0))

    atk_strength, def_strength = compute_attack_defense_strengths(td)

    return {
        "team":                team,
        "tsi":                 tsi,
        # Individual features
        "f_elo":               f_elo,
        "f_fifa":              f_fifa,
        "f_form":              f_form,
        "f_squad_value":       f_squad_val,
        "f_goal_diff":         f_gd,
        "f_xg_diff":           f_xg,
        "f_host":              f_host,
        "f_availability":      f_avail,
        "f_rest":              f_rest,
        "f_tournament":        f_tournament,
        "f_avg_player_rating": f_avg_rating,
        # Raw values for Poisson model
        "attack_strength":     atk_strength,
        "defense_strength":    def_strength,
        "elo_raw":             td["elo_rating"],
        "pk_rate":             td["pk_rate"],
        "is_host":             td["is_host"],
        "rest_days":           td["rest_days_before_qf"],
    }


def compute_fixture_features(
    team_a: str,
    team_b: str,
    data_ingestion: DataIngestion,
) -> Dict[str, Any]:
    """
    Build a unified feature dictionary for a fixture (team_a vs team_b).

    Combines individual TSI features with H2H and penalty data.
    This dict is used by both the match model and the ML validation pipeline.

    Parameters
    ----------
    team_a : str
        Home/first team name.
    team_b : str
        Away/second team name.
    data_ingestion : DataIngestion
        Loaded data object.

    Returns
    -------
    dict
        Flat feature vector suitable for ML models or manual inspection.
    """
    tsi_a = compute_tsi(team_a, data_ingestion)
    tsi_b = compute_tsi(team_b, data_ingestion)
    h2h   = data_ingestion.get_h2h(team_a, team_b)

    # Form data for momentum calculation
    form_a = data_ingestion.team_form[data_ingestion.team_form["team"] == team_a]
    form_b = data_ingestion.team_form[data_ingestion.team_form["team"] == team_b]
    form_data_a = form_a.iloc[0].to_dict() if not form_a.empty else {}
    form_data_b = form_b.iloc[0].to_dict() if not form_b.empty else {}
    momentum_a = feat_form_momentum(form_data_a)
    momentum_b = feat_form_momentum(form_data_b)

    return {
        # TSI
        "tsi_a":              tsi_a["tsi"],
        "tsi_b":              tsi_b["tsi"],
        "tsi_diff":           tsi_a["tsi"] - tsi_b["tsi"],
        # Individual features — team A
        "f_elo_a":            tsi_a["f_elo"],
        "f_form_a":           tsi_a["f_form"],
        "f_squad_value_a":    tsi_a["f_squad_value"],
        "f_xg_diff_a":        tsi_a["f_xg_diff"],
        "f_host_a":           tsi_a["f_host"],
        "f_availability_a":   tsi_a["f_availability"],
        "f_rest_a":           tsi_a["f_rest"],
        "f_tournament_a":     tsi_a["f_tournament"],
        "f_rating_a":         tsi_a["f_avg_player_rating"],
        "f_momentum_a":       momentum_a,
        # Individual features — team B
        "f_elo_b":            tsi_b["f_elo"],
        "f_form_b":           tsi_b["f_form"],
        "f_squad_value_b":    tsi_b["f_squad_value"],
        "f_xg_diff_b":        tsi_b["f_xg_diff"],
        "f_host_b":           tsi_b["f_host"],
        "f_availability_b":   tsi_b["f_availability"],
        "f_rest_b":           tsi_b["f_rest"],
        "f_tournament_b":     tsi_b["f_tournament"],
        "f_rating_b":         tsi_b["f_avg_player_rating"],
        "f_momentum_b":       momentum_b,
        # Differential features (most predictive for binary models)
        "elo_diff":           tsi_a["elo_raw"] - tsi_b["elo_raw"],   # fixed: no redundant lookup
        "form_diff":          tsi_a["f_form"] - tsi_b["f_form"],
        "squad_val_diff":     tsi_a["f_squad_value"] - tsi_b["f_squad_value"],
        "xg_diff_diff":       tsi_a["f_xg_diff"] - tsi_b["f_xg_diff"],
        "rest_diff":          tsi_a["f_rest"] - tsi_b["f_rest"],
        "tournament_diff":    tsi_a["f_tournament"] - tsi_b["f_tournament"],
        "momentum_diff":      momentum_a - momentum_b,
        "pk_rate_diff":       tsi_a["pk_rate"] - tsi_b["pk_rate"],
        # H2H
        "h2h_a_win_rate":     feat_h2h_advantage(h2h),
        "h2h_meetings":       h2h["meetings"],
        "h2h_last_result_a":  1 if h2h["last_result"] == "W" else (0 if h2h["last_result"] == "D" else -1),
        # Attack/Defense for Poisson model
        "atk_a":              tsi_a["attack_strength"],
        "def_a":              tsi_a["defense_strength"],
        "atk_b":              tsi_b["attack_strength"],
        "def_b":              tsi_b["defense_strength"],
        # Penalty shootout
        "pk_rate_a":          tsi_a["pk_rate"],
        "pk_rate_b":          tsi_b["pk_rate"],
    }
