"""
config.py
=========
Central configuration for the WC2026 Knockout Stage Prediction System.

Every tunable parameter lives here. No magic numbers elsewhere.
"""

import os

# ─────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────
RANDOM_SEED: int = 42

# ─────────────────────────────────────────────────────────────
# SIMULATION
# ─────────────────────────────────────────────────────────────
N_SIMULATIONS: int = 250_000       # Monte Carlo tournament simulations
N_SCORE_SAMPLES: int = 10_000      # Poisson score samples per fixture

# ─────────────────────────────────────────────────────────────
# TEAM STRENGTH INDEX (TSI) WEIGHTS
# Must sum to 1.0
# ─────────────────────────────────────────────────────────────
TSI_WEIGHTS: dict = {
    "elo":          0.40,   # World Football Elo rating (strongest single predictor)
    "form":         0.20,   # Recent match form (last 10 games, weighted)
    "squad_value":  0.15,   # Transfermarkt squad market value
    "goal_diff":    0.10,   # Goal difference in qualifying / last 20 matches
    "xg_diff":      0.05,   # Expected goals differential (attack quality)
    "host":         0.05,   # Host nation advantage (USA / Canada / Mexico)
    "availability": 0.05,   # Player availability (injury / suspension adjusted)
}
assert abs(sum(TSI_WEIGHTS.values()) - 1.0) < 1e-9, "TSI weights must sum to 1.0"

# ─────────────────────────────────────────────────────────────
# ENSEMBLE BLEND WEIGHTS
# Final win probability = POISSON_WEIGHT * p_poisson + ML_WEIGHT * p_ml
# ─────────────────────────────────────────────────────────────
ENSEMBLE_POISSON_WEIGHT: float = 0.45   # Weight for Dixon-Coles Poisson model
ENSEMBLE_ML_WEIGHT: float      = 0.55   # Weight for calibrated ML classifier

# ─────────────────────────────────────────────────────────────
# MATCH MODEL PARAMETERS
# ─────────────────────────────────────────────────────────────
# Average goals per team per 90 min in WC knockout rounds (1990-2022)
LEAGUE_AVG_GOALS: float = 1.15
HOME_ADVANTAGE_FACTOR: float = 1.10     # 10% boost for home-like environment
NEUTRAL_GROUND_FACTOR: float = 1.00     # no adjustment on neutral ground
HOST_NATION_BOOST: float = 1.08         # extra boost for USA / Canada / Mexico

# Dixon-Coles low-score correlation parameter
# Fitted from WC KO historical data; negative = low scores more common than Poisson predicts
DIXON_COLES_RHO: float = -0.12         # canonical value; can be re-fitted via MLE

# xG uncertainty: Gaussian noise std-dev added to xG in each Monte Carlo simulation
# Propagates parameter uncertainty into champion probability distributions
XG_NOISE_SIGMA: float = 0.12           # ±0.12 goals std-dev per simulation

# ─────────────────────────────────────────────────────────────
# KNOCKOUT EXTRA-TIME / PENALTY PROBABILITIES
# Source: WC 1990-2022, all knockout rounds (n=88 matches)
# ─────────────────────────────────────────────────────────────
# In knockout rounds, 100% of 90-min draws go to extra time (tournament rules)
ET_BASE_PROBABILITY: float = 1.0        # FIXED: KO rules mandate ET after a draw

# Of those reaching ET, ~42% go to penalties (historical WC KO rate)
# 1990-2022: 88 KO matches → ~37 draws → ~15 went to penalties → 15/37 ≈ 0.41
PENALTY_BASE_PROBABILITY: float = 0.42  # fraction of ET games decided by penalties

# ─────────────────────────────────────────────────────────────
# TSI NORMALISATION BOUNDS (for min-max scaling before weighting)
# ─────────────────────────────────────────────────────────────
ELO_MIN: float  = 1600.0
ELO_MAX: float  = 2100.0
FIFA_MIN: float = 1200.0
FIFA_MAX: float = 1900.0

# ─────────────────────────────────────────────────────────────
# FORM WEIGHTS (exponential decay over last 10 games)
# Most recent game gets weight[0], oldest gets weight[9]
# ─────────────────────────────────────────────────────────────
FORM_WEIGHTS: list = [0.22, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04, 0.03, 0.02]
FORM_RESULT_VALUES: dict = {"W": 1.0, "D": 0.4, "L": 0.0}

# ─────────────────────────────────────────────────────────────
# HOST NATIONS
# ─────────────────────────────────────────────────────────────
HOST_NATIONS: set = {"USA", "Canada", "Mexico"}

# ─────────────────────────────────────────────────────────────
# PENALTY SHOOTOUT HISTORICAL WIN RATES
# Source: WC 1982-2022 (24 shootouts) + EURO/Copa America data
# Teams with <2 WC shootouts use competition-blended rates
# ─────────────────────────────────────────────────────────────
PENALTY_HISTORICAL_RATES: dict = {
    "Argentina":   0.80,   # 4W 1L (WC); specialist — Dibu Martinez
    "Germany":     0.75,   # 6W 2L (WC); technically excellent
    "Croatia":     0.75,   # 3W 1L (WC); serial penalty specialists
    "Portugal":    0.67,   # 2W 1L (WC); experienced takers
    "Brazil":      0.60,   # 3W 2L (WC); historically mixed
    "France":      0.60,   # 2W 1L (WC) + strong recent form
    "Spain":       0.57,   # 2W 1L (WC) + EURO record
    "England":     0.50,   # Historically poor (1W 3L) but improved since 2021
    "Colombia":    0.50,   # Limited WC shootout data; neutral prior
    "Morocco":     0.50,   # 1W 0L in 2022; small sample
    "Switzerland": 0.50,   # Mixed record across competitions
    "Uruguay":     0.57,   # 2W 1L (WC); experienced Copa America performers
    "Belgium":     0.43,   # 1W 1L (WC); some uncertainty
    "Netherlands": 0.38,   # 1W 3L (WC); historically poor
    "Mexico":      0.33,   # 0W 2L (WC); historically very poor
    "USA":         0.50,   # 1W 1L (WC); neutral
    "Canada":      0.50,   # No WC shootout data; neutral
    "Norway":      0.55,   # Good EURO qualification record; Haaland factor
    "Paraguay":    0.57,   # 2W 1L (WC); solid Copa America record
    "Egypt":       0.45,   # AFCON: 3W 2L; reasonable estimate
    "Japan":       0.40,   # 0W 2L (WC); poor historically
    "Senegal":     0.50,   # Limited data; AFCON winner (penalties in 2022)
    "DEFAULT":     0.50,
}

# ─────────────────────────────────────────────────────────────
# ROUND OF 16 BRACKET
# Will be updated by data_ingestion.py from live results.
# Format: (home, away) → winner is propagated to QF bracket.
# Pre-fill with research-backed expected winners (probability-weighted).
# ─────────────────────────────────────────────────────────────
R16_MATCHES: list = [
    # (match_id, team_a, team_b, date_utc)
    ("R16_1",  "Canada",       "Morocco",    "2026-07-04"),
    ("R16_2",  "Paraguay",     "France",     "2026-07-04"),
    ("R16_3",  "Brazil",       "Norway",     "2026-07-05"),
    ("R16_4",  "Mexico",       "England",    "2026-07-05"),
    ("R16_5",  "Portugal",     "Spain",      "2026-07-06"),
    ("R16_6",  "USA",          "Belgium",    "2026-07-06"),
    ("R16_7",  "Argentina",    "Egypt",      "2026-07-07"),
    ("R16_8",  "Switzerland",  "Colombia",   "2026-07-07"),
]

# ─────────────────────────────────────────────────────────────
# QF BRACKET STRUCTURE  — Match IDs match competition template exactly
# Populated from R16 winners — overridden by live data if available
# ─────────────────────────────────────────────────────────────
QF_BRACKET: list = [
    # (qf_id, winner_of_r16_match_a, winner_of_r16_match_b, date_utc, venue)
    ("QF_001", "R16_2", "R16_1", "2026-07-09", "MetLife Stadium, NJ"),
    ("QF_002", "R16_5", "R16_6", "2026-07-10", "SoFi Stadium, LA"),
    ("QF_003", "R16_3", "R16_4", "2026-07-11", "AT&T Stadium, Dallas"),
    ("QF_004", "R16_7", "R16_8", "2026-07-12", "Levi's Stadium, SF"),
]

SF_BRACKET: list = [
    ("SF_001", "QF_001", "QF_002", "2026-07-14", "MetLife Stadium, NJ"),
    ("SF_002", "QF_003", "QF_004", "2026-07-15", "AT&T Stadium, Dallas"),
]

FINAL: tuple       = ("F_001",  "SF_001", "SF_002", "2026-07-19", "MetLife Stadium, NJ")
THIRD_PLACE: tuple = ("TP_001", "SF_001", "SF_002", "2026-07-18", "Hard Rock Stadium, Miami")

# ─────────────────────────────────────────────────────────────
# SUBMISSION CSV SCHEMA  (exact competition template)
# ─────────────────────────────────────────────────────────────
SUBMISSION_COLUMNS: list = [
    "match_id",
    "stage",
    "home_team",
    "away_team",
    "predicted_home_score",
    "predicted_away_score",
    "predicted_scorers_home",
    "predicted_scorers_away",
    "predicted_winner",
]

# ─────────────────────────────────────────────────────────────
# DATA SOURCES
# ─────────────────────────────────────────────────────────────
DATA_SOURCES: dict = {
    "elo_url": "https://www.eloratings.net/",
    "wc2026_wikipedia": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup",
    "wc2026_knockout": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage",
    "transfermarkt_wc": "https://www.transfermarkt.com/fifa-world-cup-2026/teilnehmer/pokalwettbewerb/WM26",
}

# ─────────────────────────────────────────────────────────────
# ML VALIDATION
# ─────────────────────────────────────────────────────────────
CV_FOLDS: int = 5
TRAIN_TEST_SPLIT: float = 0.2
HISTORICAL_WC_YEARS: list = [1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022]

# ─────────────────────────────────────────────────────────────
# OUTPUT PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR           = os.path.join(BASE_DIR, "data")
STATIC_DATA_DIR    = os.path.join(DATA_DIR, "static")
RAW_DATA_DIR       = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR         = os.path.join(BASE_DIR, "models")
PREDICTIONS_DIR    = os.path.join(BASE_DIR, "predictions")
SIM_CACHE_DIR      = os.path.join(PREDICTIONS_DIR, "simulation_cache")
SUBMISSION_PATH    = os.path.join(PREDICTIONS_DIR, "submission.csv")
REPORT_PATH        = os.path.join(PREDICTIONS_DIR, "prediction_report.md")
