# FIFA World Cup 2026 — Knockout Stage Prediction System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **competition-winning, publication-quality** hybrid football prediction system targeting the FIFA World Cup 2026 Knockout Stage (Quarter Finals → Final).

## 🏆 What It Does

- **Predicts** all QF, SF, and Final match outcomes with scores, jersey-numbered goalscorers, and win probabilities
- **Runs 250,000 Monte Carlo simulations** to estimate tournament-wide probability distributions
- **Combines 15 features** (Elo, FIFA rankings, xG, squad value, form, H2H, rest days, host advantage, injury, penalty history, and more)
- **Validates 6 ML models** (LogReg, RF, GBoost, XGBoost, CatBoost, LightGBM) against historical WC data
- **Generates** `submission.csv` and `prediction_report.md` automatically

---

## 🚀 Quickstart

```bash
# 1. Clone / navigate to the project directory
cd wc2026_predictor

# 2. Create a virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

# 3. Run the full pipeline
python main.py

# 4. Check outputs
# predictions/submission.csv
# predictions/prediction_report.md
```

---

## 🗂️ Repository Structure

```
wc2026_predictor/
├── data/
│   ├── raw/                    # Downloaded/scraped data (auto-populated)
│   ├── processed/              # Cleaned feature matrices (auto-populated)
│   └── static/                 # Curated research-backed fallback CSVs
│       ├── elo_ratings.csv
│       ├── fifa_rankings.csv
│       ├── squad_values.csv
│       ├── team_form.csv
│       ├── team_stats.csv
│       ├── player_data.csv
│       └── h2h_records.csv
├── models/
│   ├── saved/                  # Serialised best model
│   └── validation_results/     # Cross-validation JSON
├── predictions/
│   ├── submission.csv          # 🎯 COMPETITION SUBMISSION FILE
│   ├── prediction_report.md    # 📄 Full analysis report
│   └── simulation_cache/       # Monte Carlo result cache (pickle)
├── notebooks/
│   └── exploration.ipynb       # EDA notebook
├── src/
│   ├── __init__.py
│   ├── config.py               # All parameters & weights (no magic numbers)
│   ├── data_ingestion.py       # Live scraping + static fallback
│   ├── feature_engineering.py  # 15 features + TSI computation
│   ├── match_model.py          # Dixon-Coles Poisson + probability engine
│   ├── tournament_simulator.py # 250k Monte Carlo engine
│   ├── goalscorer_model.py     # Player-level scorer predictions
│   ├── ml_validation.py        # 6-model ML comparison
│   └── output_generator.py     # CSV + Markdown report generation
├── main.py                     # Single entry point
├── requirements.txt
└── README.md
```

---

## ⚙️ CLI Options

```bash
python main.py                    # Full pipeline (250k sims)
python main.py --offline          # Skip live scraping (pure static data)
python main.py --sims 50000       # Custom simulation count
python main.py --validate-only    # ML validation only
python main.py --dry-run          # 1000 sims, no file writes (fast test)
python main.py --no-ml-validation # Skip ML validation (faster)
python main.py --update-bracket   # Force live bracket refresh
```

---

## 🧮 The Model

### Team Strength Index (TSI)

```
TSI = 0.40 × Elo Rating
    + 0.20 × Recent Form (last 10 games, exp-weighted)
    + 0.15 × Squad Market Value (Transfermarkt, log-scaled)
    + 0.10 × Goal Difference (last 10 games)
    + 0.05 × xG Differential (FBref)
    + 0.05 × Host Nation Advantage
    + 0.05 × Player Availability (injury/suspension adjusted)
```

All weights are configurable in `src/config.py`.

### Match Model (Dixon-Coles Poisson)

```python
xG_home = attack_home × defense_away × league_avg × home_factor
xG_away = attack_away × defense_home × league_avg
```

With low-score Poisson correction (ρ = −0.12 from Dixon & Coles 1997).

### Monte Carlo Simulation

250,000 full tournament simulations → probability distributions for:
- QF qualification (all teams)
- SF qualification probability
- Final appearance probability
- Champion probability
- Runner-up probability

### Goalscorer Model

Player scoring weights:
```
weight = scoring_rate × (minutes/90) × role_weight × availability
```

Role weights: FWD=1.0, MID=0.55, DEF=0.18, GK=0.00

Penalty taker bonus applied proportionally to P(penalty shootout).

---

## 📋 The 15 Features

| # | Feature | Source | Rationale |
|---|---------|--------|-----------|
| 1 | World Football Elo | eloratings.net | Best pure historical team quality predictor |
| 2 | FIFA Ranking Points | fifa.com | Official global standing with confederation weighting |
| 3 | Recent Form | FBref / Static | Short-term momentum; beats Elo by ~5-8% |
| 4 | xG Differential | FBref | Underlying quality beyond actual goals |
| 5 | Attack Strength | FBref | Offensive output vs tournament average |
| 6 | Defensive Strength | FBref | Defensive solidity vs tournament average |
| 7 | Squad Market Value | Transfermarkt | Best public proxy for talent depth |
| 8 | Avg Player Rating | Sofascore | Individual quality aggregated to team level |
| 9 | Tournament Performance | WC history | KO-round pedigree (some teams consistently overperform) |
| 10 | Head-to-Head Record | Wikipedia | Direct matchup psychological edge |
| 11 | Rest Days | Schedule | Fatigue — <4 days rest → ~4-8% performance drop |
| 12 | Host Nation Advantage | Config | Home crowd + travel advantage for USA/Canada/Mexico |
| 13 | Injury/Suspension | News/Static | Key player absence shifts xG by up to 0.5/game |
| 14 | Penalty Shootout History | WC records | Historical PK win rates (Argentina 80%, England 40%) |
| 15 | Historical KO Performance | WC history | Win rate in all WC knockout games |

---

## 📊 ML Model Comparison

6 classifiers validated on historical WC knockout data (1990–2022):

| Model | Purpose |
|-------|---------|
| Logistic Regression | Baseline linear model |
| Random Forest | Ensemble, non-linear |
| Gradient Boosting | Boosted trees, sklearn |
| XGBoost | State-of-art boosting |
| CatBoost | Handles categorical natively |
| LightGBM | Fast gradient boosting |

5-fold stratified cross-validation. Results saved to `models/validation_results/cv_results.json`.

---

## 🔧 Configuration

All parameters in `src/config.py`:

```python
N_SIMULATIONS = 250_000        # Monte Carlo sim count
RANDOM_SEED = 42               # Full reproducibility

TSI_WEIGHTS = {
    "elo": 0.40,               # ← Change weights here
    "form": 0.20,
    "squad_value": 0.15,
    "goal_diff": 0.10,
    "xg_diff": 0.05,
    "host": 0.05,
    "availability": 0.05,
}

LEAGUE_AVG_GOALS = 1.15        # WC KO round base rate
HOME_ADVANTAGE_FACTOR = 1.10   # Home game boost
HOST_NATION_BOOST = 1.08       # Co-host boost
```

---

## 📁 Output Format

### `predictions/submission.csv`

| Column | Description |
|--------|-------------|
| match_id | QF1, QF2, SF1, FINAL, etc. |
| stage | Quarter-Final / Semi-Final / Final |
| match_date | YYYY-MM-DD |
| venue | Stadium name |
| home_team | Team A name |
| away_team | Team B name |
| home_score | Predicted home goals |
| away_score | Predicted away goals |
| winner | Predicted winning team |
| is_extra_time | 1/0 |
| is_penalties | 1/0 |
| home_scorers_jersey | Semicolon-separated jersey numbers |
| away_scorers_jersey | Semicolon-separated jersey numbers |
| home_win_prob | [0.0, 1.0] |
| draw_prob | [0.0, 1.0] |
| away_win_prob | [0.0, 1.0] |
| et_prob | [0.0, 1.0] |
| penalty_prob | [0.0, 1.0] |

---

## 📚 References

- Dixon, M., & Coles, S. (1997). *Modelling Association Football Scores and Inefficiencies in the Football Betting Market.* Journal of the Royal Statistical Society.
- Elo, A. (1978). *The Rating of Chess Players, Past and Present.* Arco Publishing.
- Hvattum, L.M., & Arntzen, H. (2010). *Using ELO ratings for match result prediction in association football.* International Journal of Forecasting.
- Poli, R. (2014). *Understanding transfers: the importance of squad value.* CIES Football Observatory.
- eloratings.net — World Football Elo Ratings
- transfermarkt.com — Squad Market Values
- fbref.com — Football Reference Statistics

---

## 🔒 Reproducibility

All random operations use `RANDOM_SEED = 42`. Running `python main.py` twice with the same static data produces **identical** outputs. Monte Carlo results are cached to `predictions/simulation_cache/` (hash-verified).

---

## License

MIT License — free for research and competition use.
