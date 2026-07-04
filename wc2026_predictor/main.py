"""
main.py
=======
Single entry point for the WC2026 Knockout Stage Prediction System.

Usage:
    python main.py                    # Full pipeline
    python main.py --offline          # Skip live data scraping
    python main.py --validate-only    # ML validation only (no simulation)
    python main.py --dry-run          # 1000 sims, no file writes
    python main.py --sims 50000       # Custom simulation count
    python main.py --no-ml-validation # Faster run, skip ML comparison
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Any

# ─────────────────────────────────────────────────────────────
# Ensure predictions/ directory exists before FileHandler init
# ─────────────────────────────────────────────────────────────
_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions")
os.makedirs(_LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(_LOG_DIR, "run.log"), mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("wc2026")

# ─────────────────────────────────────────────────────────────
# Project imports
# ─────────────────────────────────────────────────────────────
from src import config
from src.data_ingestion import DataIngestion
from src.feature_engineering import compute_tsi, compute_fixture_features
from src.match_model import MatchPredictor
from src.ensemble import EnsemblePredictor
from src.tournament_simulator import TournamentSimulator, resolve_bracket_teams, build_qf_lineup
from src.goalscorer_model import GoalscorerModel
from src.ml_validation import run_validation
from src.output_generator import (
    build_match_prediction,
    generate_submission_csv,
    generate_prediction_report,
)

BANNER = r"""
╔══════════════════════════════════════════════════════════════════╗
║   FIFA WORLD CUP 2026 — KNOCKOUT STAGE PREDICTION SYSTEM         ║
║   Hybrid TSI + Dixon-Coles Poisson + Monte Carlo (250k sims)     ║
║   Quarter Finals → Semi Finals → Third Place → Final             ║
╚══════════════════════════════════════════════════════════════════╝
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WC2026 Knockout Predictor")
    p.add_argument("--offline",          action="store_true", help="No live scraping")
    p.add_argument("--validate-only",    action="store_true", help="ML validation only")
    p.add_argument("--dry-run",          action="store_true", help="1k sims, no writes")
    p.add_argument("--sims",             type=int, default=config.N_SIMULATIONS)
    p.add_argument("--no-ml-validation", action="store_true", help="Skip ML validation")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# PIPELINE STEPS
# ─────────────────────────────────────────────────────────────

def step1_ingest(offline: bool) -> DataIngestion:
    logger.info("━━━━ STEP 1 │ Data Ingestion ━━━━")
    data = DataIngestion(use_live_data=not offline)
    data.save_processed()
    return data


def step2_tsi(data: DataIngestion) -> Dict[str, Dict]:
    logger.info("━━━━ STEP 2 │ Team Strength Index ━━━━")
    teams = set()
    for _, ta, tb, _ in config.R16_MATCHES:
        teams.add(ta); teams.add(tb)
    tsi_map = {}
    for team in sorted(teams):
        t = compute_tsi(team, data)
        tsi_map[team] = t
        logger.info("  %-15s TSI=%.3f (Elo=%.3f Form=%.3f SqVal=%.3f Atk=%.2f Def=%.2f)",
                    team, t["tsi"], t["f_elo"], t["f_form"], t["f_squad_value"],
                    t["attack_strength"], t["defense_strength"])
    return tsi_map


def step3_bracket(data: DataIngestion) -> List[tuple]:
    logger.info("━━━━ STEP 3 │ Bracket Resolution ━━━━")
    resolved = resolve_bracket_teams(data.r16_results, data)
    fixtures = build_qf_lineup(resolved)
    logger.info("Quarter-Final fixtures:")
    for fid, ta, tb, dt, venue in fixtures:
        logger.info("  %s: %-15s vs %-15s  [%s]", fid, ta, tb, dt)
    return fixtures


def step4_match_model(qf_fixtures: List[tuple], data: DataIngestion) -> Dict[str, EnsemblePredictor]:
    logger.info("━━━━ STEP 4 │ Ensemble Match Model (Poisson + ML) ━━━━")
    predictors = {}
    for fid, ta, tb, dt, venue in qf_fixtures:
        feats = compute_fixture_features(ta, tb, data)
        is_host = ta in config.HOST_NATIONS
        pred = EnsemblePredictor(ta, tb, feats, home_is_host=is_host)
        predictors[fid] = pred
        winner = ta if pred.outcome_probs["p_home_total"] >= 0.50 else tb
        logger.info("  %s │ xG: %.2f–%.2f │ P(%s)=%.0f%% │ MLS: %d-%d",
                    fid, pred.xg_home, pred.xg_away, winner,
                    max(pred.outcome_probs["p_home_total"],
                        pred.outcome_probs["p_away_total"]) * 100,
                    *pred.most_likely_score)
    return predictors


def step5_ml(skip: bool) -> Dict[str, Any]:
    if skip:
        logger.info("━━━━ STEP 5 │ ML Validation [SKIPPED] ━━━━")
        return {"best_model": "Skipped", "n_training_samples": 0}
    logger.info("━━━━ STEP 5 │ ML Model Validation ━━━━")
    return run_validation()


def step6_montecarlo(qf_fixtures: List[tuple], data: DataIngestion, n_sims: int) -> TournamentSimulator:
    logger.info("━━━━ STEP 6 │ Monte Carlo Simulation (%s runs) ━━━━", f"{n_sims:,}")
    sim = TournamentSimulator(qf_fixtures, data, n_sims=n_sims)
    sim.run()
    return sim


def step7_scorers(data: DataIngestion) -> GoalscorerModel:
    logger.info("━━━━ STEP 7 │ Goalscorer Model ━━━━")
    return GoalscorerModel(data)


def step8_outputs(
    qf_fixtures: List[tuple],
    predictors: Dict[str, MatchPredictor],
    scorer_model: GoalscorerModel,
    simulator: TournamentSimulator,
    validation_results: Dict[str, Any],
    tsi_map: Dict[str, Dict],
    data: DataIngestion,
    dry_run: bool,
) -> None:
    logger.info("━━━━ STEP 8 │ Output Generation ━━━━")
    all_predictions = []

    # ── Quarter Finals ──
    for fid, ta, tb, dt, venue in qf_fixtures:
        pred = predictors[fid]
        row = build_match_prediction(
            match_id=fid, stage="Quarter Final",
            home_team=ta, away_team=tb,
            predictor=pred, scorer_model=scorer_model,
        )
        all_predictions.append(row)

    # ── Derive QF winners ──
    qf_winners = []
    for fid, ta, tb, *_ in qf_fixtures:
        pred = predictors[fid]
        qf_winners.append(ta if pred.outcome_probs["p_home_total"] >= 0.50 else tb)

    # ── Semi Finals ──
    sf_predictors: Dict[str, MatchPredictor] = {}
    sf_winners: List[str] = []
    sf_losers:  List[str] = []

    for i, (sf_id, _, __, dt, venue) in enumerate(config.SF_BRACKET):
        ta = qf_winners[2 * i]     if 2 * i     < len(qf_winners) else "TBD"
        tb = qf_winners[2 * i + 1] if 2 * i + 1 < len(qf_winners) else "TBD"
        if ta == "TBD" or tb == "TBD":
            continue
        feats = compute_fixture_features(ta, tb, data)
        pred  = EnsemblePredictor(ta, tb, feats)
        sf_predictors[sf_id] = pred
        winner = ta if pred.outcome_probs["p_home_total"] >= 0.50 else tb
        loser  = tb if winner == ta else ta
        sf_winners.append(winner)
        sf_losers.append(loser)
        logger.info("  %s │ %s vs %s → %s", sf_id, ta, tb, winner)
        row = build_match_prediction(
            match_id=sf_id, stage="Semi Final",
            home_team=ta, away_team=tb,
            predictor=pred, scorer_model=scorer_model,
        )
        all_predictions.append(row)

    # ── Third Place ──
    if len(sf_losers) == 2:
        ta3, tb3 = sf_losers[0], sf_losers[1]
        feats3 = compute_fixture_features(ta3, tb3, data)
        pred3  = EnsemblePredictor(ta3, tb3, feats3)
        row3 = build_match_prediction(
            match_id="TP_001", stage="Third Place",
            home_team=ta3, away_team=tb3,
            predictor=pred3, scorer_model=scorer_model,
        )
        all_predictions.append(row3)
        logger.info("  TP_001 │ %s vs %s → %s", ta3, tb3, row3["predicted_winner"])

    # ── Final ──
    if len(sf_winners) == 2:
        fa, fb = sf_winners[0], sf_winners[1]
        feats_f = compute_fixture_features(fa, fb, data)
        pred_f  = EnsemblePredictor(fa, fb, feats_f)
        row_f = build_match_prediction(
            match_id="F_001", stage="Final",
            home_team=fa, away_team=fb,
            predictor=pred_f, scorer_model=scorer_model,
        )
        all_predictions.append(row_f)
        logger.info("  F_001  │ %s vs %s → 🏆 %s", fa, fb, row_f["predicted_winner"])

    # ── Write files ──
    if not dry_run:
        generate_submission_csv(all_predictions)
        generate_prediction_report(
            qf_fixtures=qf_fixtures,
            predictors=predictors,
            simulator=simulator,
            validation_results=validation_results,
            tsi_map=tsi_map,
            all_predictions=all_predictions,
        )
    else:
        logger.info("DRY RUN — no files written")

    # Console summary
    logger.info("")
    logger.info("┌─────────────────────────────────────────────────────────┐")
    logger.info("│                  FINAL PREDICTIONS                      │")
    logger.info("├──────────┬──────────────────┬────────────────────┬──────┤")
    logger.info("│ match_id │ home             │ away               │ score│")
    logger.info("├──────────┼──────────────────┼────────────────────┼──────┤")
    for r in all_predictions:
        logger.info("│ %-8s │ %-16s │ %-18s │ %d–%-2d │",
                    r["match_id"], r["home_team"], r["away_team"],
                    r["predicted_home_score"], r["predicted_away_score"])
    logger.info("└──────────┴──────────────────┴────────────────────┴──────┘")
    logger.info("")
    if all_predictions:
        champ_row = next((r for r in all_predictions if r["match_id"] == "F_001"), None)
        if champ_row:
            logger.info("🏆  PREDICTED CHAMPION: %s", champ_row["predicted_winner"])


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main() -> None:
    print(BANNER)
    args = parse_args()
    n_sims = 1_000 if args.dry_run else args.sims
    if args.dry_run:
        logger.info("DRY RUN mode — 1,000 simulations, output files suppressed")

    t0 = time.time()

    data        = step1_ingest(offline=args.offline)
    tsi_map     = step2_tsi(data)
    qf_fixtures = step3_bracket(data)
    # ML runs BEFORE match model so the saved model is available to EnsemblePredictor
    val_results = step5_ml(skip=args.no_ml_validation)
    predictors  = step4_match_model(qf_fixtures, data)
    simulator   = step6_montecarlo(qf_fixtures, data, n_sims)
    scorer_model = step7_scorers(data)
    step8_outputs(
        qf_fixtures=qf_fixtures,
        predictors=predictors,
        scorer_model=scorer_model,
        simulator=simulator,
        validation_results=val_results,
        tsi_map=tsi_map,
        data=data,
        dry_run=args.dry_run,
    )

    elapsed = time.time() - t0
    logger.info("✅ Complete in %.1fs", elapsed)
    if not args.dry_run:
        logger.info("   📄 predictions/submission.csv")
        logger.info("   📊 predictions/prediction_report.md")
        logger.info("   🤖 models/saved/best_model.joblib")


if __name__ == "__main__":
    main()
