"""
src/data_ingestion.py
=====================
Multi-source data fetcher for the WC2026 Knockout Prediction System.

Attempts live scraping from eloratings.net, Wikipedia, and Transfermarkt.
Gracefully falls back to curated static CSVs on any failure.

All external calls are wrapped in try/except with logging.
"""

import os
import logging
import time
import hashlib
import json
from typing import Optional, Dict, Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src import config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# HTTP HELPERS
# ─────────────────────────────────────────────────────────────

HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
REQUEST_TIMEOUT: int = 15  # seconds
REQUEST_DELAY: float = 1.5  # seconds between requests (polite scraping)


def _fetch_url(url: str, timeout: int = REQUEST_TIMEOUT) -> Optional[str]:
    """
    Fetch the HTML content of a URL.

    Parameters
    ----------
    url : str
        Target URL.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    str or None
        HTML text on success, None on failure.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        time.sleep(REQUEST_DELAY)
        return response.text
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None


# ─────────────────────────────────────────────────────────────
# STATIC FALLBACK LOADERS
# ─────────────────────────────────────────────────────────────

def load_static_elo() -> pd.DataFrame:
    """Load World Football Elo ratings from the curated static CSV."""
    path = os.path.join(config.STATIC_DATA_DIR, "elo_ratings.csv")
    df = pd.read_csv(path, comment="#")
    logger.info("Loaded static Elo ratings (%d teams)", len(df))
    return df


def load_static_fifa_rankings() -> pd.DataFrame:
    """Load FIFA World Rankings from the curated static CSV."""
    path = os.path.join(config.STATIC_DATA_DIR, "fifa_rankings.csv")
    df = pd.read_csv(path, comment="#")
    logger.info("Loaded static FIFA rankings (%d teams)", len(df))
    return df


def load_static_squad_values() -> pd.DataFrame:
    """Load Transfermarkt squad market values from the curated static CSV."""
    path = os.path.join(config.STATIC_DATA_DIR, "squad_values.csv")
    df = pd.read_csv(path, comment="#")
    logger.info("Loaded static squad values (%d teams)", len(df))
    return df


def load_static_team_form() -> pd.DataFrame:
    """Load team form (last 10 matches) from the curated static CSV."""
    path = os.path.join(config.STATIC_DATA_DIR, "team_form.csv")
    df = pd.read_csv(path, comment="#")
    logger.info("Loaded static team form (%d teams)", len(df))
    return df


def load_static_team_stats() -> pd.DataFrame:
    """Load comprehensive team statistics from the curated static CSV."""
    path = os.path.join(config.STATIC_DATA_DIR, "team_stats.csv")
    df = pd.read_csv(path, comment="#")
    logger.info("Loaded static team stats (%d teams)", len(df))
    return df


def load_static_player_data() -> pd.DataFrame:
    """Load player data (goalscorer model inputs) from the curated static CSV."""
    path = os.path.join(config.STATIC_DATA_DIR, "player_data.csv")
    df = pd.read_csv(path, comment="#")
    logger.info("Loaded static player data (%d players)", len(df))
    return df


def load_static_h2h() -> pd.DataFrame:
    """Load head-to-head records from the curated static CSV."""
    path = os.path.join(config.STATIC_DATA_DIR, "h2h_records.csv")
    df = pd.read_csv(path, comment="#")
    logger.info("Loaded static H2H records (%d matchups)", len(df))
    return df


# ─────────────────────────────────────────────────────────────
# LIVE SCRAPERS
# ─────────────────────────────────────────────────────────────

def scrape_elo_ratings() -> Optional[pd.DataFrame]:
    """
    Attempt to scrape current World Football Elo ratings from eloratings.net.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns [team, elo_rating, elo_rank] or None on failure.
    """
    url = config.DATA_SOURCES["elo_url"]
    html = _fetch_url(url)
    if html is None:
        return None

    try:
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table", {"id": "ranking"})
        if table is None:
            table = soup.find("table")
        if table is None:
            return None

        rows = []
        for tr in table.find_all("tr")[1:]:
            cells = tr.find_all("td")
            if len(cells) >= 3:
                rank = cells[0].get_text(strip=True)
                team = cells[1].get_text(strip=True)
                rating = cells[2].get_text(strip=True).replace(",", "")
                try:
                    rows.append({
                        "team": team,
                        "elo_rating": float(rating),
                        "elo_rank": int(rank),
                        "source_date": "live",
                    })
                except ValueError:
                    continue

        if rows:
            df = pd.DataFrame(rows)
            logger.info("Scraped live Elo ratings (%d teams)", len(df))
            return df
    except Exception as exc:
        logger.warning("Elo scraping failed: %s", exc)

    return None


def scrape_wc_bracket() -> Optional[Dict[str, str]]:
    """
    Attempt to scrape the live WC 2026 bracket from Wikipedia.

    Returns
    -------
    dict or None
        Maps match_id (e.g. "R16_1") to the winning team name, or None on failure.
    """
    url = config.DATA_SOURCES["wc2026_knockout"]
    html = _fetch_url(url)
    if html is None:
        return None

    try:
        soup = BeautifulSoup(html, "lxml")
        # Wikipedia knockout bracket tables vary — look for result tables
        tables = pd.read_html(html)
        results: Dict[str, str] = {}

        # Try to identify R16 results by table content heuristics
        for tbl in tables:
            cols_lower = [str(c).lower() for c in tbl.columns]
            if any("score" in c or "result" in c for c in cols_lower):
                logger.info("Found potential bracket table: %s", tbl.columns.tolist())
                # Parsing is heuristic — exact structure depends on Wikipedia edit state
                break

        logger.info("Bracket scrape returned %d confirmed results", len(results))
        return results if results else None

    except Exception as exc:
        logger.warning("Bracket scraping failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────
# MAIN INGESTION ORCHESTRATOR
# ─────────────────────────────────────────────────────────────

class DataIngestion:
    """
    Orchestrates data loading: tries live scraping, falls back to static CSVs.

    Attributes
    ----------
    elo : pd.DataFrame
    fifa : pd.DataFrame
    squad_values : pd.DataFrame
    team_form : pd.DataFrame
    team_stats : pd.DataFrame
    players : pd.DataFrame
    h2h : pd.DataFrame
    r16_results : dict  # match_id -> winning team (from live scrape or config)
    """

    def __init__(self, use_live_data: bool = True) -> None:
        """
        Parameters
        ----------
        use_live_data : bool
            If True, attempt live scraping before falling back to static data.
            If False, skip live scraping entirely (offline mode).
        """
        self.use_live_data = use_live_data
        self._load_all()

    def _load_all(self) -> None:
        """Load all data sources, attempting live then falling back to static."""
        logger.info("=== Data Ingestion Starting ===")

        # Elo Ratings
        if self.use_live_data:
            self.elo = scrape_elo_ratings() or load_static_elo()
        else:
            self.elo = load_static_elo()

        # All other sources use static CSVs (most reliable for model consistency)
        self.fifa         = load_static_fifa_rankings()
        self.squad_values = load_static_squad_values()
        self.team_form    = load_static_team_form()
        self.team_stats   = load_static_team_stats()
        self.players      = load_static_player_data()
        self.h2h          = load_static_h2h()

        # Live bracket results
        self.r16_results: Dict[str, str] = {}
        if self.use_live_data:
            live_bracket = scrape_wc_bracket()
            if live_bracket:
                self.r16_results = live_bracket
                logger.info("Using live bracket results for %d R16 matches",
                            len(live_bracket))

        logger.info("=== Data Ingestion Complete ===")

    def get_team_data(self, team: str) -> Dict[str, Any]:
        """
        Retrieve a unified data dictionary for a specific team.

        Parameters
        ----------
        team : str
            Team name (must match names in static CSVs).

        Returns
        -------
        dict
            Merged data from all sources for the team.
        """
        def _first(df: pd.DataFrame, col: str, default: Any) -> Any:
            row = df[df["team"] == team]
            if row.empty:
                return default
            return row.iloc[0].get(col, default)

        return {
            "team": team,
            "elo_rating":           _first(self.elo,          "elo_rating",              1800.0),
            "elo_rank":             _first(self.elo,          "elo_rank",                50),
            "fifa_rank":            _first(self.fifa,         "fifa_rank",               50),
            "fifa_points":          _first(self.fifa,         "fifa_points",             1500.0),
            "squad_value_eur_m":    _first(self.squad_values, "squad_value_eur_m",       300.0),
            "avg_player_value":     _first(self.squad_values, "avg_player_value_eur_m",  10.0),
            "form_score":           _first(self.team_form,    "form_score",              0.60),
            "goals_for_10":         _first(self.team_form,    "goals_for_10",            12),
            "goals_against_10":     _first(self.team_form,    "goals_against_10",        10),
            "attack_strength":      _first(self.team_stats,   "attack_strength",         1.0),
            "defense_strength":     _first(self.team_stats,   "defense_strength",        1.0),
            "xg_for":               _first(self.team_stats,   "xg_for",                  1.5),
            "xg_against":           _first(self.team_stats,   "xg_against",              1.5),
            "avg_possession":       _first(self.team_stats,   "avg_possession",          50.0),
            "ko_win_rate":          _first(self.team_stats,   "ko_win_rate",             0.50),
            "penalty_ko_rate":      _first(self.team_stats,   "penalty_ko_rate",         0.50),
            "avg_player_rating":    _first(self.team_stats,   "avg_player_rating",       7.0),
            "injury_factor":        _first(self.team_stats,   "injury_suspension_factor",1.0),
            "rest_days_before_qf":  _first(self.team_stats,   "rest_days_before_qf",     5),
            "is_host":              team in config.HOST_NATIONS,
            "pk_rate":              config.PENALTY_HISTORICAL_RATES.get(
                                        team, config.PENALTY_HISTORICAL_RATES["DEFAULT"]
                                    ),
        }

    def get_h2h(self, team_a: str, team_b: str) -> Dict[str, Any]:
        """
        Retrieve head-to-head statistics for a fixture.

        Parameters
        ----------
        team_a, team_b : str
            Team names.

        Returns
        -------
        dict
            H2H statistics from team_a's perspective.
        """
        mask_ab = (self.h2h["team_a"] == team_a) & (self.h2h["team_b"] == team_b)
        mask_ba = (self.h2h["team_a"] == team_b) & (self.h2h["team_b"] == team_a)

        if mask_ab.any():
            row = self.h2h[mask_ab].iloc[0]
            return {
                "meetings":    int(row["meetings"]),
                "a_wins":      int(row["team_a_wins"]),
                "draws":       int(row["draws"]),
                "b_wins":      int(row["team_b_wins"]),
                "a_goals":     int(row["team_a_goals"]),
                "b_goals":     int(row["team_b_goals"]),
                "last_result": str(row["last_meeting_result"]),
                "a_win_rate":  row["team_a_wins"] / max(row["meetings"], 1),
            }
        elif mask_ba.any():
            row = self.h2h[mask_ba].iloc[0]
            last_map = {"W": "L", "L": "W", "D": "D"}
            return {
                "meetings":    int(row["meetings"]),
                "a_wins":      int(row["team_b_wins"]),
                "draws":       int(row["draws"]),
                "b_wins":      int(row["team_a_wins"]),
                "a_goals":     int(row["team_b_goals"]),
                "b_goals":     int(row["team_a_goals"]),
                "last_result": last_map.get(str(row["last_meeting_result"]), "D"),
                "a_win_rate":  row["team_b_wins"] / max(row["meetings"], 1),
            }
        else:
            return {
                "meetings": 0, "a_wins": 0, "draws": 0, "b_wins": 0,
                "a_goals": 0, "b_goals": 0, "last_result": "D", "a_win_rate": 0.5,
            }

    def save_processed(self) -> None:
        """Persist processed DataFrames to data/processed/ for audit trail."""
        os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
        self.elo.to_csv(
            os.path.join(config.PROCESSED_DATA_DIR, "elo_used.csv"), index=False)
        self.fifa.to_csv(
            os.path.join(config.PROCESSED_DATA_DIR, "fifa_used.csv"), index=False)
        self.team_stats.to_csv(
            os.path.join(config.PROCESSED_DATA_DIR, "team_stats_used.csv"), index=False)
        logger.info("Processed data saved to %s", config.PROCESSED_DATA_DIR)
