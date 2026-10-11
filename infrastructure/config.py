# -*- coding: utf-8 -*-
"""
infrastructure/config.py
------------------------
Cargador de configuración centralizada.
Lee config.yaml y provee acceso a los parámetros del sistema.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """Cargador de configuración singleton."""

    _instance: Optional["Config"] = None
    _config: Dict[str, Any] = {}

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        config_path = self._find_config_file()
        if config_path and config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
        else:
            self._config = self._default_config()

    def _find_config_file(self) -> Optional[Path]:
        candidates = [
            Path("config.yaml"),
            Path(__file__).parent.parent / "config.yaml",
            Path(os.environ.get("NBA_PREDICTOR_CONFIG", "config.yaml")),
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _default_config(self) -> Dict[str, Any]:
        return {
            "weights": {
                "PTS": 0.30, "FG_PCT": 0.20, "FG3_PCT": 0.15,
                "REB": 0.10, "AST": 0.10, "TOV": -0.05,
                "STL": 0.05, "BLK": 0.05, "PLUS_MINUS": 0.10,
            },
            "normalization": {
                "PTS": 20.0, "FG_PCT": 0.1, "FG3_PCT": 0.1,
                "REB": 10.0, "AST": 8.0, "TOV": 5.0,
                "STL": 4.0, "BLK": 3.0, "PLUS_MINUS": 15.0,
            },
            "home_advantage": 0.05,
            "prediction": {
                "engine": "heuristic",
                "min_probability": 0.1,
                "max_probability": 0.9,
            },
            "cache": {"ttl_hours": 2},
            "logging": {"level": "INFO"},
        }

    @property
    def weights(self) -> Dict[str, float]:
        return self._config.get("weights", {})

    @property
    def normalization(self) -> Dict[str, float]:
        return self._config.get("normalization", {})

    @property
    def home_advantage(self) -> float:
        return self._config.get("home_advantage", 0.05)

    @property
    def min_probability(self) -> float:
        return self._config.get("prediction", {}).get("min_probability", 0.1)

    @property
    def max_probability(self) -> float:
        return self._config.get("prediction", {}).get("max_probability", 0.9)

    @property
    def confidence_threshold(self) -> float:
        return self._config.get("prediction", {}).get("confidence_threshold", 0.3)

    @property
    def significant_factor_threshold(self) -> float:
        return self._config.get("prediction", {}).get("significant_factor_threshold", 0.3)

    @property
    def recent_form_weight(self) -> float:
        return self._config.get("prediction", {}).get("recent_form_weight", 2.0)

    @property
    def injury_impact(self) -> float:
        return self._config.get("prediction", {}).get("injury_impact", 5.0)

    @property
    def h2h_weight(self) -> float:
        return self._config.get("prediction", {}).get("h2h_weight", 0.15)

    @property
    def h2h_min_games(self) -> int:
        return self._config.get("prediction", {}).get("h2h_min_games", 3)

    @property
    def ml_weight(self) -> float:
        return self._config.get("prediction", {}).get("ml_weight", 0.5)

    @property
    def hybrid_ml_weight(self) -> float:
        return self.ml_weight

    @property
    def model_path(self) -> str:
        return self._config.get("prediction", {}).get("model_path", "xgboost_model.json")

    @property
    def cache_ttl_hours(self) -> int:
        return self._config.get("cache", {}).get("ttl_hours", 2)

    @property
    def log_level(self) -> str:
        return self._config.get("logging", {}).get("level", "INFO")

    @property
    def log_format(self) -> str:
        return self._config.get("logging", {}).get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    @property
    def team_csv(self) -> str:
        return self._config.get("repository", {}).get("team_csv", "team_data.csv")

    @property
    def history_csv(self) -> str:
        return self._config.get("repository", {}).get("history_csv", "prediction_history.csv")

    @property
    def repository_type(self) -> str:
        return self._config.get("repository", {}).get("type", "sqlite")

    @property
    def database_path(self) -> str:
        return self._config.get("database", {}).get("path", "nba_predictor.db")

    @property
    def bet_tracking_excel_path(self) -> str:
        return os.environ.get(
            "NBA_BET_TRACKING_EXCEL",
            self._config.get("bet_tracking", {}).get(
                "excel_path",
                r"C:\Users\mblad\Documents\Codigoes\nba_bet_tracking.xlsx",
            ),
        )

    @property
    def odds_api_key(self) -> Optional[str]:
        raw_key = (
            os.environ.get("ODDS_API_KEY")
            or self._config.get("odds_api_key")
            or self._config.get("odds_api", {}).get("api_key")
        )
        if raw_key and str(raw_key).strip():
            return str(raw_key).strip()
        return None


config = Config()