# -*- coding: utf-8 -*-
"""
infrastructure/csv_repositories.py
------------------------------------
Implementaciones concretas de TeamRepository y PredictionRepository
usando archivos CSV como almacenamiento.

Correcciones respecto al código original:
- Serialización con json.dumps/loads (no ast.literal_eval).
- Timestamps consistentes en UTC ISO 8601.
- PredictionRepository usa modo append (no reescritura total).
- Bloqueo con threading.Lock para seguridad en acceso concurrente.
- TeamRepository usa un cache en memoria para evitar doble lectura de disco.
"""

from __future__ import annotations
import csv
import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from domain.entities import H2HStats, Odds, PlayerStat, PredictionResult, Team, TeamStats
from domain.repositories import H2HRepository, PredictionRepository, TeamRepository

logger = logging.getLogger(__name__)


class CSVTeamRepository(TeamRepository):
    """
    Almacena datos de equipos en un CSV con una fila por (team_id, season).
    Usa un dict en memoria como caché de primer nivel para evitar lecturas repetidas.
    Thread-safe mediante Lock.
    """

    _COLUMNS = [
        "team_id", "full_name", "abbreviation", "season",
        "stats_json", "top_players_json", "fetched_at", "source",
    ]

    def __init__(self, path: str = "team_data.csv") -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._cache: dict[tuple, Team] = {}
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not self._path.exists():
            with open(self._path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._COLUMNS)
                writer.writeheader()

    def find(self, team_id: int, season: str) -> Optional[Team]:
        key = (team_id, season)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            team = self._read_from_disk(team_id, season)
            if team:
                self._cache[key] = team
            return team

    def save(self, team: Team) -> None:
        with self._lock:
            rows = self._load_all_rows()
            key = (str(team.team_id), team.season)
            # Eliminar fila existente para ese equipo/temporada
            rows = [r for r in rows if not (r["team_id"] == key[0] and r["season"] == key[1])]
            rows.append(self._to_row(team))
            self._write_all_rows(rows)
            self._cache[(team.team_id, team.season)] = team

    def delete(self, team_id: int, season: str) -> None:
        with self._lock:
            rows = self._load_all_rows()
            rows = [r for r in rows if not (r["team_id"] == str(team_id) and r["season"] == season)]
            self._write_all_rows(rows)
            self._cache.pop((team_id, season), None)

    def _read_from_disk(self, team_id: int, season: str) -> Optional[Team]:
        rows = self._load_all_rows()
        for row in rows:
            if row["team_id"] == str(team_id) and row["season"] == season:
                return self._from_row(row)
        return None

    def _load_all_rows(self) -> List[dict]:
        if not self._path.exists():
            return []
        with open(self._path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _write_all_rows(self, rows: List[dict]) -> None:
        with open(self._path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _to_row(team: Team) -> dict:
        return {
            "team_id": str(team.team_id),
            "full_name": team.full_name,
            "abbreviation": team.abbreviation,
            "season": team.season,
            "stats_json": json.dumps(team.stats.to_dict()),
            "top_players_json": json.dumps([
                {"name": p.name, "avg_points": p.avg_points, "avg_minutes": p.avg_minutes}
                for p in team.top_players
            ]),
            "fetched_at": team.fetched_at.isoformat() if team.fetched_at else "",
            "source": team.source,
        }

    @staticmethod
    def _from_row(row: dict) -> Team:
        try:
            stats = TeamStats.from_dict(json.loads(row.get("stats_json", "{}")))
        except (json.JSONDecodeError, KeyError):
            logger.warning("No se pudo deserializar stats para team_id=%s", row.get("team_id"))
            stats = TeamStats()

        try:
            players_data = json.loads(row.get("top_players_json", "[]"))
            players = [
                PlayerStat(
                    name=p["name"],
                    avg_points=p.get("avg_points", 0.0),
                    avg_minutes=p.get("avg_minutes", 0.0),
                )
                for p in players_data
            ]
        except (json.JSONDecodeError, KeyError):
            players = []

        fetched_at = None
        raw_ts = row.get("fetched_at", "")
        if raw_ts:
            try:
                fetched_at = datetime.fromisoformat(raw_ts)
            except ValueError:
                pass

        return Team(
            team_id=int(row["team_id"]),
            full_name=row.get("full_name", ""),
            abbreviation=row.get("abbreviation", ""),
            season=row.get("season", ""),
            stats=stats,
            top_players=players,
            fetched_at=fetched_at,
            source=row.get("source", "unknown"),
        )


class CSVPredictionRepository(PredictionRepository):
    """
    Historial de predicciones en CSV.
    Usa modo append para añadir filas sin reescribir el archivo completo.
    Thread-safe mediante Lock.
    """

    _COLUMNS = [
        "predicted_at", "home_team", "away_team",
        "home_win_prob", "away_win_prob", "confidence",
        "odds_home", "odds_away",
        "home_expected_value", "away_expected_value",
        "recommended_bet", "key_factors",
    ]

    def __init__(self, path: str = "prediction_history.csv") -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not self._path.exists():
            with open(self._path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._COLUMNS)
                writer.writeheader()

    def save(self, prediction: PredictionResult) -> None:
        with self._lock:
            with open(self._path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._COLUMNS)
                writer.writerow(self._to_row(prediction))

    def list_all(self) -> List[PredictionResult]:
        with self._lock:
            if not self._path.exists():
                return []
            with open(self._path, newline="", encoding="utf-8") as f:
                return [self._from_row(r) for r in csv.DictReader(f)]

    @staticmethod
    def _to_row(p: PredictionResult) -> dict:
        return {
            "predicted_at": p.predicted_at.isoformat(),
            "home_team": p.home_team,
            "away_team": p.away_team,
            "home_win_prob": p.home_win_prob,
            "away_win_prob": p.away_win_prob,
            "confidence": p.confidence,
            "odds_home": p.odds.home_decimal if p.odds else "",
            "odds_away": p.odds.away_decimal if p.odds else "",
            "home_expected_value": p.home_expected_value if p.home_expected_value is not None else "",
            "away_expected_value": p.away_expected_value if p.away_expected_value is not None else "",
            "recommended_bet": p.recommended_bet,
            "key_factors": " | ".join(p.key_factors),
        }

    @staticmethod
    def _from_row(row: dict) -> PredictionResult:
        odds = None
        if row.get("odds_home") and row.get("odds_away"):
            try:
                odds = Odds(
                    home_decimal=float(row["odds_home"]),
                    away_decimal=float(row["odds_away"]),
                )
            except ValueError:
                pass

        return PredictionResult(
            home_team=row.get("home_team", ""),
            away_team=row.get("away_team", ""),
            home_win_prob=float(row.get("home_win_prob", 0)),
            away_win_prob=float(row.get("away_win_prob", 0)),
            confidence=float(row.get("confidence", 0)),
            key_factors=row.get("key_factors", "").split(" | "),
            predicted_at=datetime.fromisoformat(row["predicted_at"]) if row.get("predicted_at") else datetime.utcnow(),
            odds=odds,
            home_expected_value=float(row["home_expected_value"]) if row.get("home_expected_value") else None,
            away_expected_value=float(row["away_expected_value"]) if row.get("away_expected_value") else None,
            recommended_bet=row.get("recommended_bet", "No bet"),
        )


class CSVH2HRepository(H2HRepository):
    """
    Repositorio de historial head-to-head derivando desde prediction_history.csv.
    Construye estadísticas H2H desde partidos jugados (no predicciones).
    """

    def __init__(self, history_path: str = "h2h_history.csv") -> None:
        self._path = Path(history_path)
        self._lock = threading.Lock()

    def get_h2h(self, home_team: str, away_team: str) -> Optional[H2HStats]:
        with self._lock:
            if not self._path.exists():
                return None
            try:
                with open(self._path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    matches = [
                        r for r in reader
                        if (r.get("home_team") == home_team and r.get("away_team") == away_team)
                        or (r.get("home_team") == away_team and r.get("away_team") == home_team)
                    ]
            except Exception:
                return None

        if not matches:
            return None

        home_wins = 0
        away_wins = 0
        for match in matches:
            if not match.get("actual_result"):
                continue
            result = match.get("actual_result", "").lower()
            is_home_team_home = match.get("home_team") == home_team
            if result == "home_win":
                if is_home_team_home:
                    home_wins += 1
                else:
                    away_wins += 1
            elif result == "away_win":
                if is_home_team_home:
                    away_wins += 1
                else:
                    home_wins += 1

        total = home_wins + away_wins
        if total == 0:
            return None

        return H2HStats(
            home_team=home_team,
            away_team=away_team,
            home_wins=home_wins,
            away_wins=away_wins,
            total_games=total,
        )

    def save_result(self, home_team: str, away_team: str, actual_result: str) -> None:
        """Registra un resultado real del partido."""
        with self._lock:
            cols = ["home_team", "away_team", "actual_result", "played_at"]
            file_exists = self._path.exists()
            with open(self._path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=cols)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    "home_team": home_team,
                    "away_team": away_team,
                    "actual_result": actual_result,
                    "played_at": datetime.utcnow().isoformat(),
                })
