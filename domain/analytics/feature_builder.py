# -*- coding: utf-8 -*-
"""
domain/analytics/feature_builder.py
-----------------------------------
Pipeline de ingeniería de características temporales y contextuales para la NBA.

Garantiza:
1. Ponderación temporal mediante EWMA (Exponentially Weighted Moving Average)
   con decaimiento configurable (halflife de 10 a 15 partidos).
2. Estricta ausencia de Data Leakage mediante el desplazamiento temporal (.shift(1))
   en todas las métricas acumuladas antes del partido evaluado.
3. Derivación de variables de fatiga y calendario: rest_days, is_b2b y rest_advantage.
4. Generación de vectores diferenciales (Home - Away) y targets para entrenamiento ML.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import structlog

from domain.analytics.advanced_metrics import (
    calculate_possessions,
    calculate_offensive_rating,
    calculate_defensive_rating,
    calculate_net_rating,
    calculate_efg_pct,
    calculate_tov_pct,
    calculate_orb_pct,
    calculate_drb_pct,
    calculate_ft_rate,
)
from domain.analytics.travel_context import (
    calculate_travel_miles,
    calculate_time_zones_crossed,
    is_altitude_advantage,
    calculate_travel_fatigue_index,
)
from infrastructure.database import Database, get_db

logger = structlog.get_logger()

# Métricas canónicas y Four Factors que serán procesadas mediante EWMA
CANONICAL_METRICS = [
    "poss",
    "ortg",
    "drtg",
    "net_rtg",
    "efg_pct",
    "tov_pct",
    "orb_pct",
    "drb_pct",
    "ft_rate",
    "opp_efg_pct",
    "opp_tov_pct",
    "opp_ft_rate",
    "pts",
    "opp_pts",
]


class FeatureBuilder:
    """Constructor de características temporales y avanzadas de baloncesto."""

    def __init__(
        self,
        db: Optional[Database] = None,
        halflife: float = 12.0,
        min_games: int = 5,
    ) -> None:
        """
        Args:
            db: Instancia del gestor de base de datos SQLite.
            halflife: Número de partidos para que el peso de un partido decaiga a la mitad (EWMA).
            min_games: Partidos mínimos previos requeridos para que un partido se incluya en entrenamiento.
        """
        self._db = db or get_db()
        self.halflife = halflife
        self.min_games = min_games

    def load_raw_data_from_db(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extrae historical_games y historical_team_boxscores desde SQLite."""
        logger.info("feature-builder-cargando-db")
        games_rows = self._db.execute(
            """
            SELECT game_id, season, game_date, home_team_id, away_team_id,
                   home_score, away_score, season_type
            FROM historical_games
            ORDER BY game_date ASC, game_id ASC
            """
        )
        boxscores_rows = self._db.execute(
            """
            SELECT game_id, team_id, is_home, pts, fgm, fga, fg3m, fg3a,
                   ftm, fta, oreb, dreb, reb, ast, stl, blk, tov, pf, plus_minus
            FROM historical_team_boxscores
            ORDER BY game_id ASC, is_home DESC
            """
        )

        games_df = pd.DataFrame([dict(r) for r in games_rows])
        boxscores_df = pd.DataFrame([dict(r) for r in boxscores_rows])

        logger.info(
            "feature-builder-datos-cargados",
            games_count=len(games_df),
            boxscores_count=len(boxscores_df),
        )
        return games_df, boxscores_df

    def pair_team_and_opponent_boxscores(
        self,
        games_df: pd.DataFrame,
        boxscores_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Empareja cada registro de boxscore con las estadísticas del oponente en el mismo partido.
        Retorna un DataFrame a nivel de equipo-partido con métricas propias y del rival.
        """
        if games_df.empty or boxscores_df.empty:
            return pd.DataFrame()

        # Separar boxscores en local y visitante
        home_box = boxscores_df[boxscores_df["is_home"] == 1].copy()
        away_box = boxscores_df[boxscores_df["is_home"] == 0].copy()

        # Crear sufijos para equipo y oponente
        # 1. Perspectiva Local: team = home, opp = away
        opp_cols = {
            "team_id": "opp_team_id",
            "pts": "opp_pts",
            "fgm": "opp_fgm",
            "fga": "opp_fga",
            "fg3m": "opp_fg3m",
            "fg3a": "opp_fg3a",
            "ftm": "opp_ftm",
            "fta": "opp_fta",
            "oreb": "opp_oreb",
            "dreb": "opp_dreb",
            "reb": "opp_reb",
            "ast": "opp_ast",
            "stl": "opp_stl",
            "blk": "opp_blk",
            "tov": "opp_tov",
            "pf": "opp_pf",
            "plus_minus": "opp_plus_minus",
        }

        away_renamed_for_home = away_box.rename(columns=opp_cols).drop(columns=["is_home"], errors="ignore")
        home_perspective = pd.merge(home_box, away_renamed_for_home, on="game_id", how="inner")

        # 2. Perspectiva Visitante: team = away, opp = home
        home_renamed_for_away = home_box.rename(columns=opp_cols).drop(columns=["is_home"], errors="ignore")
        away_perspective = pd.merge(away_box, home_renamed_for_away, on="game_id", how="inner")

        combined = pd.concat([home_perspective, away_perspective], ignore_index=True)

        # Unir metadatos del partido (fecha, temporada, scores)
        meta_cols = ["game_id", "season", "game_date", "season_type", "home_team_id", "away_team_id", "home_score", "away_score"]
        final_paired = pd.merge(combined, games_df[meta_cols], on="game_id", how="inner")

        return final_paired

    def compute_single_game_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula posesiones, ratings por 100 posesiones y Four Factors en cada partido individual."""
        if df.empty:
            return df

        df = df.copy()

        # 1. Posesiones estimadas según fórmula canónica
        # Poss = 0.5 * (FGA + 0.44 * FTA - OREB + TOV)
        # Promediamos las estimaciones de ambos bandos para balance posesional idéntico
        poss_team = calculate_possessions(df["fga"], df["fta"], df["oreb"], df["tov"])
        poss_opp = calculate_possessions(df["opp_fga"], df["opp_fta"], df["opp_oreb"], df["opp_tov"])
        df["poss"] = 0.5 * (poss_team + poss_opp)

        # 2. Ratings ofensivo, defensivo y neto
        df["ortg"] = calculate_offensive_rating(df["pts"], df["poss"])
        df["drtg"] = calculate_defensive_rating(df["opp_pts"], df["poss"])
        df["net_rtg"] = calculate_net_rating(df["ortg"], df["drtg"])

        # 3. Los Cuatro Factores de Dean Oliver (Ofensivos)
        df["efg_pct"] = calculate_efg_pct(df["fgm"], df["fg3m"], df["fga"])
        df["tov_pct"] = calculate_tov_pct(df["tov"], df["fga"], df["fta"])
        df["orb_pct"] = calculate_orb_pct(df["oreb"], df["opp_dreb"])
        df["drb_pct"] = calculate_drb_pct(df["dreb"], df["opp_oreb"])
        df["ft_rate"] = calculate_ft_rate(df["ftm"], df["fga"])

        # 4. Los Cuatro Factores Permitidos al Rival (Defensivos)
        df["opp_efg_pct"] = calculate_efg_pct(df["opp_fgm"], df["opp_fg3m"], df["opp_fga"])
        df["opp_tov_pct"] = calculate_tov_pct(df["opp_tov"], df["opp_fga"], df["opp_fta"])
        df["opp_ft_rate"] = calculate_ft_rate(df["opp_ftm"], df["opp_fga"])

        return df

    def compute_temporal_and_fatigue_features(
        self,
        df: pd.DataFrame,
        halflife: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Calcula EWMA sin fuga de datos (.shift(1)) y variables de descanso/fatiga por equipo.
        """
        if df.empty:
            return df

        hl = halflife or self.halflife
        df = df.copy()

        # Asegurar formato datetime y orden cronológico estricto
        df["game_date_dt"] = pd.to_datetime(df["game_date"])
        df = df.sort_values(by=["team_id", "game_date_dt", "game_id"]).reset_index(drop=True)

        # Crear columnas destino
        df["game_num"] = 0
        df["rest_days"] = 7.0
        df["is_b2b"] = 0

        for metric in CANONICAL_METRICS:
            df[f"ewma_{metric}"] = np.nan

        # Agrupar cronológicamente por equipo
        processed_groups = []
        for team_id, group in df.groupby("team_id", sort=False):
            grp = group.copy()

            # Número correlativo de juego del equipo en la muestra
            grp["game_num"] = np.arange(1, len(grp) + 1)

            # Variables de descanso y fatiga
            prev_dates = grp["game_date_dt"].shift(1)
            raw_rest = (grp["game_date_dt"] - prev_dates).dt.days
            # Imputar inicio de temporada / descansos largos a 7 días y topar a máximo 10
            grp["rest_days"] = raw_rest.fillna(7.0).clip(lower=0.0, upper=10.0)
            grp["is_b2b"] = (grp["rest_days"] == 1.0).astype(int)

            # ZERO DATA LEAKAGE EWMA:
            # Calculamos ewm().mean() y desplazamos 1 partido con .shift(1).
            # Para el partido K, solo se usan los partidos 1 .. K-1.
            for metric in CANONICAL_METRICS:
                ewma_series = grp[metric].ewm(halflife=hl, min_periods=1).mean()
                grp[f"ewma_{metric}"] = ewma_series.shift(1)

            processed_groups.append(grp)

        final_df = pd.concat(processed_groups, ignore_index=True)
        return final_df

    def build_matchup_dataset(self, team_metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combina las features pre-partido calculadas para el equipo local y el visitante,
        produciendo las variables diferenciales (Home - Away) y los targets de entrenamiento.
        """
        if team_metrics_df.empty:
            return pd.DataFrame()

        # Separar filas locales y visitantes
        home_side = team_metrics_df[team_metrics_df["is_home"] == 1].copy()
        away_side = team_metrics_df[team_metrics_df["is_home"] == 0].copy()

        # Prefijos para columnas específicas de local y visitante
        feature_cols = [f"ewma_{m}" for m in CANONICAL_METRICS] + ["rest_days", "is_b2b", "game_num"]

        home_rename = {col: f"home_{col}" for col in feature_cols}
        away_rename = {col: f"away_{col}" for col in feature_cols}

        home_subset = home_side[["game_id", "season", "game_date", "season_type", "home_team_id", "away_team_id", "home_score", "away_score"] + feature_cols].rename(columns=home_rename)
        away_subset = away_side[["game_id"] + feature_cols].rename(columns=away_rename)

        matchups = pd.merge(home_subset, away_subset, on="game_id", how="inner")

        # 1. Variables de descanso diferenciales
        matchups["rest_advantage"] = matchups["home_rest_days"] - matchups["away_rest_days"]

        # Variables geográficas y contextuales de viaje
        matchups["is_altitude_advantage"] = matchups["home_team_id"].apply(
            lambda tid: 1 if is_altitude_advantage(tid) else 0
        )
        matchups["travel_miles_away"] = [
            calculate_travel_miles(away, home)
            for away, home in zip(matchups["away_team_id"], matchups["home_team_id"])
        ]
        matchups["time_zones_crossed_away"] = [
            calculate_time_zones_crossed(away, home)
            for away, home in zip(matchups["away_team_id"], matchups["home_team_id"])
        ]
        matchups["travel_fatigue_away"] = [
            calculate_travel_fatigue_index(dist, tz, rest, b2b)
            for dist, tz, rest, b2b in zip(
                matchups["travel_miles_away"],
                matchups["time_zones_crossed_away"],
                matchups["away_rest_days"],
                matchups["away_is_b2b"],
            )
        ]

        # 2. Variables diferenciales clave (Home - Away)
        for metric in CANONICAL_METRICS:
            home_col = f"home_ewma_{metric}"
            away_col = f"away_ewma_{metric}"
            matchups[f"diff_{metric}"] = matchups[home_col] - matchups[away_col]

        # 3. Targets de entrenamiento
        matchups["point_spread"] = matchups["home_score"] - matchups["away_score"]
        matchups["home_win"] = (matchups["point_spread"] > 0).astype(int)
        matchups["total_points"] = matchups["home_score"] + matchups["away_score"]

        # 4. Filtrar partidos iniciales con historial insuficiente (Zero Cold-Start)
        valid_mask = (matchups["home_game_num"] >= self.min_games) & (matchups["away_game_num"] >= self.min_games)
        filtered_matchups = matchups[valid_mask].copy().reset_index(drop=True)

        logger.info(
            "feature-builder-dataset-ensamblado",
            total_games=len(matchups),
            filtered_games=len(filtered_matchups),
            features_count=len(filtered_matchups.columns),
        )
        return filtered_matchups

    def build_dataset_pipeline(self) -> pd.DataFrame:
        """Ejecuta el pipeline completo de ingeniería de características desde la base de datos."""
        games_df, boxscores_df = self.load_raw_data_from_db()
        if games_df.empty or boxscores_df.empty:
            logger.warning("feature-builder-sin-datos")
            return pd.DataFrame()

        paired_df = self.pair_team_and_opponent_boxscores(games_df, boxscores_df)
        single_metrics_df = self.compute_single_game_metrics(paired_df)
        temporal_df = self.compute_temporal_and_fatigue_features(single_metrics_df)
        matchup_dataset = self.build_matchup_dataset(temporal_df)

        return matchup_dataset
