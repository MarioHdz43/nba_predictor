# -*- coding: utf-8 -*-
"""
domain/analytics/advanced_metrics.py
------------------------------------
Módulo de cálculo de métricas canónicas y avanzadas para baloncesto profesional (NBA).
Funciones puras, vectorizadas y protegidas ante divisiones por cero.

Métricas implementadas:
- Posesiones estimadas por partido y por equipo
- Offensive Rating (ORtg), Defensive Rating (DRtg) y Net Rating (NRtg)
- Pace ajustado a 48 minutos reglamentarios
- Los Cuatro Factores de Dean Oliver: eFG%, TOV%, ORB%/DRB%, FT Rate
"""

from __future__ import annotations
from typing import Union
import numpy as np
import pandas as pd

Numeric = Union[float, int, np.ndarray, pd.Series]


def _safe_divide(
    numerator: Numeric,
    denominator: Numeric,
    fill_value: float = 0.0,
) -> Numeric:
    """División protegida contra ceros e infinitos para escalares, arrays y series de pandas."""
    if isinstance(numerator, pd.Series) or isinstance(denominator, pd.Series):
        num = pd.Series(numerator) if not isinstance(numerator, pd.Series) else numerator
        den = pd.Series(denominator) if not isinstance(denominator, pd.Series) else denominator
        res = num / den.replace(0, np.nan)
        return res.fillna(fill_value).replace([np.inf, -np.inf], fill_value)

    if isinstance(numerator, np.ndarray) or isinstance(denominator, np.ndarray):
        num = np.asarray(numerator, dtype=float)
        den = np.asarray(denominator, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            res = np.where(den != 0, num / den, fill_value)
            res = np.nan_to_num(res, nan=fill_value, posinf=fill_value, neginf=fill_value)
        return res

    # Escalares (float, int)
    if denominator == 0 or np.isnan(denominator):
        return fill_value
    val = numerator / denominator
    return fill_value if (np.isnan(val) or np.isinf(val)) else val


def calculate_possessions(
    fga: Numeric,
    fta: Numeric,
    oreb: Numeric,
    tov: Numeric,
) -> Numeric:
    """
    Calcula las posesiones estimadas según la fórmula canónica de Dean Oliver:
        Poss = 0.5 * (FGA + 0.44 * FTA - OREB + TOV)
    """
    return 0.5 * (fga + 0.44 * fta - oreb + tov)


def calculate_game_possessions(
    home_fga: Numeric,
    home_fta: Numeric,
    home_oreb: Numeric,
    home_tov: Numeric,
    away_fga: Numeric,
    away_fta: Numeric,
    away_oreb: Numeric,
    away_tov: Numeric,
) -> Numeric:
    """
    Calcula el ritmo posesional conjunto de un partido promediando las estimaciones
    de posesiones de ambos contendientes:
        Poss_game = 0.5 * (Poss_home_raw + Poss_away_raw)
    """
    poss_home = fga_to_poss = home_fga + 0.44 * home_fta - home_oreb + home_tov
    poss_away = away_fga + 0.44 * away_fta - away_oreb + away_tov
    return 0.5 * (poss_home + poss_away)


def calculate_offensive_rating(
    pts: Numeric,
    poss: Numeric,
) -> Numeric:
    """
    Calcula el Offensive Rating (puntos anotados por cada 100 posesiones):
        ORtg = 100 * (PTS / Poss)
    """
    return _safe_divide(pts * 100.0, poss, fill_value=0.0)


def calculate_defensive_rating(
    opp_pts: Numeric,
    poss: Numeric,
) -> Numeric:
    """
    Calcula el Defensive Rating (puntos permitidos al oponente por cada 100 posesiones):
        DRtg = 100 * (Opponent_PTS / Poss)
    """
    return _safe_divide(opp_pts * 100.0, poss, fill_value=0.0)


def calculate_net_rating(
    ortg: Numeric,
    drtg: Numeric,
) -> Numeric:
    """
    Calcula el Net Rating (diferencial de eficiencia neta por 100 posesiones):
        NRtg = ORtg - DRtg
    """
    return ortg - drtg


def calculate_pace(
    poss_home: Numeric,
    poss_away: Numeric,
    team_minutes: Numeric = 240.0,
) -> Numeric:
    """
    Calcula el Pace proyectado a 48 minutos reglamentarios:
        Pace = 48 * (Poss_home + Poss_away) / (2 * (team_minutes / 5))
    En regulación (48 min por jugador * 5 = 240 min de equipo), equivale a:
        (Poss_home + Poss_away) / 2
    """
    game_minutes_norm = _safe_divide(team_minutes, 5.0, fill_value=48.0)
    denominator = 2.0 * game_minutes_norm
    return 48.0 * _safe_divide(poss_home + poss_away, denominator, fill_value=0.0)


def calculate_efg_pct(
    fgm: Numeric,
    fg3m: Numeric,
    fga: Numeric,
) -> Numeric:
    """
    Effective Field Goal Percentage (eFG%):
        eFG% = (FGM + 0.5 * 3PM) / FGA
    """
    return _safe_divide(fgm + 0.5 * fg3m, fga, fill_value=0.0)


def calculate_tov_pct(
    tov: Numeric,
    fga: Numeric,
    fta: Numeric,
) -> Numeric:
    """
    Turnover Percentage (TOV%): Porcentaje de posesiones que terminan en pérdida:
        TOV% = TOV / (FGA + 0.44 * FTA + TOV)
    """
    denominator = fga + 0.44 * fta + tov
    return _safe_divide(tov, denominator, fill_value=0.0)


def calculate_orb_pct(
    oreb_team: Numeric,
    dreb_opp: Numeric,
) -> Numeric:
    """
    Offensive Rebounding Percentage (ORB%):
        ORB% = OREB_team / (OREB_team + DREB_opp)
    """
    return _safe_divide(oreb_team, oreb_team + dreb_opp, fill_value=0.0)


def calculate_drb_pct(
    dreb_team: Numeric,
    oreb_opp: Numeric,
) -> Numeric:
    """
    Defensive Rebounding Percentage (DRB%):
        DRB% = DREB_team / (DREB_team + OREB_opp)
    """
    return _safe_divide(dreb_team, dreb_team + oreb_opp, fill_value=0.0)


def calculate_ft_rate(
    ftm: Numeric,
    fga: Numeric,
) -> Numeric:
    """
    Free Throw Rate (FT Rate):
        FT Rate = FTM / FGA
    """
    return _safe_divide(ftm, fga, fill_value=0.0)
