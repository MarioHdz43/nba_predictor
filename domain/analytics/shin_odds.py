# -*- coding: utf-8 -*-
"""
domain/analytics/shin_odds.py
------------------------------
Extracción de probabilidades implícitas verdaderas (No-Vig / Fair Odds)
mediante el Método de Shin (Shin, 1992, 1993; Jevons Smith, 2006).

A diferencia de la normalización multiplicativa clásica (que reparte el margen
proporcionalmente), el Método de Shin postula la presencia de una proporción latente
'z' de apostadores con información privilegiada (insiders).

El bookmaker expande su margen asimétricamente para protegerse de los insiders,
creando el sesgo favorito-underdog (Favorite-Longshot Bias). Shin resuelve
simultáneamente la proporción z y las probabilidades reales libres de margen.

Formulación canónica:
    Dado un vector de cuotas decimales O con inversas pi_i = 1 / O_i
    y overround beta = sum(pi_i):

    p_i(z) = ( sqrt( z^2 + 4 * (1 - z) * (pi_i^2) / beta ) - z ) / ( 2 * (1 - z) )

    Se determina z tal que:
    sum_{i=1}^n p_i(z) = 1.0
"""

from __future__ import annotations
from typing import List, Sequence, Tuple, Union
import numpy as np
from scipy.optimize import root_scalar
import structlog

logger = structlog.get_logger()


def calculate_market_vigorish(odds: Sequence[float]) -> float:
    """
    Calcula el margen total (overround / juice) aplicado por el bookmaker.
    Vigorish = sum(1 / odds_i) - 1.0
    """
    inv_odds = [1.0 / float(o) for o in odds if float(o) > 0.0]
    booksum = sum(inv_odds)
    return max(0.0, booksum - 1.0)


def calculate_shin_probabilities(
    odds: Union[Sequence[float], np.ndarray],
    max_iter: int = 100,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, float, float]:
    """
    Calcula las probabilidades reales sin vigorish usando el Método de Shin.

    Args:
        odds: Lista o arreglo de cuotas decimales (ej. [1.91, 1.91] o [2.6, 2.4, 4.3]).
        max_iter: Número máximo de iteraciones para el solver numérico.
        tol: Tolerancia de convergencia en la raíz.

    Returns:
        Tuple conteniendo:
            - true_probabilities (np.ndarray): Probabilidades no-vig sumando exactamente 1.0.
            - z (float): Proporción estimada de apostadores informados (insiders).
            - vigorish (float): Margen total del mercado (booksum - 1.0).
    """
    odds_arr = np.asarray(odds, dtype=float)
    if np.any(odds_arr <= 1.0):
        raise ValueError(f"Las cuotas decimales deben ser estrictamente superiores a 1.0: {odds_arr}")

    inv_odds = 1.0 / odds_arr
    beta = float(np.sum(inv_odds))
    vigorish = max(0.0, beta - 1.0)

    # Si no hay vigorish (mercado justo o con arbitraje), no hay insider bias
    if beta <= 1.000001:
        p_fair = inv_odds / beta
        return p_fair, 0.0, vigorish

    # Definir la función de balance de Shin
    def shin_objective(z_val: float) -> float:
        if z_val >= 1.0:
            return 1.0
        one_minus_z = 1.0 - z_val
        terms = np.sqrt(z_val**2 + 4.0 * one_minus_z * (inv_odds**2) / beta) - z_val
        probs = terms / (2.0 * one_minus_z)
        return float(np.sum(probs) - 1.0)

    # Resolver numéricamente z en el intervalo [0.0, 0.499]
    try:
        # Evaluar extremos
        f_0 = shin_objective(0.0)
        # Si en z=0 ya la suma es <= 1, el margen es insignificante
        if f_0 <= 0.0:
            z_star = 0.0
        else:
            upper_bound = min(0.45, max(0.05, vigorish * 1.5))
            f_upper = shin_objective(upper_bound)

            # Expandir el intervalo si no cambia de signo
            attempts = 0
            while f_upper > 0.0 and upper_bound < 0.49 and attempts < 5:
                upper_bound = min(0.49, upper_bound + 0.1)
                f_upper = shin_objective(upper_bound)
                attempts += 1

            if f_0 * f_upper < 0.0:
                sol = root_scalar(
                    shin_objective,
                    bracket=[0.0, upper_bound],
                    method="brentq",
                    maxiter=max_iter,
                    rtol=tol,
                )
                z_star = float(sol.root)
            else:
                # Fallback analítico aproximado si no se encuentra raíz estricta
                z_star = max(0.0, min(0.3, vigorish * 0.25))
    except Exception as e:
        logger.warning("shin-solver-fallback-proporcional", error=str(e))
        z_star = 0.0

    # Calcular vector final de probabilidades con z*
    one_minus_z = 1.0 - z_star
    if one_minus_z > 1e-6:
        terms = np.sqrt(z_star**2 + 4.0 * one_minus_z * (inv_odds**2) / beta) - z_star
        p_true = terms / (2.0 * one_minus_z)
    else:
        p_true = inv_odds / beta

    # Normalizar estrictamente para garantizar suma exactamente igual a 1.0
    p_true = p_true / np.sum(p_true)
    return p_true, float(z_star), round(vigorish, 4)


def shin_two_way_probabilities(
    home_odds: float,
    away_odds: float,
) -> Tuple[float, float, float, float]:
    """
    Extracción especializada para mercados bidireccionales de la NBA (Home / Away).

    Returns:
        (home_no_vig_prob, away_no_vig_prob, z, market_vigorish)
    """
    probs, z, vig = calculate_shin_probabilities([home_odds, away_odds])
    return float(probs[0]), float(probs[1]), z, vig
