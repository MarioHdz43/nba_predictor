# -*- coding: utf-8 -*-
"""
domain/analytics/beta_calibration.py
------------------------------------
Calibrador probabilístico paramétrico mediante Beta Calibration (Kull et al., 2017).

Formulación matemática:
    ln( q / (1 - q) ) = a * ln(p) - b * ln(1 - p) + c

Donde:
    - p es la probabilidad bruta no calibrada en [0, 1].
    - q es la probabilidad calibrada resultante.
    - a >= 0 y b >= 0 garantizan la estricta monotonicidad de la calibración.
    - c es el término de sesgo (intercepto) en la escala logit.

La estimación de parámetros se realiza minimizando la pérdida logarítmica
(Binary Cross-Entropy / Log-Loss) sobre las observaciones reales de validación.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize
import structlog

logger = structlog.get_logger()

EPSILON = 1e-7


class BetaCalibrator:
    """
    Calibrador Beta paramétrico para probabilidades de victoria en el intervalo [0, 1].
    Preserva monotonicidad y corrige tanto sobreconfianza como sesgo sistémico.
    """

    def __init__(
        self,
        a: float = 1.0,
        b: float = 1.0,
        c: float = 0.0,
    ) -> None:
        self.a: float = float(a)
        self.b: float = float(b)
        self.c: float = float(c)
        self.is_fitted: bool = False

    def fit(
        self,
        y_prob: Union[List[float], np.ndarray],
        y_true: Union[List[int], np.ndarray],
    ) -> "BetaCalibrator":
        """
        Ajusta los parámetros (a, b, c) minimizando la binary cross-entropy.
        """
        p = np.clip(np.asarray(y_prob, dtype=float), EPSILON, 1.0 - EPSILON)
        y = np.asarray(y_true, dtype=float)

        if len(p) != len(y):
            raise ValueError(f"Longitud inconsistente: len(y_prob)={len(p)} vs len(y_true)={len(y)}")

        if len(p) < 10:
            logger.warning("beta-calibrator-muestras-insuficientes", count=len(p))
            self.is_fitted = True
            return self

        log_p = np.log(p)
        log_1_p = np.log(1.0 - p)

        def loss_function(params: np.ndarray) -> float:
            param_a, param_b, param_c = params
            logit = param_a * log_p - param_b * log_1_p + param_c
            # Proteger contra overflow numérico en exp(-logit)
            logit_clipped = np.clip(logit, -35.0, 35.0)
            q = 1.0 / (1.0 + np.exp(-logit_clipped))
            q = np.clip(q, EPSILON, 1.0 - EPSILON)
            loss = -np.mean(y * np.log(q) + (1.0 - y) * np.log(1.0 - q))
            return float(loss)

        # Restricciones: a >= 0, b >= 0 para asegurar ordenación monotónica
        initial_params = [1.0, 1.0, 0.0]
        bounds = [(0.0, 15.0), (0.0, 15.0), (-15.0, 15.0)]

        res = minimize(
            loss_function,
            x0=initial_params,
            bounds=bounds,
            method="L-BFGS-B",
        )

        if res.success or res.fun < loss_function(np.array(initial_params)):
            self.a = float(res.x[0])
            self.b = float(res.x[1])
            self.c = float(res.x[2])
            self.is_fitted = True
            logger.info("beta-calibrator-ajustado", a=self.a, b=self.b, c=self.c, loss=res.fun)
        else:
            logger.warning("beta-calibrator-fallo-optimizacion", message=res.message)
            self.a = 1.0
            self.b = 1.0
            self.c = 0.0
            self.is_fitted = True

        return self

    def predict_proba(
        self,
        y_prob: Union[float, List[float], np.ndarray],
    ) -> Union[float, np.ndarray]:
        """
        Transforma probabilidades brutas a probabilidades calibradas.
        """
        is_scalar = np.isscalar(y_prob)
        p_arr = np.clip(np.asarray(y_prob, dtype=float), EPSILON, 1.0 - EPSILON)

        if not self.is_fitted:
            return float(p_arr) if is_scalar else p_arr

        logit = self.a * np.log(p_arr) - self.b * np.log(1.0 - p_arr) + self.c
        logit_clipped = np.clip(logit, -35.0, 35.0)
        q = 1.0 / (1.0 + np.exp(-logit_clipped))
        q_clipped = np.clip(q, 0.01, 0.99)

        if is_scalar:
            return float(q_clipped.item())
        return q_clipped

    def to_dict(self) -> Dict[str, Any]:
        """Serializa los parámetros del calibrador."""
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BetaCalibrator":
        """Instancia un calibrador desde un diccionario de parámetros."""
        calibrator = cls(
            a=data.get("a", 1.0),
            b=data.get("b", 1.0),
            c=data.get("c", 0.0),
        )
        calibrator.is_fitted = data.get("is_fitted", True)
        return calibrator
