# -*- coding: utf-8 -*-
"""
domain/analytics/clv.py
-----------------------
Módulo de Closing Line Value (CLV) y observabilidad de eficiencia de mercado.

En las apuestas deportivas cuantitativas, batir la línea de cierre institucional
(Pinnacle / exchanges) de manera consistente es la prueba estadística más robusta
de poseer una ventaja matemática real (Edge) sobre el mercado.

Formulación matemática:
    Dada una cuota tomada O_tomada y una cuota de cierre con su contraparte
    (O_cierre, O_cierre_oponente):
    
    1. Se calcula la probabilidad verdadera de cierre mediante el método de Shin:
       p_cierre_no_vig = shin(O_cierre, O_cierre_oponente)
    
    2. El CLV relativo a la línea justa (No-Vig CLV) es:
       CLV = (O_tomada * p_cierre_no_vig) - 1.0

    3. Si no se dispone de la cuota del oponente para desviggear:
       CLV_crudo = (O_tomada / O_cierre) - 1.0

    4. En apuestas de margen de puntos (Point Spread):
       Spread_CLV = taken_spread - closing_spread
       (Positivo indica que se aseguró un colchón de puntos superior al cierre).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import structlog

from domain.analytics.shin_odds import shin_two_way_probabilities

logger = structlog.get_logger()


def calculate_clv(
    taken_odds: float,
    closing_odds: float,
    closing_opp_odds: Optional[float] = None,
) -> float:
    """
    Calcula el Closing Line Value (CLV) respecto al mercado de cierre.

    Args:
        taken_odds: Cuota decimal obtenida al momento de colocar la apuesta.
        closing_odds: Cuota decimal de cierre para esa misma selección.
        closing_opp_odds: Cuota decimal de cierre del desenlace contrario (para desviggear vía Shin).

    Returns:
        float: Retorno porcentual esperado frente al cierre (ej. +0.035 = +3.5% CLV).
    """
    if taken_odds <= 1.0 or closing_odds <= 1.0:
        return 0.0

    if closing_opp_odds is not None and closing_opp_odds > 1.0:
        # Extraer probabilidad verdadera sin vig de la línea eficiente de cierre
        p_close_no_vig, _, _, _ = shin_two_way_probabilities(closing_odds, closing_opp_odds)
        clv = (taken_odds * p_close_no_vig) - 1.0
    else:
        # Comparación directa de cuotas
        clv = (taken_odds / closing_odds) - 1.0

    return round(float(clv), 4)


def calculate_spread_clv(
    taken_spread: float,
    closing_spread: float,
) -> float:
    """
    Calcula la ventaja de handicap obtenida frente a la línea de cierre.
    
    Ejemplos:
        - Tomar +5.5 cuando cerró en +3.5: +2.0 puntos de CLV (más colchón).
        - Tomar -3.5 cuando cerró en -5.5: +2.0 puntos de CLV (menos exigencia).
    """
    return round(float(taken_spread - closing_spread), 2)


@dataclass
class CLVBetRecord:
    """Registro individual de una apuesta auditada contra la línea de cierre."""
    bet_id: str
    game_id: str
    selection: str  # "Home" o "Away"
    taken_odds: float
    closing_odds: float
    closing_opp_odds: Optional[float] = None
    taken_spread: Optional[float] = None
    closing_spread: Optional[float] = None
    won: Optional[bool] = None
    clv: float = field(init=False)
    spread_clv: Optional[float] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.clv = calculate_clv(self.taken_odds, self.closing_odds, self.closing_opp_odds)
        if self.taken_spread is not None and self.closing_spread is not None:
            self.spread_clv = calculate_spread_clv(self.taken_spread, self.closing_spread)

    @property
    def beat_closing_line(self) -> bool:
        """Determina si la apuesta superó la línea eficiente de cierre."""
        return self.clv > 0.0


@dataclass
class CLVReport:
    """Reporte agregado del rendimiento frente a la línea de cierre institucional."""
    total_bets: int
    beat_closing_count: int
    beat_closing_rate: float
    average_clv: float
    positive_clv_profit_margin: float


class CLVPerformanceTracker:
    """Rastreador de desempeño de mercado mediante CLV acumulado."""

    def __init__(self) -> None:
        self.records: List[CLVBetRecord] = []

    def add_record(self, record: CLVBetRecord) -> None:
        self.records.append(record)

    def generate_report(self) -> CLVReport:
        if not self.records:
            return CLVReport(
                total_bets=0,
                beat_closing_count=0,
                beat_closing_rate=0.0,
                average_clv=0.0,
                positive_clv_profit_margin=0.0,
            )

        total = len(self.records)
        beat_count = sum(1 for r in self.records if r.beat_closing_line)
        beat_rate = beat_count / total
        avg_clv = float(np.mean([r.clv for r in self.records]))
        pos_clvs = [r.clv for r in self.records if r.beat_closing_line]
        avg_pos_clv = float(np.mean(pos_clvs)) if pos_clvs else 0.0

        return CLVReport(
            total_bets=total,
            beat_closing_count=beat_count,
            beat_closing_rate=round(beat_rate, 4),
            average_clv=round(avg_clv, 4),
            positive_clv_profit_margin=round(avg_pos_clv, 4),
        )
