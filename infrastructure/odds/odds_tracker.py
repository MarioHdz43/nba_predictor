# -*- coding: utf-8 -*-
"""
infrastructure/odds/odds_tracker.py
------------------------------------
Gestor y rastreador de series temporales de cuotas de mercado (Odds Tracking).

Monitorea la evolución de cuotas desde la apertura (Opening Line) hasta el
cierre institucional (Closing Line) para auditar la liquidez, el movimiento de línea
y posibilitar el cálculo formal de CLV (Closing Line Value).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import structlog

from domain.analytics.clv import calculate_clv, calculate_spread_clv

logger = structlog.get_logger()


@dataclass(frozen=True)
class OddsSnapshot:
    """Instantánea de cuotas de mercado para un partido en un instante de tiempo."""
    game_id: str
    bookmaker: str
    home_odds: float
    away_odds: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    home_spread: Optional[float] = None
    away_spread: Optional[float] = None
    total_line: Optional[float] = None

    @property
    def market_overround(self) -> float:
        """Suma de probabilidades implícitas del bookmaker."""
        return (1.0 / self.home_odds) + (1.0 / self.away_odds)


@dataclass
class GameMarketHistory:
    """Historial de mercado completo para un partido determinado."""
    game_id: str
    opening_odds: Optional[OddsSnapshot] = None
    closing_odds: Optional[OddsSnapshot] = None
    snapshots: List[OddsSnapshot] = field(default_factory=list)

    def add_snapshot(
        self,
        snapshot: OddsSnapshot,
        is_opening: bool = False,
        is_closing: bool = False,
    ) -> None:
        self.snapshots.append(snapshot)
        if is_opening or self.opening_odds is None:
            self.opening_odds = snapshot
        if is_closing:
            self.closing_odds = snapshot


class MarketOddsTracker:
    """
    Rastreador de cuotas institucionales y cálculo de CLV sobre posiciones ejecutadas.
    """

    def __init__(self, reference_bookmaker: str = "Pinnacle") -> None:
        self.reference_bookmaker = reference_bookmaker
        self._markets: Dict[str, GameMarketHistory] = {}

    def record_snapshot(
        self,
        snapshot: OddsSnapshot,
        is_opening: bool = False,
        is_closing: bool = False,
    ) -> None:
        """Registra un snapshot de cuotas para un partido."""
        if snapshot.game_id not in self._markets:
            self._markets[snapshot.game_id] = GameMarketHistory(game_id=snapshot.game_id)
        
        self._markets[snapshot.game_id].add_snapshot(
            snapshot,
            is_opening=is_opening,
            is_closing=is_closing,
        )

    def get_game_history(self, game_id: str) -> Optional[GameMarketHistory]:
        return self._markets.get(game_id)

    def get_closing_odds(self, game_id: str) -> Optional[OddsSnapshot]:
        history = self._markets.get(game_id)
        if not history:
            return None
        return history.closing_odds or (history.snapshots[-1] if history.snapshots else None)

    def get_opening_odds(self, game_id: str) -> Optional[OddsSnapshot]:
        history = self._markets.get(game_id)
        if not history:
            return None
        return history.opening_odds or (history.snapshots[0] if history.snapshots else None)

    def compute_bet_clv(
        self,
        game_id: str,
        selection: str,  # "Home" o "Away"
        taken_odds: float,
        taken_spread: Optional[float] = None,
    ) -> Dict[str, Optional[float]]:
        """
        Calcula el CLV de una apuesta frente a la línea de cierre registrada.
        """
        closing = self.get_closing_odds(game_id)
        if not closing:
            logger.warning("odds-tracker-sin-cuotas-de-cierre", game_id=game_id)
            return {"clv": None, "spread_clv": None}

        sel_upper = selection.strip().upper()
        if "HOME" in sel_upper:
            close_odds = closing.home_odds
            close_opp = closing.away_odds
            close_spread = closing.home_spread
        else:
            close_odds = closing.away_odds
            close_opp = closing.home_odds
            close_spread = closing.away_spread

        odds_clv = calculate_clv(taken_odds, close_odds, close_opp)
        spread_val: Optional[float] = None
        if taken_spread is not None and close_spread is not None:
            spread_val = calculate_spread_clv(taken_spread, close_spread)

        return {
            "clv": odds_clv,
            "spread_clv": spread_val,
            "closing_odds": close_odds,
            "closing_opp_odds": close_opp,
        }
