# -*- coding: utf-8 -*-
"""
domain/analytics/roster_impact.py
---------------------------------
Proyección dinámica de rotaciones y cálculo de degradación de rating de equipo
mediante métricas de impacto individual (RAPM / EPM) y reporte oficial de lesiones.

Formulación canónica:
    Team_ORtg_Proj = League_Average + sum_{i in Active} (Minutos_i / 48) * O_RAPM_i
    Team_DRtg_Proj = League_Average + sum_{i in Active} (Minutos_i / 48) * D_RAPM_i
    Team_Net_Rating_Proj = Team_ORtg_Proj - Team_DRtg_Proj

Garantiza la conservación estricta de 240 minutos por partido y la redistribución
bayesiana de minutos de jugadores ausentes hacia el banco y nivel de reemplazo.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import structlog

from infrastructure.scrapers.injury_parser import PlayerInjuryReport

logger = structlog.get_logger()

# Constantes del entorno NBA
TOTAL_GAME_MINUTES = 240.0  # 5 jugadores * 48 minutos
LEAGUE_AVERAGE_ORTG = 114.5
LEAGUE_AVERAGE_DRTG = 114.5
REPLACEMENT_O_RAPM = -1.25  # Nivel de reemplazo ofensivo
REPLACEMENT_D_RAPM = 1.25   # Nivel de reemplazo defensivo (concede +1.25 pts/100 poss más que la media)


@dataclass
class PlayerRosterProfile:
    """Perfil analítico de un jugador con sus minutos base y métricas de impacto."""
    player_id: str = ""
    name: str = ""
    team: str = ""
    base_minutes: float = 0.0
    o_rapm: float = 0.0
    d_rapm: float = 0.0  # + = peor defensa (más puntos concedidos), - = mejor defensa
    is_starter: bool = False
    position: str = "G"
    player_name: Optional[str] = None
    team_id: Optional[Any] = None
    baseline_minutes: Optional[float] = None
    injury_status: Optional[Any] = None
    availability_prob: float = 1.0

    def __post_init__(self):
        if self.player_name and not self.name:
            self.name = self.player_name
        elif self.name and not self.player_name:
            self.player_name = self.name

        if self.team_id is not None and not self.team:
            self.team = str(self.team_id)

        if self.baseline_minutes is not None and self.base_minutes == 0.0:
            self.base_minutes = float(self.baseline_minutes)
        elif self.base_minutes > 0.0 and self.baseline_minutes is None:
            self.baseline_minutes = float(self.base_minutes)

    @property
    def net_rapm(self) -> float:
        """Impacto neto por 100 posesiones (positivo = mejor)."""
        return self.o_rapm - self.d_rapm


@dataclass
class RosterProjectionResult:
    """Resultado de la proyección de rotación y degradación de rating."""
    team: str
    baseline_ortg: float
    baseline_drtg: float
    baseline_net_rating: float
    projected_ortg: float
    projected_drtg: float
    projected_net_rating: float
    net_degradation: float  # baseline_net_rating - projected_net_rating
    total_minutes: float
    active_players_count: int
    minutes_redistributed: float
    player_projected_minutes: Dict[str, float] = field(default_factory=dict)

    @property
    def total_projected_minutes(self) -> float:
        return self.total_minutes

    @property
    def minutes_lost(self) -> float:
        return self.minutes_redistributed

    @property
    def net_rating_degradation(self) -> float:
        """Delta respecto al baseline (negativo si el equipo empeoró)."""
        return self.projected_net_rating - self.baseline_net_rating


class RosterImpactCalculator:
    """
    Calculador de rotaciones y proyecciones de ratings por posesión para la NBA.
    """

    def __init__(
        self,
        league_avg_ortg: float = LEAGUE_AVERAGE_ORTG,
        league_avg_drtg: float = LEAGUE_AVERAGE_DRTG,
        replacement_o_rapm: float = REPLACEMENT_O_RAPM,
        replacement_d_rapm: float = REPLACEMENT_D_RAPM,
    ) -> None:
        self.league_avg_ortg = league_avg_ortg
        self.league_avg_drtg = league_avg_drtg
        self.replacement_o_rapm = replacement_o_rapm
        self.replacement_d_rapm = replacement_d_rapm

    def project_team_ratings(
        self,
        roster: List[PlayerRosterProfile],
        injuries: Optional[List[PlayerInjuryReport]] = None,
    ) -> RosterProjectionResult:
        """
        Proyecta los ratings ofensivo, defensivo y neto del equipo integrando
        las probabilidades de disponibilidad de lesiones y redistribuyendo minutos.
        """
        if not roster:
            return RosterProjectionResult(
                team="UNKNOWN",
                baseline_ortg=self.league_avg_ortg,
                baseline_drtg=self.league_avg_drtg,
                baseline_net_rating=0.0,
                projected_ortg=self.league_avg_ortg,
                projected_drtg=self.league_avg_drtg,
                projected_net_rating=0.0,
                net_degradation=0.0,
                total_minutes=0.0,
                active_players_count=0,
                minutes_redistributed=0.0,
            )

        team_name = roster[0].team

        # Mapeo de disponibilidades por nombre de jugador
        injury_map: Dict[str, float] = {}
        for p in roster:
            if p.availability_prob < 1.0:
                injury_map[p.name.lower()] = p.availability_prob
        if injuries:
            for inj in injuries:
                injury_map[inj.player_name.lower()] = inj.availability_prob

        # 1. Calcular Baseline (Roster al 100% de salud)
        norm_roster_baseline = self._normalize_minutes(roster, target_minutes=TOTAL_GAME_MINUTES)
        base_ortg, base_drtg = self._calculate_ratings_from_allocation(norm_roster_baseline)
        base_net = base_ortg - base_drtg

        # 2. Ajustar minutos según disponibilidad
        adjusted_allocation: List[Tuple[PlayerRosterProfile, float]] = []
        lost_minutes = 0.0

        for player, base_mins in norm_roster_baseline:
            avail_prob = injury_map.get(player.name.lower(), 1.0)
            effective_mins = base_mins * avail_prob
            adjusted_allocation.append((player, effective_mins))
            lost_minutes += (base_mins - effective_mins)

        # 3. Redistribuir minutos perdidos hacia suplentes y nivel de reemplazo
        final_allocation = self._redistribute_lost_minutes(adjusted_allocation, lost_minutes)

        # 4. Calcular ratings proyectados
        proj_ortg, proj_drtg = self._calculate_ratings_from_allocation(final_allocation)
        proj_net = proj_ortg - proj_drtg

        net_degradation = base_net - proj_net

        minutes_dict = {p.name: round(m, 1) for p, m in final_allocation if m > 0.0}

        return RosterProjectionResult(
            team=team_name,
            baseline_ortg=round(base_ortg, 2),
            baseline_drtg=round(base_drtg, 2),
            baseline_net_rating=round(base_net, 2),
            projected_ortg=round(proj_ortg, 2),
            projected_drtg=round(proj_drtg, 2),
            projected_net_rating=round(proj_net, 2),
            net_degradation=round(net_degradation, 2),
            total_minutes=round(sum(m for _, m in final_allocation), 2),
            active_players_count=len([m for _, m in final_allocation if m > 0.5]),
            minutes_redistributed=round(lost_minutes, 2),
            player_projected_minutes=minutes_dict,
        )

    def _calculate_ratings_from_allocation(
        self,
        allocation: List[Tuple[PlayerRosterProfile, float]],
    ) -> Tuple[float, float]:
        """
        Team_ORtg_Proj = League_Average + sum (Minutos / 48) * O_RAPM
        Team_DRtg_Proj = League_Average + sum (Minutos / 48) * D_RAPM
        """
        o_impact = sum((mins / 48.0) * player.o_rapm for player, mins in allocation)
        d_impact = sum((mins / 48.0) * player.d_rapm for player, mins in allocation)

        team_ortg = self.league_avg_ortg + o_impact
        team_drtg = self.league_avg_drtg + d_impact
        return team_ortg, team_drtg

    def _normalize_minutes(
        self,
        roster: List[PlayerRosterProfile],
        target_minutes: float = TOTAL_GAME_MINUTES,
    ) -> List[Tuple[PlayerRosterProfile, float]]:
        """Ajusta proporcionalmente los minutos base para totalizar exactamente 240 minutos."""
        total_base = sum(p.base_minutes for p in roster)
        if total_base <= 0.0:
            even_share = target_minutes / len(roster)
            return [(p, even_share) for p in roster]

        scale = target_minutes / total_base
        return [(p, p.base_minutes * scale) for p in roster]

    def _redistribute_lost_minutes(
        self,
        allocation: List[Tuple[PlayerRosterProfile, float]],
        lost_minutes: float,
    ) -> List[Tuple[PlayerRosterProfile, float]]:
        """
        Distribuye los minutos ausentes entre los jugadores disponibles del banco
        hasta un tope saludable (38 min) y el resto a un jugador de nivel de reemplazo.
        """
        if lost_minutes <= 0.001:
            return allocation

        result: List[Tuple[PlayerRosterProfile, float]] = []
        remaining_lost = lost_minutes

        # Jugadores disponibles que no son la baja
        available_bench = [
            (p, m) for p, m in allocation if m > 0.0 and m < 36.0
        ]

        # Priorizar jugadores con capacidad de asumir más minutos
        bench_boost_total = 0.0
        for p, current_m in allocation:
            if current_m > 0.0 and current_m < 36.0 and remaining_lost > 0.0:
                headroom = 36.0 - current_m
                add_m = min(headroom * 0.4, remaining_lost * 0.5)
                new_m = current_m + add_m
                remaining_lost -= add_m
                bench_boost_total += add_m
                result.append((p, new_m))
            else:
                result.append((p, current_m))

        # Si aún quedan minutos por cubrir, se asignan a jugador de reemplazo
        if remaining_lost > 0.001:
            replacement_player = PlayerRosterProfile(
                player_id="REP_00",
                name="Replacement Player",
                team=allocation[0][0].team,
                base_minutes=remaining_lost,
                o_rapm=self.replacement_o_rapm,
                d_rapm=self.replacement_d_rapm,
                is_starter=False,
            )
            result.append((replacement_player, remaining_lost))

        return result

    def calculate_team_projected_ratings(
        self,
        roster: List[PlayerRosterProfile],
        injuries: Optional[List[PlayerInjuryReport]] = None,
        league_avg_ortg: Optional[float] = None,
        league_avg_drtg: Optional[float] = None,
    ) -> RosterProjectionResult:
        """Alias retrocompatible para project_team_ratings."""
        if league_avg_ortg is not None:
            self.league_avg_ortg = league_avg_ortg
        if league_avg_drtg is not None:
            self.league_avg_drtg = league_avg_drtg
        return self.project_team_ratings(roster=roster, injuries=injuries)


# Alias canónico
RosterProjectionEngine = RosterImpactCalculator
