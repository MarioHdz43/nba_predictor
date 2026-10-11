# -*- coding: utf-8 -*-
"""
infrastructure/scrapers/injury_parser.py
----------------------------------------
Parser estructurado para el Official NBA Injury Report.
Convierte las designaciones cualitativas oficiales en probabilidades
cuantitativas bayesianas a priori de disponibilidad:
    - OUT: 0.0 (0%)
    - DOUBTFUL: 0.10 (10%)
    - QUESTIONABLE: 0.50 (50%)
    - PROBABLE: 0.90 (90%)
    - AVAILABLE: 1.0 (100%)
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Tuple
import structlog

logger = structlog.get_logger()


class InjuryStatus(str, Enum):
    OUT = "OUT"
    DOUBTFUL = "DOUBTFUL"
    QUESTIONABLE = "QUESTIONABLE"
    PROBABLE = "PROBABLE"
    AVAILABLE = "AVAILABLE"


# Probabilidades cuantitativas a priori de jugar según designación oficial
INJURY_PROBABILITIES: Dict[InjuryStatus, float] = {
    InjuryStatus.OUT: 0.0,
    InjuryStatus.DOUBTFUL: 0.10,
    InjuryStatus.QUESTIONABLE: 0.50,
    InjuryStatus.PROBABLE: 0.90,
    InjuryStatus.AVAILABLE: 1.0,
}


@dataclass(frozen=True)
class PlayerInjuryReport:
    """Representación estructurada del estado de lesión de un jugador."""
    player_name: str
    team: str
    status: InjuryStatus
    availability_prob: float
    reason: str = ""
    game_date: Optional[str] = None
    notes: Optional[str] = None

    @property
    def team_name(self) -> str:
        """Alias para el nombre del equipo."""
        return self.team

    @property
    def expected_availability(self) -> float:
        """Probabilidad esperada de jugar (availability_prob)."""
        return self.availability_prob

    @property
    def expected_absence_rate(self) -> float:
        """Probabilidad esperada de no participar en el partido (1 - P)."""
        return 1.0 - self.availability_prob


def parse_injury_status(status_str: str) -> InjuryStatus:
    """Normaliza una cadena de texto y retorna el enum InjuryStatus correspondiente."""
    status, _ = normalize_status(status_str)
    return status


def get_availability_probability(status: InjuryStatus | str) -> float:
    """Retorna la probabilidad a priori cuantitativa de disponibilidad."""
    if isinstance(status, str):
        status, prob = normalize_status(status)
        return prob
    return INJURY_PROBABILITIES.get(status, 0.50)


def normalize_status(status_str: str) -> Tuple[InjuryStatus, float]:
    """
    Normaliza una cadena de texto de estatus a la designación oficial
    y retorna la tupla (InjuryStatus, availability_prob).
    """
    cleaned = status_str.strip().upper()

    if "OUT" in cleaned or "INJURY" in cleaned or "G LEAGUE" in cleaned or "SURGERY" in cleaned:
        status = InjuryStatus.OUT
    elif "DOUBTFUL" in cleaned:
        status = InjuryStatus.DOUBTFUL
    elif "QUESTIONABLE" in cleaned:
        status = InjuryStatus.QUESTIONABLE
    elif "PROBABLE" in cleaned:
        status = InjuryStatus.PROBABLE
    elif "AVAILABLE" in cleaned or "ACTIVE" in cleaned:
        status = InjuryStatus.AVAILABLE
    else:
        # Por defecto, si no se especifica explícitamente pero figura en la lista de lesiones
        logger.debug("injury-status-desconocido", raw=status_str)
        status = InjuryStatus.QUESTIONABLE

    return status, INJURY_PROBABILITIES[status]


def parse_injury_records(records: List[Dict[str, Any]]) -> List[PlayerInjuryReport]:
    """
    Parsea una lista de diccionarios (JSON o respuesta de feed) al formato tipado.
    Espera llaves flexibles como 'player', 'name', 'team', 'status', 'reason'.
    """
    parsed: List[PlayerInjuryReport] = []

    for r in records:
        name = r.get("player") or r.get("player_name") or r.get("Player") or ""
        team = r.get("team") or r.get("Team") or ""
        raw_status = r.get("status") or r.get("Current Status") or r.get("Status") or "Questionable"
        reason = r.get("reason") or r.get("Reason") or r.get("injury") or ""
        date = r.get("game_date") or r.get("Game Date") or None

        if not name:
            continue

        status, prob = normalize_status(str(raw_status))
        parsed.append(
            PlayerInjuryReport(
                player_name=name.strip(),
                team=team.strip(),
                status=status,
                availability_prob=prob,
                reason=reason.strip(),
                game_date=date,
            )
        )

    logger.info("injury-records-parseados", count=len(parsed))
    return parsed


def parse_injury_text(raw_text: str) -> List[PlayerInjuryReport]:
    """
    Parsea texto tabular, exportado o formateado del Official NBA Injury Report.
    Formatos soportados:
        - "Team Name: Player1 (Status - Reason); Player2 (Status - Reason)"
        - "Team | Player | Status | Reason"
        - "Date\tTeam\tMatchup\tPlayer\tStatus\tReason"
        - "LeBron James (LAL) - Out (Ankle)"
    """
    lines = [line.strip() for line in raw_text.strip().splitlines() if line.strip()]
    reports: List[PlayerInjuryReport] = []

    for line in lines:
        # Omitir cabeceras comunes
        if any(h in line.lower() for h in ["game date", "page ", "injury report", "not yet submitted"]):
            continue

        # Formato de reporte agrupado por equipo: "Los Angeles Lakers: LeBron James (Questionable - ...); Anthony Davis (Probable - ...)"
        if ":" in line and ("(" in line and ")" in line):
            team_header, _, players_str = line.partition(":")
            team_name = team_header.strip()
            player_chunks = [c.strip() for c in players_str.split(";") if c.strip()]
            for chunk in player_chunks:
                chunk_match = re.match(r"^(.*?)\s*\((.*?)\)$", chunk)
                if chunk_match:
                    p_name, status_reason = chunk_match.groups()
                    if "-" in status_reason:
                        st_part, r_part = status_reason.split("-", 1)
                    else:
                        st_part, r_part = status_reason, ""
                    status, prob = normalize_status(st_part)
                    reports.append(
                        PlayerInjuryReport(
                            player_name=p_name.strip(),
                            team=team_name,
                            status=status,
                            availability_prob=prob,
                            reason=r_part.strip(),
                        )
                    )
            if player_chunks:
                continue

        parts = [p.strip() for p in re.split(r"\t|\|", line) if p.strip()]

        if len(parts) >= 4:
            # Formato tabular con al menos 4 campos
            status_idx = -1
            for idx, p in enumerate(parts):
                up = p.upper()
                if any(s in up for s in ["OUT", "QUESTIONABLE", "DOUBTFUL", "PROBABLE", "AVAILABLE"]):
                    status_idx = idx
                    break

            if status_idx != -1:
                raw_status = parts[status_idx]
                status, prob = normalize_status(raw_status)
                player_name = parts[status_idx - 1] if status_idx > 0 else parts[0]
                team = parts[0] if status_idx > 1 else ""
                reason = parts[status_idx + 1] if status_idx + 1 < len(parts) else ""

                reports.append(
                    PlayerInjuryReport(
                        player_name=player_name,
                        team=team,
                        status=status,
                        availability_prob=prob,
                        reason=reason,
                    )
                )
        else:
            # Línea con formato libre: "LeBron James (LAL) - Out (Ankle)"
            match = re.search(r"([A-Za-z\s\.\'\-]+?)(?:\s*\(([A-Z]{3})\))?\s*[-:]\s*(Out|Questionable|Doubtful|Probable|Available)(?:\s*[\-\(](.*?)[\)]?)?$", line, re.IGNORECASE)
            if match:
                name, team, raw_status, reason = match.groups()
                status, prob = normalize_status(raw_status)
                reports.append(
                    PlayerInjuryReport(
                        player_name=name.strip(),
                        team=team or "",
                        status=status,
                        availability_prob=prob,
                        reason=(reason or "").strip(),
                    )
                )

    logger.info("injury-text-parseado", lines_processed=len(lines), reports_found=len(reports))
    return reports


# Alias para retrocompatibilidad
parse_injury_text_report = parse_injury_text
