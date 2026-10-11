# -*- coding: utf-8 -*-
"""
domain/analytics/travel_context.py
----------------------------------
Variables contextuales geográficas, fatiga de viaje y efecto altitud para la NBA.

Modelado:
- Coordenadas geodésicas y husos horarios (UTC offset) para las 30 franquicias.
- Distancia ortodrómica en millas mediante la fórmula de Haversine.
- Conteo de husos horarios cruzados.
- Efecto altitud para partidos en Denver (Ball Arena) y Salt Lake City (Delta Center).
- Índice de fatiga circadiana y de viaje.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple, Union

EARTH_RADIUS_MILES = 3958.8


@dataclass(frozen=True)
class TeamLocation:
    """Información geográfica y de huso horario de una franquicia de la NBA."""
    abbreviation: str
    city: str
    arena_name: str
    latitude: float
    longitude: float
    utc_offset: int  # UTC offset en horas (ej: ET = -5, CT = -6, MT = -7, PT = -8)
    elevation_feet: float
    team_id: Optional[int] = None

    @property
    def is_high_altitude(self) -> bool:
        """Indica si el estadio está ubicado a más de 4,000 pies de altura sobre el nivel del mar."""
        return self.elevation_feet >= 4000.0


# Registro canónico de las 30 franquicias de la NBA
NBA_LOCATIONS: Dict[str, TeamLocation] = {
    "ATL": TeamLocation("ATL", "Atlanta", "State Farm Arena", 33.7573, -84.3963, -5, 1050, 1610612737),
    "BOS": TeamLocation("BOS", "Boston", "TD Garden", 42.3662, -71.0621, -5, 14, 1610612738),
    "BKN": TeamLocation("BKN", "Brooklyn", "Barclays Center", 40.6826, -73.9754, -5, 50, 1610612751),
    "CHA": TeamLocation("CHA", "Charlotte", "Spectrum Center", 35.2251, -80.8392, -5, 751, 1610612766),
    "CHI": TeamLocation("CHI", "Chicago", "United Center", 41.8807, -87.6742, -6, 597, 1610612741),
    "CLE": TeamLocation("CLE", "Cleveland", "Rocket Mortgage FieldHouse", 41.4965, -81.6882, -5, 653, 1610612739),
    "DAL": TeamLocation("DAL", "Dallas", "American Airlines Center", 32.7905, -96.8103, -6, 430, 1610612742),
    "DEN": TeamLocation("DEN", "Denver", "Ball Arena", 39.7487, -105.0076, -7, 5280, 1610612743),
    "DET": TeamLocation("DET", "Detroit", "Little Caesars Arena", 42.3411, -83.0553, -5, 600, 1610612765),
    "GSW": TeamLocation("GSW", "San Francisco", "Chase Center", 37.7680, -122.3877, -8, 10, 1610612744),
    "HOU": TeamLocation("HOU", "Houston", "Toyota Center", 29.7508, -95.3621, -6, 49, 1610612745),
    "IND": TeamLocation("IND", "Indianapolis", "Gainbridge Fieldhouse", 39.7640, -86.1555, -5, 715, 1610612754),
    "LAC": TeamLocation("LAC", "Los Angeles", "Intuit Dome", 33.9583, -118.3417, -8, 100, 1610612746),
    "LAL": TeamLocation("LAL", "Los Angeles", "Crypto.com Arena", 34.0430, -118.2673, -8, 233, 1610612747),
    "MEM": TeamLocation("MEM", "Memphis", "FedExForum", 35.1381, -90.0506, -6, 337, 1610612763),
    "MIA": TeamLocation("MIA", "Miami", "Kaseya Center", 25.7814, -80.1870, -5, 6, 1610612748),
    "MIL": TeamLocation("MIL", "Milwaukee", "Fiserv Forum", 43.0451, -87.9172, -6, 617, 1610612749),
    "MIN": TeamLocation("MIN", "Minneapolis", "Target Center", 44.9795, -93.2761, -6, 830, 1610612750),
    "NOP": TeamLocation("NOP", "New Orleans", "Smoothie King Center", 29.9490, -90.0821, -6, 3, 1610612740),
    "NYK": TeamLocation("NYK", "New York", "Madison Square Garden", 40.7505, -73.9934, -5, 33, 1610612752),
    "OKC": TeamLocation("OKC", "Oklahoma City", "Paycom Center", 35.4634, -97.5151, -6, 1200, 1610612760),
    "ORL": TeamLocation("ORL", "Orlando", "Kia Center", 28.5392, -81.3839, -5, 82, 1610612753),
    "PHI": TeamLocation("PHI", "Philadelphia", "Wells Fargo Center", 39.9012, -75.1720, -5, 20, 1610612755),
    "PHX": TeamLocation("PHX", "Phoenix", "Footprint Center", 33.4457, -112.0712, -7, 1086, 1610612756),
    "POR": TeamLocation("POR", "Portland", "Moda Center", 45.5316, -122.6668, -8, 50, 1610612757),
    "SAC": TeamLocation("SAC", "Sacramento", "Golden 1 Center", 38.5802, -121.4997, -8, 25, 1610612758),
    "SAS": TeamLocation("SAS", "San Antonio", "Frost Bank Center", 29.4270, -98.4375, -6, 650, 1610612759),
    "TOR": TeamLocation("TOR", "Toronto", "Scotiabank Arena", 43.6435, -79.3791, -5, 250, 1610612761),
    "UTA": TeamLocation("UTA", "Salt Lake City", "Delta Center", 40.7683, -111.9011, -7, 4226, 1610612762),
    "WAS": TeamLocation("WAS", "Washington", "Capital One Arena", 38.8982, -77.0209, -5, 25, 1610612764),
}

# Mapeo alternativo por Team ID
TEAM_ID_TO_LOCATION: Dict[int, TeamLocation] = {
    loc.team_id: loc for loc in NBA_LOCATIONS.values() if loc.team_id is not None
}


def get_team_location(team_identifier: Union[str, int]) -> Optional[TeamLocation]:
    """Obtiene la ubicación de una franquicia por abreviatura o ID numérico."""
    if isinstance(team_identifier, int):
        return TEAM_ID_TO_LOCATION.get(team_identifier)

    code = str(team_identifier).strip().upper()
    if code in NBA_LOCATIONS:
        return NBA_LOCATIONS[code]

    # Búsqueda por subcadena
    for abbr, loc in NBA_LOCATIONS.items():
        if abbr in code or loc.city.upper() in code:
            return loc
    return None


def calculate_haversine_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> float:
    """Calcula la distancia geodésica en millas terrestres entre dos coordenadas."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_MILES * c


def calculate_travel_miles(
    origin_team: Union[str, int],
    destination_team: Union[str, int],
) -> float:
    """Calcula las millas estimadas de vuelo entre dos sedes de equipos."""
    loc_orig = get_team_location(origin_team)
    loc_dest = get_team_location(destination_team)

    if not loc_orig or not loc_dest:
        return 0.0

    if loc_orig.abbreviation == loc_dest.abbreviation:
        return 0.0

    dist = calculate_haversine_distance(
        loc_orig.latitude, loc_orig.longitude,
        loc_dest.latitude, loc_dest.longitude,
    )
    return round(dist, 1)


def calculate_time_zones_crossed(
    origin_team: Union[str, int],
    destination_team: Union[str, int],
) -> int:
    """Calcula el número absoluto de husos horarios cruzados entre dos sedes."""
    loc_orig = get_team_location(origin_team)
    loc_dest = get_team_location(destination_team)

    if not loc_orig or not loc_dest:
        return 0

    return abs(loc_dest.utc_offset - loc_orig.utc_offset)


def is_altitude_advantage(home_team: Union[str, int]) -> bool:
    """
    Retorna True si el partido se juega en Denver (Ball Arena) o Utah (Delta Center).
    Ambos estadios se sitúan a más de 4,200 pies de altura, produciendo una ventaja fisiológica comprobada.
    """
    loc = get_team_location(home_team)
    if not loc:
        return False
    return loc.is_high_altitude


def calculate_travel_fatigue_index(
    travel_miles: float,
    time_zones_crossed: int = 0,
    rest_days: float = 1.0,
    is_b2b: int = 0,
    time_zones: Optional[int] = None,
) -> float:
    """
    Calcula un índice compuesto de fatiga de viaje (0.0 a 10.0+):
    Pondera distancia (> 1,500 mi), disrupción circadiana (jet lag) y falta de descanso (B2B).
    """
    tz_val = time_zones if time_zones is not None else time_zones_crossed

    # Componente distancia: 1 punto por cada 1,000 millas
    dist_factor = travel_miles / 1000.0

    # Componente circadiano: 1.5 puntos por cada huso horario cruzado
    tz_factor = tz_val * 1.5

    # Penalización por back-to-back: +3.0 puntos
    b2b_penalty = 3.0 if is_b2b == 1 else 0.0

    # Alivio por descanso: -1.0 punto por cada día extra de descanso más allá de 1 día
    rest_relief = max(0.0, (rest_days - 1.0) * 1.0)

    total_fatigue = dist_factor + tz_factor + b2b_penalty - rest_relief
    return round(max(0.0, total_fatigue), 2)


# Alias canónico
NBA_TEAM_COORDINATES = NBA_LOCATIONS
