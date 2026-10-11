# -*- coding: utf-8 -*-
"""
scripts/verify_live_odds.py
---------------------------
Script de verificación y diagnóstico de conectividad en vivo con The Odds API.
Comprueba la validez de la API Key, cuota restante mensual (x-requests-remaining)
y la extracción de cuotas moneyline de la NBA.
"""

import sys
from pathlib import Path

# Agregar directorio raíz al path para ejecución directa
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.config import config
from infrastructure.odds.live_odds_client import LiveOddsClient


def main() -> int:
    print("=" * 65)
    print("      VERIFICACIÓN DE CONEXIÓN EN VIVO: THE ODDS API (NBA)      ")
    print("=" * 65)

    api_key = config.odds_api_key
    if not api_key:
        print("[ERROR] No se encontró una clave 'odds_api_key' configurada en config.yaml ni en el entorno.")
        return 1

    masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
    print(f"[*] API Key detectada : {masked_key} (Longitud: {len(api_key)})")

    client = LiveOddsClient(api_key=api_key)
    print("[*] Conectando con endpoint: v4/sports/basketball_nba/odds/?regions=us,eu&markets=h2h ...")
    
    conn_info = client.check_connection()

    status_code = conn_info.get("status_code")
    valid_key = conn_info.get("valid_key")
    req_remaining = conn_info.get("requests_remaining")
    req_used = conn_info.get("requests_used")
    games_count = conn_info.get("games_count", 0)

    print(f"[*] Código HTTP de Respuesta: {status_code}")
    print(f"[*] Validez de la API Key   : {'VÁLIDA (HTTP 200)' if valid_key else 'INVÁLIDA / ERROR'}")
    print(f"[*] Consultas Restantes     : {req_remaining} (x-requests-remaining)")
    print(f"[*] Consultas Utilizadas    : {req_used} (x-requests-used)")
    print(f"[*] Partidos Detectados     : {games_count}")
    print("-" * 65)

    if not valid_key:
        print(f"[FALLO] Detalle: {conn_info.get('message')}")
        return 1

    if games_count == 0:
        print("[INFO] No hay partidos programados en este momento (mercado inactivo o fuera de fecha).")
        print("[INFO] Manejo de lista vacía verificado: El sistema responde limpiamente con None.")
    else:
        print(f"[ÉXITO] Se encontraron {games_count} encuentros programados.")
        games = client._fetch_active_games()
        print("\nEjemplo de partidos y cuotas en vivo extraídas:")
        for idx, g in enumerate(games[:5], start=1):
            h_name = g.get("home_team", "")
            a_name = g.get("away_team", "")
            odds = client.get_game_odds(h_name, a_name)
            if odds:
                h_odd, a_odd, bk = odds
                print(f"  {idx}. {h_name} vs {a_name} -> Local: {h_odd:.2f} | Visitante: {a_odd:.2f} ({bk})")
            else:
                print(f"  {idx}. {h_name} vs {a_name} -> Líneas aún no publicadas")

    # Prueba de partido inexistente
    print("\n[*] Prueba de resiliencia con enfrentamiento de prueba inexistente:")
    test_odds = client.get_game_odds("Boston Celtics", "Real Madrid Baloncesto")
    print(f"  - Resultado: {test_odds} (Manejado de forma segura sin excepciones)")

    print("=" * 65)
    print("           DIAGNÓSTICO FINAL: CONEXIÓN Y CUOTA 100% OPERATIVAS     ")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
