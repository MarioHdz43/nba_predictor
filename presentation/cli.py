# -*- coding: utf-8 -*-
"""
presentation/cli.py
--------------------
Controlador de interfaz de línea de comandos.
Solo responsable de I/O: parsear argumentos, mostrar resultados, leer input.
No contiene lógica de negocio ni de infraestructura.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from application.use_cases import PredictGameUseCase
from domain.entities import Odds, PredictionResult

logger = logging.getLogger(__name__)


def load_teams_index() -> pd.DataFrame:
    """Carga el índice de equipos NBA desde nba_api (con fallback estático)."""
    try:
        from nba_api.stats.static import teams
        df = pd.DataFrame(teams.get_teams())[["id", "full_name", "abbreviation"]]
    except Exception:
        logger.warning("nba_api no disponible, usando lista estática de equipos.")
        df = pd.DataFrame(_FALLBACK_TEAMS)

    return df.sort_values("full_name").reset_index(drop=True)


def resolve_team(team_df: pd.DataFrame, query: str) -> Tuple[int, str, str]:
    """
    Resuelve un identificador de equipo (ID numérico, abreviatura, nombre o alias).
    Retorna (team_id, full_name, abbreviation).
    """
    q = str(query).strip()

    if q.isdigit():
        row = team_df[team_df["id"] == int(q)]
        if not row.empty:
            r = row.iloc[0]
            return int(r["id"]), str(r["full_name"]), str(r["abbreviation"])

    # Matching canónico institucional
    try:
        from infrastructure.odds.live_odds_client import canonicalize_team
        c_abbr = canonicalize_team(q)
        if c_abbr:
            row = team_df[team_df["abbreviation"].str.upper() == c_abbr.upper()]
            if not row.empty:
                r = row.iloc[0]
                return int(r["id"]), str(r["full_name"]), str(r["abbreviation"])
    except Exception:
        pass

    row = team_df[team_df["abbreviation"].str.upper() == q.upper()]
    if not row.empty:
        r = row.iloc[0]
        return int(r["id"]), str(r["full_name"]), str(r["abbreviation"])

    row = team_df[team_df["full_name"].str.contains(q, case=False, na=False)]
    if not row.empty:
        r = row.iloc[0]
        return int(r["id"]), str(r["full_name"]), str(r["abbreviation"])

    raise ValueError(f"Equipo no encontrado: '{query}'")


def format_game_menu_entry(idx: int, game: dict) -> str:
    """Formatea la entrada de un partido para el menú de The Odds API."""
    h_team = str(game.get("home_team", ""))
    a_team = str(game.get("away_team", ""))
    h_short = h_team.split()[-1]
    a_short = a_team.split()[-1]

    h_odd = game.get("home_odds")
    a_odd = game.get("away_odds")
    if h_odd is not None and a_odd is not None:
        odds_str = f"Cuotas: {h_odd:.2f} ({h_short}) vs {a_odd:.2f} ({a_short})"
    else:
        odds_str = "Cuotas: No publicadas aún"

    status_tag = game.get("status_tag") or ("[DISPONIBLE]" if game.get("is_pregame") else "[EN JUGADA / CERRADO]")
    loc_time = game.get("local_time_str", "")

    return f"[{idx}] {h_team} vs {a_team}\n    Hora: {loc_time} | {odds_str} {status_tag}"


def parse_batch_selection(
    user_input: str,
    max_items: int,
) -> Tuple[List[int], Optional[str]]:
    """
    Parsea la selección del usuario en el menú interactivo de partidos.

    Retorna una tupla: (indices_seleccionados, comando_de_vista)
    - Comandos de cambio de vista:
        * 'm' / 'mañana' / 'manana' -> ([], 'tomorrow')
        * 't' -> ([], 'all_view')
        * 'h' / 'hoy' -> ([], 'today')
        * 'l' / 'liquidar' -> ([], 'settle_manual')
    - Selección de partidos:
        * '0' -> ([0], None) (modo manual)
        * 'todos' / 'all' / '*' -> (list(1..max_items), None)
        * '1' o '1, 3, 5' -> ([1, 3, 5], None)
    - Estados de error o control:
        * '' -> ([], 'empty')
        * Out of range -> ([], 'out_of_range')
        * Formato inválido -> ([], 'invalid')
    """
    raw = str(user_input).strip()
    if not raw:
        return [], "empty"

    raw_lower = raw.lower()

    if raw_lower in ("m", "mañana", "manana"):
        return [], "tomorrow"
    if raw_lower == "t":
        return [], "all_view"
    if raw_lower in ("h", "hoy"):
        return [], "today"
    if raw_lower in ("l", "liquidar"):
        return [], "settle_manual"

    if raw_lower in ("todos", "all", "*"):
        if max_items <= 0:
            return [], "out_of_range"
        return list(range(1, max_items + 1)), None

    if raw == "0":
        return [0], None

    tokens = [tok.strip() for tok in raw.replace(";", ",").split(",") if tok.strip()]
    if not tokens:
        return [], "invalid"

    selected: List[int] = []
    for tok in tokens:
        if not tok.isdigit():
            return [], "invalid"
        val = int(tok)
        if val < 1 or val > max_items:
            return [], "out_of_range"
        if val not in selected:
            selected.append(val)

    return selected, None


def prompt_teams(team_df: pd.DataFrame) -> Tuple[int, str, str, int, str, str]:
    """
    Muestra la lista de equipos y pide al usuario seleccionar local y visitante.
    Retorna los datos de ambos equipos.
    """
    print("\n=== EQUIPOS NBA ===")
    for idx, row in team_df.iterrows():
        print(f"  {idx+1:2d}. {row['full_name']} ({row['abbreviation']})")

    home_num = int(input("\nLOCAL (número): ")) - 1
    away_num = int(input("VISITANTE (número): ")) - 1

    home = team_df.iloc[home_num]
    away = team_df.iloc[away_num]

    return (
        int(home["id"]), str(home["full_name"]), str(home["abbreviation"]),
        int(away["id"]), str(away["full_name"]), str(away["abbreviation"]),
    )


def prompt_manual_odds() -> Optional[Odds]:
    """Permite el ingreso manual de cuotas decimales."""
    use_odds = input("  ¿Ingresar cuotas manualmente? (s/N): ").strip().lower()
    if use_odds not in ("s", "si", "sí", "y", "yes"):
        return None

    try:
        home_odd = float(input("  Cuota LOCAL (decimal, ej: 1.85): ").replace(",", "."))
        away_odd = float(input("  Cuota VISITANTE (decimal, ej: 2.10): ").replace(",", "."))
        if home_odd > 1.0 and away_odd > 1.0:
            return Odds(home_decimal=home_odd, away_decimal=away_odd)
        print("  Las cuotas deben ser mayores a 1.0. Continuando sin ellas.")
        return None
    except ValueError:
        print("  Cuotas inválidas. Continuando sin ellas.")
        return None


def prompt_odds(home_name: Optional[str] = None, away_name: Optional[str] = None) -> Optional[Odds]:
    """
    Consulta cuotas en vivo de la NBA automáticamente vía The Odds API.
    Si no están disponibles, permite ingresar cuotas manualmente de forma opcional.
    """
    if home_name and away_name:
        try:
            from infrastructure.odds.live_odds_client import find_market_match
            match = find_market_match(home_name, away_name)
            if match and match.get("home_odds") and match.get("away_odds"):
                h_odd = match["home_odds"] if not match["is_inverted"] else match["away_odds"]
                a_odd = match["away_odds"] if not match["is_inverted"] else match["home_odds"]
                bk = match.get("bookmaker") or "Consenso"
                print(f"\n  [Mercado] Cuotas en vivo detectadas ({bk}): Local {h_odd:.2f} | Visitante {a_odd:.2f}")
                return Odds(home_decimal=h_odd, away_decimal=a_odd)
        except Exception as exc:
            logger.debug(f"live-odds-fetch-error: {exc}")

        print("\n  [Mercado] Cuotas no disponibles / Partido de prueba")

    return prompt_manual_odds()


def display_result(result: PredictionResult) -> None:
    """Presenta el resultado de la predicción en consola."""
    sep = "=" * 54

    print(f"\n{sep}")
    print(f"  PREDICCION: {result.home_team} vs {result.away_team}")
    print(sep)
    print(f"  {result.home_team:<28} {result.home_win_prob*100:5.1f}%")
    print(f"  {result.away_team:<28} {result.away_win_prob*100:5.1f}%")
    print(f"  Confianza del modelo:          {result.confidence*100:5.1f}%")

    print(f"\n  FACTORES CLAVE")
    print(f"  {'-'*50}")
    for factor in result.key_factors:
        print(f"  - {factor}")

    print(f"\n  VALOR DE APUESTA")
    print(f"  {'-'*50}")
    if result.odds:
        print(f"  Cuota local:       {result.odds.home_decimal:.2f}")
        print(f"  Cuota visitante:   {result.odds.away_decimal:.2f}")
        if result.home_no_vig_prob is not None and result.away_no_vig_prob is not None:
            print(f"  No-Vig (Shin):     Local {result.home_no_vig_prob*100:5.1f}% | Visitante {result.away_no_vig_prob*100:5.1f}%")
        print(f"  EV local:          {result.home_expected_value:+.4f}")
        print(f"  EV visitante:      {result.away_expected_value:+.4f}")
        if result.kelly_stake_home is not None or result.kelly_stake_away is not None:
            k_h = (result.kelly_stake_home or 0.0) * 100
            k_a = (result.kelly_stake_away or 0.0) * 100
            print(f"  Quarter Kelly:     Local {k_h:4.1f}% | Visitante {k_a:4.1f}%")
    else:
        print("  Sin cuotas (predicción puramente estadística del modelo)")
    print(f"  Recomendacion:     {result.recommended_bet}")
    print(sep)


class CLIController:
    """
    Controlador principal del CLI.
    Coordina el flujo de la sesión de predicciones.
    """

    def __init__(
        self,
        predict_game: PredictGameUseCase,
        excel_path: Optional[str] = None,
    ) -> None:
        self._predict_game = predict_game
        self._excel_path = excel_path

    def execute_batch(
        self,
        matches: List[Dict[str, Any]],
        season: str = "2024-25",
        force_fresh: bool = False,
        is_batch: bool = False,
    ) -> Tuple[int, int]:
        """
        Ejecuta el procesamiento de una lista de partidos seleccionados (individual o por lote):
        - Predicción cuantitativa del encuentro
        - Visualización en consola
        - Registro en libro mayor Excel (.xlsx)
        - Resumen consolidado del lote si len(matches) > 1 o is_batch=True

        Retorna (analyzed_count, recommended_bets_count).
        """
        total_matches = len(matches)
        analyzed_count = 0
        recommended_bets_count = 0

        for b_idx, match_info in enumerate(matches, start=1):
            if total_matches > 1:
                print(f"\n============================================================")
                print(f"  PARTIDO [{b_idx}/{total_matches}]: {match_info['home_name']} vs {match_info['away_name']}")
                print(f"============================================================")

            odds = match_info.get("odds")
            if odds is not None:
                bk = match_info.get("bookmaker") or "Consenso"
                print(f"\n  [Mercado] Cuotas en vivo detectadas ({bk}): Local {odds.home_decimal:.2f} | Visitante {odds.away_decimal:.2f}")
            else:
                if total_matches == 1:
                    odds = prompt_odds(home_name=match_info["home_name"], away_name=match_info["away_name"])
                else:
                    print("\n  [Mercado] Cuotas no publicadas aún para este partido. Continuando con análisis estadístico.")
                    odds = None

            print(f"\nPrediciendo: {match_info['home_name']} (local) vs {match_info['away_name']} (visitante)...")

            result = self._predict_game.execute(
                home_team_id=match_info["home_id"],
                home_abbr=match_info["home_abbr"],
                away_team_id=match_info["away_id"],
                away_abbr=match_info["away_abbr"],
                season=season,
                force_fresh=force_fresh,
                odds=odds,
            )

            display_result(result)
            analyzed_count += 1

            # Registro en el libro mayor Excel
            try:
                from infrastructure.reporting.bet_tracker_excel import (
                    get_excel_path,
                    log_bet_prediction,
                    is_bet_recommended,
                )
                is_rec = is_bet_recommended(result) and (odds is not None)
                if is_rec:
                    recommended_bets_count += 1

                logged = log_bet_prediction(
                    prediction=result,
                    odds=odds,
                    game_id=match_info.get("game_id"),
                    excel_path=self._excel_path,
                )
                if logged:
                    target_file = get_excel_path(self._excel_path)
                    if is_rec:
                        print(f"\n  [Bitácora] Apuesta recomendada (+EV) registrada en Excel: {target_file}")
                    else:
                        print(f"\n  [Bitácora] Partido registrado en Excel (Sin valor de apuesta): {target_file}")
            except Exception as exc:
                logger.warning(f"excel-logging-error: {exc}")

        # Resumen consolidado del lote
        if total_matches > 1 or is_batch:
            print("\n" + "=" * 60)
            print(f"[Lote Completo] {analyzed_count} partidos analizados | {recommended_bets_count} apuestas recomendadas registradas en Excel.")
            print("=" * 60)

        return analyzed_count, recommended_bets_count

    def prompt_manual_settlement(self) -> bool:
        """
        Permite seleccionar un partido pendiente registrado en nba_bet_tracking.xlsx
        e ingresar los puntos reales anotados por cada equipo para liquidarlo de forma manual.
        """
        try:
            from infrastructure.reporting.settle_bets import get_pending_bets, settle_bet_manually
            from infrastructure.reporting.bet_tracker_excel import get_excel_path
        except ImportError as exc:
            print(f"Error cargando módulo de liquidación: {exc}")
            return False

        target_file = get_excel_path(self._excel_path)
        pending_list = get_pending_bets(target_file)

        if not pending_list:
            print("\n============================================================")
            print("          LIQUIDACIÓN MANUAL DE PARTIDOS PENDIENTES")
            print("============================================================")
            print("No hay partidos con estado 'PENDIENTE' en el libro mayor.")
            print(f"Archivo verificado: {target_file}")
            print("============================================================\n")
            return False

        print("\n============================================================")
        print("          PARTIDOS PENDIENTES EN BITÁCORA EXCEL")
        print("============================================================")
        for p in pending_list:
            print(
                f"  [{p['id']}] {p['partido']} (Fecha: {p['fecha']} | "
                f"Pronóstico: {p['probabilidad']} | Decisión: {p['decision']} | Cuota: {p['cuota']})"
            )
        print("------------------------------------------------------------")
        print("  [0] Volver al menú principal")
        print("============================================================")

        try:
            choice_str = input("Selecciona el número del partido a liquidar: ").strip()
            if not choice_str or choice_str == "0":
                print("Liquidación manual cancelada.")
                return False

            choice_num = int(choice_str)
            target = next((p for p in pending_list if p["id"] == choice_num), None)
            if target is None:
                print("Número de partido no válido. Operación cancelada.")
                return False

            print(f"\n--- Liquidando: {target['partido']} (Local: {target['team_1']} | Visitante: {target['team_2']}) ---")

            h_input = input(f"Puntos Local ({target['team_1']}): ").strip()
            a_input = input(f"Puntos Visitante ({target['team_2']}): ").strip()

            h_pts = int(h_input)
            a_pts = int(a_input)

            if h_pts < 0 or a_pts < 0:
                print("Los puntos no pueden ser negativos. Operación cancelada.")
                return False

            res = settle_bet_manually(
                match_identifier=target["id"],
                home_score=h_pts,
                away_score=a_pts,
                excel_path=target_file,
            )

            if res.get("success"):
                print("\n" + "=" * 60)
                print(f"[Liquidación Manual] Partido '{res['partido']}' liquidado con éxito.")
                print(f"Marcador Oficial: {res['score']} -> Resultado del Modelo: {res['status']}")
                print(f"P&L Asociado: {res['pnl']:+.4f} unidades.")
                print(f"Libro mayor y 'Resumen_Metricas' actualizados: {res['excel_path']}")
                print("=" * 60 + "\n")
                return True
            else:
                print(f"\nError en liquidación: {res.get('message')}")
                return False
        except ValueError:
            print("Entrada inválida (se esperaba un número entero). Operación cancelada.")
            return False
        except Exception as exc:
            logger.warning(f"manual-settlement-error: {exc}")
            print(f"Error procesando liquidación: {exc}")
            return False

    def run(
        self,
        home_query: Optional[str] = None,
        away_query: Optional[str] = None,
        season: str = "2024-25",
        force_fresh: bool = False,
    ) -> None:
        try:
            from infrastructure.reporting.settle_bets import auto_reconcile_pending_bets
            auto_reconcile_pending_bets(excel_path=self._excel_path)
        except Exception as exc:
            logger.debug(f"auto-reconcile-cli-error: {exc}")

        team_df = load_teams_index()
        current_view = "today"


        while True:
            try:
                # Caso 1: Se pasaron equipos por argumentos CLI (--home, --away)
                if home_query and away_query:
                    home_id, home_name, home_abbr = resolve_team(team_df, home_query)
                    away_id, away_name, away_abbr = resolve_team(team_df, away_query)
                    home_query = away_query = None

                    # Verificación y corrección de localía en mercado
                    try:
                        from infrastructure.odds.live_odds_client import find_market_match
                        m_match = find_market_match(home_name, away_name)
                        if m_match and m_match["is_inverted"]:
                            off_h = m_match["official_home"]
                            off_a = m_match["official_away"]
                            print(f"\n  [Mercado] Ajustando localía oficial: Local = {off_h}, Visitante = {off_a}")
                            home_id, home_name, home_abbr = resolve_team(team_df, off_h)
                            away_id, away_name, away_abbr = resolve_team(team_df, off_a)
                    except Exception as exc:
                        logger.debug(f"market-match-error: {exc}")

                    matches_to_run = [{
                        "home_id": home_id, "home_name": home_name, "home_abbr": home_abbr,
                        "away_id": away_id, "away_name": away_name, "away_abbr": away_abbr,
                        "odds": None,
                        "game_id": None,
                        "is_pregame": True,
                    }]
                    self.execute_batch(matches_to_run, season=season, force_fresh=force_fresh, is_batch=False)

                else:
                    # Caso 2: Menú interactivo con filtro dinámico por fechas y procesamiento por lotes
                    available_games = []
                    try:
                        from infrastructure.odds.live_odds_client import get_games_by_filter
                        available_games = get_games_by_filter(filter_type=current_view)
                    except Exception as exc:
                        logger.debug(f"get-games-by-filter-error: {exc}")

                    # Si NO hay partidos programados para la vista actual
                    if not available_games:
                        if current_view == "today":
                            print("\n============================================================")
                            print("No hay partidos de la NBA programados para hoy.")
                            print("============================================================")
                            print("[1] Ver partidos de Mañana")
                            print("[2] Ver todos los próximos partidos programados")
                            print("[L] Liquidar manualmente un partido pendiente")
                            print("[0] Salir/Manual")
                            print("============================================================")
                            sub_choice = input("Elige una opción: ").strip()
                            if sub_choice in ("1", "m", "M"):
                                current_view = "tomorrow"
                                continue
                            elif sub_choice in ("2", "t", "T"):
                                current_view = "all"
                                continue
                            elif sub_choice in ("l", "L", "liquidar"):
                                self.prompt_manual_settlement()
                                continue
                            elif sub_choice == "0":
                                # Ingreso manual
                                (
                                    home_id, home_name, home_abbr,
                                    away_id, away_name, away_abbr,
                                ) = prompt_teams(team_df)
                                matches_to_run = [{
                                    "home_id": home_id, "home_name": home_name, "home_abbr": home_abbr,
                                    "away_id": away_id, "away_name": away_name, "away_abbr": away_abbr,
                                    "odds": None,
                                    "game_id": None,
                                    "is_pregame": True,
                                }]
                                self.execute_batch(matches_to_run, season=season, force_fresh=force_fresh, is_batch=False)
                            else:
                                print("Opción inválida. Intenta nuevamente.")
                                continue
                        else:
                            view_labels = {"tomorrow": "Mañana", "all": "todos los próximos"}
                            lbl = view_labels.get(current_view, current_view)
                            print(f"\n[Mercado] No hay partidos programados para {lbl}.")
                            print("[1] Ver partidos de Hoy | [2] Ver todos los próximos | [L] Liquidar manual | [0] Manual")
                            sub_choice = input("Elige una opción: ").strip()
                            if sub_choice in ("1", "h", "H"):
                                current_view = "today"
                                continue
                            elif sub_choice in ("2", "t", "T"):
                                current_view = "all"
                                continue
                            elif sub_choice in ("l", "L", "liquidar"):
                                self.prompt_manual_settlement()
                                continue
                            elif sub_choice == "0":
                                (
                                    home_id, home_name, home_abbr,
                                    away_id, away_name, away_abbr,
                                ) = prompt_teams(team_df)
                                matches_to_run = [{
                                    "home_id": home_id, "home_name": home_name, "home_abbr": home_abbr,
                                    "away_id": away_id, "away_name": away_name, "away_abbr": away_abbr,
                                    "odds": None,
                                    "game_id": None,
                                    "is_pregame": True,
                                }]
                                self.execute_batch(matches_to_run, season=season, force_fresh=force_fresh, is_batch=False)
                            else:
                                print("Opción inválida. Intenta nuevamente.")
                                continue

                    else:
                        # Hay partidos disponibles en la vista actual
                        max_display = 15
                        displayed = available_games[:max_display]

                        view_titles = {
                            "today": "PARTIDOS DE HOY EN EL MERCADO (THE ODDS API)",
                            "tomorrow": "PARTIDOS DE MAÑANA EN EL MERCADO (THE ODDS API)",
                            "all": "TODOS LOS PRÓXIMOS PARTIDOS EN EL MERCADO (THE ODDS API)",
                        }
                        title = view_titles.get(current_view, "PARTIDOS DISPONIBLES EN EL MERCADO (THE ODDS API)")

                        print("\n============================================================")
                        print(f"         {title}")
                        print("============================================================")
                        for idx, g in enumerate(displayed, start=1):
                            print(format_game_menu_entry(idx, g))
                        if len(available_games) > max_display:
                            print(f"    ... [Mostrando los primeros {max_display} de {len(available_games)} partidos disponibles]")

                        cmd_opts = []
                        if current_view != "today":
                            cmd_opts.append("[H] Ver Hoy")
                        if current_view != "tomorrow":
                            cmd_opts.append("[M] Ver Mañana")
                        if current_view != "all":
                            cmd_opts.append("[T] Ver Todos")
                        cmd_opts.append("[L] Liquidar Manual")
                        cmd_opts.append("[0] Manual")

                        print("------------------------------------------------------------")
                        print(f"Comandos rápidos: {' | '.join(cmd_opts)}")
                        print("Selección: Un partido [1], varios [1, 3, 5] o lote completo [todos]")
                        print("============================================================")

                        choice_raw = input("Elige el número del partido: ").strip()
                        selected_indices, action = parse_batch_selection(choice_raw, len(displayed))

                        if action == "tomorrow":
                            current_view = "tomorrow"
                            continue
                        elif action == "all_view":
                            current_view = "all"
                            continue
                        elif action == "today":
                            current_view = "today"
                            continue
                        elif action == "settle_manual":
                            self.prompt_manual_settlement()
                            continue
                        elif action == "out_of_range":
                            print(f"Número(s) fuera de rango (1 - {len(displayed)}). Intenta nuevamente.")
                            continue
                        elif action in ("invalid", "empty"):
                            print("Opción inválida. Intenta nuevamente.")
                            continue

                        if selected_indices == [0]:
                            # Modo manual
                            (
                                home_id, home_name, home_abbr,
                                away_id, away_name, away_abbr,
                            ) = prompt_teams(team_df)
                            # Verificación y corrección de localía
                            try:
                                from infrastructure.odds.live_odds_client import find_market_match
                                m_match = find_market_match(home_name, away_name)
                                if m_match and m_match["is_inverted"]:
                                    off_h = m_match["official_home"]
                                    off_a = m_match["official_away"]
                                    print(f"\n  [Mercado] Ajustando localía oficial: Local = {off_h}, Visitante = {off_a}")
                                    home_id, home_name, home_abbr = resolve_team(team_df, off_h)
                                    away_id, away_name, away_abbr = resolve_team(team_df, off_a)
                            except Exception as exc:
                                logger.debug(f"market-match-error: {exc}")

                            matches_to_run = [{
                                "home_id": home_id, "home_name": home_name, "home_abbr": home_abbr,
                                "away_id": away_id, "away_name": away_name, "away_abbr": away_abbr,
                                "odds": None,
                                "game_id": None,
                                "is_pregame": True,
                            }]
                            self.execute_batch(matches_to_run, season=season, force_fresh=force_fresh, is_batch=False)

                        else:
                            # 1 o más partidos seleccionados del menú
                            matches_to_run = []
                            skip_selection = False
                            is_batch_mode = len(selected_indices) > 1 or choice_raw.lower() in ("todos", "all")

                            for s_idx in selected_indices:
                                sel_game = displayed[s_idx - 1]

                                if not sel_game["is_pregame"]:
                                    print(f"\n  [ADVERTENCIA] El partido {sel_game['home_team']} vs {sel_game['away_team']} ya ha comenzado o su mercado está cerrado.")
                                    if len(selected_indices) == 1:
                                        confirm = input("  ¿Deseas analizarlo de todos modos con fines estadísticos? (s/N): ").strip().lower()
                                        if confirm not in ("s", "si", "sí", "y", "yes"):
                                            skip_selection = True
                                            break
                                    else:
                                        print("  Se analizará únicamente con fines estadísticos.")

                                h_id, h_name, h_abbr = resolve_team(team_df, sel_game["home_team"])
                                a_id, a_name, a_abbr = resolve_team(team_df, sel_game["away_team"])

                                odds_obj = None
                                if sel_game.get("home_odds") and sel_game.get("away_odds"):
                                    odds_obj = Odds(
                                        home_decimal=sel_game["home_odds"],
                                        away_decimal=sel_game["away_odds"],
                                    )

                                matches_to_run.append({
                                    "home_id": h_id,
                                    "home_name": h_name,
                                    "home_abbr": h_abbr,
                                    "away_id": a_id,
                                    "away_name": a_name,
                                    "away_abbr": a_abbr,
                                    "odds": odds_obj,
                                    "bookmaker": sel_game.get("bookmaker"),
                                    "game_id": sel_game.get("game_id"),
                                    "is_pregame": sel_game.get("is_pregame"),
                                })

                            if skip_selection:
                                continue

                            self.execute_batch(matches_to_run, season=season, force_fresh=force_fresh, is_batch=is_batch_mode)

            except (ValueError, RuntimeError) as exc:
                print(f"\nError: {exc}")
            except (IndexError, KeyError):
                print("\nSeleccion invalida. Intenta nuevamente.")

            repeat = input("\n¿Hacer otra prediccion? (s/N): ").strip().lower()
            if repeat not in ("s", "si", "sí", "y", "yes"):
                print("Hasta luego.")
                break


# ---------------------------------------------------------------------------
# Fallback estático de equipos (si nba_api no está disponible)
# ---------------------------------------------------------------------------
_FALLBACK_TEAMS = [
    {"id": 1610612737, "full_name": "Atlanta Hawks",            "abbreviation": "ATL"},
    {"id": 1610612738, "full_name": "Boston Celtics",           "abbreviation": "BOS"},
    {"id": 1610612751, "full_name": "Brooklyn Nets",            "abbreviation": "BKN"},
    {"id": 1610612766, "full_name": "Charlotte Hornets",        "abbreviation": "CHA"},
    {"id": 1610612741, "full_name": "Chicago Bulls",            "abbreviation": "CHI"},
    {"id": 1610612739, "full_name": "Cleveland Cavaliers",      "abbreviation": "CLE"},
    {"id": 1610612742, "full_name": "Dallas Mavericks",         "abbreviation": "DAL"},
    {"id": 1610612743, "full_name": "Denver Nuggets",           "abbreviation": "DEN"},
    {"id": 1610612765, "full_name": "Detroit Pistons",          "abbreviation": "DET"},
    {"id": 1610612744, "full_name": "Golden State Warriors",    "abbreviation": "GSW"},
    {"id": 1610612745, "full_name": "Houston Rockets",          "abbreviation": "HOU"},
    {"id": 1610612754, "full_name": "Indiana Pacers",           "abbreviation": "IND"},
    {"id": 1610612746, "full_name": "Los Angeles Clippers",     "abbreviation": "LAC"},
    {"id": 1610612747, "full_name": "Los Angeles Lakers",       "abbreviation": "LAL"},
    {"id": 1610612763, "full_name": "Memphis Grizzlies",        "abbreviation": "MEM"},
    {"id": 1610612748, "full_name": "Miami Heat",               "abbreviation": "MIA"},
    {"id": 1610612749, "full_name": "Milwaukee Bucks",          "abbreviation": "MIL"},
    {"id": 1610612750, "full_name": "Minnesota Timberwolves",   "abbreviation": "MIN"},
    {"id": 1610612740, "full_name": "New Orleans Pelicans",     "abbreviation": "NOP"},
    {"id": 1610612752, "full_name": "New York Knicks",          "abbreviation": "NYK"},
    {"id": 1610612760, "full_name": "Oklahoma City Thunder",    "abbreviation": "OKC"},
    {"id": 1610612753, "full_name": "Orlando Magic",            "abbreviation": "ORL"},
    {"id": 1610612755, "full_name": "Philadelphia 76ers",       "abbreviation": "PHI"},
    {"id": 1610612756, "full_name": "Phoenix Suns",             "abbreviation": "PHX"},
    {"id": 1610612757, "full_name": "Portland Trail Blazers",   "abbreviation": "POR"},
    {"id": 1610612758, "full_name": "Sacramento Kings",         "abbreviation": "SAC"},
    {"id": 1610612759, "full_name": "San Antonio Spurs",        "abbreviation": "SAS"},
    {"id": 1610612761, "full_name": "Toronto Raptors",          "abbreviation": "TOR"},
    {"id": 1610612762, "full_name": "Utah Jazz",                "abbreviation": "UTA"},
    {"id": 1610612764, "full_name": "Washington Wizards",       "abbreviation": "WAS"},
]
