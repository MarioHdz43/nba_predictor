# -*- coding: utf-8 -*-
"""
infrastructure/reporting/bet_tracker_excel.py
---------------------------------------------
Gestor y Bitácora cuantitativa de seguimiento de apuestas y pronósticos de la NBA.
Implementa una maquetación ejecutiva limpia de 7 columnas en bloques de 2 filas por encuentro:
  - Fila 1 (Pronóstico): Fecha (DD/MM/YYYY), Partido ("PHI - BOS"), Cuota, Probabilidad, Decisión, "-" en Marcador_Final, "PENDIENTE" en Estado
  - Fila 2 (Liquidación): "-" en Fecha, Partido, Cuota, Probabilidad, Decisión; "Por definir" o marcador real en Marcador_Final; celda de Estado combinada con Fila 1
  - Fila 3 (Separador): Fila completamente en blanco de separación visual
Registra TODO partido analizado en el sistema sin spam visual ni redundancias.
"""

from __future__ import annotations
from datetime import datetime, timedelta, timezone
import io
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import zoneinfo
except ImportError:
    zoneinfo = None

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd
import structlog

from domain.entities import Odds, PredictionResult
from infrastructure.config import config
from infrastructure.odds.live_odds_client import canonicalize_team

logger = structlog.get_logger()

DEFAULT_EXCEL_PATH = r"C:\Users\mblad\Documents\Codigoes\nba_bet_tracking.xlsx"

SHEET_BETS = "Registro_Apuestas"
SHEET_METRICS = "Resumen_Metricas"

# Esquema limpio de 7 columnas principales
COLUMNS_LEDGER: List[str] = [
    "Fecha",
    "Partido",
    "Cuota",
    "Probabilidad",
    "Decision",
    "Marcador_Final",
    "Estado",
]


def get_user_timezone() -> Any:
    """Obtiene la zona horaria institucional de referencia (America/Bogota, UTC-5 o zona local del sistema)."""
    if zoneinfo is not None:
        try:
            return zoneinfo.ZoneInfo("America/Bogota")
        except Exception:
            pass
    sys_tz = datetime.now().astimezone().tzinfo
    if sys_tz is not None:
        return sys_tz
    return timezone(timedelta(hours=-5))


def get_local_now() -> datetime:
    """Obtiene el datetime actual en la zona horaria local (America/Bogota, UTC-5)."""
    tz = get_user_timezone()
    return datetime.now(tz)


def format_local_timestamp(dt: Optional[datetime] = None) -> str:
    """
    Formatea la fecha y hora en la zona horaria local con estándar explícito:
    'DD/MM/YYYY HH:MM:SS' (ej: '10/10/2026 21:04:26').
    """
    target = dt or get_local_now()
    if target.tzinfo is None:
        target = target.replace(tzinfo=get_user_timezone())
    else:
        target = target.astimezone(get_user_timezone())
    return target.strftime("%d/%m/%Y %H:%M:%S")


def format_date_ddmmyyyy(date_val: Optional[Union[str, datetime]]) -> str:
    """
    Convierte cualquier representación de fecha a formato explícito 'DD/MM/YYYY'
    respetando estrictamente la zona horaria local (America/Bogota, UTC-5).
    Evita el desfase o salto prematuro de día provocado por UTC después de las 7:00 PM.
    """
    tz_local = get_user_timezone()
    if not date_val:
        return get_local_now().strftime("%d/%m/%Y")

    if isinstance(date_val, datetime):
        if date_val.tzinfo is not None:
            date_val = date_val.astimezone(tz_local)
        return date_val.strftime("%d/%m/%Y")

    s = str(date_val).strip()
    if not s or s in ("-", "None", "nan"):
        return get_local_now().strftime("%d/%m/%Y")

    # Si ya tiene formato explícito DD/MM/YYYY
    if re.match(r"^\d{2}/\d{2}/\d{4}$", s):
        return s

    # Si viene con timestamp ISO con hora (ej: '2026-10-11T01:00:00Z', '2026-10-10T23:30:00+00:00')
    if "T" in s or "Z" in s:
        try:
            iso_s = s.replace("Z", "+00:00")
            dt = datetime.fromisoformat(iso_s)
            if dt.tzinfo is not None:
                dt = dt.astimezone(tz_local)
            return dt.strftime("%d/%m/%Y")
        except Exception:
            pass

    # Si viene en formato YYYY-MM-DD
    m_iso = re.match(r"^(\d{4})-(\d{2})-(\d{2})", s)
    if m_iso:
        yyyy, mm, dd = m_iso.groups()
        return f"{dd}/{mm}/{yyyy}"

    # Si viene en formato DD-MM-YYYY
    m_dash = re.match(r"^(\d{2})-(\d{2})-(\d{4})", s)
    if m_dash:
        dd, mm, yyyy = m_dash.groups()
        return f"{dd}/{mm}/{yyyy}"

    for fmt in ("%Y/%m/%d", "%m/%d/%Y", "%Y%m%d"):
        try:
            dt = datetime.strptime(s[:10], fmt)
            return dt.strftime("%d/%m/%Y")
        except ValueError:
            pass

    return get_local_now().strftime("%d/%m/%Y")



def get_excel_path(custom_path: Optional[Union[str, Path]] = None) -> Path:
    """Resuelve la ruta absoluta del archivo Excel de bitácora."""
    if custom_path:
        return Path(custom_path)
    configured = getattr(config, "bet_tracking_excel_path", None)
    if configured:
        return Path(configured)
    return Path(DEFAULT_EXCEL_PATH)


def load_ledger(excel_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Carga el historial de apuestas desde la hoja Registro_Apuestas.
    Si el archivo no existe o está vacío, retorna un DataFrame vacío con las 7 columnas oficiales.
    """
    target = get_excel_path(excel_path)
    if not target.exists():
        return pd.DataFrame(columns=COLUMNS_LEDGER).astype(object)

    try:
        with open(target, "rb") as f:
            excel_bytes = io.BytesIO(f.read())
            xl = pd.ExcelFile(excel_bytes)
            sheet_to_read = (
                SHEET_BETS
                if SHEET_BETS in xl.sheet_names
                else xl.sheet_names[0]
            )
            df = pd.read_excel(
                excel_bytes,
                sheet_name=sheet_to_read,
                dtype=str,
            )

        # Mapeo retrocompatible por si proviniese de esquemas anteriores
        col_rename = {}
        for c in df.columns:
            c_str = str(c).strip().lower()
            if c_str in ("fecha", "date", "fecha_partido"):
                col_rename[c] = "Fecha"
            elif c_str in ("partido", "match", "matchup"):
                col_rename[c] = "Partido"
            elif c_str in ("cuota", "cuota_tomada", "odds"):
                col_rename[c] = "Cuota"
            elif c_str in ("probabilidad", "pronostico_proyectado", "prob"):
                col_rename[c] = "Probabilidad"
            elif c_str in ("decision", "decisión"):
                col_rename[c] = "Decision"
            elif c_str in ("marcador_final", "resultado_marcador", "marcador", "score"):
                col_rename[c] = "Marcador_Final"
            elif c_str in ("estado", "estado_verificacion", "status"):
                col_rename[c] = "Estado"

        if col_rename:
            df = df.rename(columns=col_rename)

        for col in COLUMNS_LEDGER:
            if col not in df.columns:
                df[col] = None

        df = df[COLUMNS_LEDGER].astype(object)

        # Consolidación y migración de esquemas bi-fila residuales a 1 sola fila por evento
        consolidated_rows = []
        n_rows = len(df)
        i = 0
        while i < n_rows:
            row = df.iloc[i].to_dict()
            partido_val = str(row.get("Partido") or "").strip()
            fecha_val = str(row.get("Fecha") or "").strip()

            # Si es una fila principal de partido
            if partido_val and partido_val not in ("-", "None", "nan") and fecha_val not in ("-", "None", "nan"):
                # Si la fila siguiente era una fila secundaria bi-fila con marcador real
                if i + 1 < n_rows:
                    next_row = df.iloc[i + 1]
                    next_partido = str(next_row.get("Partido") or "").strip()
                    next_marcador = str(next_row.get("Marcador_Final") or "").strip()
                    next_estado = str(next_row.get("Estado") or "").strip()
                    if next_partido in ("-", "", "None", "nan") and next_marcador not in ("-", "Por definir", "None", "nan", ""):
                        if str(row.get("Marcador_Final") or "").strip() in ("-", "Por definir", ""):
                            row["Marcador_Final"] = next_marcador
                    if next_estado and next_estado not in ("-", "None", "nan", "") and not str(row.get("Estado") or "").strip():
                        row["Estado"] = next_estado
                consolidated_rows.append(row)
            i += 1

        if consolidated_rows:
            df = pd.DataFrame(consolidated_rows)
        else:
            mask_valid = (
                df["Partido"].notna()
                & (df["Partido"].astype(str).str.strip() != "")
                & (df["Partido"].astype(str).str.strip() != "-")
            )
            df = df[mask_valid]

        for col in COLUMNS_LEDGER:
            if col not in df.columns:
                df[col] = None

        return df[COLUMNS_LEDGER].astype(object).reset_index(drop=True)
    except Exception as exc:
        logger.warning(f"ledger-read-error: {exc}")
        return pd.DataFrame(columns=COLUMNS_LEDGER).astype(object)


def calculate_metrics_summary(ledger_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula KPIs cuantitativos agregados para la hoja Resumen_Metricas.
    Distingue la precisión del modelo (Accuracy de pronósticos) y el rendimiento de apuestas.
    """
    if ledger_df.empty:
        records = [
            ("Total Encuentros Analizados", 0),
            ("Encuentros Pendientes", 0),
            ("Encuentros Liquidados", 0),
            ("Pronósticos Acertados", 0),
            ("Pronósticos Fallados", 0),
            ("Tasa de Acierto del Modelo (Accuracy %)", "0.0%"),
            ("Recomendaciones 'APOSTAR'", 0),
            ("Recomendaciones 'NO APOSTAR'", 0),
            ("Apuestas Ganadas", 0),
            ("Apuestas Perdidas", 0),
            ("Apuestas Push / Empate", 0),
            ("Win Rate de Apuestas (%)", "0.0%"),
            ("Total Unidades Apostadas", 0.0),
            ("P&L Neto Estimado (Unidades)", 0.0),
            ("Rendimiento (ROI / Yield %)", "0.0%"),
            ("Última Actualización", format_local_timestamp()),
        ]
        return pd.DataFrame(records, columns=["Métrica", "Valor"])

    # Identificar filas Fila 1 (Pronóstico): Partido != '-' y notna
    mask_game = (
        ledger_df["Partido"].notna()
        & (ledger_df["Partido"].astype(str).str.strip() != "")
        & (ledger_df["Partido"].astype(str).str.strip() != "-")
    )
    games_df = ledger_df[mask_game].reset_index(drop=True)
    total_games = len(games_df)

    status_series = games_df["Estado"].astype(str).str.strip().str.upper()
    settled_mask = status_series.isin(["ACERTÓ", "ACERTO", "FALLÓ", "FALLO", "PUSH", "GANADA", "PERDIDA"])

    settled_count = int(settled_mask.sum())
    pending_count = total_games - settled_count

    aciertos = int(status_series.isin(["ACERTÓ", "ACERTO", "GANADA"]).sum())
    fallos = int(status_series.isin(["FALLÓ", "FALLO", "PERDIDA"]).sum())
    push_count = int((status_series == "PUSH").sum())

    model_acc = (aciertos / (aciertos + fallos) * 100.0) if (aciertos + fallos) > 0 else 0.0

    apostar_count = int((games_df["Decision"].astype(str).str.strip().str.upper() == "APOSTAR").sum())
    no_apostar_count = total_games - apostar_count

    # Métricas financieras de apuestas recomendadas
    bets_won = 0
    bets_lost = 0
    bets_push = 0
    total_stakes = 0.0
    total_pnl = 0.0

    for _, row in games_df.iterrows():
        dec = str(row.get("Decision") or "").strip().upper()
        est = str(row.get("Estado") or "").strip().upper()

        if dec == "APOSTAR" and est in ("ACERTÓ", "ACERTO", "FALLÓ", "FALLO", "PUSH", "GANADA", "PERDIDA"):
            cuota_str = str(row.get("Cuota") or "").replace(",", ".").strip()
            m_cuota = re.search(r"(\d+(?:\.\d+)?)", cuota_str)
            cuota_val = float(m_cuota.group(1)) if m_cuota else 1.90

            stake_unit = 1.0
            total_stakes += stake_unit

            if est in ("ACERTÓ", "ACERTO", "GANADA"):
                bets_won += 1
                total_pnl += stake_unit * (cuota_val - 1.0)
            elif est in ("FALLÓ", "FALLO", "PERDIDA"):
                bets_lost += 1
                total_pnl -= stake_unit
            elif est == "PUSH":
                bets_push += 1

    bet_win_rate = (bets_won / (bets_won + bets_lost) * 100.0) if (bets_won + bets_lost) > 0 else 0.0
    roi_yield = (total_pnl / total_stakes * 100.0) if total_stakes > 0 else 0.0

    records = [
        ("Total Encuentros Analizados", total_games),
        ("Encuentros Pendientes", pending_count),
        ("Encuentros Liquidados", settled_count),
        ("Pronósticos Acertados", aciertos),
        ("Pronósticos Fallados", fallos),
        ("Tasa de Acierto del Modelo (Accuracy %)", f"{model_acc:.2f}%"),
        ("Recomendaciones 'APOSTAR'", apostar_count),
        ("Recomendaciones 'NO APOSTAR'", no_apostar_count),
        ("Apuestas Ganadas", bets_won),
        ("Apuestas Perdidas", bets_lost),
        ("Apuestas Push / Empate", bets_push),
        ("Win Rate de Apuestas (%)", f"{bet_win_rate:.2f}%"),
        ("Total Unidades Apostadas", round(total_stakes, 2)),
        ("P&L Neto Estimado (Unidades)", round(total_pnl, 4)),
        ("Rendimiento (ROI / Yield %)", f"{roi_yield:+.2f}%"),
        ("Última Actualización", format_local_timestamp()),
    ]
    return pd.DataFrame(records, columns=["Métrica", "Valor"])


def save_ledger_workbook(
    ledger_df: pd.DataFrame,
    excel_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Guarda el ledger de 7 columnas y la hoja de métricas en el archivo Excel aplicando
    la maquetación ejecutiva con separadores y estilos cromáticos.
    """
    target = get_excel_path(excel_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    metrics_df = calculate_metrics_summary(ledger_df)

    with pd.ExcelWriter(target, engine="openpyxl") as writer:
        ledger_df.to_excel(writer, sheet_name=SHEET_BETS, index=False)
        metrics_df.to_excel(writer, sheet_name=SHEET_METRICS, index=False)

    wb = None
    try:
        wb = openpyxl.load_workbook(target, data_only=False)
        _style_ledger_sheet(wb[SHEET_BETS])
        _style_metrics_sheet(wb[SHEET_METRICS])
        wb.save(target)
    except Exception as exc:
        logger.warning(f"excel-styling-warning: {exc}")
    finally:
        if wb is not None:
            wb.close()

    logger.info("ledger-saved", path=str(target), rows=len(ledger_df))
    return target


def _style_ledger_sheet(ws) -> None:
    """Aplica formato visual ejecutivo limpio para las 7 columnas (1 fila por partido)."""
    header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    data_font = Font(name="Calibri", size=10)
    bold_data_font = Font(name="Calibri", size=10, bold=True)
    thin_border = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )

    # Estilos cromáticos ejecutivos
    win_fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
    win_font = Font(name="Calibri", size=10, bold=True, color="274E13")

    loss_fill = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")
    loss_font = Font(name="Calibri", size=10, bold=True, color="C65911")

    pending_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
    pending_font = Font(name="Calibri", size=10, bold=True, color="806000")

    push_fill = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
    push_font = Font(name="Calibri", size=10, bold=True, color="595959")

    # Cabecera
    ws.row_dimensions[1].height = 28
    for col_idx in range(1, ws.max_column + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    # Recorrer filas de datos (1 fila por encuentro)
    for row_idx in range(2, ws.max_row + 1):
        ws.row_dimensions[row_idx].height = 22

        for col_idx in range(1, ws.max_column + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.font = data_font
            cell.border = thin_border
            cell.alignment = Alignment(vertical="center", horizontal="center")

            val_str = str(cell.value or "").strip()

            # Columna 2: Partido ("PHI - BOS")
            if col_idx == 2 and val_str and val_str != "-":
                cell.font = bold_data_font

            # Columna 5: Decision ("APOSTAR" / "NO APOSTAR")
            if col_idx == 5:
                if val_str == "APOSTAR":
                    cell.fill = win_fill
                    cell.font = win_font
                elif val_str == "NO APOSTAR":
                    cell.fill = push_fill
                    cell.font = Font(name="Calibri", size=10, color="595959")

            # Columna 6: Marcador_Final
            if col_idx == 6:
                if val_str and val_str not in ("-", "Por definir"):
                    cell.font = bold_data_font

            # Columna 7: Estado ("PENDIENTE", "ACERTÓ", "FALLÓ", "PUSH")
            if col_idx == 7:
                u_val = val_str.upper()
                if u_val in ("ACERTÓ", "ACERTO", "GANADA"):
                    cell.fill = win_fill
                    cell.font = win_font
                elif u_val in ("FALLÓ", "FALLO", "PERDIDA"):
                    cell.fill = loss_fill
                    cell.font = loss_font
                elif u_val == "PENDIENTE":
                    cell.fill = pending_fill
                    cell.font = pending_font
                elif u_val == "PUSH":
                    cell.fill = push_fill
                    cell.font = push_font

    # Anchos ejecutivos estables para las 7 columnas
    col_widths = {
        1: 15,  # Fecha: "10/10/2026"
        2: 18,  # Partido: "PHI - BOS"
        3: 18,  # Cuota: "1.85 (BOS)"
        4: 22,  # Probabilidad: "81.4% (BOS)"
        5: 18,  # Decision: "APOSTAR"
        6: 18,  # Marcador_Final: "112 - 104"
        7: 18,  # Estado: "PENDIENTE" / "ACERTÓ"
    }
    for col_idx, width in col_widths.items():
        ws.column_dimensions[get_column_letter(col_idx)].width = width


def _style_metrics_sheet(ws) -> None:
    """Aplica formato visual al dashboard de KPIs."""
    header_fill = PatternFill(start_color="2F5597", end_color="2F5597", fill_type="solid")
    header_font = Font(name="Calibri", size=12, bold=True, color="FFFFFF")
    metric_font = Font(name="Calibri", size=11, bold=True, color="1F4E79")
    val_font = Font(name="Calibri", size=11, bold=True)
    thin_border = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )

    ws.row_dimensions[1].height = 26
    for col_idx in range(1, ws.max_column + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")

    for row_idx in range(2, ws.max_row + 1):
        ws.row_dimensions[row_idx].height = 22
        cell_m = ws.cell(row=row_idx, column=1)
        cell_v = ws.cell(row=row_idx, column=2)

        cell_m.font = metric_font
        cell_m.border = thin_border
        cell_m.alignment = Alignment(horizontal="left", vertical="center")

        cell_v.font = val_font
        cell_v.border = thin_border
        cell_v.alignment = Alignment(horizontal="right", vertical="center")

    ws.column_dimensions["A"].width = 38
    ws.column_dimensions["B"].width = 24


def is_bet_recommended(prediction: PredictionResult) -> bool:
    """Evalúa si la predicción generó una recomendación activa de apuesta con Edge positivo."""
    rec = str(prediction.recommended_bet or "").strip()
    if not rec or rec.lower() in ("no bet", "none", "false"):
        return False

    has_positive_ev = (
        (prediction.home_expected_value is not None and prediction.home_expected_value > 0)
        or (prediction.away_expected_value is not None and prediction.away_expected_value > 0)
    )
    has_positive_stake = (
        (prediction.kelly_stake_home is not None and prediction.kelly_stake_home > 0)
        or (prediction.kelly_stake_away is not None and prediction.kelly_stake_away > 0)
    )

    return has_positive_ev or has_positive_stake


def log_bet_prediction(
    prediction: PredictionResult,
    odds: Optional[Odds] = None,
    game_id: Optional[str] = None,
    game_date: Optional[str] = None,
    excel_path: Optional[Union[str, Path]] = None,
    closing_odds: Optional[Odds] = None,
    selection_type: Optional[str] = None,
) -> bool:
    """
    Registra TODO partido analizado en el libro mayor Excel en formato limpio de 7 columnas:
      - Fila 1: Fecha (DD/MM/YYYY), Partido ("PHI - BOS"), Cuota, Probabilidad, Decisión, Marcador "-", Estado "PENDIENTE"
      - Fila 2: Celdas "-", Marcador "Por definir", Estado ""
      - Fila 3: Fila de separación vacía garantizada
    """
    effective_pred = prediction
    if effective_pred.odds is None and odds is not None:
        effective_pred = effective_pred.attach_odds(odds, closing_odds=closing_odds)

    eff_odds = effective_pred.odds or odds
    has_odds = eff_odds is not None
    has_positive_edge = is_bet_recommended(effective_pred)

    decision = "APOSTAR" if (has_odds and has_positive_edge) else "NO APOSTAR"

    home_abbr = canonicalize_team(effective_pred.home_team) or effective_pred.home_team[:3].upper()
    away_abbr = canonicalize_team(effective_pred.away_team) or effective_pred.away_team[:3].upper()
    partido_label = f"{home_abbr} - {away_abbr}"

    # Probabilidad del equipo proyectado a ganar y su cuota
    if effective_pred.home_win_prob >= 0.5:
        winner_abbr = home_abbr
        winner_prob = effective_pred.home_win_prob
        winner_odd = eff_odds.home_decimal if has_odds else None
    else:
        winner_abbr = away_abbr
        winner_prob = effective_pred.away_win_prob
        winner_odd = eff_odds.away_decimal if has_odds else None

    if has_odds and winner_odd is not None:
        cuota_taken = f"{winner_odd:.2f} ({winner_abbr})"
    else:
        cuota_taken = "NO DISPONIBLE"

    prob_label = f"{winner_prob * 100:.1f}% ({winner_abbr})"
    date_str = format_date_ddmmyyyy(game_date)

    # Registro de 1 sola fila por partido en el esquema de 7 columnas
    row_data = {
        "Fecha": date_str,
        "Partido": partido_label,
        "Cuota": cuota_taken,
        "Probabilidad": prob_label,
        "Decision": decision,
        "Marcador_Final": "-",
        "Estado": "PENDIENTE",
    }

    # Cargar bitácora actual (con consolidación de 1 fila)
    df = load_ledger(excel_path)

    if not df.empty and "Partido" in df.columns:
        # Buscar si ya existe este encuentro pendiente
        mask_match = (
            (df["Partido"].astype(str).str.strip().str.upper() == partido_label.upper())
            & (df["Estado"].astype(str).str.strip().str.upper() == "PENDIENTE")
        )
        if mask_match.any():
            idx_match = df[mask_match].index[0]
            for k, v in row_data.items():
                df.at[idx_match, k] = v
        else:
            df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
    else:
        df = pd.DataFrame([row_data])

    save_ledger_workbook(df, excel_path)
    logger.info(
        "game-logged-to-excel",
        fecha=date_str,
        partido=partido_label,
        decision=decision,
    )
    return True


def reset_ledger(excel_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Reinicia el archivo Excel con los nuevos encabezados limpios
    y sin registros de prueba previos.
    """
    target = get_excel_path(excel_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df_empty = pd.DataFrame(columns=COLUMNS_LEDGER)
    save_ledger_workbook(df_empty, excel_path=target)
    logger.info("ledger-reset", path=str(target))
    return target
