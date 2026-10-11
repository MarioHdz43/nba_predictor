# AGENT LOG
Este es el diario de memoria de mi ingeniero de IA.

# CONFIGURACIÓN
- **Modelo Principal:** MiniMax M2.5 Free
- **Memoria:** Engram MCP habilitado
- **Modo:** Single Orchestrator

# REGLAS DE ORO
1. Mantener siempre el contexto de los cambios anteriores.
2. Usar español para las explicaciones e interfaz.
3. Al finalizar una tarea, actualizar este log con lo aprendido.

# HISTORIAL DE TAREAS Y APRENDIZAJES

## [2026-09-21] Auditoría Técnica Completa de Arquitectura y Machine Learning
- **Alcance:** Inspección y diagnóstico integral de código, pipeline E2E, modelos (Heurístico, XGBoost, Híbrido), base de datos SQLite y tests.
- **Hallazgos Críticos:**
  1. *Entrenamiento Sintético:* El modelo XGBoost (`xgboost_model.json`) fue entrenado con 500 muestras sintéticas aleatorias sin datos históricos reales de la NBA, sin validación temporal (Time-Series Split) y evaluado sobre el mismo conjunto sintético.
  2. *Inyección Hardcodeada:* `main.py` y `api.py` instancian `HeuristicEngine()` ignorando la configuración en `config.yaml`.
  3. *Error de ejecución en API:* `api.py` llama a `prediction_engine.is_trained()` sobre `HeuristicEngine`, lo que causa `AttributeError`.
  4. *Dependencia rota:* `requirements.txt` especifica `scikit-learn>=2.0.0` (versión inexistente).
  5. *Tests unitarios desfasados:* `test_entities.py` falla en el cálculo y lógica de EV (`test_attach_odds_home_win` y `test_attach_odds_away_win`) por discrepancia entre valor esperado y win probability.
  6. *Puntos ciegos de analítica:* Ausencia de métricas avanzadas por posesión (Four Factors), reportes de lesiones en tiempo real, proyección de minutos/rotaciones y fatiga de calendario (back-to-back/viajes).

## [2026-09-22] Bloque 1 de Refactorización: Corrección de Bugs Críticos, Deuda Técnica y Dependencias
- **Alcance:** Resolución de dependencias rotas, corrección de inicio en API, propagación de claves primarias en endpoints, sincronización de configuración y corrección de tests de Expected Value.
- **Acciones Realizadas:**
  1. *requirements.txt:* Se actualizó `scikit-learn>=1.4.0,<1.7.0` y se garantizaron dependencias del ecosistema (`lightgbm`, `xgboost`, `pandas`, `numpy`, `scipy`, `fastapi`).
  2. *api.py y repositorios:*
     - Se aseguró la invocación de `is_trained()` con `getattr(prediction_engine, 'is_trained', lambda: True)()` para prevenir `AttributeError` al usar `HeuristicEngine`.
     - Se integraron los campos opcionales `id` y `actual_result` en `PredictionResult` y se mapeó `row["id"]` en `SQLitePredictionRepository._row_to_prediction`, permitiendo que `GET /predictions/recent` devuelva la clave primaria real de SQLite para correlacionar con `POST /result`.
  3. *domain/engines/hybrid_engine.py & infrastructure/config.py:* Sincronización del parámetro de peso a `ml_weight` con alias retrocompatible `hybrid_ml_weight` en `Config` y porcentaje dinámico en factores clave.
  4. *domain/engines/xgboost_engine.py:* Eliminación de código muerto duplicado tras la sentencia `return` en el método `train()`.
  5. *tests/test_entities.py:*
     - Corrección matemática de los tests de EV (`test_attach_odds_home_win` y `test_attach_odds_away_win`) conforme a la fórmula real \(EV = P \times (odds - 1) - (1 - P) \times 1\).
     - Corrección de `test_is_stale_with_old_data` usando `timedelta` para garantizar robustez ante lecturas temporales inmediatas.
- **Resultado:** 100% de la suite de tests unitarios e integración pasando con éxito (34/34 tests superados).

## [2026-09-22] Bloque 2 de Rediseño: Pipeline de Ingesta y Persistencia Histórica Real (nba_api)
- **Alcance:** Creación del esquema relacional en SQLite para datos históricos de partidos y boxscores, implementación de repositorio transaccional por lotes, script CLI de ingesta incremental con rate limiting/backoff y suite de tests unitarios.
- **Acciones Realizadas:**
  1. *infrastructure/database.py:*
     - Creación de tablas `historical_games` (con `game_id` PK, season, game_date, home/away IDs y scores) y `historical_team_boxscores` (con PK compuesta `(game_id, team_id)`, FK a `historical_games` con ON DELETE CASCADE, y 15 métricas de boxscore).
     - Creación de 4 índices estratégicos (`idx_historical_games_date`, `idx_historical_games_season`, `idx_historical_games_teams`, `idx_historical_boxscores_team`).
     - Habilitación de `PRAGMA foreign_keys = ON;` y soporte de rutas personalizadas en `Database`.
  2. *infrastructure/historical_repository.py:*
     - Implementación de `SQLiteHistoricalRepository` con inserción transaccional atómica por lotes (`INSERT OR IGNORE`) y consulta de IDs existentes para ingesta incremental.
  3. *scripts/fetch_historical_data.py:*
     - Script ejecutable con `argparse` (`--seasons`, `--season-types`, `--db-path`, `--delay-min`, `--delay-max`, `--max-retries`, `--dry-run`).
     - Parser robusto que empareja local y visitante mediante `MATCHUP` ('vs.' vs '@').
     - Tolerancia a fallos: retardos de cortesía (0.6 - 1.2s) y retroceso exponencial ante bloqueos o errores de red.
     - Registro de progreso visual con soporte opcional para `tqdm` y logs estructurados.
  4. *tests/test_historical_ingest.py:*
     - 8 tests unitarios cubriendo creación de esquema, restricción de integridad foránea, parsing de DataFrames, idempotencia en persistencia, pipeline completo simulado y reintentos ante fallos de red.
- **Resultado:** 100% de la suite global pasando con éxito (42/42 tests superados). Verificación exitosa en vivo con `stats.nba.com` en modo `--dry-run` procesando 1,230 partidos y 2,460 boxscores de la temporada 2023-24.

## [2026-09-22] Bloque 3 de Rediseño: Feature Engineering Matemático para Baloncesto Profesional
- **Alcance:** Implementación de métricas canónicas avanzadas por posesión (Dean Oliver Four Factors, ORtg, DRtg, Pace), pipeline temporal EWMA con Zero Data Leakage (.shift(1)), variables de fatiga/descanso (rest_days, is_b2b, rest_advantage) y generador de dataset de entrenamiento serializable en Parquet/CSV.
- **Acciones Realizadas:**
  1. *requirements.txt:* Incorporación de `pyarrow>=15.0.0` para serialización de alta eficiencia.
  2. *domain/analytics/advanced_metrics.py:*
     - Cálculo vectorizado y protegido ante divisiones por cero de: Posesiones ($Poss$), $ORtg$, $DRtg$, $NRtg$, $Pace$ ajustado a 48 min, y los Cuatro Factores ($eFG\%$, $TOV\%$, $ORB\%$, $DRB\%$, $FT\text{ Rate}$) ofensivos y permitidos al oponente.
  3. *domain/analytics/feature_builder.py:*
     - Emparejamiento de boxscores propios y del oponente por partido.
     - Ponderación exponencial temporal (EWMA con half-life de 12 partidos) aplicando estrictamente `.shift(1)` para evitar fuga de información hacia el partido evaluado.
     - Extracción de variables de fatiga: `rest_days`, `is_b2b` y `rest_advantage`.
     - Fusión de enfrentamiento (Home vs Away) produciendo métricas diferenciales ($\Delta = Home - Away$) y targets (`point_spread`, `home_win`, `total_points`).
  4. *scripts/build_training_dataset.py:*
     - CLI ejecutable con `argparse` para procesar SQLite y exportar `data/processed_features.parquet` y `data/processed_features.csv`, con análisis de correlaciones frente al spread.
  5. *tests/test_feature_engineering.py:*
     - 8 tests unitarios exhaustivos validando precisión matemática de fórmulas, cálculo de descanso/B2B, demostración formal de ausencia de Data Leakage en EWMA y ensamblado de variables diferenciales.
- **Resultado:** 100% de la suite global pasando con éxito (50/50 tests superados).

## [2026-09-22] Bloque 4 de Rediseño: Modelado Predictivo de Point Spread, Walk-Forward, Calibración y Criterio de Kelly
- **Alcance:** Implementación de modelado continuo sobre margen de victoria (Point Spread), validación temporal continua Walk-Forward (Expanding Window), calibración probabilística vía CDF Normal ($\Phi$), gestión financiera con Criterio de Kelly Fraccional (0.25 / Quarter Kelly), nuevo motor `MLEngine` y suite de tests.
- **Acciones Realizadas:**
  1. *domain/entities.py:*
     - Incorporación de `kelly_stake_home` y `kelly_stake_away` en `PredictionResult`.
     - Implementación de Quarter Kelly (0.25): \(f^* = \max(0, \frac{P \times b - (1 - P)}{b}) \times 0.25\) donde \(b = \text{odds} - 1\).
  2. *scripts/train_model.py:*
     - Pipeline cuantitativo que sustituye a `train_xgboost.py`.
     - Target continuo `point_spread = home_score - away_score`.
     - Modelos LightGBM Regressor (fallback a XGBoost) con regularización L1/L2 y control de profundidad.
     - Walk-Forward Validation cronológico sin filtración temporal.
     - Métricas evaluadas: MAE y RMSE de margen; Accuracy, Brier Score y Log-Loss en clasificación.
     - Calibración de probabilidad de victoria: \(P(\text{Home Win}) = \Phi(\frac{\mu_{MOV}}{\sigma_{MOV}})\).
     - Serialización de modelo y metadatos en `models/nba_spread_model.joblib` y `models/model_metadata.json`.
  3. *domain/engines/ml_engine.py & domain/engines/xgboost_engine.py:*
     - Creación de `MLEngine` con inferencia sobre margen y CDF Normal.
     - Adaptación retrocompatible de `XGBoostEngine` con prioridad para modelos `.joblib` de Point Spread.
  4. *tests/test_model_pipeline.py:*
     - 5 tests unitarios verificando cálculo de EV y Quarter Kelly, transformación CDF Normal, estructura Walk-Forward y pipeline end-to-end de entrenamiento e inferencia.
- **Resultado:** 100% de la suite global pasando con éxito (55/55 tests superados).

## [2026-09-24] Bloque 5 de Rediseño: Parser de Lesiones, Proyección de Rotaciones (RAPM) y Contexto Geográfico
- **Alcance:** Implementación del parser probabilístico para el Official NBA Injury Report con probabilidades bayesianas a priori, proyección dinámica de rotaciones preservando 240 minutos con degradación RAPM/EPM, modelado de variables contextuales geográficas (Haversine, husos horarios cruzados, altitud de Denver/Utah y fatiga circadiana) e integración en el vector de características.
- **Acciones Realizadas:**
  1. *infrastructure/scrapers/injury_parser.py:*
     - Modelado de `InjuryStatus` (`OUT`, `DOUBTFUL`, `QUESTIONABLE`, `PROBABLE`, `AVAILABLE`) y asignación cuantitativa bayesiana a priori (\(P(\text{Play}) = [0.0, 0.10, 0.50, 0.90, 1.00]\)).
     - Parsers flexibles para formatos estructurados (JSON/dict), texto tabular oficial (PDF/HTML export) y agrupaciones de plantilla por equipo.
  2. *domain/analytics/roster_impact.py:*
     - Implementación canónica de `RosterImpactCalculator` / `RosterProjectionEngine` garantizando la estricta conservación de 240 minutos reglamentarios.
     - Formulación matemática:
       $$\text{Team\_ORtg}_{\text{proj}} = \text{League\_Avg} + \sum_{i} \left(\frac{\text{Min}_i}{48}\right) \times \text{O\_RAPM}_i$$
       $$\text{Team\_DRtg}_{\text{proj}} = \text{League\_Avg} + \sum_{i} \left(\frac{\text{Min}_i}{48}\right) \times \text{D\_RAPM}_i$$
       $$\text{Team\_Net\_Rating}_{\text{proj}} = \text{Team\_ORtg}_{\text{proj}} - \text{Team\_DRtg}_{\text{proj}}$$
     - Redistribución de minutos ausentes hacia jugadores elegibles del banquillo y nivel de reemplazo (\(\text{O\_RAPM} = -1.25, \text{D\_RAPM} = +1.25\)).
  3. *domain/analytics/travel_context.py:*
     - Georreferenciación de las 30 franquicias de la NBA con latitud, longitud, huso horario UTC y elevación en pies.
     - Cálculo de distancia de vuelo geodésica mediante Haversine (\(R = 3958.8\) mi) y desfase de husos horarios.
     - Detección de ventaja fisiológica por altitud para estadios a \(> 4,000\) pies (Denver Nuggets y Utah Jazz).
     - Modelado compuesto de índice de fatiga circadiana y de viaje.
  4. *domain/analytics/feature_builder.py & scripts/train_model.py:*
     - Integración automática en `build_matchup_dataset` de `is_altitude_advantage`, `travel_miles_away`, `time_zones_crossed_away` y `travel_fatigue_away`.
     - Inclusión de las variables contextuales en `FEATURE_COLUMNS` del pipeline de entrenamiento ML.
  5. *tests/test_roster_and_context.py:*
     - 14 tests unitarios exhaustivos validando probabilidades bayesianas, parseo de reportes de lesiones, conservación de 240 minutos, degradación de rating por baja de estrellas, precisión de Haversine (~2600 mi LAL-BOS), husos horarios cruzados y generación de dataset de matchups.
- **Resultado:** 100% de la suite global pasando con éxito (69/69 tests superados).

## [2026-09-24] Bloque 6 de Rediseño: Calibración Beta, Método de Shin y Observabilidad de Mercado (CLV)
- **Alcance:** Culminación del 100% de la arquitectura técnica cuantitativa institucional. Implementación de Calibración Beta paramétrica (Kull et al., 2017) con minimización de log-loss y monotonicidad estricta, extracción de probabilidades implícitas verdaderas No-Vig mediante el Método de Shin (Shin, 1993; Jevons Smith, 2006) eliminando el sesgo favorito-underdog, cálculo formal de Closing Line Value (CLV) y observabilidad de cuotas de mercado.
- **Acciones Realizadas:**
  1. *domain/analytics/beta_calibration.py:*
     - Formulación paramétrica: \(\ln(q / (1 - q)) = a \ln(p) - b \ln(1 - p) + c\) con \(a, b \ge 0\).
     - Ajuste mediante `scipy.optimize.minimize` (`L-BFGS-B`) minimizando la entropía cruzada binaria (binary cross-entropy / log-loss) con protección numérica por clipping.
     - Métodos `fit(y_prob, y_true)` y `predict_proba(y_prob)`, con serialización `to_dict()` y `from_dict()`.
  2. *domain/analytics/shin_odds.py:*
     - Resolución numérica del parámetro latente \(z\) de apostadores informados mediante `scipy.optimize.root_scalar` (`brentq` en \([0.0, 0.499]\)).
     - Cálculo de la probabilidad verdadera \(p_i(z) = \frac{\sqrt{z^2 + 4(1 - z)\pi_i^2 / \beta} - z}{2(1 - z)}\) con suma normalizada a 1.0.
     - Funciones `calculate_shin_probabilities`, `shin_two_way_probabilities` y `calculate_market_vigorish`.
  3. *domain/analytics/clv.py & infrastructure/odds/odds_tracker.py:*
     - Implementación de \(CLV = \text{Odds}_{\text{tomadas}} \times \pi_{\text{cierre, no-vig}} - 1.0\).
     - Diferencial de margen de puntos (Point Spread CLV): \(\text{Spread}_{\text{tomado}} - \text{Spread}_{\text{cierre}}\).
     - Clases `CLVPerformanceTracker`, `CLVBetRecord`, `OddsSnapshot` y `MarketOddsTracker` para captura temporal de apertura, snapshots en vivo y cierre de referencia institucional (Pinnacle).
  4. *domain/entities.py & domain/engines/ml_engine.py & scripts/train_model.py:*
     - En `PredictionResult`: incorporación de campos `home_no_vig_prob`, `away_no_vig_prob`, `market_vigorish` y `expected_clv`.
     - En `attach_odds()`: desviggeado con Shin y cómputo de EV, Quarter Kelly (0.25) y CLV si se proporcionan cuotas de cierre.
     - En `MLEngine`: integración del artefacto `_beta_calibrator` en la inferencia sobre la CDF normal.
     - En `scripts/train_model.py`: ajuste del calibrador Beta sobre las probabilidades fuera de muestra de Walk-Forward y serialización en `nba_spread_model.joblib`.
  5. *tests/test_shin_and_market.py:*
     - 17 tests unitarios exhaustivos validando el caso canónico de Shin \([2.6, 2.4, 4.3]\), mercados simétricos NBA, corrección del sesgo favorito-underdog, monotonicidad y corrección de sobreconfianza en Beta Calibration, cálculo de CLV (+/- y spread) y ciclo completo en `attach_odds`.
- **Resultado:** 100% de la suite global pasando con éxito (86/86 tests superados). Arquitectura institucional completada al 100%.

## [2026-10-06] Subsistema de Auditoría y Seguimiento de Apuestas (Bet Tracking & Settlement Ledger en Excel)
- **Alcance:** Implementación de un subsistema cuantitativo automatizado de Auditoría, Seguimiento de Apuestas (Bet Tracking) y Liquidación de Resultados (Settlement Ledger) en formato Excel (.xlsx) en `C:\Users\mblad\Documents\Codigoes\nba_bet_tracking.xlsx`.
- **Acciones Realizadas:**
  1. *infrastructure/reporting/bet_tracker_excel.py:*
     - Gestor institucional con soporte para manipulación segura sin corrupción de registros previos mediante `pandas` y `openpyxl`.
     - Estructura bi-hoja:
       * `Registro_Apuestas`: 17 columnas canónicas (`ID_Partido`, `Fecha_Partido`, `Local`, `Visitante`, `Seleccion_Apostada`, `Margen_Proyectado_Local`, `Probabilidad_Calibrada`, `Cuota_Tomada`, `No_Vig_Prob`, `EV_Pct`, `Kelly_Stake_Pct`, `Resultado_Marcador`, `Margen_Real`, `Estado_Apuesta`, `Unidades_Profit_Loss`, `CLV_Obtenido`, `Fecha_Registro`).
       * `Resumen_Metricas`: Dashboard cuantitativo agregado con métricas clave (Total Apuestas, Pendientes, Liquidadas, Ganadas, Perdidas, Push, Win Rate %, Total Unidades Apostadas, P&L Neto en Unidades, ROI/Yield % y CLV Promedio).
     - Formato visual corporativo: cabeceras con relleno Navy Blue institucional (`#1F4E79`), auto-ajuste dinámico de ancho de columnas y formato condicional cromático para estados (`GANADA` en verde suave, `PERDIDA` en salmón suave, `PENDIENTE` en ámbar suave y `PUSH` en gris neutro).
     - Regla estricta de filtrado: `log_bet_prediction` ignora de forma determinista pronósticos con `No bet`, sin cuotas o con EV no positivo, evitando ensuciar la bitácora.
  2. *infrastructure/reporting/settle_bets.py:*
     - Motor de liquidación matemática `settle_pending_bets(db_path, excel_path)` que lee las filas `PENDIENTE` y cruza contra `historical_games` y `predictions` en SQLite (o proveedores externos/fallback).
     - Algoritmo de emparejamiento tolerante a formatos de ID (con y sin ceros a la izquierda) y resolución automática de nombres/abreviaturas de las 30 franquicias de la NBA.
     - Liquidación matemática rigurosa:
       * Ganada: \(+ \text{Stake} \times (\text{Cuota} - 1)\).
       * Perdida: \(- \text{Stake}\).
       * Push / Empate: \(0.0\).
       * Margen real vs proyectado: \(\text{Margen Real} = \text{Puntos Local} - \text{Puntos Visitante}\).
       * Actualización de CLV ex-post si se suministran cuotas de cierre.
  3. *presentation/cli.py & main.py:*
     - Integración transparente en `CLIController`: registro automático en el Excel tras predicciones con recomendación positiva de apuesta.
     - Parámetros CLI añadidos en `main.py`: `--excel-path`, `--settle-bets` y `--db-path`.
  4. *scripts/update_bet_results.py:*
     - Script CLI ejecutable para que el operador quant liquide en cualquier momento sus apuestas pendientes contra la base de datos de partidos finalizados con visualización de tabla en consola.
  5. *domain/entities.py & domain/engines/ml_engine.py & requirements.txt:*
     - Incorporación del campo `projected_margin` en la entidad `PredictionResult` y preservación en `attach_odds`.
     - Adición de dependencia `openpyxl>=3.1.0` y configuración en `config.yaml`.
  6. *tests/test_bet_tracker_excel.py:*
     - 12 tests unitarios e integrados cubriendo registro condicional (filtrado de no bet y EV negativo), persistencia y estilos openpyxl, cálculo de KPIs agregados, liquidación matemática de apuestas ganadas, perdidas, empate y cruce end-to-end con SQLite.
- **Resultado:** 100% de la suite global pasando con éxito (98/98 tests superados).

## [2026-10-10] Integración de Cuotas en Vivo (The Odds API) y Rediseño Bi-Fila del Libro Mayor en Excel
- **Alcance:** Conexión con The Odds API (`v4/sports/basketball_nba/odds`), configuración segura de API Key institucional, extracción automática de cuotas decimales con jerarquía de bookmakers (Pinnacle, DraftKings), seguimiento de cuota de solicitudes (`x-requests-remaining`), reestructuración visual del Excel a bloques de 2 filas por encuentro (Pre y Post partido) con separador, y validación en vivo.
- **Acciones Realizadas:**
  1. *infrastructure/config.py & config.yaml:*
     - Registro de clave en `config.yaml` (`odds_api_key: "693fbbf0a05943b5d95e17b8d63c00d1"` y bloque `odds_api`).
     - Propiedad `odds_api_key` con resolución multinivel (variable de entorno `ODDS_API_KEY` -> `odds_api_key` raíz -> `odds_api.api_key`).
  2. *infrastructure/odds/live_odds_client.py:*
     - Cliente `LiveOddsClient` con mapeo de alias de las 30 franquicias de la NBA (`canonicalize_team`), caché en memoria y selección preferencial de casas de apuestas (Pinnacle, DraftKings, etc.).
     - Captura de métricas de cuota desde cabeceras HTTP (`x-requests-remaining` y `x-requests-used`).
     - Método de diagnóstico `check_connection()` con validación limpia de estados vacíos (fuera de temporada o sin líneas publicadas).
  3. *infrastructure/reporting/bet_tracker_excel.py & infrastructure/reporting/settle_bets.py:*
     - Formato de 2 filas por encuentro: Fila 1 (Fecha, Local, Visitante, Pronóstico proyectado, Cuota, Decisión APOSTAR/NO APOSTAR, Estado PENDIENTE), Fila 2 (Puntos Local/Visitante, Marcador Oficial, Margen Real, Estado ACERTÓ/FALLÓ/PUSH, Unidades P&L), Fila 3 (Separador en blanco).
     - Corrección de tipado en pandas/pyarrow (`.astype(object)`) para asignación sin errores de tipos mixtos (texto y punto flotante).
  4. *scripts/verify_live_odds.py:*
     - Herramienta de consola para auditoría inmediata de la conexión en vivo, verificación HTTP 200, visualización de cuota restante e inspección de partidos disponibles.
  5. *tests/test_bet_tracker_excel.py:*
     - 16 tests cubriendo extracción de cuotas en vivo, mock de conexión y cuota, manejo seguro de listas vacías, registro bi-fila, liquidación de aciertos/fallos y cruce con SQLite.
- **Resultado:** 100% de la suite global pasando con éxito (102/102 tests superados). Verificación en vivo exitosa contra The Odds API con 46 partidos detectados y 494 consultas restantes.

## [2026-10-10] Actualización de CLI: Menú Interactivo en Vivo, Estado Temporal y Corrección de Localía
- **Alcance:** Transformación integral de la experiencia de usuario en consola (`presentation/cli.py`) y del cliente de cuotas (`infrastructure/odds/live_odds_client.py`) incorporando extracción y conversión de horarios UTC a hora local (America/Bogota, UTC-5), etiquetado dinámico de estado pre-partido (`[DISPONIBLE]` vs `[EN JUGADA / CERRADO]`), menú interactivo visual en el CLI, y detección y corrección estricta de localía oficial para preservar la integridad matemática del modelo.
- **Acciones Realizadas:**
  1. *infrastructure/odds/live_odds_client.py:*
     - Función `parse_commence_time()` que convierte timestamps ISO 8601 UTC a la zona horaria local (`America/Bogota`), determina `is_pregame` comparando contra la hora actual del sistema, y genera etiquetas legibles (`Hoy HH:MM PM`, `Mañana HH:MM PM`, `[DISPONIBLE - HH:MM PM]` y `[EN JUGADA / CERRADO]`).
     - Método `list_available_games()` que devuelve la lista de partidos ordenada cronológicamente con cuotas de consenso/bookmaker preferente (Pinnacle/DraftKings).
     - Método `find_market_match(team_a, team_b)` que evalúa emparejamientos y detecta inversiones de localía (`is_inverted=True`), devolviendo los nombres de franquicias oficiales `official_home` y `official_away`.
     - Actualización de `get_game_odds()` y exportación de funciones de acceso directo en `infrastructure/odds/__init__.py`.
  2. *presentation/cli.py:*
     - Incorporación de `format_game_menu_entry()` y renderizado del menú interactivo numerado de partidos en vivo al iniciar `main.py`.
     - Selección con un solo toque (tecla numérica): carga automática de equipos en el orden oficial, cuotas en vivo, ejecución matemática de la predicción y persistencia en `nba_bet_tracking.xlsx`.
     - Manejo de advertencia para partidos ya iniciados (`[ADVERTENCIA] Este partido ya ha comenzado...`).
     - Detección y corrección automática de localía en ingreso manual o flags CLI: emite `[Mercado] Ajustando localía oficial: Local = Real, Visitante = Real` e intercambia variables para garantizar que la ventaja de localía (`is_home=True`) se atribuya al equipo anfitrión verdadero.
     - Robustecimiento de `resolve_team()` con matching canónico de alias para todas las 30 franquicias.
  3. *tests/test_bet_tracker_excel.py:*
     - 6 nuevos tests unitarios verificando `parse_commence_time` (pre-partido, partidos cerrados y entradas nulas), `list_available_games` (estructura y orden cronológico), detección directa e invertida en `find_market_match`, y formateo de menú con `format_game_menu_entry`.
- **Resultado:** 100% de la suite global pasando con éxito (108/108 tests superados). Verificación end-to-end completada exitosamente en consola con selección interactiva y corrección en vivo de localía.

## [2026-10-10] Selector Dinámico por Fechas y Procesamiento por Lotes (Batch Predictions)
- **Alcance:** Mejora integral del selector de partidos en `presentation/cli.py` y `infrastructure/odds/live_odds_client.py` con filtro dinámico por fechas relativas (`HOY`, `MAÑANA`, `PRÓXIMOS`, `TODOS`), agrupación por fecha local (America/Bogota, UTC-5), flujo de pizarra interactivo con manejo elegante de pizarras vacías y navegación rápida (`[H]`, `[M]`, `[T]`), y soporte completo de selección y ejecución por lotes (Batch Predictions) con registro en el libro mayor Excel y resumen consolidado.
- **Acciones Realizadas:**
  1. *infrastructure/odds/live_odds_client.py & infrastructure/odds/__init__.py:*
     - Categorización temporal relativa en `list_available_games()` agregando el campo `date_category` (`HOY`, `MAÑANA`, `PRÓXIMOS`, `AYER`, `PASADOS`).
     - Soporte para tiempo de referencia determinista `reference_now_utc` para pruebas unitarias.
     - Implementación del método `get_games_by_filter(filter_type, target_timezone, only_pregame, reference_now_utc)` compatible con `'today'`/`'hoy'`, `'tomorrow'`/`'manana'`/`'mañana'`, `'upcoming'`/`'proximos'`/`'próximos'`, `'all'`/`'todos'` y fechas exactas `'YYYY-MM-DD'` o `'DD/MM'`.
     - Implementación del método `group_games_by_date()` que retorna un diccionario estructurado `{"HOY": [...], "MAÑANA": [...], "PRÓXIMOS": [...]}`.
     - Funciones de acceso directo a nivel de módulo `get_games_by_filter()` y `group_games_by_date()`, exportadas en `__init__.py`.
  2. *presentation/cli.py:*
     - Implementación de `parse_batch_selection(user_input, max_items)` soportando:
       * Modalidad individual: `'1'`
       * Modalidad lista separada por comas: `'1, 3, 5'` o `'1,3,5'` (con deduplicación preservando orden)
       * Modalidad lote completo: `'todos'`, `'all'` o `'*'`
       * Comandos rápidos de navegación de vista: `'[M]'` (Mañana), `'[T]'` (Todos), `'[H]'` (Hoy)
       * Modo manual: `'0'`
       * Manejo de errores por fuera de rango y entradas no reconocidas.
     - Flujo de pizarra en `CLIController.run`:
       * Inicia por defecto consultando partidos de `HOY`.
       * Si no hay partidos hoy: Muestra mensaje `"No hay partidos de la NBA programados para hoy."` y menú rápido: `[1] Ver partidos de Mañana | [2] Ver todos los próximos partidos programados | [0] Salir/Manual`.
       * Títulos dinámicos según la vista actual (`PARTIDOS DE HOY...`, `PARTIDOS DE MAÑANA...`, `TODOS LOS PRÓXIMOS PARTIDOS...`).
       * Comandos rápidos visibles en pie de menú según la vista activa.
     - Método `execute_batch(matches, season, force_fresh, is_batch)`:
       * Procesa secuencialmente cada partido seleccionado.
       * Muestra predicción estadística, probabilidades, factores clave y valor de apuesta.
       * Registra automáticamente en `nba_bet_tracking.xlsx` marcando decisión `"APOSTAR"` solo para pronósticos con Edge positivo (+EV y Quarter Kelly).
       * Al finalizar el lote, imprime el resumen consolidado institucional:
         `[Lote Completo] X partidos analizados | Y apuestas recomendadas registradas en Excel.`
  3. *tests/test_bet_tracker_excel.py:*
     - 8 nuevos tests unitarios en la clase `TestBatchSelectionAndDateFiltering`:
       * `test_parse_batch_selection_single`: Selección unitaria `1` y `4`.
       * `test_parse_batch_selection_comma_list`: Listas `1, 3, 5` y deduplicación `2, 1, 2`.
       * `test_parse_batch_selection_todos_keyword`: Palabras clave `todos`, `TODOS` y `all`.
       * `test_parse_batch_selection_navigation_commands`: Comandos `m`, `M`, `t`, `T`, `h`, `H`, `0`.
       * `test_parse_batch_selection_errors`: Detección de out_of_range, invalid y empty.
       * `test_get_games_by_filter_categories`: Filtros `today`, `tomorrow`, `upcoming` y `all` contra mock de API.
       * `test_group_games_by_date`: Diccionario de fechas `HOY`, `MAÑANA`, `PRÓXIMOS`.
       * `test_cli_execute_batch_predictions_summary`: Ejecución por lote E2E con registro en Excel, conteo de apuestas recomendadas y resumen consolidado.
- **Resultado:** 100% de la suite global pasando con éxito (116/116 tests superados). Experiencia CLI institucional completada al 100%.

## [2026-10-10] Corrección de Logging en CLI y Sincronización de Dependencias (openpyxl)
- **Alcance:** Resolución del bug de incompatibilidad en llamadas al logger estándar de Python en `presentation/cli.py` y aseguramiento del registro formal de dependencias (`openpyxl`, `requests`, `scipy`, `numpy`, `pyarrow`) en `pyproject.toml` y `requirements.txt`.
- **Acciones Realizadas:**
  1. *presentation/cli.py:*
     - Corrección de llamadas con kwargs arbitrarios incompatibles con `logging.Logger`:
       * `logger.warning("excel-logging-error", error=str(exc))` sustituido por `logger.warning(f"excel-logging-error: {exc}")`.
       * Corrección de llamadas `logger.debug("...", error=str(exc))` en manejo de cuotas, filtros y localía por strings formateados `logger.debug(f"...: {exc}")`.
  2. *pyproject.toml & requirements.txt:*
     - Inclusión formal de `openpyxl>=3.1.0` y `requests>=2.28.0` en `pyproject.toml` bajo `project.dependencies`, garantizando paridad con `requirements.txt`.
  3. *Verificación:*
     - 30/30 tests de `tests/test_bet_tracker_excel.py` superados al 100%.
     - 116/116 tests de la suite global superados al 100%.
- **Resultado:** 100% de la suite global pasando con éxito (116/116 tests superados). Zero warnings ni errores de logging.


## [2026-10-10] Rediseño Ejecutivo del Libro Mayor en Excel (8 Columnas) y Auto-Liquidación Silenciosa al Iniciar
- **Alcance:** Transformación del libro mayor en `infrastructure/reporting/bet_tracker_excel.py` a un formato ejecutivo limpio de 8 columnas, implementación del motor de auto-liquidación silenciosa al arranque (`main.py` y `presentation/cli.py`) con consulta en vivo a stats.nba.com (`ScoreboardV2`), eliminación de redundancias visuales y reseteo del archivo Excel en vivo.
- **Acciones Realizadas:**
  1. *infrastructure/reporting/bet_tracker_excel.py:*
     - Rediseño al esquema ejecutivo canónico de 8 columnas (`COLUMNS_LEDGER`):
       1. `Partido` (ej. `"BOS - MIA"`)
       2. `Local` (ej. `"BOS"`)
       3. `Visitante` (ej. `"MIA"`)
       4. `Cuota` (ej. `"1.85"` o `"NO DISPONIBLE"`)
       5. `Probabilidad` (ej. `"73.4% (BOS)"`)
       6. `Decision` (`"APOSTAR"` verde institucional `#E2EFDA` / `"NO APOSTAR"` gris neutro `#F2F2F2`)
       7. `Marcador_Final` (Guión `"-"` en Fila 1; `"Por definir"` en Fila 2 mientras esté pendiente, y marcador oficial real `"112 - 104"` al liquidar)
       8. `Estado` (Celda verticalmente combinada `H{r}:H{r+1}`: `"PENDIENTE"` ámbar `#FFF2CC`, `"ACERTÓ"` verde `#E2EFDA`, `"FALLÓ"` salmón `#FCE4D6`, `"PUSH"` gris `#F2F2F2`)
     - Estructura bi-fila estricta con fila separadora completamente vacía (3 filas por bloque de encuentro).
     - Función `reset_ledger()` para reiniciar la bitácora limpia con el nuevo esquema.
     - Métricas ejecutivas adaptadas en `calculate_metrics_summary()` evaluando Accuracy del modelo y ROI/Yield de apuestas.
  2. *infrastructure/reporting/settle_bets.py:*
     - Adaptación completa de `settle_pending_bets()` para operar sobre el esquema de 8 columnas: emparejamiento por abreviaturas canónicas, resolución de ganadores por `Probabilidad`, actualización de `Marcador_Final` en Fila 2 y de `Estado` en Fila 1.
     - Implementación de `auto_reconcile_pending_bets(excel_path, db_path, fallback_results)`:
       * Chequeo instantáneo O(1): si no existen filas con `Estado == "PENDIENTE"`, finaliza de inmediato sin realizar llamadas a APIs ni bases de datos.
       * Si existen pendientes: consulta los marcadores finales de la NBA vía `fetch_scoreboard_results()` en `infrastructure/nba_api_client.py`, liquida los encuentros finalizados y emite mensaje en consola:
         `[Auto-Liquidación] X partidos finalizados fueron liquidados en el Excel.`
  3. *main.py & presentation/cli.py:*
     - Invocación automática de `auto_reconcile_pending_bets()` al arranque de `main.py` y en el inicio del método `CLIController.run()`.
  4. *Limpieza de Bitácora en Vivo:*
     - Ejecución de `reset_ledger()` sobre `C:\Users\mblad\Documents\Codigoes\nba_bet_tracking.xlsx`, eliminando registros sintéticos previos e inicializando la plantilla de 8 columnas.
  5. *tests/test_bet_tracker_excel.py:*
     - 33 tests unitarios e integrados pasando al 100%: registro bi-fila de 8 columnas, liquidación de aciertos, fallos, push y no-apostar, cálculo de KPIs agregados, persistencia y combinación de celdas openpyxl, cruce con SQLite y auto-reconciliación silenciosa con mock de API.
- **Resultado:** 100% de la suite global pasando con éxito (119/119 tests superados).


## [2026-10-10] Actualización de Esquema Ejecutivo del Libro Mayor a 7 Columnas y Formato de Fecha Explícito
- **Alcance:** Modificación de la hoja `Registro_Apuestas` en `infrastructure/reporting/bet_tracker_excel.py` y del motor de liquidación en `infrastructure/reporting/settle_bets.py`, eliminando las columnas redundantes `Local` y `Visitante`, reubicando la `Fecha` en formato estricto `DD/MM/YYYY` al inicio, ajustando las fórmulas de auto-liquidación y combinaciones de celda de Estado a la columna 7, y regenerando el archivo Excel en vivo `C:\Users\mblad\Documents\Codigoes\nba_bet_tracking.xlsx`.
- **Acciones Realizadas:**
  1. *infrastructure/reporting/bet_tracker_excel.py:*
     - Redefinición de `COLUMNS_LEDGER` a 7 columnas principales:
       1. `Fecha` (formato estricto "DD/MM/YYYY", ej. "10/10/2026")
       2. `Partido` (identificador limpio en mayúsculas, ej. "PHI - BOS")
       3. `Cuota` (cuota tomada o "NO DISPONIBLE")
       4. `Probabilidad` (porcentaje proyectado con equipo asignado, ej. "81.4% (BOS)")
       5. `Decision` ("APOSTAR" verde `#E2EFDA` / "NO APOSTAR" gris `#F2F2F2`)
       6. `Marcador_Final` ("-" en Fila 1; marcador real en Fila 2 al liquidar, o "Por definir")
       7. `Estado` (celda verticalmente combinada `G{r}:G{r+1}` para el bloque: "PENDIENTE", "ACERTÓ", "FALLÓ" o "PUSH")
     - Función `format_date_ddmmyyyy()` para parseo tolerante y conversión uniforme a "DD/MM/YYYY".
     - Estructura bi-fila estricta: Fila 1 con datos, Fila 2 con guiones en columnas 1 a 5 y marcador en columna 6, Fila 3 en blanco.
     - Adaptación de `_style_ledger_sheet()` para las 7 columnas y combinación de celdas en columna 7 (`G`).
     - Cálculo de KPIs en `calculate_metrics_summary()` utilizando `Partido != "-"` como selector de filas maestras.
  2. *infrastructure/reporting/settle_bets.py:*
     - Adaptación de lectura en `settle_pending_bets()`: extracción de equipos anfitrión y visitante directamente desde `Partido` (separador ` - `), desviggeado, cruce contra SQLite y marcadores en vivo de `stats.nba.com`.
     - Actualización de `mask_pending` sobre `Partido != "-"` y `Estado == "PENDIENTE"`.
  3. *Regeneración en Vivo:*
     - Ejecución de `reset_ledger()` en `C:\Users\mblad\Documents\Codigoes\nba_bet_tracking.xlsx`, verificando en disco las 7 columnas: `['Fecha', 'Partido', 'Cuota', 'Probabilidad', 'Decision', 'Marcador_Final', 'Estado']`.
  4. *tests/test_bet_tracker_excel.py:*
     - Actualización completa de la suite para verificar las 7 columnas, formato explícito de fecha "DD/MM/YYYY", comprobación de guiones en Fila 2 y persistencia en openpyxl con `max_column == 7`.
- **Resultado:** 100% de la suite global pasando con éxito (119/119 tests superados).

## [2026-10-10] Sincronización Estricta de Zona Horaria Local y Corrección de Fechas en Excel (America/Bogota)
- **Alcance:** Resolución del desfase temporal provocado por el estándar UTC en el registro de fechas y horas en el libro mayor de Excel (`infrastructure/reporting/bet_tracker_excel.py`), estandarización de la métrica `Última Actualización` al formato 24h local (`DD/MM/YYYY HH:MM:SS`) y prevención del salto prematuro de fecha en partidos jugados después de las 7:00 PM local.
- **Acciones Realizadas:**
  1. *infrastructure/reporting/bet_tracker_excel.py:*
     - Implementación de `get_user_timezone()` para resolución institucional de `zoneinfo.ZoneInfo("America/Bogota")` (con fallback dinámico a zona local del sistema o UTC-5).
     - Implementación de `get_local_now()` devolviendo `datetime` timezone-aware en la hora local del usuario.
     - Implementación de `format_local_timestamp(dt)` garantizando el formato uniforme `"DD/MM/YYYY HH:MM:SS"` (ej. `"10/10/2026 21:04:26"`).
     - Actualización de `format_date_ddmmyyyy(date_val)`: conversión explícita de strings ISO 8601 con UTC (`"T"`, `"Z"`, `"+00:00"`) a `America/Bogota` previo a la extracción de la fecha. Esto asegura que partidos que inician a las 8:30 PM local (01:30 UTC del día siguiente) se registren con la fecha correcta del día del encuentro (`"10/10/2026"`) y no al día siguiente.
     - Actualización de `calculate_metrics_summary(ledger_df)`: reemplazo de `datetime.utcnow().strftime(...)` por `format_local_timestamp()` tanto para DataFrames vacíos como con registros acumulados.
  2. *infrastructure/reporting/__init__.py:*
     - Exportación pública de `format_date_ddmmyyyy`, `format_local_timestamp`, `get_local_now` y `get_user_timezone`.
  3. *tests/test_bet_tracker_excel.py:*
     - Creación de la clase `TestLocalTimezoneAndExcelFormatting` con 6 nuevos tests unitarios:
       * `test_format_local_timestamp_structure`: validación del patrón regex `^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}$`.
       * `test_format_local_timestamp_utc_conversion`: conversión matemática precisa de UTC a UTC-5.
       * `test_format_date_ddmmyyyy_iso_utc_evening_no_day_drift`: verificación de ausencia de desfase de día para horarios nocturnos ("2026-10-11T01:30:00Z" -> "10/10/2026").
       * `test_format_date_ddmmyyyy_standard_formats`: compatibilidad con DD/MM/YYYY y YYYY-MM-DD.
       * `test_format_date_ddmmyyyy_fallback_local_now`: asignación consistente de fecha de hoy local en ausencia de datos.
       * `test_empty_dataframe_metrics_summary_has_local_timestamp`: presencia y formato local en resumen vacío.
       * Verificación de `Última Actualización` en `test_metrics_summary_aggregate_calculation`.
- **Resultado:** 100% de la suite global pasando con éxito (125/125 tests superados). Sincronización horaria institucional culminada sin fisuras.


## [2026-10-10] Auto-Liquidación de Partidos de Hoy, Cruce Bidireccional y Transparencia en Consola
- **Alcance:** Mejora integral del motor de auto-liquidación (`infrastructure/reporting/settle_bets.py`, `infrastructure/nba_api_client.py`, `main.py`) permitiendo liquidación en el mismo día (`game_date <= local_today`), cruce bidireccional simétrico de emparejamientos (directo e invertido), jerarquía estricta de fuentes evitando colisiones con partidos de temporadas pasadas, y logs informativos transparentes al inicio del CLI.
- **Acciones Realizadas:**
  1. *infrastructure/nba_api_client.py:*
     - Adaptación de `fetch_scoreboard_results(game_date)` para soportar `Union[str, List[str]]` y consultar lotes de fechas sin duplicados.
     - Detección precisa de finalización oficial: `(GAME_STATUS_ID == 3)` o `GAME_STATUS_TEXT` conteniendo `"final"` o `"final/ot"`.
  2. *infrastructure/reporting/settle_bets.py:*
     - Implementación de `parse_ledger_date(date_val)` para conversión tolerante de cualquier representación de fecha a `datetime.date`.
     - Filtro temporal: partidos con `game_date <= local_today` se evalúan inmediatamente al arrancar. Partidos futuros (`game_date > local_today`) se mantienen pendientes de forma segura sin consultas prematuras a la API.
     - Cruce Bidireccional de Equipos:
       * Para cualquier registro en el Excel (ej. `"PHI - BOS"`), se evalúa si la API o DB reporta `(PHI, BOS)` O `(BOS, PHI)`.
       * Se extraen los puntajes asignando correctamente los puntos al local y visitante oficial respetando el orden del registro en Fila 2 (ej. `"106 - 114"`).
       * Se calcula la verificación `ACERTÓ` o `FALLÓ` comparando el equipo proyectado en `Probabilidad` contra el ganador real del encuentro.
     - Jerarquía estricta de fuentes: `api_results` (en vivo) > `fallback_results` > SQLite por fecha exacta `game_date` > DB fallback. Esto previene que partidos de temporadas anteriores en `historical_games` liquiden prematuramente partidos de hoy no finalizados.
     - Logs informativos de transparencia en consola:
       * Al arrancar: `[Auto-Liquidación] Verificando X partido(s) pendiente(s)...`
       * Al liquidar: `[Auto-Liquidación] Partido 'EQUIPO1 - EQUIPO2' finalizado (Marcador: X - Y) -> Liquidado como ACERTÓ/FALLÓ.`
       * Si partidos siguen en juego o sin resultado: `[Auto-Liquidación] X partido(s) aún no cuentan con marcador final oficial en NBA API.`
  3. *infrastructure/reporting/__init__.py & main.py:*
     - Exportación pública de `parse_ledger_date`.
     - Limpieza de ejecución en `main.py` delegando la llamada de auto-liquidación al inicio de `CLIController.run()`.
  4. *tests/test_bet_tracker_excel.py:*
     - Incorporación de 7 tests unitarios cubriendo partidos finalizados hoy (`game_date == today`), cruce bidireccional en victoria, cruce bidireccional en derrota, preservación de partidos futuros, captura de logs de consola con `capsys` y validación de `parse_ledger_date`.
- **Resultado:** 100% de la suite global pasando con éxito (132/132 tests superados). Auto-liquidación en vivo 100% robusta y bidireccional.

## [2026-10-11] Subsistema de Liquidación Manual Forzada (get_pending_bets, settle_bet_manually y CLI)
- **Alcance:** Implementación de liquidación manual forzada para encuentros pendientes o de prueba en `nba_bet_tracking.xlsx`, accesible tanto a través del menú interactivo del CLI (`[L] Liquidar manualmente un partido pendiente`) como vía argumento directo CLI (`--settle-manual`), permitiendo ingresar los marcadores reales anotados por cada equipo cuando no existen en el calendario oficial de la NBA API o para auditoría directa del operador cuantitativo.
- **Acciones Realizadas:**
  1. *infrastructure/reporting/settle_bets.py:*
     - Función `get_pending_bets(excel_path)`: Escanea el libro mayor e identifica filas en estado `PENDIENTE`, extrayendo una lista estructurada de diccionarios con metadatos del partido (`id`, `row_idx`, `fecha`, `partido`, `cuota`, `probabilidad`, `decision`, `team_1`, `team_2`).
     - Función `settle_bet_manually(match_identifier, home_score, away_score, excel_path)`: Localiza el partido por ID numérico (1, 2, ...) o nombre de encuentro (ej. `"PHI - BOS"`), actualiza la Fila 1 con el estado resultante (`ACERTÓ` / `FALLÓ` / `PUSH`), actualiza la Fila 2 con el marcador oficial (`"PTS_H - PTS_A"`), calcula el P&L en unidades respetando la recomendación (`APOSTAR` vs `NO APOSTAR`), recalcula la hoja `Resumen_Metricas` mediante `save_ledger_workbook` y preserva la integridad visual y estilos institucionales openpyxl.
  2. *infrastructure/reporting/__init__.py:*
     - Exportación de `get_pending_bets` y `settle_bet_manually` en `__all__`.
  3. *presentation/cli.py:*
     - Soporte en `parse_batch_selection`: reconocimiento de tokens `'l'` y `'liquidar'` retornando acción `'settle_manual'`.
     - Integración visual del comando `[L] Liquidar manualmente un partido pendiente` / `[L] Liquidar Manual` tanto en la pizarra sin partidos como en la lista de partidos programados.
     - Método `CLIController.prompt_manual_settlement()`: Lista los encuentros pendientes con su número, fecha, pronóstico proyectado y decisión. Solicita al usuario los puntos del equipo local y visitante con validación estricta de enteros y valores no negativos, ejecutando la liquidación inmediata y presentando un resumen de confirmación con el resultado, marcador y P&L asociado.
  4. *main.py:*
     - Registro del argumento CLI `--settle-manual` en `parse_args()`.
     - Ejecución directa de `controller.prompt_manual_settlement()` al invocar `python main.py --settle-manual`.
  5. *tests/test_bet_tracker_excel.py:*
     - 4 nuevos tests unitarios en `TestManualBetSettlement`:
       * `test_get_pending_bets_returns_structured_list`: verificación de lista estructurada y claves de metadatos en partidas registradas con y sin cuota.
       * `test_settle_bet_manually_won_updates_rows_and_metrics`: verificación de liquidación de acierto, actualización de Fila 1 y Fila 2, y recálculo de métricas cuantitativas en `Resumen_Metricas`.
       * `test_settle_bet_manually_lost_by_name`: verificación de liquidación de fallo mediante identificador textual del partido.
       * `test_prompt_manual_settlement_cli_flow`: simulación mock de interacción en consola con usuario ingresando ID y puntuaciones, verificando liquidación exitosa en Excel.
- **Resultado:** 100% de la suite global pasando con éxito (136/136 tests superados). Subsistema de liquidación manual validado end-to-end.


## [2026-10-11] Arquitectura de Liquidación en Cascada (Fallback Cascade) y Motor de Red Resiliente
- **Alcance:** Erradicación definitiva de vulnerabilidades por caídas HTTP 500 y bloqueos en `ScoreboardV2` (`stats.nba.com`) implementando una arquitectura resiliente de 3 niveles de respaldo para marcadores finales, normalización léxica exhaustiva y preservación íntegra de fórmulas en Excel.
- **Acciones Realizadas:**
  1. *infrastructure/nba_api_client.py:*
     - Implementación de `create_resilient_session()`: Sesión `requests.Session()` con `HTTPAdapter` y política `Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])`, `User-Agent` de navegador moderno y timeouts explícitos `(5.0, 15.0)`.
     - Diccionario `TEAM_CANONICAL_MAP` y función `standardize_team_code()`: Estandarización de variantes léxicas (ESPN/históricas como `GS`, `PHO`, `SA`, `NY`, `NO`, `NJ`, `WSH`, `UTAH`, nombres completos de las 30 franquicias y apodos) a códigos tricodificados oficiales de 3 letras.
     - Implementación de la Cascada en `fetch_scoreboard_results()`:
       * **Tier 1 (Primaria - API Pública de ESPN):** `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={YYYYMMDD}`, filtrando únicamente partidos con `status.type.name == 'STATUS_FINAL'`.
       * **Tier 2 (Secundaria - NBA Live CDN Edge):** `https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_v2.json`, extrayendo marcadores cuando `gameStatus == 3`.
       * **Tier 3 (Contingencia - The Odds API / ScoreboardV2):** Consulta de endpoint `/scores` si existe API key y fallback encapsulado en `try/except` a `ScoreboardV2` sin interrumpir el flujo.
  2. *infrastructure/reporting/settle_bets.py & infrastructure/reporting/bet_tracker_excel.py:*
     - En `auto_reconcile_pending_bets()`: emisión de log transparente `[Auto-Liquidación] Consultando marcadores vía ESPN/NBA Network para fecha YYYY-MM-DD...`.
     - En `save_ledger_workbook()`: paso explícito de `openpyxl.load_workbook(target, data_only=False)` para asegurar la preservación intacta de fórmulas en la hoja `Resumen_Metricas`.
  3. *tests/test_nba_api_client.py & tests/test_bet_tracker_excel.py:*
     - Creación de `tests/test_nba_api_client.py` con 9 tests exhaustivos cubriendo:
       * Normalización canónica de variantes ESPN e históricas.
       * Configuración de reintentos y headers de sesión resiliente.
       * Éxito en Tier 1 (ESPN) con filtrado de partidos finalizados.
       * Degradación exitosa a Tier 2 (NBA CDN) ante caídas de Tier 1.
       * Degradación exitosa a Tier 3 (ScoreboardV2) ante caídas de Tier 1 y Tier 2.
       * Manejo seguro de fallo total retornando diccionario vacío sin excepciones.
     - Actualización de `test_bet_tracker_excel.py` verificando emisión del log informativo de ESPN/NBA Network.
- **Resultado:** 100% de la suite global pasando con éxito (145/145 tests superados). Marcadores oficiales blindados contra fallos de red.


## [2026-10-11] Refactorización de Arquitectura del Libro Mayor a 1 Fila por Partido (Tabular Estricto)
- **Alcance:** Refactorización integral del libro mayor `Registro_Apuestas` en `infrastructure/reporting/bet_tracker_excel.py`, del motor de liquidación en `infrastructure/reporting/settle_bets.py`, del flujo CLI en `presentation/cli.py` y de la suite de pruebas para adoptar una arquitectura tabular estricta de **1 sola fila por partido**, eliminando la fila secundaria de guiones `'-'`, eliminando filas separadoras y suprimiendo cualquier combinación de celdas (`merge_cells`) en la columna `Estado`.
- **Acciones Realizadas:**
  1. *infrastructure/reporting/bet_tracker_excel.py:*
     - Modificación de `log_bet_prediction()`: cada encuentro analizado o apostado se registra en una única fila de 7 columnas (`[Fecha, Partido, Cuota, Probabilidad, Decision, Marcador_Final, Estado]`).
     - Al registrar un pronóstico pendiente: `Marcador_Final` se inicializa como `'-'`, `Estado` como `'PENDIENTE'` (fondo ámbar suave `#FFF2CC`), y `Cuota` almacena el valor decimal junto a la franquicia recomendada (ej. `'1.42 (BOS)'`) o `'NO DISPONIBLE'`.
     - Modificación de `_style_ledger_sheet()`: eliminación definitiva del bucle de combinación de celdas `merge_cells`. Cada celda de `Estado` y `Decision` se estiliza individualmente. Altura de fila de datos fijada en 22 pt con bordes y fuentes ejecutivas.
     - En `calculate_metrics_summary()`: integración de extracción numérica robusta por expresión regular (`re.search(r"(\d+(?:\.\d+)?)", cuota_raw)`) para tolerar cadenas formateadas como `'1.42 (BOS)'`.
     - En `load_ledger()`: implementación de migración y consolidación automática hacia atrás: colapsa registros bi-fila residuales previos conservando los marcadores liquidados (`Marcador_Final`) en la fila principal y descartando filas vacías o de guiones.
  2. *infrastructure/reporting/settle_bets.py:*
     - Refactorización de `settle_pending_bets()`: bucle simplificado iterando fila por fila (`step = 1`). Al liquidar un encuentro, actualiza directamente en esa misma fila `ledger_df.at[idx_1, "Marcador_Final"] = score_text` y `ledger_df.at[idx_1, "Estado"] = verificacion` (`ACERTÓ`, `FALLÓ` o `PUSH`).
     - Refactorización de `settle_bet_manually()`: actualización del marcador oficial y estado en la fila maestra seleccionada (`idx_1`), eliminando referencias a `idx_2`.
  3. *Migración del Libro Mayor en Disco:*
     - Consolidación y reescritura exitosa de `C:\Users\mblad\Documents\Codigoes\nba_bet_tracking.xlsx` al nuevo estándar tabular estricto de 1 fila por evento, preservando el marcador histórico de prueba `106 - 144` y el estado `ACERTÓ`.
  4. *tests/test_bet_tracker_excel.py:*
     - Actualización exhaustiva de todos los tests de registro, liquidación automática, liquidación manual, procesamiento por lotes y persistencia openpyxl, validando la estructura estricta de 1 fila por encuentro (`len(df) == 1`, `ws.max_row == 2` para un solo partido, y marcadores en `df.iloc[0]["Marcador_Final"]`).
- **Resultado:** 100% de la suite global pasando con éxito (145/145 tests superados). Libro mayor 100% tabular, limpio y optimizado para análisis cuantitativo.


## [2026-10-11] Auditoría de Seguridad, Blindaje de Privacidad y Preparación de Repositorio Git
- **Alcance:** Configuración y auditoría exhaustiva del control de versiones para la subida institucional a GitHub: erradicación de fuga de secretos y API keys, exclusión de bases de datos locales, datasets procesados pesados y binarios compilados, diseño de plantilla `config.example.yaml` y generación del primer commit limpio en `main`.
- **Acciones Realizadas:**
  1. *Plantilla Pública de Configuración (`config.example.yaml`):*
     - Desacople de credenciales privadas: sustitución de la API key de The Odds API y rutas absolutas locales de Windows por placeholders (`"YOUR_ODDS_API_KEY_HERE"`, `"nba_bet_tracking.xlsx"`).
  2. *Auditoría y Blindaje de `.gitignore`:*
     - Entornos virtuales y compilados: `.venv/`, `venv/`, `env/`, `__pycache__/`, `*.py[cod]`.
     - Testing y cobertura: `.pytest_cache/`, `.coverage`, `.coverage.*`, `coverage.xml`, `htmlcov/`.
     - Secretos y configuración local: `.env`, `*.env`, `.env.*`, `config.yaml`.
     - Bases de datos y almacenamiento: `*.db`, `*.sqlite`, `*.sqlite3`, `nba_predictor.db`.
     - Hojas de cálculo y métricas locales: `*.xlsx`, `*.xls`, `*.csv`.
     - Modelos y datasets de gran tamaño: `*.parquet`, `*.joblib`, `*.pkl`, `xgboost_model.json`, `data/`, `models/*.joblib`.
     - Archivos de IDE y tooling temporal: `.vscode/`, `.idea/`, `.DS_Store`, `.gemini/`, `scratch/`, `.gga`, `*.log`.
  3. *Inicialización e Inspección de Git:*
     - Inicialización con rama estándar `main` (`git init -b main`).
     - Verificación mediante `git status --ignored` confirmando la exclusión limpia de bases de datos SQLite (`nba_predictor.db`), credenciales (`config.yaml`), binarios pesados (`nba_spread_model.joblib`, `processed_features.*`) y cachés.
  4. *Validación Técnica:*
     - Ejecución completa de la suite de pruebas (145/145 tests pasando en 12.75s).
- **Resultado:** Repositorio blindado contra filtraciones de credenciales y listo para vinculación con remote de GitHub.