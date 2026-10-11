# AGENT LOG
Este es el diario de memoria de mi ingeniero de IA.

# CONFIGURACIÓN
- **Modelo Principal:** MiniMax M2.5 Free
- **Memoria:** Engram MCP habilitado
- **Modo:** Single Orchestrator
- **Test Framework:** pytest con Strict TDD

# REGLAS DE ORO
1. Mantener siempre el contexto de los cambios anteriores.
2. Usar español para las explicaciones e interfaz.
3. Al finalizar una tarea, actualizar este log con lo aprendido.

# INFRAESTRUCTURA ACTUAL (2026-04-26)
- Database: SQLite (nba_predictor.db)
- Repositorios: SQLiteTeamRepository, SQLitePredictionRepository, SQLiteH2HRepository
- Motor ML: XGBoost (entrenado con 500 samples)
- Engine: HybridEngine (Heuristic + XGBoost)
- Tests: 25 tests passing (Strict TDD)
- API: FastAPI con structlog

# CÓMO LEVANTAR EL SERVIDOR
```bash
# Instalar dependencias
pip install fastapi uvicorn structlog

# Arrancar servidor
python api.py

# Endpoints disponibles:
# - GET /health           # Health check
# - POST /predict         # Hacer predicción
# - GET /accuracy        # Ver accuracy del modelo
# - POST /result         # Actualizar resultado real
# - GET /predictions/recent  # Ver predicciones recientes
```

# EJEMPLO DE USO API
```bash
# Predicción
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team": "LAL", "away_team": "BOS", "odds_home": 1.85, "odds_away": 2.10}'

# Accuracy
curl http://localhost:8000/accuracy
```

# PRODUCCIÓN
```bash
# CI/CD (GitHub Actions)
- .github/workflows/ci.yml corre automáticamente en PRs

# Security Review
python security_review.py
```