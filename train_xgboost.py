# -*- coding: utf-8 -*-
"""
train_xgboost.py
-------------
Entrena el modelo XGBoost con datos históricos.
Genera datos sintéticos realistas basados en lógica de NBA.
"""

import random
from pathlib import Path

from domain.engines.xgboost_engine import XGBoostEngine, FEATURE_NAMES


def generate_training_data(n_samples: int = 500) -> tuple:
    """
    Genera datos de entrenamiento sintéticos realistas.
    Cada muestra representa un partido.
    """
    X = []
    y = []

    for _ in range(n_samples):
        pts_diff = random.uniform(-20, 20)
        fg_pct_diff = random.uniform(-0.08, 0.08)
        fg3_pct_diff = random.uniform(-0.06, 0.06)
        reb_diff = random.uniform(-15, 15)
        ast_diff = random.uniform(-10, 10)
        tov_diff = random.uniform(-5, 5)
        stl_diff = random.uniform(-4, 4)
        blk_diff = random.uniform(-3, 3)
        plus_minus_diff = random.uniform(-15, 15)

        home_win_pct = random.uniform(0.25, 0.75)
        away_win_pct = random.uniform(0.25, 0.75)

        home_injury_impact = random.choice([0, 0, 0, -5, -8, -10])
        away_injury_impact = random.choice([0, 0, 0, -5, -8, -10])

        h2h_home_win_pct = random.uniform(0.3, 0.7)

        home_advantage = 0.05

        features = [
            pts_diff, fg_pct_diff, fg3_pct_diff, reb_diff, ast_diff,
            tov_diff, stl_diff, blk_diff, plus_minus_diff,
            home_win_pct, away_win_pct,
            home_injury_impact, away_injury_impact,
            h2h_home_win_pct, home_advantage
        ]

        score = 0.5
        score += pts_diff * 0.25 * 0.01
        score += fg_pct_diff * 0.15 * 10
        score += fg3_pct_diff * 0.10 * 10
        score += reb_diff * 0.10 * 0.1
        score += ast_diff * 0.08 * 0.1
        score += tov_diff * (-0.05) * 0.1
        score += plus_minus_diff * 0.12 * 0.1
        score += home_win_pct * 0.05
        score += away_win_pct * (-0.05)
        score += home_injury_impact * (-0.02)
        score += away_injury_impact * 0.02
        score += h2h_home_win_pct * 0.03
        score += home_advantage

        score += random.uniform(-0.05, 0.05)

        result = 1 if score > 0.5 else 0

        X.append(features)
        y.append(result)

    return X, y


def main():
    print("=" * 50)
    print("ENTRENAMIENTO XGBOOST")
    print("=" * 50)

    n_samples = 500
    print(f"\nGenerando {n_samples} muestras de entrenamiento...")

    X, y = generate_training_data(n_samples)

    puntos_local = sum(1 for r in y if r == 1)
    puntos_away = sum(1 for r in y if r == 0)
    print(f"  - Partidos local gana: {puntos_local} ({puntos_local/n_samples*100:.1f}%)")
    print(f"  - Partidos away gana: {puntos_away} ({puntos_away/n_samples*100:.1f}%)")

    print("\nEntrenando modelo XGBoost...")
    engine = XGBoostEngine()

    result = engine.train(X, y)

    print(f"\n[OK] Entrenamiento completado!")
    print(f"  - Accuracy: {result['train_accuracy']*100:.1f}%")
    print(f"  - Muestras: {result['samples']}")
    print(f"  - Archivo: {engine._model_path}")

    print("\n" + "=" * 50)
    print("PRUEBA DEL MODELO")
    print("=" * 50)

    test_cases = [
        ([10, 0.03, 0.02, 5, 3, -1, 1, 1, 5, 0.7, 0.3, 0, 0, 0.6, 0.05], "Lakers fuerte local"),
        ([-10, -0.03, -0.02, -5, -3, 1, -1, -1, -5, 0.3, 0.7, -8, 0, 0.4, 0.05], "Celtics weak away"),
        ([2, 0.01, 0.00, 1, 0, 0, 0, 0, 1, 0.5, 0.5, 0, 0, 0.5, 0.05], "Partido parejo"),
    ]

    print("\nPredicciones de prueba:")
    for features, description in test_cases:
        prob = engine._model.predict_proba([features])[0][1]
        print(f"  {description}: {prob*100:.1f}% para local")


if __name__ == "__main__":
    main()