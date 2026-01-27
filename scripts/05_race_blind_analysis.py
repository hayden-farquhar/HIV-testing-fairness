#!/usr/bin/env python3
"""
05_race_blind_analysis.py - Test proxy discrimination hypothesis

This script:
1. Trains models excluding race features
2. Compares DPD with race-inclusive models
3. Quantifies proxy discrimination
4. Tests if removing race reduces disparities

Output:
    - results/race_blind_comparison.csv
    - results/race_blind_models.pkl
"""

import os
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from src.data_utils import get_feature_columns
from src.fairness_metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    group_metrics,
    selection_rate_gap
)


def train_race_blind_models(X_train, y_train):
    """Train models without race features."""
    models = {}

    print("\nTraining race-blind Logistic Regression...")
    models['Logistic Regression'] = LogisticRegression(
        max_iter=1000, random_state=42
    )
    models['Logistic Regression'].fit(X_train, y_train)

    print("Training race-blind Random Forest...")
    models['Random Forest'] = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    )
    models['Random Forest'].fit(X_train, y_train)

    print("Training race-blind XGBoost...")
    models['XGBoost'] = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1
    )
    models['XGBoost'].fit(X_train, y_train)

    print("Training race-blind Gradient Boosting...")
    models['Gradient Boosting'] = GradientBoostingClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42
    )
    models['Gradient Boosting'].fit(X_train, y_train)

    return models


def main():
    """Main race-blind analysis pipeline."""
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data' / 'processed' / 'analysis_dataset.csv'
    results_dir = project_root / 'results'
    predictions_path = results_dir / 'model_predictions.pkl'

    print("=" * 60)
    print("HIV Testing Fairness - Race-Blind Analysis")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from {data_path}...")
    if not data_path.exists():
        print("Error: Processed data not found. Run 01_data_loading.py first.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} records")

    # Load baseline predictions for comparison
    if not predictions_path.exists():
        print("Error: Baseline predictions not found. Run 02_baseline_models.py first.")
        sys.exit(1)

    with open(predictions_path, 'rb') as f:
        baseline_data = pickle.load(f)

    # Prepare features - EXCLUDE race variables
    race_blind_features = get_feature_columns(include_race=False)
    full_features = get_feature_columns(include_race=True)

    print(f"\nRace-blind features ({len(race_blind_features)}):")
    for feat in race_blind_features:
        print(f"  - {feat}")

    print(f"\nExcluded race features:")
    race_features = [f for f in full_features if f not in race_blind_features]
    for feat in race_features:
        print(f"  - {feat}")

    # Prepare data
    X_blind = df[race_blind_features]
    y = df['hiv_tested']
    sensitive = df['race']

    # Use same train/test split as baseline
    X_train_blind, X_test_blind, y_train, y_test, sens_train, sens_test = train_test_split(
        X_blind, y, sensitive, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set: {len(X_train_blind):,} records")
    print(f"Test set: {len(X_test_blind):,} records")

    # Train race-blind models
    print("\n" + "-" * 40)
    print("Training Race-Blind Models")
    print("-" * 40)

    blind_models = train_race_blind_models(X_train_blind, y_train)

    # Evaluate and compare
    print("\n" + "-" * 40)
    print("Comparing Race-Blind vs Race-Aware Models")
    print("-" * 40)

    comparison_results = []

    for name in blind_models.keys():
        # Race-blind predictions
        y_pred_blind = blind_models[name].predict(X_test_blind)
        y_prob_blind = blind_models[name].predict_proba(X_test_blind)[:, 1]

        # Race-aware predictions (from baseline)
        y_pred_aware = baseline_data[f'{name}_pred']
        y_prob_aware = baseline_data[f'{name}_prob']

        # Calculate metrics for race-blind
        dpd_blind = demographic_parity_difference(y_pred_blind, sens_test.values)
        eod_blind = equalized_odds_difference(y_test.values, y_pred_blind, sens_test.values)
        acc_blind = accuracy_score(y_test, y_pred_blind)
        auc_blind = roc_auc_score(y_test, y_prob_blind)

        # Calculate metrics for race-aware
        dpd_aware = demographic_parity_difference(y_pred_aware, sens_test.values)
        eod_aware = equalized_odds_difference(y_test.values, y_pred_aware, sens_test.values)
        acc_aware = accuracy_score(y_test, y_pred_aware)
        auc_aware = roc_auc_score(y_test, y_prob_aware)

        # Selection rate gaps
        black_blind, white_blind, gap_blind = selection_rate_gap(
            y_pred_blind, sens_test.values, 'Black', 'White'
        )
        black_aware, white_aware, gap_aware = selection_rate_gap(
            y_pred_aware, sens_test.values, 'Black', 'White'
        )

        # Calculate proxy discrimination percentage
        # (proportion of disparity that persists without race)
        proxy_pct = (dpd_blind / dpd_aware * 100) if dpd_aware > 0 else 0

        result = {
            'model': name,
            'dpd_race_aware': dpd_aware,
            'dpd_race_blind': dpd_blind,
            'dpd_reduction_pct': (1 - dpd_blind/dpd_aware) * 100 if dpd_aware > 0 else 0,
            'proxy_discrimination_pct': proxy_pct,
            'eod_race_aware': eod_aware,
            'eod_race_blind': eod_blind,
            'accuracy_race_aware': acc_aware,
            'accuracy_race_blind': acc_blind,
            'auc_race_aware': auc_aware,
            'auc_race_blind': auc_blind,
            'bw_gap_aware': gap_aware,
            'bw_gap_blind': gap_blind
        }
        comparison_results.append(result)

        print(f"\n{name}:")
        print(f"  Race-aware DPD: {dpd_aware:.4f}")
        print(f"  Race-blind DPD: {dpd_blind:.4f}")
        print(f"  Proxy discrimination: {proxy_pct:.1f}% of disparity persists")
        print(f"  Black-White gap (aware): {gap_aware*100:.1f} pp")
        print(f"  Black-White gap (blind): {gap_blind*100:.1f} pp")
        print(f"  Accuracy change: {acc_aware:.4f} -> {acc_blind:.4f}")

    comparison_df = pd.DataFrame(comparison_results)

    # Group-level analysis for race-blind XGBoost
    print("\n" + "-" * 40)
    print("Race-Blind XGBoost Group Metrics")
    print("-" * 40)

    xgb_blind = blind_models['XGBoost']
    y_pred_xgb = xgb_blind.predict(X_test_blind)
    y_prob_xgb = xgb_blind.predict_proba(X_test_blind)[:, 1]

    gm = group_metrics(y_test.values, y_pred_xgb, y_prob_xgb, sens_test.values)
    for _, row in gm.iterrows():
        print(f"  {row['group']}: selection_rate={row['selection_rate']:.4f}, TPR={row['tpr']:.4f}")

    # Save results
    print("\n" + "-" * 40)
    print("Saving Results")
    print("-" * 40)

    comparison_path = results_dir / 'race_blind_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Saved comparison results to {comparison_path}")

    # Save race-blind models
    blind_models_path = results_dir / 'race_blind_models.pkl'
    blind_data = {
        'models': blind_models,
        'X_test': X_test_blind,
        'y_test': y_test,
        'sensitive_test': sens_test
    }
    for name, model in blind_models.items():
        blind_data[f'{name}_pred'] = model.predict(X_test_blind)
        blind_data[f'{name}_prob'] = model.predict_proba(X_test_blind)[:, 1]

    with open(blind_models_path, 'wb') as f:
        pickle.dump(blind_data, f)
    print(f"Saved race-blind models to {blind_models_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Race-Blind Analysis Summary")
    print("=" * 60)

    print("\nExpected values (from manuscript):")
    print("  Race-blind model DPD: ~0.154-0.169")
    print("  ~70% of baseline disparity persists")

    mean_proxy = comparison_df['proxy_discrimination_pct'].mean()
    mean_dpd_blind = comparison_df['dpd_race_blind'].mean()

    print(f"\nActual values:")
    print(f"  Mean race-blind DPD: {mean_dpd_blind:.4f}")
    print(f"  Mean proxy discrimination: {mean_proxy:.1f}%")

    print("\nConclusion:")
    if mean_proxy > 50:
        print("  Significant proxy discrimination detected.")
        print("  Removing race features does NOT eliminate racial disparities.")
        print("  Other features (income, geography, etc.) act as proxies for race.")
    else:
        print("  Race features explain most of the disparity.")

    return comparison_df


if __name__ == "__main__":
    main()
