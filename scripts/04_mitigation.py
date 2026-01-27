#!/usr/bin/env python3
"""
04_mitigation.py - Apply fairness mitigation methods

This script:
1. Applies ThresholdOptimizer (post-processing) with demographic parity constraint
2. Applies ExponentiatedGradient (in-processing) with DP and EO constraints
3. Tests multiple epsilon values
4. Calculates DPD reduction percentages
5. Identifies Pareto-optimal models

Output:
    - results/mitigation_results.csv
    - results/pareto_frontier.csv
    - results/epsilon_grid_search.csv
"""

import os
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.mitigation import (
    apply_threshold_optimizer,
    apply_exponentiated_gradient,
    evaluate_mitigation,
    grid_search_epsilon,
    pareto_frontier,
    run_mitigation_experiment
)
from src.fairness_metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    group_metrics
)


def main():
    """Main mitigation pipeline."""
    project_root = Path(__file__).parent.parent
    results_dir = project_root / 'results'
    predictions_path = results_dir / 'model_predictions.pkl'

    print("=" * 60)
    print("HIV Testing Fairness - Fairness Mitigation")
    print("=" * 60)

    # Load data
    if not predictions_path.exists():
        print("Error: Model predictions not found. Run 02_baseline_models.py first.")
        sys.exit(1)

    with open(predictions_path, 'rb') as f:
        data = pickle.load(f)

    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    sensitive_train = data['sensitive_train']
    sensitive_test = data['sensitive_test']
    models = data['models']

    print(f"\nTraining set: {len(X_train):,} records")
    print(f"Test set: {len(X_test):,} records")

    # Run mitigation experiments
    print("\n" + "-" * 40)
    print("ThresholdOptimizer Mitigation")
    print("-" * 40)

    all_results = []

    for name, model in models.items():
        print(f"\nProcessing {name}...")

        # Baseline predictions
        y_pred_baseline = data[f'{name}_pred']
        y_prob_baseline = data[f'{name}_prob']

        baseline_dpd = demographic_parity_difference(y_pred_baseline, sensitive_test.values)
        baseline_eod = equalized_odds_difference(y_test.values, y_pred_baseline, sensitive_test.values)
        baseline_acc = accuracy_score(y_test, y_pred_baseline)

        all_results.append({
            'model': name,
            'method': 'baseline',
            'accuracy': baseline_acc,
            'dpd': baseline_dpd,
            'eod': baseline_eod,
            'dpd_reduction_pct': 0.0
        })

        # ThresholdOptimizer - Demographic Parity
        try:
            print(f"  Applying ThresholdOptimizer (DP)...")
            to_dp = apply_threshold_optimizer(
                model, X_train, y_train, sensitive_train,
                constraint='demographic_parity', prefit=True
            )
            y_pred_to = to_dp.predict(X_test, sensitive_features=sensitive_test)

            to_dpd = demographic_parity_difference(y_pred_to, sensitive_test.values)
            to_eod = equalized_odds_difference(y_test.values, y_pred_to, sensitive_test.values)
            to_acc = accuracy_score(y_test, y_pred_to)
            dpd_reduction = (baseline_dpd - to_dpd) / baseline_dpd * 100 if baseline_dpd > 0 else 0

            all_results.append({
                'model': name,
                'method': 'ThresholdOptimizer_DP',
                'accuracy': to_acc,
                'dpd': to_dpd,
                'eod': to_eod,
                'dpd_reduction_pct': dpd_reduction
            })

            print(f"    DPD: {baseline_dpd:.4f} -> {to_dpd:.4f} ({dpd_reduction:.1f}% reduction)")
            print(f"    Accuracy: {baseline_acc:.4f} -> {to_acc:.4f}")

        except Exception as e:
            print(f"    ThresholdOptimizer DP failed: {e}")

        # ThresholdOptimizer - Equalized Odds
        try:
            print(f"  Applying ThresholdOptimizer (EO)...")
            to_eo = apply_threshold_optimizer(
                model, X_train, y_train, sensitive_train,
                constraint='equalized_odds', prefit=True
            )
            y_pred_to_eo = to_eo.predict(X_test, sensitive_features=sensitive_test)

            to_eo_dpd = demographic_parity_difference(y_pred_to_eo, sensitive_test.values)
            to_eo_eod = equalized_odds_difference(y_test.values, y_pred_to_eo, sensitive_test.values)
            to_eo_acc = accuracy_score(y_test, y_pred_to_eo)
            eod_reduction = (baseline_eod - to_eo_eod) / baseline_eod * 100 if baseline_eod > 0 else 0

            all_results.append({
                'model': name,
                'method': 'ThresholdOptimizer_EO',
                'accuracy': to_eo_acc,
                'dpd': to_eo_dpd,
                'eod': to_eo_eod,
                'dpd_reduction_pct': (baseline_dpd - to_eo_dpd) / baseline_dpd * 100 if baseline_dpd > 0 else 0
            })

        except Exception as e:
            print(f"    ThresholdOptimizer EO failed: {e}")

    # ExponentiatedGradient with epsilon grid search
    print("\n" + "-" * 40)
    print("ExponentiatedGradient Epsilon Grid Search")
    print("-" * 40)

    epsilons = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    print(f"Testing epsilon values: {epsilons}")

    # Use Logistic Regression as base estimator for ExponentiatedGradient
    base_lr = LogisticRegression(max_iter=1000, random_state=42)
    baseline_lr_dpd = demographic_parity_difference(data['Logistic Regression_pred'], sensitive_test.values)

    eg_results = []
    for eps in epsilons:
        try:
            print(f"\nEpsilon = {eps}...")
            lr_clone = clone(base_lr)
            eg = apply_exponentiated_gradient(
                lr_clone, X_train, y_train, sensitive_train,
                constraint='demographic_parity', eps=eps
            )
            y_pred_eg = eg.predict(X_test)

            eg_dpd = demographic_parity_difference(y_pred_eg, sensitive_test.values)
            eg_eod = equalized_odds_difference(y_test.values, y_pred_eg, sensitive_test.values)
            eg_acc = accuracy_score(y_test, y_pred_eg)
            dpd_reduction = (baseline_lr_dpd - eg_dpd) / baseline_lr_dpd * 100 if baseline_lr_dpd > 0 else 0

            result = {
                'model': 'Logistic Regression',
                'method': f'ExponentiatedGradient_eps{eps}',
                'epsilon': eps,
                'accuracy': eg_acc,
                'dpd': eg_dpd,
                'eod': eg_eod,
                'dpd_reduction_pct': dpd_reduction
            }
            eg_results.append(result)
            all_results.append(result)

            print(f"  DPD: {eg_dpd:.4f} (reduction: {dpd_reduction:.1f}%)")
            print(f"  Accuracy: {eg_acc:.4f}")

        except Exception as e:
            print(f"  Failed: {e}")

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # Identify Pareto frontier
    print("\n" + "-" * 40)
    print("Pareto Frontier Analysis")
    print("-" * 40)

    pareto_df = pareto_frontier(results_df, accuracy_col='accuracy', fairness_col='dpd')
    print(f"\nPareto-optimal models ({len(pareto_df)}):")
    for _, row in pareto_df.iterrows():
        print(f"  {row['model']} ({row['method']}): Acc={row['accuracy']:.4f}, DPD={row['dpd']:.4f}")

    # Save results
    print("\n" + "-" * 40)
    print("Saving Results")
    print("-" * 40)

    results_path = results_dir / 'mitigation_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"Saved mitigation results to {results_path}")

    pareto_path = results_dir / 'pareto_frontier.csv'
    pareto_df.to_csv(pareto_path, index=False)
    print(f"Saved Pareto frontier to {pareto_path}")

    if eg_results:
        eg_df = pd.DataFrame(eg_results)
        eg_path = results_dir / 'epsilon_grid_search.csv'
        eg_df.to_csv(eg_path, index=False)
        print(f"Saved epsilon grid search to {eg_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Mitigation Summary")
    print("=" * 60)

    print("\nDPD Reduction by Method:")
    summary = results_df.groupby('method').agg({
        'dpd': 'mean',
        'accuracy': 'mean',
        'dpd_reduction_pct': 'mean'
    }).round(4)
    print(summary.to_string())

    print("\nExpected values (from manuscript):")
    print("  ThresholdOptimizer DPD reduction: 86-97%")

    to_results = results_df[results_df['method'] == 'ThresholdOptimizer_DP']
    if len(to_results) > 0:
        mean_reduction = to_results['dpd_reduction_pct'].mean()
        print(f"\nActual ThresholdOptimizer DPD reduction: {mean_reduction:.1f}%")

    return results_df, pareto_df


if __name__ == "__main__":
    main()
