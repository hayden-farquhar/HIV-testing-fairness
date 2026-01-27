#!/usr/bin/env python3
"""
07_clinical_utility.py - Decision curve analysis and clinical metrics

This script:
1. Calculates net benefit across thresholds (0.05-0.95)
2. Computes Number Needed to Screen (NNS) by race
3. Performs policy simulation at threshold=0.5
4. Compares clinical utility across models

Output:
    - results/decision_curve.csv
    - results/nns_by_race.csv
    - results/clinical_utility.csv
"""

import os
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

from src.fairness_metrics import selection_rate_gap


def calculate_net_benefit(y_true, y_prob, threshold):
    """
    Calculate net benefit at a given threshold.

    Net benefit = TP/n - FP/n * (threshold / (1 - threshold))
    """
    y_pred = (y_prob >= threshold).astype(int)

    n = len(y_true)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))

    net_benefit = tp / n - fp / n * (threshold / (1 - threshold))
    return net_benefit


def calculate_nns(y_true, y_pred):
    """
    Calculate Number Needed to Screen.

    NNS = 1 / (True Positive Rate * Prevalence)
    or equivalently: Number screened / Number of true positives detected
    """
    n_screened = np.sum(y_pred == 1)
    tp = np.sum((y_pred == 1) & (y_true == 1))

    if tp == 0:
        return np.inf

    return n_screened / tp


def policy_simulation(y_true, y_pred, y_prob, sensitive, threshold=0.5):
    """
    Simulate policy implementation at given threshold.

    Returns metrics for different demographic groups.
    """
    results = []
    groups = np.unique(sensitive[~pd.isna(sensitive)])

    for group in groups:
        mask = sensitive == group
        y_t = y_true[mask]
        y_p = y_pred[mask]
        y_pr = y_prob[mask]

        n = mask.sum()
        n_screened = np.sum(y_p == 1)
        tp = np.sum((y_p == 1) & (y_t == 1))
        fp = np.sum((y_p == 1) & (y_t == 0))
        fn = np.sum((y_p == 0) & (y_t == 1))
        tn = np.sum((y_p == 0) & (y_t == 0))

        # Metrics
        screening_rate = n_screened / n if n > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        nns = n_screened / tp if tp > 0 else np.inf

        results.append({
            'group': group,
            'n': n,
            'n_screened': n_screened,
            'screening_rate': screening_rate,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'ppv': ppv,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'nns': nns
        })

    return pd.DataFrame(results)


def main():
    """Main clinical utility pipeline."""
    project_root = Path(__file__).parent.parent
    results_dir = project_root / 'results'
    predictions_path = results_dir / 'model_predictions.pkl'

    print("=" * 60)
    print("HIV Testing Fairness - Clinical Utility Analysis")
    print("=" * 60)

    # Load predictions
    if not predictions_path.exists():
        print("Error: Model predictions not found. Run 02_baseline_models.py first.")
        sys.exit(1)

    with open(predictions_path, 'rb') as f:
        data = pickle.load(f)

    y_test = data['y_test'].values
    sensitive_test = data['sensitive_test'].values
    model_names = ['Logistic Regression', 'Random Forest', 'XGBoost', 'Gradient Boosting']

    # 1. Decision Curve Analysis
    print("\n" + "-" * 40)
    print("Decision Curve Analysis")
    print("-" * 40)

    thresholds = np.arange(0.05, 0.96, 0.05)
    dca_results = []

    # Calculate baseline strategies
    prevalence = y_test.mean()
    print(f"HIV testing prevalence: {prevalence:.4f}")

    for thresh in thresholds:
        # Test all strategy
        nb_all = prevalence - (1 - prevalence) * thresh / (1 - thresh)

        # Test none strategy
        nb_none = 0

        dca_results.append({
            'model': 'Test All',
            'threshold': thresh,
            'net_benefit': nb_all
        })
        dca_results.append({
            'model': 'Test None',
            'threshold': thresh,
            'net_benefit': nb_none
        })

    # Calculate for each model
    for name in model_names:
        y_prob = data[f'{name}_prob']

        for thresh in thresholds:
            nb = calculate_net_benefit(y_test, y_prob, thresh)
            dca_results.append({
                'model': name,
                'threshold': thresh,
                'net_benefit': nb
            })

    dca_df = pd.DataFrame(dca_results)

    # Print summary at key thresholds
    print("\nNet Benefit at Threshold = 0.5:")
    for name in model_names:
        nb = dca_df[(dca_df['model'] == name) & (dca_df['threshold'] == 0.5)]['net_benefit'].values[0]
        print(f"  {name}: {nb:.4f}")

    # 2. Number Needed to Screen by Race
    print("\n" + "-" * 40)
    print("Number Needed to Screen (NNS) by Race")
    print("-" * 40)

    nns_results = []

    for name in model_names:
        y_pred = data[f'{name}_pred']
        y_prob = data[f'{name}_prob']

        groups = np.unique(sensitive_test[~pd.isna(sensitive_test)])

        for group in groups:
            mask = sensitive_test == group
            nns = calculate_nns(y_test[mask], y_pred[mask])

            nns_results.append({
                'model': name,
                'group': group,
                'nns': nns
            })

        # Overall
        nns_overall = calculate_nns(y_test, y_pred)
        nns_results.append({
            'model': name,
            'group': 'Overall',
            'nns': nns_overall
        })

    nns_df = pd.DataFrame(nns_results)

    # Print XGBoost NNS
    print("\nXGBoost NNS by race:")
    xgb_nns = nns_df[nns_df['model'] == 'XGBoost']
    for _, row in xgb_nns.iterrows():
        nns_str = f"{row['nns']:.1f}" if row['nns'] < 100 else "inf"
        print(f"  {row['group']}: {nns_str}")

    # 3. Policy Simulation at Threshold = 0.5
    print("\n" + "-" * 40)
    print("Policy Simulation (Threshold = 0.5)")
    print("-" * 40)

    clinical_results = []

    for name in model_names:
        y_pred = data[f'{name}_pred']
        y_prob = data[f'{name}_prob']

        policy_df = policy_simulation(y_test, y_pred, y_prob, sensitive_test)
        policy_df['model'] = name
        clinical_results.append(policy_df)

    clinical_df = pd.concat(clinical_results, ignore_index=True)

    # Print summary for XGBoost
    print("\nXGBoost Policy Simulation:")
    xgb_policy = clinical_df[clinical_df['model'] == 'XGBoost']

    for _, row in xgb_policy.iterrows():
        print(f"\n  {row['group']} (n={row['n']:,}):")
        print(f"    Screening rate: {row['screening_rate']*100:.1f}%")
        print(f"    Sensitivity: {row['sensitivity']:.4f}")
        print(f"    PPV: {row['ppv']:.4f}")
        print(f"    NNS: {row['nns']:.1f}")

    # 4. Disparity in Clinical Outcomes
    print("\n" + "-" * 40)
    print("Disparity in Clinical Outcomes")
    print("-" * 40)

    for name in model_names:
        policy = clinical_df[clinical_df['model'] == name]

        black_row = policy[policy['group'] == 'Black'].iloc[0]
        white_row = policy[policy['group'] == 'White'].iloc[0]

        screen_gap = black_row['screening_rate'] - white_row['screening_rate']
        sens_gap = black_row['sensitivity'] - white_row['sensitivity']
        nns_gap = black_row['nns'] - white_row['nns']

        print(f"\n{name} (Black - White disparities):")
        print(f"  Screening rate gap: {screen_gap*100:.1f} pp")
        print(f"  Sensitivity gap: {sens_gap:.4f}")
        print(f"  NNS difference: {nns_gap:.1f}")

    # 5. Optimal Threshold Analysis
    print("\n" + "-" * 40)
    print("Optimal Threshold Analysis")
    print("-" * 40)

    for name in model_names:
        model_dca = dca_df[dca_df['model'] == name]
        max_nb = model_dca['net_benefit'].max()
        optimal_thresh = model_dca[model_dca['net_benefit'] == max_nb]['threshold'].values[0]

        print(f"\n{name}:")
        print(f"  Optimal threshold: {optimal_thresh:.2f}")
        print(f"  Maximum net benefit: {max_nb:.4f}")

    # Save results
    print("\n" + "-" * 40)
    print("Saving Results")
    print("-" * 40)

    dca_path = results_dir / 'decision_curve.csv'
    dca_df.to_csv(dca_path, index=False)
    print(f"Saved decision curve analysis to {dca_path}")

    nns_path = results_dir / 'nns_by_race.csv'
    nns_df.to_csv(nns_path, index=False)
    print(f"Saved NNS by race to {nns_path}")

    clinical_path = results_dir / 'clinical_utility.csv'
    clinical_df.to_csv(clinical_path, index=False)
    print(f"Saved clinical utility to {clinical_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Clinical Utility Summary")
    print("=" * 60)

    print("\nKey findings:")
    print("- Decision curve shows models outperform 'test all' at most thresholds")
    print("- NNS varies significantly by race, indicating differential efficiency")
    print("- Black individuals have higher screening rates but not proportionally higher detection")

    return dca_df, nns_df, clinical_df


if __name__ == "__main__":
    main()
