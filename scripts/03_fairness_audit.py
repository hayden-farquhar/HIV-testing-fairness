#!/usr/bin/env python3
"""
03_fairness_audit.py - Comprehensive fairness metrics calculation

This script:
1. Calculates DPD and EOD for all baseline models
2. Computes group-wise metrics by race/ethnicity
3. Performs intersectional analysis (race x sex)
4. Calculates calibration curves by race
5. Audits Denver HIV Risk Score coefficients

Output:
    - results/fairness_metrics.csv
    - results/intersectional_analysis.csv
    - results/calibration_by_race.csv
    - results/group_metrics.csv
"""

import os
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.fairness_metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    group_metrics,
    calibration_by_group,
    cohens_h,
    selection_rate_gap
)
from src.data_utils import create_intersectional_groups


def main():
    """Main fairness audit pipeline."""
    project_root = Path(__file__).parent.parent
    results_dir = project_root / 'results'
    predictions_path = results_dir / 'model_predictions.pkl'

    print("=" * 60)
    print("HIV Testing Fairness - Fairness Audit")
    print("=" * 60)

    # Load predictions
    if not predictions_path.exists():
        print("Error: Model predictions not found. Run 02_baseline_models.py first.")
        sys.exit(1)

    with open(predictions_path, 'rb') as f:
        data = pickle.load(f)

    y_test = data['y_test']
    sensitive_test = data['sensitive_test']
    models = data['models']

    # Get model names
    model_names = list(models.keys())
    print(f"\nAnalyzing {len(model_names)} models:")
    for name in model_names:
        print(f"  - {name}")

    # Calculate DPD and EOD for each model
    print("\n" + "-" * 40)
    print("Demographic Parity and Equalized Odds")
    print("-" * 40)

    fairness_results = []
    for name in model_names:
        y_pred = data[f'{name}_pred']
        y_prob = data[f'{name}_prob']

        dpd = demographic_parity_difference(y_pred, sensitive_test)
        eod = equalized_odds_difference(y_test.values, y_pred, sensitive_test.values)

        # Black-White selection rate gap
        black_rate, white_rate, bw_gap = selection_rate_gap(
            y_pred, sensitive_test.values, 'Black', 'White'
        )

        # Cohen's h effect size
        h = cohens_h(black_rate, white_rate)

        result = {
            'model': name,
            'dpd': dpd,
            'eod': eod,
            'selection_rate_black': black_rate,
            'selection_rate_white': white_rate,
            'black_white_gap': bw_gap,
            'cohens_h': h
        }
        fairness_results.append(result)

        print(f"\n{name}:")
        print(f"  DPD: {dpd:.4f}")
        print(f"  EOD: {eod:.4f}")
        print(f"  Black selection rate: {black_rate:.4f}")
        print(f"  White selection rate: {white_rate:.4f}")
        print(f"  Black-White gap: {bw_gap:.4f} ({bw_gap*100:.1f} pp)")
        print(f"  Cohen's h: {h:.4f}")

    fairness_df = pd.DataFrame(fairness_results)

    # Group-wise metrics
    print("\n" + "-" * 40)
    print("Group-wise Metrics")
    print("-" * 40)

    all_group_metrics = []
    for name in model_names:
        y_pred = data[f'{name}_pred']
        y_prob = data[f'{name}_prob']

        gm = group_metrics(y_test.values, y_pred, y_prob, sensitive_test.values)
        gm['model'] = name
        all_group_metrics.append(gm)

    group_metrics_df = pd.concat(all_group_metrics, ignore_index=True)

    # Print summary for XGBoost (primary model)
    print("\nXGBoost Group Metrics:")
    xgb_groups = group_metrics_df[group_metrics_df['model'] == 'XGBoost']
    for _, row in xgb_groups.iterrows():
        print(f"\n  {row['group']} (n={row['n']:,}):")
        print(f"    Selection rate: {row['selection_rate']:.4f}")
        print(f"    TPR: {row['tpr']:.4f}")
        print(f"    FPR: {row['fpr']:.4f}")
        print(f"    Base rate: {row['base_rate']:.4f}")

    # Intersectional analysis (race x sex)
    print("\n" + "-" * 40)
    print("Intersectional Analysis (Race x Sex)")
    print("-" * 40)

    # Load full test data for intersectional groups
    data_path = project_root / 'data' / 'processed' / 'analysis_dataset.csv'
    df = pd.read_csv(data_path)

    # Get test indices
    from sklearn.model_selection import train_test_split
    from src.data_utils import get_feature_columns

    feature_cols = get_feature_columns(include_race=True)
    X = df[feature_cols]
    y = df['hiv_tested']
    sensitive = df['race']

    _, X_test, _, y_test_full, _, sens_test_full = train_test_split(
        X, y, sensitive, test_size=0.2, random_state=42, stratify=y
    )

    # Create intersectional groups
    test_indices = X_test.index
    df_test = df.loc[test_indices].copy()
    df_test['intersectional_group'] = create_intersectional_groups(df_test)

    intersectional_results = []
    for name in model_names:
        y_pred = data[f'{name}_pred']
        y_prob = data[f'{name}_prob']

        groups = df_test['intersectional_group'].unique()
        for group in groups:
            mask = df_test['intersectional_group'] == group
            if mask.sum() > 0:
                group_pred = y_pred[mask.values]
                group_true = y_test.values[mask.values]
                group_prob = y_prob[mask.values]

                intersectional_results.append({
                    'model': name,
                    'group': group,
                    'n': mask.sum(),
                    'selection_rate': group_pred.mean(),
                    'tpr': group_pred[group_true == 1].mean() if (group_true == 1).sum() > 0 else np.nan,
                    'fpr': group_pred[group_true == 0].mean() if (group_true == 0).sum() > 0 else np.nan,
                    'base_rate': group_true.mean()
                })

    intersectional_df = pd.DataFrame(intersectional_results)

    # Print summary
    print("\nXGBoost Intersectional Selection Rates:")
    xgb_inter = intersectional_df[intersectional_df['model'] == 'XGBoost'].sort_values(
        'selection_rate', ascending=False
    )
    for _, row in xgb_inter.head(10).iterrows():
        print(f"  {row['group']}: {row['selection_rate']:.4f} (n={row['n']:,})")

    # Calibration by race
    print("\n" + "-" * 40)
    print("Calibration Analysis by Race")
    print("-" * 40)

    calibration_results = []
    for name in model_names:
        y_prob = data[f'{name}_prob']

        cal_data = calibration_by_group(
            y_test.values, y_prob, sensitive_test.values, n_bins=10
        )

        for group, cal in cal_data.items():
            calibration_results.append({
                'model': name,
                'group': group,
                'brier_score': cal['brier_score'],
                'n': cal['n']
            })

    calibration_df = pd.DataFrame(calibration_results)

    # Print summary
    print("\nXGBoost Brier Scores by Race:")
    xgb_cal = calibration_df[calibration_df['model'] == 'XGBoost']
    for _, row in xgb_cal.iterrows():
        print(f"  {row['group']}: {row['brier_score']:.4f}")

    # Denver HIV Risk Score coefficient audit
    print("\n" + "-" * 40)
    print("Denver HIV Risk Score Coefficient Audit")
    print("-" * 40)

    # The Denver score assigns points based on risk factors
    # Here we audit the logistic regression coefficients
    lr_model = models['Logistic Regression']
    coef_df = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': lr_model.coef_[0],
        'odds_ratio': np.exp(lr_model.coef_[0])
    }).sort_values('coefficient', ascending=False)

    print("\nLogistic Regression Coefficients (proxy for Denver score audit):")
    for _, row in coef_df.iterrows():
        print(f"  {row['feature']}: coef={row['coefficient']:.4f}, OR={row['odds_ratio']:.4f}")

    # Check race coefficients specifically
    print("\nRace-related coefficients:")
    race_coefs = coef_df[coef_df['feature'].str.startswith('race_')]
    for _, row in race_coefs.iterrows():
        print(f"  {row['feature']}: coef={row['coefficient']:.4f}, OR={row['odds_ratio']:.4f}")

    # Save results
    print("\n" + "-" * 40)
    print("Saving Results")
    print("-" * 40)

    fairness_path = results_dir / 'fairness_metrics.csv'
    fairness_df.to_csv(fairness_path, index=False)
    print(f"Saved fairness metrics to {fairness_path}")

    group_path = results_dir / 'group_metrics.csv'
    group_metrics_df.to_csv(group_path, index=False)
    print(f"Saved group metrics to {group_path}")

    inter_path = results_dir / 'intersectional_analysis.csv'
    intersectional_df.to_csv(inter_path, index=False)
    print(f"Saved intersectional analysis to {inter_path}")

    cal_path = results_dir / 'calibration_by_race.csv'
    calibration_df.to_csv(cal_path, index=False)
    print(f"Saved calibration data to {cal_path}")

    # Verification
    print("\n" + "=" * 60)
    print("Fairness Audit Summary")
    print("=" * 60)
    print("\nExpected values (from manuscript):")
    print("  XGBoost DPD: ~0.539")
    print("  Black-White gap: ~49.7 percentage points")

    xgb_result = fairness_df[fairness_df['model'] == 'XGBoost'].iloc[0]
    print(f"\nActual values:")
    print(f"  XGBoost DPD: {xgb_result['dpd']:.4f}")
    print(f"  Black-White gap: {xgb_result['black_white_gap']*100:.1f} percentage points")

    return fairness_df, group_metrics_df, intersectional_df


if __name__ == "__main__":
    main()
