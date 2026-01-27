#!/usr/bin/env python3
"""
06_statistical_tests.py - Robustness and statistical validation

This script:
1. Bootstrap confidence intervals for DPD (500 iterations)
2. 5-fold cross-validation
3. 10 random seed stability analysis
4. McNemar's test (baseline vs mitigated)
5. Chi-square independence tests
6. Cohen's h effect sizes
7. Two-proportion z-tests with Bonferroni correction

Output:
    - results/bootstrap_ci.csv
    - results/cv_results.csv
    - results/statistical_tests.csv
    - results/seed_stability.csv
"""

import os
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from src.data_utils import get_feature_columns
from src.fairness_metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    cohens_h,
    selection_rate_gap
)
from src.mitigation import apply_threshold_optimizer


def bootstrap_dpd(y_pred, sensitive, n_iterations=500, alpha=0.05):
    """Calculate bootstrap confidence interval for DPD."""
    n = len(y_pred)
    dpd_samples = []

    for _ in range(n_iterations):
        idx = np.random.choice(n, size=n, replace=True)
        dpd = demographic_parity_difference(y_pred[idx], sensitive[idx])
        dpd_samples.append(dpd)

    dpd_samples = np.array(dpd_samples)
    lower = np.percentile(dpd_samples, 100 * alpha / 2)
    upper = np.percentile(dpd_samples, 100 * (1 - alpha / 2))
    mean = np.mean(dpd_samples)
    std = np.std(dpd_samples)

    return mean, std, lower, upper


def mcnemar_test(y_pred1, y_pred2, y_true):
    """Perform McNemar's test comparing two classifiers."""
    # Create contingency table
    # b = correct by model1, incorrect by model2
    # c = incorrect by model1, correct by model2
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)

    b = np.sum(correct1 & ~correct2)
    c = np.sum(~correct1 & correct2)

    # McNemar's test statistic
    if b + c == 0:
        return np.nan, 1.0

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, 1)

    return chi2, p_value


def two_proportion_z_test(p1, n1, p2, n2):
    """Two-proportion z-test."""
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

    if se == 0:
        return np.nan, 1.0

    z = (p1 - p2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return z, p_value


def main():
    """Main statistical testing pipeline."""
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data' / 'processed' / 'analysis_dataset.csv'
    results_dir = project_root / 'results'
    predictions_path = results_dir / 'model_predictions.pkl'

    print("=" * 60)
    print("HIV Testing Fairness - Statistical Tests")
    print("=" * 60)

    # Load data
    print(f"\nLoading data...")
    df = pd.read_csv(data_path)

    with open(predictions_path, 'rb') as f:
        pred_data = pickle.load(f)

    y_test = pred_data['y_test']
    sensitive_test = pred_data['sensitive_test']
    X_train = pred_data['X_train']
    y_train = pred_data['y_train']
    sensitive_train = pred_data['sensitive_train']
    X_test = pred_data['X_test']

    # 1. Bootstrap Confidence Intervals
    print("\n" + "-" * 40)
    print("Bootstrap Confidence Intervals (500 iterations)")
    print("-" * 40)

    bootstrap_results = []
    model_names = ['Logistic Regression', 'Random Forest', 'XGBoost', 'Gradient Boosting']

    for name in model_names:
        print(f"\nProcessing {name}...")
        y_pred = pred_data[f'{name}_pred']

        mean, std, lower, upper = bootstrap_dpd(
            y_pred, sensitive_test.values, n_iterations=500
        )

        bootstrap_results.append({
            'model': name,
            'method': 'baseline',
            'dpd_mean': mean,
            'dpd_std': std,
            'dpd_ci_lower': lower,
            'dpd_ci_upper': upper
        })

        print(f"  DPD: {mean:.4f} [{lower:.4f}, {upper:.4f}]")

    bootstrap_df = pd.DataFrame(bootstrap_results)

    # 2. Cross-Validation
    print("\n" + "-" * 40)
    print("5-Fold Cross-Validation")
    print("-" * 40)

    feature_cols = get_feature_columns(include_race=True)
    X = df[feature_cols]
    y = df['hiv_tested']
    sensitive = df['race']

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []

    for name in ['Logistic Regression', 'XGBoost']:
        print(f"\nProcessing {name}...")

        if name == 'Logistic Regression':
            model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            model = XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1
            )

        fold_dpds = []
        fold_accs = []
        fold_dpd_reductions = []

        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_fold_train = X.iloc[train_idx]
            y_fold_train = y.iloc[train_idx]
            X_fold_test = X.iloc[test_idx]
            y_fold_test = y.iloc[test_idx]
            sens_fold_train = sensitive.iloc[train_idx]
            sens_fold_test = sensitive.iloc[test_idx]

            # Train and predict
            model.fit(X_fold_train, y_fold_train)
            y_pred = model.predict(X_fold_test)

            # Calculate metrics
            dpd = demographic_parity_difference(y_pred, sens_fold_test.values)
            acc = accuracy_score(y_fold_test, y_pred)
            fold_dpds.append(dpd)
            fold_accs.append(acc)

            # Apply mitigation
            try:
                to = apply_threshold_optimizer(
                    model, X_fold_train, y_fold_train, sens_fold_train,
                    constraint='demographic_parity', prefit=True
                )
                y_pred_mit = to.predict(X_fold_test, sensitive_features=sens_fold_test)
                dpd_mit = demographic_parity_difference(y_pred_mit, sens_fold_test.values)
                reduction = (dpd - dpd_mit) / dpd * 100 if dpd > 0 else 0
                fold_dpd_reductions.append(reduction)
            except Exception:
                fold_dpd_reductions.append(np.nan)

        cv_results.append({
            'model': name,
            'dpd_mean': np.mean(fold_dpds),
            'dpd_std': np.std(fold_dpds),
            'accuracy_mean': np.mean(fold_accs),
            'accuracy_std': np.std(fold_accs),
            'dpd_reduction_mean': np.nanmean(fold_dpd_reductions),
            'dpd_reduction_std': np.nanstd(fold_dpd_reductions)
        })

        print(f"  DPD: {np.mean(fold_dpds):.4f} +/- {np.std(fold_dpds):.4f}")
        print(f"  Accuracy: {np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}")
        print(f"  DPD Reduction: {np.nanmean(fold_dpd_reductions):.1f}% +/- {np.nanstd(fold_dpd_reductions):.1f}%")

    cv_df = pd.DataFrame(cv_results)

    # 3. Random Seed Stability
    print("\n" + "-" * 40)
    print("Random Seed Stability (10 seeds)")
    print("-" * 40)

    seeds = list(range(10))
    seed_results = []

    for seed in seeds:
        X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
            X, y, sensitive, test_size=0.2, random_state=seed, stratify=y
        )

        model = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=seed, use_label_encoder=False, eval_metric='logloss', n_jobs=-1
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        dpd = demographic_parity_difference(y_pred, s_te.values)
        acc = accuracy_score(y_te, y_pred)

        seed_results.append({
            'seed': seed,
            'dpd': dpd,
            'accuracy': acc
        })

    seed_df = pd.DataFrame(seed_results)
    print(f"\nXGBoost across seeds:")
    print(f"  DPD: {seed_df['dpd'].mean():.4f} +/- {seed_df['dpd'].std():.4f}")
    print(f"  Accuracy: {seed_df['accuracy'].mean():.4f} +/- {seed_df['accuracy'].std():.4f}")

    # 4. McNemar's Test
    print("\n" + "-" * 40)
    print("McNemar's Test (Baseline vs Mitigated)")
    print("-" * 40)

    mcnemar_results = []

    for name in model_names:
        y_pred_base = pred_data[f'{name}_pred']

        # Apply mitigation
        try:
            model = pred_data['models'][name]
            to = apply_threshold_optimizer(
                model, X_train, y_train, sensitive_train,
                constraint='demographic_parity', prefit=True
            )
            y_pred_mit = to.predict(X_test, sensitive_features=sensitive_test)

            chi2, p_val = mcnemar_test(y_pred_base, y_pred_mit, y_test.values)

            mcnemar_results.append({
                'model': name,
                'chi2': chi2,
                'p_value': p_val,
                'significant': p_val < 0.05
            })

            print(f"\n{name}:")
            print(f"  Chi-square: {chi2:.4f}")
            print(f"  P-value: {p_val:.4f}")
            print(f"  Significant: {p_val < 0.05}")

        except Exception as e:
            print(f"\n{name}: Failed - {e}")

    # 5. Chi-Square Independence Tests
    print("\n" + "-" * 40)
    print("Chi-Square Tests (Selection Rate Independence)")
    print("-" * 40)

    chi_results = []

    for name in model_names:
        y_pred = pred_data[f'{name}_pred']

        # Create contingency table: race x prediction
        contingency = pd.crosstab(sensitive_test, y_pred)
        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)

        chi_results.append({
            'model': name,
            'chi2': chi2,
            'dof': dof,
            'p_value': p_val,
            'significant': p_val < 0.05
        })

        print(f"\n{name}:")
        print(f"  Chi-square: {chi2:.2f}")
        print(f"  P-value: {p_val:.2e}")

    # 6. Cohen's h Effect Sizes
    print("\n" + "-" * 40)
    print("Cohen's h Effect Sizes (Black vs White)")
    print("-" * 40)

    effect_sizes = []

    for name in model_names:
        y_pred = pred_data[f'{name}_pred']

        black_rate, white_rate, gap = selection_rate_gap(
            y_pred, sensitive_test.values, 'Black', 'White'
        )
        h = cohens_h(black_rate, white_rate)

        effect_sizes.append({
            'model': name,
            'black_rate': black_rate,
            'white_rate': white_rate,
            'cohens_h': h,
            'interpretation': 'Large' if abs(h) >= 0.8 else ('Medium' if abs(h) >= 0.5 else 'Small')
        })

        print(f"\n{name}:")
        print(f"  Black rate: {black_rate:.4f}")
        print(f"  White rate: {white_rate:.4f}")
        print(f"  Cohen's h: {h:.4f} ({effect_sizes[-1]['interpretation']})")

    # 7. Two-Proportion Z-Tests with Bonferroni Correction
    print("\n" + "-" * 40)
    print("Two-Proportion Z-Tests (Bonferroni Corrected)")
    print("-" * 40)

    races = ['Black', 'Hispanic', 'Asian', 'AIAN', 'NHPI', 'Multiracial', 'Other']
    n_comparisons = len(races)
    bonferroni_alpha = 0.05 / n_comparisons

    z_test_results = []

    for name in model_names[:1]:  # Just XGBoost for brevity
        y_pred = pred_data[f'{name}_pred']

        white_mask = sensitive_test == 'White'
        white_rate = y_pred[white_mask.values].mean()
        white_n = white_mask.sum()

        print(f"\n{name} (vs White, Bonferroni alpha = {bonferroni_alpha:.4f}):")

        for race in races:
            race_mask = sensitive_test == race
            if race_mask.sum() > 0:
                race_rate = y_pred[race_mask.values].mean()
                race_n = race_mask.sum()

                z, p_val = two_proportion_z_test(race_rate, race_n, white_rate, white_n)

                z_test_results.append({
                    'model': name,
                    'comparison': f'{race} vs White',
                    'z_statistic': z,
                    'p_value': p_val,
                    'significant_bonferroni': p_val < bonferroni_alpha
                })

                sig = '*' if p_val < bonferroni_alpha else ''
                print(f"  {race}: z={z:.2f}, p={p_val:.2e} {sig}")

    # Save all results
    print("\n" + "-" * 40)
    print("Saving Results")
    print("-" * 40)

    bootstrap_path = results_dir / 'bootstrap_ci.csv'
    bootstrap_df.to_csv(bootstrap_path, index=False)
    print(f"Saved bootstrap CIs to {bootstrap_path}")

    cv_path = results_dir / 'cv_results.csv'
    cv_df.to_csv(cv_path, index=False)
    print(f"Saved CV results to {cv_path}")

    seed_path = results_dir / 'seed_stability.csv'
    seed_df.to_csv(seed_path, index=False)
    print(f"Saved seed stability to {seed_path}")

    # Combine statistical tests
    stat_tests = {
        'mcnemar': mcnemar_results,
        'chi_square': chi_results,
        'effect_sizes': effect_sizes,
        'z_tests': z_test_results
    }
    stat_path = results_dir / 'statistical_tests.pkl'
    with open(stat_path, 'wb') as f:
        pickle.dump(stat_tests, f)
    print(f"Saved statistical tests to {stat_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Statistical Testing Summary")
    print("=" * 60)

    print("\nExpected values (from manuscript):")
    print("  CV DPD reduction: 90.9% +/- 3.1%")

    if cv_df[cv_df['model'] == 'XGBoost']['dpd_reduction_mean'].values:
        actual = cv_df[cv_df['model'] == 'XGBoost']['dpd_reduction_mean'].values[0]
        actual_std = cv_df[cv_df['model'] == 'XGBoost']['dpd_reduction_std'].values[0]
        print(f"\nActual values:")
        print(f"  CV DPD reduction (XGBoost): {actual:.1f}% +/- {actual_std:.1f}%")

    return bootstrap_df, cv_df, seed_df


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    main()
