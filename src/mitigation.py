"""
Fairness Mitigation Wrapper Functions

Implements wrappers for fairlearn's ThresholdOptimizer and ExponentiatedGradient.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds


def apply_threshold_optimizer(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    sensitive: np.ndarray,
    constraint: str = 'demographic_parity',
    prefit: bool = True
) -> ThresholdOptimizer:
    """
    Apply ThresholdOptimizer post-processing for fairness.

    Args:
        estimator: Trained sklearn estimator with predict_proba
        X: Feature matrix
        y: True labels
        sensitive: Sensitive attribute values
        constraint: 'demographic_parity' or 'equalized_odds'
        prefit: Whether the estimator is already fitted

    Returns:
        Fitted ThresholdOptimizer
    """
    threshold_optimizer = ThresholdOptimizer(
        estimator=estimator,
        constraints=constraint,
        prefit=prefit,
        predict_method='predict_proba'
    )

    threshold_optimizer.fit(X, y, sensitive_features=sensitive)

    return threshold_optimizer


def apply_exponentiated_gradient(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    sensitive: np.ndarray,
    constraint: str = 'demographic_parity',
    eps: float = 0.01
) -> ExponentiatedGradient:
    """
    Apply ExponentiatedGradient in-processing for fairness.

    Args:
        estimator: Unfitted sklearn estimator
        X: Feature matrix
        y: True labels
        sensitive: Sensitive attribute values
        constraint: 'demographic_parity' or 'equalized_odds'
        eps: Constraint tolerance (smaller = stricter fairness)

    Returns:
        Fitted ExponentiatedGradient model
    """
    if constraint == 'demographic_parity':
        constraints = DemographicParity(difference_bound=eps)
    elif constraint == 'equalized_odds':
        constraints = EqualizedOdds(difference_bound=eps)
    else:
        raise ValueError(f"Unknown constraint: {constraint}")

    mitigator = ExponentiatedGradient(
        estimator=estimator,
        constraints=constraints
    )

    mitigator.fit(X, y, sensitive_features=sensitive)

    return mitigator


def evaluate_mitigation(
    y_true: np.ndarray,
    y_pred_baseline: np.ndarray,
    y_pred_mitigated: np.ndarray,
    sensitive: np.ndarray
) -> Dict[str, float]:
    """
    Compare baseline and mitigated predictions.

    Args:
        y_true: True labels
        y_pred_baseline: Baseline predictions
        y_pred_mitigated: Mitigated predictions
        sensitive: Sensitive attribute values

    Returns:
        Dictionary with comparison metrics
    """
    from .fairness_metrics import demographic_parity_difference, equalized_odds_difference
    from sklearn.metrics import accuracy_score

    dpd_baseline = demographic_parity_difference(y_pred_baseline, sensitive)
    dpd_mitigated = demographic_parity_difference(y_pred_mitigated, sensitive)

    eod_baseline = equalized_odds_difference(y_true, y_pred_baseline, sensitive)
    eod_mitigated = equalized_odds_difference(y_true, y_pred_mitigated, sensitive)

    acc_baseline = accuracy_score(y_true, y_pred_baseline)
    acc_mitigated = accuracy_score(y_true, y_pred_mitigated)

    dpd_reduction = (dpd_baseline - dpd_mitigated) / dpd_baseline * 100 if dpd_baseline > 0 else 0

    return {
        'dpd_baseline': dpd_baseline,
        'dpd_mitigated': dpd_mitigated,
        'dpd_reduction_pct': dpd_reduction,
        'eod_baseline': eod_baseline,
        'eod_mitigated': eod_mitigated,
        'accuracy_baseline': acc_baseline,
        'accuracy_mitigated': acc_mitigated,
        'accuracy_drop': acc_baseline - acc_mitigated
    }


def grid_search_epsilon(
    estimator: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sensitive_train: np.ndarray,
    sensitive_test: np.ndarray,
    epsilons: List[float] = None,
    constraint: str = 'demographic_parity'
) -> pd.DataFrame:
    """
    Grid search over epsilon values for ExponentiatedGradient.

    Args:
        estimator: Unfitted sklearn estimator (will be cloned)
        X_train, y_train: Training data
        X_test, y_test: Test data
        sensitive_train, sensitive_test: Sensitive attributes
        epsilons: List of epsilon values to try
        constraint: Constraint type

    Returns:
        DataFrame with results for each epsilon
    """
    from sklearn.base import clone
    from sklearn.metrics import accuracy_score
    from .fairness_metrics import demographic_parity_difference, equalized_odds_difference

    if epsilons is None:
        epsilons = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

    results = []

    for eps in epsilons:
        try:
            est_clone = clone(estimator)
            mitigator = apply_exponentiated_gradient(
                est_clone, X_train, y_train, sensitive_train,
                constraint=constraint, eps=eps
            )

            y_pred = mitigator.predict(X_test)

            dpd = demographic_parity_difference(y_pred, sensitive_test)
            eod = equalized_odds_difference(y_test, y_pred, sensitive_test)
            acc = accuracy_score(y_test, y_pred)

            results.append({
                'epsilon': eps,
                'dpd': dpd,
                'eod': eod,
                'accuracy': acc
            })
        except Exception as e:
            print(f"Error with epsilon={eps}: {e}")
            results.append({
                'epsilon': eps,
                'dpd': np.nan,
                'eod': np.nan,
                'accuracy': np.nan
            })

    return pd.DataFrame(results)


def pareto_frontier(
    results_df: pd.DataFrame,
    accuracy_col: str = 'accuracy',
    fairness_col: str = 'dpd',
    minimize_fairness: bool = True
) -> pd.DataFrame:
    """
    Identify Pareto-optimal models (maximize accuracy, optimize fairness).

    Args:
        results_df: DataFrame with model results
        accuracy_col: Column name for accuracy metric
        fairness_col: Column name for fairness metric
        minimize_fairness: Whether lower fairness values are better

    Returns:
        DataFrame with only Pareto-optimal rows
    """
    df = results_df.copy()
    df = df.dropna(subset=[accuracy_col, fairness_col])

    # Sort by accuracy descending
    df = df.sort_values(accuracy_col, ascending=False).reset_index(drop=True)

    pareto_mask = np.zeros(len(df), dtype=bool)

    if minimize_fairness:
        # For each point, check if any other point dominates it
        # Point A dominates B if A has better accuracy AND better (lower) fairness
        best_fairness = np.inf
        for i, row in df.iterrows():
            if row[fairness_col] < best_fairness:
                pareto_mask[i] = True
                best_fairness = row[fairness_col]
    else:
        # Higher fairness is better
        best_fairness = -np.inf
        for i, row in df.iterrows():
            if row[fairness_col] > best_fairness:
                pareto_mask[i] = True
                best_fairness = row[fairness_col]

    return df[pareto_mask]


def run_mitigation_experiment(
    models: Dict[str, BaseEstimator],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sensitive_train: np.ndarray,
    sensitive_test: np.ndarray
) -> pd.DataFrame:
    """
    Run full mitigation experiment on multiple models.

    Args:
        models: Dictionary of {name: fitted_model}
        X_train, y_train: Training data
        X_test, y_test: Test data
        sensitive_train, sensitive_test: Sensitive attributes

    Returns:
        DataFrame with all results
    """
    from sklearn.metrics import accuracy_score
    from .fairness_metrics import demographic_parity_difference, equalized_odds_difference

    results = []

    for name, model in models.items():
        # Baseline
        y_pred_baseline = model.predict(X_test)
        y_prob_baseline = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        baseline_metrics = {
            'model': name,
            'method': 'baseline',
            'accuracy': accuracy_score(y_test, y_pred_baseline),
            'dpd': demographic_parity_difference(y_pred_baseline, sensitive_test),
            'eod': equalized_odds_difference(y_test, y_pred_baseline, sensitive_test)
        }
        results.append(baseline_metrics)

        # ThresholdOptimizer - Demographic Parity
        try:
            to_dp = apply_threshold_optimizer(
                model, X_train, y_train, sensitive_train,
                constraint='demographic_parity'
            )
            y_pred_to_dp = to_dp.predict(X_test, sensitive_features=sensitive_test)

            results.append({
                'model': name,
                'method': 'ThresholdOptimizer_DP',
                'accuracy': accuracy_score(y_test, y_pred_to_dp),
                'dpd': demographic_parity_difference(y_pred_to_dp, sensitive_test),
                'eod': equalized_odds_difference(y_test, y_pred_to_dp, sensitive_test)
            })
        except Exception as e:
            print(f"ThresholdOptimizer DP failed for {name}: {e}")

        # ThresholdOptimizer - Equalized Odds
        try:
            to_eo = apply_threshold_optimizer(
                model, X_train, y_train, sensitive_train,
                constraint='equalized_odds'
            )
            y_pred_to_eo = to_eo.predict(X_test, sensitive_features=sensitive_test)

            results.append({
                'model': name,
                'method': 'ThresholdOptimizer_EO',
                'accuracy': accuracy_score(y_test, y_pred_to_eo),
                'dpd': demographic_parity_difference(y_pred_to_eo, sensitive_test),
                'eod': equalized_odds_difference(y_test, y_pred_to_eo, sensitive_test)
            })
        except Exception as e:
            print(f"ThresholdOptimizer EO failed for {name}: {e}")

    return pd.DataFrame(results)
