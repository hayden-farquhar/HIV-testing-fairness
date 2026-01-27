"""
Fairness Metrics Calculation Functions

Implements demographic parity difference, equalized odds difference,
calibration metrics, and statistical effect sizes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def demographic_parity_difference(
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    reference_group: Optional[str] = None
) -> float:
    """
    Calculate Demographic Parity Difference.

    DPD = max|P(Y_hat=1|A=a) - P(Y_hat=1|A=b)| for all group pairs

    Args:
        y_pred: Predicted labels (binary)
        sensitive: Sensitive attribute values
        reference_group: Optional reference group for comparison

    Returns:
        Maximum absolute difference in selection rates
    """
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)

    groups = np.unique(sensitive[~pd.isna(sensitive)])
    selection_rates = {}

    for group in groups:
        mask = sensitive == group
        if mask.sum() > 0:
            selection_rates[group] = y_pred[mask].mean()

    if len(selection_rates) < 2:
        return 0.0

    rates = list(selection_rates.values())
    if reference_group and reference_group in selection_rates:
        ref_rate = selection_rates[reference_group]
        return max(abs(r - ref_rate) for r in rates)

    return max(rates) - min(rates)


def equalized_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray
) -> float:
    """
    Calculate Equalized Odds Difference.

    EOD = max of TPR difference and FPR difference across groups

    Args:
        y_true: True labels
        y_pred: Predicted labels (binary)
        sensitive: Sensitive attribute values

    Returns:
        Maximum of TPR and FPR differences
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)

    groups = np.unique(sensitive[~pd.isna(sensitive)])
    tpr_by_group = {}
    fpr_by_group = {}

    for group in groups:
        mask = sensitive == group
        y_t = y_true[mask]
        y_p = y_pred[mask]

        # TPR = TP / (TP + FN) = P(Y_hat=1 | Y=1)
        pos_mask = y_t == 1
        if pos_mask.sum() > 0:
            tpr_by_group[group] = y_p[pos_mask].mean()

        # FPR = FP / (FP + TN) = P(Y_hat=1 | Y=0)
        neg_mask = y_t == 0
        if neg_mask.sum() > 0:
            fpr_by_group[group] = y_p[neg_mask].mean()

    tpr_diff = 0.0
    fpr_diff = 0.0

    if len(tpr_by_group) >= 2:
        tpr_values = list(tpr_by_group.values())
        tpr_diff = max(tpr_values) - min(tpr_values)

    if len(fpr_by_group) >= 2:
        fpr_values = list(fpr_by_group.values())
        fpr_diff = max(fpr_values) - min(fpr_values)

    return max(tpr_diff, fpr_diff)


def group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    sensitive: np.ndarray
) -> pd.DataFrame:
    """
    Calculate comprehensive metrics by group.

    Args:
        y_true: True labels
        y_pred: Predicted labels (binary)
        y_prob: Predicted probabilities
        sensitive: Sensitive attribute values

    Returns:
        DataFrame with metrics for each group
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    sensitive = np.asarray(sensitive)

    groups = np.unique(sensitive[~pd.isna(sensitive)])
    results = []

    for group in groups:
        mask = sensitive == group
        y_t = y_true[mask]
        y_p = y_pred[mask]
        y_proba = y_prob[mask]

        n = mask.sum()
        n_positive = (y_t == 1).sum()
        n_negative = (y_t == 0).sum()

        # Selection rate
        selection_rate = y_p.mean()

        # True positive rate
        tpr = y_p[y_t == 1].mean() if n_positive > 0 else np.nan

        # False positive rate
        fpr = y_p[y_t == 0].mean() if n_negative > 0 else np.nan

        # Precision
        predicted_positive = (y_p == 1).sum()
        precision = ((y_t == 1) & (y_p == 1)).sum() / predicted_positive if predicted_positive > 0 else np.nan

        # Brier score
        brier = brier_score_loss(y_t, y_proba)

        # Base rate (actual positive rate)
        base_rate = y_t.mean()

        results.append({
            'group': group,
            'n': n,
            'n_positive': n_positive,
            'n_negative': n_negative,
            'selection_rate': selection_rate,
            'tpr': tpr,
            'fpr': fpr,
            'precision': precision,
            'brier_score': brier,
            'base_rate': base_rate
        })

    return pd.DataFrame(results)


def cohens_h(p1: float, p2: float) -> float:
    """
    Calculate Cohen's h effect size for difference between proportions.

    h = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))

    Interpretation:
        - Small: 0.2
        - Medium: 0.5
        - Large: 0.8

    Args:
        p1: First proportion
        p2: Second proportion

    Returns:
        Cohen's h effect size
    """
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    return phi1 - phi2


def calibration_by_group(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sensitive: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Dict]:
    """
    Calculate calibration curves by group.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        sensitive: Sensitive attribute values
        n_bins: Number of bins for calibration curve

    Returns:
        Dictionary with calibration data for each group
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    sensitive = np.asarray(sensitive)

    groups = np.unique(sensitive[~pd.isna(sensitive)])
    calibration_data = {}

    for group in groups:
        mask = sensitive == group
        y_t = y_true[mask]
        y_p = y_prob[mask]

        try:
            prob_true, prob_pred = calibration_curve(
                y_t, y_p, n_bins=n_bins, strategy='uniform'
            )
            brier = brier_score_loss(y_t, y_p)

            calibration_data[group] = {
                'prob_true': prob_true,
                'prob_pred': prob_pred,
                'brier_score': brier,
                'n': mask.sum()
            }
        except ValueError:
            # Not enough samples for calibration curve
            calibration_data[group] = {
                'prob_true': np.array([]),
                'prob_pred': np.array([]),
                'brier_score': np.nan,
                'n': mask.sum()
            }

    return calibration_data


def selection_rate_gap(
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    group1: str,
    group2: str
) -> Tuple[float, float, float]:
    """
    Calculate selection rate gap between two specific groups.

    Args:
        y_pred: Predicted labels
        sensitive: Sensitive attribute values
        group1: First group name
        group2: Second group name

    Returns:
        Tuple of (rate1, rate2, gap)
    """
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)

    mask1 = sensitive == group1
    mask2 = sensitive == group2

    rate1 = y_pred[mask1].mean() if mask1.sum() > 0 else np.nan
    rate2 = y_pred[mask2].mean() if mask2.sum() > 0 else np.nan
    gap = rate1 - rate2

    return rate1, rate2, gap


def disparate_impact_ratio(
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    privileged_group: str,
    unprivileged_group: str
) -> float:
    """
    Calculate Disparate Impact Ratio.

    DIR = P(Y_hat=1|A=unprivileged) / P(Y_hat=1|A=privileged)

    The 4/5ths rule considers DIR < 0.8 as evidence of adverse impact.

    Args:
        y_pred: Predicted labels
        sensitive: Sensitive attribute values
        privileged_group: Name of privileged group
        unprivileged_group: Name of unprivileged group

    Returns:
        Disparate impact ratio
    """
    rate_priv, rate_unpriv, _ = selection_rate_gap(
        y_pred, sensitive, privileged_group, unprivileged_group
    )

    if rate_priv == 0:
        return np.inf if rate_unpriv > 0 else 1.0

    return rate_unpriv / rate_priv


def calculate_all_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    sensitive: np.ndarray,
    reference_group: str = 'White'
) -> Dict[str, float]:
    """
    Calculate comprehensive fairness metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        sensitive: Sensitive attribute values
        reference_group: Reference group for comparisons

    Returns:
        Dictionary with all fairness metrics
    """
    dpd = demographic_parity_difference(y_pred, sensitive, reference_group)
    eod = equalized_odds_difference(y_true, y_pred, sensitive)

    # Calculate group metrics
    gm = group_metrics(y_true, y_pred, y_prob, sensitive)

    # Get reference group selection rate
    ref_row = gm[gm['group'] == reference_group]
    ref_rate = ref_row['selection_rate'].values[0] if len(ref_row) > 0 else np.nan

    # Calculate effect sizes vs reference
    effect_sizes = {}
    for _, row in gm.iterrows():
        if row['group'] != reference_group:
            h = cohens_h(row['selection_rate'], ref_rate)
            effect_sizes[f"cohens_h_{row['group']}_vs_{reference_group}"] = h

    return {
        'dpd': dpd,
        'eod': eod,
        'max_selection_rate': gm['selection_rate'].max(),
        'min_selection_rate': gm['selection_rate'].min(),
        'reference_selection_rate': ref_rate,
        **effect_sizes
    }
