"""
Visualization Utilities for HIV Testing Fairness Analysis

Functions for creating publication-quality figures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# Publication-quality settings
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Color palette for race/ethnicity groups
RACE_COLORS = {
    'White': '#1f77b4',
    'Black': '#ff7f0e',
    'Hispanic': '#2ca02c',
    'Asian': '#d62728',
    'AIAN': '#9467bd',
    'NHPI': '#8c564b',
    'Multiracial': '#e377c2',
    'Other': '#7f7f7f'
}

# Model colors
MODEL_COLORS = {
    'Logistic Regression': '#1f77b4',
    'Random Forest': '#ff7f0e',
    'XGBoost': '#2ca02c',
    'Gradient Boosting': '#d62728'
}


def plot_fairness_comparison(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5)
) -> plt.Figure:
    """
    Create bar chart comparing fairness metrics across models.

    Args:
        results_df: DataFrame with columns ['model', 'method', 'accuracy', 'dpd', 'eod']
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    baseline_df = results_df[results_df['method'] == 'baseline'].copy()
    x = np.arange(len(baseline_df))
    width = 0.6

    # AUC/Accuracy
    axes[0].bar(x, baseline_df['accuracy'], width, color=[MODEL_COLORS.get(m, '#333') for m in baseline_df['model']])
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Model')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(baseline_df['model'], rotation=45, ha='right')
    axes[0].set_ylim(0, 1)
    axes[0].set_title('A. Model Accuracy')

    # DPD
    axes[1].bar(x, baseline_df['dpd'], width, color=[MODEL_COLORS.get(m, '#333') for m in baseline_df['model']])
    axes[1].set_ylabel('Demographic Parity Difference')
    axes[1].set_xlabel('Model')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(baseline_df['model'], rotation=45, ha='right')
    axes[1].set_title('B. Fairness (DPD)')
    axes[1].axhline(y=0.1, color='r', linestyle='--', alpha=0.7, label='Threshold')

    # EOD
    axes[2].bar(x, baseline_df['eod'], width, color=[MODEL_COLORS.get(m, '#333') for m in baseline_df['model']])
    axes[2].set_ylabel('Equalized Odds Difference')
    axes[2].set_xlabel('Model')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(baseline_df['model'], rotation=45, ha='right')
    axes[2].set_title('C. Fairness (EOD)')
    axes[2].axhline(y=0.1, color='r', linestyle='--', alpha=0.7, label='Threshold')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Saved figure to {save_path}")

    return fig


def plot_selection_rates(
    group_metrics_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    baseline_label: str = 'Baseline',
    mitigated_label: str = 'Mitigated'
) -> plt.Figure:
    """
    Create grouped bar chart of selection rates by race.

    Args:
        group_metrics_df: DataFrame with columns ['group', 'selection_rate', 'method']
        save_path: Optional path to save figure
        figsize: Figure size
        baseline_label: Label for baseline bars
        mitigated_label: Label for mitigated bars

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get unique groups and methods
    groups = group_metrics_df['group'].unique()
    methods = group_metrics_df['method'].unique()

    x = np.arange(len(groups))
    width = 0.35

    # Plot bars for each method
    for i, method in enumerate(methods):
        method_data = group_metrics_df[group_metrics_df['method'] == method]
        # Ensure correct order
        rates = [method_data[method_data['group'] == g]['selection_rate'].values[0]
                 if g in method_data['group'].values else 0 for g in groups]
        offset = (i - len(methods)/2 + 0.5) * width
        label = baseline_label if 'baseline' in method.lower() else mitigated_label
        ax.bar(x + offset, rates, width, label=label, alpha=0.8)

    ax.set_ylabel('Selection Rate (Predicted HIV Testing)')
    ax.set_xlabel('Race/Ethnicity')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.set_title('Selection Rates by Race/Ethnicity')

    # Add horizontal line at overall average
    ax.axhline(y=group_metrics_df['selection_rate'].mean(), color='gray',
               linestyle='--', alpha=0.5, label='Overall Average')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Saved figure to {save_path}")

    return fig


def plot_pareto_frontier(
    results_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6)
) -> plt.Figure:
    """
    Create Pareto frontier plot (accuracy vs fairness trade-off).

    Args:
        results_df: DataFrame with all model results
        pareto_df: DataFrame with Pareto-optimal points
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot all points
    ax.scatter(results_df['dpd'], results_df['accuracy'],
               alpha=0.5, s=50, label='All Models', c='gray')

    # Highlight Pareto-optimal points
    ax.scatter(pareto_df['dpd'], pareto_df['accuracy'],
               s=100, c='red', marker='*', label='Pareto Optimal', zorder=5)

    # Connect Pareto points
    pareto_sorted = pareto_df.sort_values('dpd')
    ax.plot(pareto_sorted['dpd'], pareto_sorted['accuracy'],
            'r--', alpha=0.7, linewidth=2)

    ax.set_xlabel('Demographic Parity Difference (lower is fairer)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy-Fairness Trade-off (Pareto Frontier)')
    ax.legend()

    # Add fairness threshold line
    ax.axvline(x=0.1, color='green', linestyle=':', alpha=0.7, label='DPD=0.1')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Saved figure to {save_path}")

    return fig


def plot_calibration_curves(
    calibration_data: Dict[str, Dict],
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 8)
) -> plt.Figure:
    """
    Create calibration curves by race/ethnicity.

    Args:
        calibration_data: Dictionary from fairness_metrics.calibration_by_group()
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')

    for group, data in calibration_data.items():
        if len(data['prob_pred']) > 0:
            color = RACE_COLORS.get(group, '#333')
            ax.plot(data['prob_pred'], data['prob_true'],
                    marker='o', label=f"{group} (Brier={data['brier_score']:.3f})",
                    color=color)

    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Curves by Race/Ethnicity')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Saved figure to {save_path}")

    return fig


def plot_bootstrap_ci(
    bootstrap_results: pd.DataFrame,
    metric: str = 'dpd',
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6)
) -> plt.Figure:
    """
    Create forest plot of bootstrap confidence intervals.

    Args:
        bootstrap_results: DataFrame with columns ['model', 'method', metric, f'{metric}_ci_lower', f'{metric}_ci_upper']
        metric: Metric to plot
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    df = bootstrap_results.copy()
    y_positions = np.arange(len(df))

    # Plot confidence intervals
    for i, row in df.iterrows():
        ci_lower = row.get(f'{metric}_ci_lower', row[metric] * 0.9)
        ci_upper = row.get(f'{metric}_ci_upper', row[metric] * 1.1)

        ax.plot([ci_lower, ci_upper], [i, i], 'b-', linewidth=2)
        ax.plot(row[metric], i, 'bo', markersize=8)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{row['model']} ({row['method']})" for _, row in df.iterrows()])
    ax.set_xlabel(f'{metric.upper()}')
    ax.set_title(f'Bootstrap 95% Confidence Intervals for {metric.upper()}')

    # Reference line at 0 for DPD
    if metric == 'dpd':
        ax.axvline(x=0, color='green', linestyle='--', alpha=0.7)
        ax.axvline(x=0.1, color='red', linestyle='--', alpha=0.7, label='Threshold')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Saved figure to {save_path}")

    return fig


def plot_intersectional_heatmap(
    intersectional_df: pd.DataFrame,
    metric: str = 'selection_rate',
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8)
) -> plt.Figure:
    """
    Create heatmap of metrics by intersectional groups (race x sex).

    Args:
        intersectional_df: DataFrame with intersectional group data
        metric: Metric to display
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Pivot data for heatmap
    # Assuming groups are formatted as "Race_Sex"
    df = intersectional_df.copy()
    df['race'] = df['group'].str.rsplit('_', n=1).str[0]
    df['sex'] = df['group'].str.rsplit('_', n=1).str[1]

    pivot = df.pivot(index='race', columns='sex', values=metric)

    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlBu_r',
                ax=ax, cbar_kws={'label': metric.replace('_', ' ').title()})

    ax.set_title(f'{metric.replace("_", " ").title()} by Race and Sex')
    ax.set_xlabel('Sex')
    ax.set_ylabel('Race/Ethnicity')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Saved figure to {save_path}")

    return fig


def plot_roc_curves_by_group(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sensitive: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 8)
) -> plt.Figure:
    """
    Create ROC curves for each demographic group.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        sensitive: Sensitive attribute values
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=figsize)

    groups = np.unique(sensitive[~pd.isna(sensitive)])

    for group in groups:
        mask = sensitive == group
        y_t = y_true[mask]
        y_p = y_prob[mask]

        fpr, tpr, _ = roc_curve(y_t, y_p)
        roc_auc = auc(fpr, tpr)

        color = RACE_COLORS.get(group, '#333')
        ax.plot(fpr, tpr, color=color, label=f'{group} (AUC={roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves by Race/Ethnicity')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Saved figure to {save_path}")

    return fig


def plot_decision_curve(
    thresholds: np.ndarray,
    net_benefits: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6)
) -> plt.Figure:
    """
    Create decision curve analysis plot.

    Args:
        thresholds: Array of threshold probabilities
        net_benefits: Dictionary of {model_name: net_benefit_array}
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot "treat all" and "treat none" reference lines
    prevalence = 0.5  # Placeholder - should be calculated from data
    treat_all_benefit = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
    ax.plot(thresholds, treat_all_benefit, 'k--', label='Test All', alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='-', label='Test None')

    # Plot net benefits for each model
    for name, nb in net_benefits.items():
        ax.plot(thresholds, nb, label=name, linewidth=2)

    ax.set_xlabel('Threshold Probability')
    ax.set_ylabel('Net Benefit')
    ax.set_title('Decision Curve Analysis')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 1)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Saved figure to {save_path}")

    return fig


def create_summary_table(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create formatted summary table for publication.

    Args:
        results_df: DataFrame with model results
        save_path: Optional path to save as CSV

    Returns:
        Formatted DataFrame
    """
    summary = results_df.copy()

    # Round numeric columns
    numeric_cols = ['accuracy', 'dpd', 'eod', 'auc', 'precision', 'recall', 'f1', 'brier_score']
    for col in numeric_cols:
        if col in summary.columns:
            summary[col] = summary[col].round(3)

    # Rename columns for publication
    rename_map = {
        'accuracy': 'Accuracy',
        'dpd': 'DPD',
        'eod': 'EOD',
        'auc': 'AUC',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1',
        'brier_score': 'Brier Score',
        'model': 'Model',
        'method': 'Method'
    }
    summary = summary.rename(columns={k: v for k, v in rename_map.items() if k in summary.columns})

    if save_path:
        summary.to_csv(save_path, index=False)
        print(f"Saved table to {save_path}")

    return summary
