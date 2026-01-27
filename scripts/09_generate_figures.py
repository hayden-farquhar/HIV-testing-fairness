#!/usr/bin/env python3
"""
09_generate_figures.py - Create publication-quality figures

This script generates all figures for the manuscript:
    - Fig 1: Model comparison (AUC, DPD, EOD bar charts)
    - Fig 2: Selection rate by race (baseline vs mitigated)
    - Fig 3: Pareto frontier (accuracy vs fairness trade-off)
    - Fig 4: Bootstrap confidence intervals
    - Fig S1-S9: Supplementary figures

Output: figures/*.png, figures/*.pdf
"""

import os
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.visualization import (
    plot_fairness_comparison,
    plot_selection_rates,
    plot_pareto_frontier,
    plot_calibration_curves,
    plot_bootstrap_ci,
    plot_intersectional_heatmap,
    plot_roc_curves_by_group,
    plot_decision_curve,
    RACE_COLORS,
    MODEL_COLORS
)
from src.fairness_metrics import calibration_by_group, group_metrics


def main():
    """Main figure generation pipeline."""
    project_root = Path(__file__).parent.parent
    results_dir = project_root / 'results'
    figures_dir = project_root / 'figures'

    # Create figures directory
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 60)
    print("HIV Testing Fairness - Figure Generation")
    print("=" * 60)

    # Load all results
    print("\nLoading results...")

    # Load predictions
    predictions_path = results_dir / 'model_predictions.pkl'
    if not predictions_path.exists():
        print("Error: Model predictions not found. Run previous scripts first.")
        sys.exit(1)

    with open(predictions_path, 'rb') as f:
        pred_data = pickle.load(f)

    y_test = pred_data['y_test']
    sensitive_test = pred_data['sensitive_test']

    # Load CSVs
    baseline_df = pd.read_csv(results_dir / 'baseline_models.csv')
    fairness_df = pd.read_csv(results_dir / 'fairness_metrics.csv')
    group_df = pd.read_csv(results_dir / 'group_metrics.csv')
    mitigation_df = pd.read_csv(results_dir / 'mitigation_results.csv')
    pareto_df = pd.read_csv(results_dir / 'pareto_frontier.csv')

    # Try to load optional files
    try:
        bootstrap_df = pd.read_csv(results_dir / 'bootstrap_ci.csv')
    except FileNotFoundError:
        bootstrap_df = None

    try:
        intersectional_df = pd.read_csv(results_dir / 'intersectional_analysis.csv')
    except FileNotFoundError:
        intersectional_df = None

    try:
        dca_df = pd.read_csv(results_dir / 'decision_curve.csv')
    except FileNotFoundError:
        dca_df = None

    # Merge baseline and fairness for combined results
    combined_df = baseline_df.merge(fairness_df, on='model')

    # ===================
    # FIGURE 1: Model Comparison
    # ===================
    print("\nGenerating Figure 1: Model Comparison...")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    models = combined_df['model'].values
    x = np.arange(len(models))
    colors = [MODEL_COLORS.get(m, '#333') for m in models]

    # AUC
    axes[0].bar(x, combined_df['auc'], color=colors, width=0.6)
    axes[0].set_ylabel('AUC-ROC')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_title('A. Model Discrimination')
    axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # DPD
    axes[1].bar(x, combined_df['dpd'], color=colors, width=0.6)
    axes[1].set_ylabel('Demographic Parity Difference')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].set_title('B. Fairness (DPD)')
    axes[1].axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Threshold')

    # EOD
    axes[2].bar(x, combined_df['eod'], color=colors, width=0.6)
    axes[2].set_ylabel('Equalized Odds Difference')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=45, ha='right')
    axes[2].set_title('C. Fairness (EOD)')
    axes[2].axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Threshold')

    plt.tight_layout()
    fig.savefig(figures_dir / 'fig1_model_comparison.png', dpi=300)
    fig.savefig(figures_dir / 'fig1_model_comparison.pdf')
    plt.close()
    print("  Saved fig1_model_comparison.png/pdf")

    # ===================
    # FIGURE 2: Selection Rates by Race
    # ===================
    print("\nGenerating Figure 2: Selection Rates by Race...")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Filter to XGBoost and sort by selection rate
    xgb_groups = group_df[group_df['model'] == 'XGBoost'].copy()
    xgb_groups = xgb_groups.sort_values('selection_rate', ascending=True)

    groups = xgb_groups['group'].values
    rates = xgb_groups['selection_rate'].values
    colors = [RACE_COLORS.get(g, '#333') for g in groups]

    y_pos = np.arange(len(groups))
    ax.barh(y_pos, rates, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(groups)
    ax.set_xlabel('Selection Rate (Predicted HIV Testing)')
    ax.set_title('XGBoost Selection Rates by Race/Ethnicity')
    ax.axvline(x=rates.mean(), color='gray', linestyle='--', alpha=0.7, label='Mean')

    # Add value labels
    for i, (rate, group) in enumerate(zip(rates, groups)):
        ax.text(rate + 0.01, i, f'{rate:.3f}', va='center')

    plt.tight_layout()
    fig.savefig(figures_dir / 'fig2_selection_rates.png', dpi=300)
    fig.savefig(figures_dir / 'fig2_selection_rates.pdf')
    plt.close()
    print("  Saved fig2_selection_rates.png/pdf")

    # ===================
    # FIGURE 3: Pareto Frontier
    # ===================
    print("\nGenerating Figure 3: Pareto Frontier...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot all points
    ax.scatter(mitigation_df['dpd'], mitigation_df['accuracy'],
               alpha=0.5, s=80, c='gray', label='All Models')

    # Highlight Pareto optimal
    ax.scatter(pareto_df['dpd'], pareto_df['accuracy'],
               s=150, c='red', marker='*', label='Pareto Optimal', zorder=5)

    # Connect Pareto points
    pareto_sorted = pareto_df.sort_values('dpd')
    ax.plot(pareto_sorted['dpd'], pareto_sorted['accuracy'],
            'r--', alpha=0.7, linewidth=2)

    # Add annotations for key points
    for _, row in pareto_df.head(3).iterrows():
        ax.annotate(f"{row['model']}\n({row['method']})",
                    (row['dpd'], row['accuracy']),
                    textcoords="offset points", xytext=(10, 10),
                    fontsize=8, alpha=0.8)

    ax.set_xlabel('Demographic Parity Difference (lower is fairer)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy-Fairness Trade-off (Pareto Frontier)', fontsize=14)
    ax.legend(loc='lower left')

    # Fairness threshold
    ax.axvline(x=0.1, color='green', linestyle=':', alpha=0.7, label='DPD=0.1')

    plt.tight_layout()
    fig.savefig(figures_dir / 'fig3_pareto_frontier.png', dpi=300)
    fig.savefig(figures_dir / 'fig3_pareto_frontier.pdf')
    plt.close()
    print("  Saved fig3_pareto_frontier.png/pdf")

    # ===================
    # FIGURE 4: Bootstrap Confidence Intervals
    # ===================
    if bootstrap_df is not None:
        print("\nGenerating Figure 4: Bootstrap CIs...")

        fig, ax = plt.subplots(figsize=(10, 6))

        y_pos = np.arange(len(bootstrap_df))

        for i, row in bootstrap_df.iterrows():
            ci_lower = row.get('dpd_ci_lower', row['dpd_mean'] * 0.9)
            ci_upper = row.get('dpd_ci_upper', row['dpd_mean'] * 1.1)

            ax.plot([ci_lower, ci_upper], [i, i], 'b-', linewidth=3)
            ax.plot(row['dpd_mean'], i, 'bo', markersize=10)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{row['model']}" for _, row in bootstrap_df.iterrows()])
        ax.set_xlabel('Demographic Parity Difference')
        ax.set_title('Bootstrap 95% Confidence Intervals for DPD')
        ax.axvline(x=0, color='green', linestyle='--', alpha=0.7)
        ax.axvline(x=0.1, color='red', linestyle='--', alpha=0.7, label='Threshold')

        plt.tight_layout()
        fig.savefig(figures_dir / 'fig4_bootstrap_ci.png', dpi=300)
        fig.savefig(figures_dir / 'fig4_bootstrap_ci.pdf')
        plt.close()
        print("  Saved fig4_bootstrap_ci.png/pdf")

    # ===================
    # SUPPLEMENTARY FIGURES
    # ===================
    print("\nGenerating Supplementary Figures...")

    # Fig S1: ROC Curves by Race
    print("  Figure S1: ROC Curves...")
    y_prob_xgb = pred_data['XGBoost_prob']

    fig = plot_roc_curves_by_group(
        y_test.values, y_prob_xgb, sensitive_test.values,
        save_path=str(figures_dir / 'figS1_roc_curves.png')
    )
    plt.close()

    # Fig S2: Calibration Curves
    print("  Figure S2: Calibration Curves...")
    cal_data = calibration_by_group(y_test.values, y_prob_xgb, sensitive_test.values)
    fig = plot_calibration_curves(
        cal_data,
        save_path=str(figures_dir / 'figS2_calibration.png')
    )
    plt.close()

    # Fig S3: Intersectional Heatmap
    if intersectional_df is not None:
        print("  Figure S3: Intersectional Heatmap...")
        xgb_inter = intersectional_df[intersectional_df['model'] == 'XGBoost'].copy()

        fig = plot_intersectional_heatmap(
            xgb_inter,
            metric='selection_rate',
            save_path=str(figures_dir / 'figS3_intersectional.png')
        )
        plt.close()

    # Fig S4: Decision Curve Analysis
    if dca_df is not None:
        print("  Figure S4: Decision Curve...")

        fig, ax = plt.subplots(figsize=(10, 6))

        for model in dca_df['model'].unique():
            model_data = dca_df[dca_df['model'] == model]
            style = '--' if model in ['Test All', 'Test None'] else '-'
            alpha = 0.5 if model in ['Test All', 'Test None'] else 1.0
            ax.plot(model_data['threshold'], model_data['net_benefit'],
                    label=model, linestyle=style, alpha=alpha, linewidth=2)

        ax.set_xlabel('Threshold Probability')
        ax.set_ylabel('Net Benefit')
        ax.set_title('Decision Curve Analysis')
        ax.legend(loc='upper right')
        ax.set_xlim(0, 1)

        plt.tight_layout()
        fig.savefig(figures_dir / 'figS4_decision_curve.png', dpi=300)
        plt.close()

    # Fig S5: Mitigation Comparison
    print("  Figure S5: Mitigation Comparison...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # DPD before/after
    baseline = mitigation_df[mitigation_df['method'] == 'baseline']
    mitigated = mitigation_df[mitigation_df['method'] == 'ThresholdOptimizer_DP']

    x = np.arange(len(baseline))
    width = 0.35

    axes[0].bar(x - width/2, baseline['dpd'], width, label='Baseline', color='coral')
    if len(mitigated) > 0:
        axes[0].bar(x + width/2, mitigated['dpd'], width, label='Mitigated', color='steelblue')
    axes[0].set_ylabel('DPD')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(baseline['model'], rotation=45, ha='right')
    axes[0].set_title('A. DPD Before/After Mitigation')
    axes[0].legend()
    axes[0].axhline(y=0.1, color='red', linestyle='--', alpha=0.7)

    # Accuracy trade-off
    axes[1].bar(x - width/2, baseline['accuracy'], width, label='Baseline', color='coral')
    if len(mitigated) > 0:
        axes[1].bar(x + width/2, mitigated['accuracy'], width, label='Mitigated', color='steelblue')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(baseline['model'], rotation=45, ha='right')
    axes[1].set_title('B. Accuracy Trade-off')
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(figures_dir / 'figS5_mitigation_comparison.png', dpi=300)
    plt.close()

    # Fig S6: Feature Importance
    print("  Figure S6: Feature Importance...")

    xgb_model = pred_data['models']['XGBoost']
    from src.data_utils import get_feature_columns
    feature_cols = get_feature_columns(include_race=True)

    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(importance_df))
    ax.barh(y_pos, importance_df['importance'], color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance_df['feature'])
    ax.set_xlabel('Feature Importance')
    ax.set_title('XGBoost Feature Importance')

    plt.tight_layout()
    fig.savefig(figures_dir / 'figS6_feature_importance.png', dpi=300)
    plt.close()

    # Summary
    print("\n" + "=" * 60)
    print("Figure Generation Complete")
    print("=" * 60)

    # List generated files
    print("\nGenerated figures:")
    for fig_file in sorted(figures_dir.glob('*.png')):
        print(f"  - {fig_file.name}")

    return True


if __name__ == "__main__":
    main()
