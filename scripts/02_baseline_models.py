#!/usr/bin/env python3
"""
02_baseline_models.py - Train and evaluate baseline ML classifiers

This script:
1. Loads processed BRFSS data
2. Trains Logistic Regression, Random Forest, XGBoost, and Gradient Boosting
3. Evaluates model performance (AUC, accuracy, precision, recall, F1, Brier)
4. Saves trained models and predictions

Output:
    - results/baseline_models.csv
    - results/model_predictions.pkl
"""

import os
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss
)

from src.data_utils import get_feature_columns


def train_baseline_models(X_train, y_train):
    """Train all baseline models with manuscript hyperparameters."""
    models = {}

    print("\nTraining Logistic Regression (with CV tuning)...")
    models['Logistic Regression'] = LogisticRegressionCV(
        cv=5,
        max_iter=1000,
        random_state=42,
        solver='lbfgs',
        penalty='l2'
    )
    models['Logistic Regression'].fit(X_train, y_train)

    print("Training Random Forest...")
    models['Random Forest'] = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    models['Random Forest'].fit(X_train, y_train)

    print("Training XGBoost...")
    models['XGBoost'] = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    )
    models['XGBoost'].fit(X_train, y_train)

    print("Training Gradient Boosting...")
    models['Gradient Boosting'] = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    models['Gradient Boosting'].fit(X_train, y_train)

    return models


def evaluate_model(model, X_test, y_test, model_name):
    """Calculate performance metrics for a model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc': roc_auc_score(y_test, y_prob),
        'brier_score': brier_score_loss(y_test, y_prob)
    }


def main():
    """Main training pipeline."""
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data' / 'processed' / 'analysis_dataset.csv'
    results_dir = project_root / 'results'

    print("=" * 60)
    print("HIV Testing Fairness - Baseline Model Training")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from {data_path}...")
    if not data_path.exists():
        print("Error: Processed data not found. Run 01_data_loading.py first.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} records")

    # Prepare features and target
    feature_cols = get_feature_columns(include_race=True)
    X = df[feature_cols]
    y = df['hiv_tested']
    sensitive = df['race']

    print(f"\nFeatures ({len(feature_cols)}):")
    for feat in feature_cols:
        print(f"  - {feat}")

    # Train/test split (80/20, stratified)
    print("\nSplitting data (80/20, stratified)...")
    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"Training set: {len(X_train):,} records")
    print(f"Test set: {len(X_test):,} records")

    # Train models
    print("\n" + "-" * 40)
    print("Training Baseline Models")
    print("-" * 40)
    models = train_baseline_models(X_train, y_train)

    # Evaluate models
    print("\n" + "-" * 40)
    print("Evaluating Models")
    print("-" * 40)

    results = []
    predictions = {
        'X_test': X_test,
        'y_test': y_test,
        'sensitive_test': sens_test
    }

    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)

        # Store predictions
        predictions[f'{name}_pred'] = model.predict(X_test)
        predictions[f'{name}_prob'] = model.predict_proba(X_test)[:, 1]

        print(f"\n{name}:")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  Brier Score: {metrics['brier_score']:.4f}")

    # Feature importance (from XGBoost)
    print("\n" + "-" * 40)
    print("Feature Importance (XGBoost)")
    print("-" * 40)
    xgb_model = models['XGBoost']
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    for _, row in importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    # Save results
    os.makedirs(results_dir, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_path = results_dir / 'baseline_models.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved results to {results_path}")

    # Save models and predictions
    predictions['models'] = models
    predictions['X_train'] = X_train
    predictions['y_train'] = y_train
    predictions['sensitive_train'] = sens_train

    predictions_path = results_dir / 'model_predictions.pkl'
    with open(predictions_path, 'wb') as f:
        pickle.dump(predictions, f)
    print(f"Saved models and predictions to {predictions_path}")

    # Summary table
    print("\n" + "=" * 60)
    print("Baseline Model Results Summary")
    print("=" * 60)
    print(results_df.to_string(index=False))

    return models, results_df


if __name__ == "__main__":
    main()
