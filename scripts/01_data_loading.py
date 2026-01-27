#!/usr/bin/env python3
"""
01_data_loading.py - Load and preprocess BRFSS 2024 data

This script:
1. Parses the BRFSS ASCII file using codebook column positions
2. Extracts relevant variables for HIV testing fairness analysis
3. Creates binary features for modeling
4. Handles missing data and outputs processed dataset

Output: data/processed/analysis_dataset.csv
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.data_utils import (
    load_brfss,
    create_features,
    get_feature_columns,
    preprocess_for_modeling
)


def main():
    """Main data loading pipeline."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    raw_data_path = project_root / 'data' / 'raw' / 'LLCP2024.ASC'
    output_path = project_root / 'data' / 'processed' / 'analysis_dataset.csv'

    print("=" * 60)
    print("HIV Testing Fairness - Data Loading Pipeline")
    print("=" * 60)

    # Check if raw data exists
    if not raw_data_path.exists():
        print(f"\nError: BRFSS data file not found at {raw_data_path}")
        print("\nPlease download BRFSS 2024 data from:")
        print("https://www.cdc.gov/brfss/annual_data/annual_2024.html")
        print("\nPlace the LLCP2024.ASC file in data/raw/")
        sys.exit(1)

    # Load raw BRFSS data
    print(f"\nLoading BRFSS data from {raw_data_path}...")
    print("This may take several minutes for the full dataset...")

    try:
        df_raw = load_brfss(str(raw_data_path))
        print(f"Loaded {len(df_raw):,} records")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Create features
    print("\nCreating analysis features...")
    df = create_features(df_raw)

    # Report initial statistics
    print("\n" + "-" * 40)
    print("Initial Data Summary")
    print("-" * 40)
    print(f"Total records: {len(df):,}")
    print(f"\nHIV Testing (target variable):")
    print(df['hiv_tested'].value_counts(dropna=False))

    print(f"\nRace/Ethnicity distribution:")
    print(df['race'].value_counts(dropna=False))

    print(f"\nSex distribution:")
    print(f"Male: {df['male'].sum():,.0f} ({df['male'].mean()*100:.1f}%)")
    print(f"Female: {(df['male'] == 0).sum():,.0f} ({(df['male'] == 0).mean()*100:.1f}%)")

    # Handle missing data
    print("\n" + "-" * 40)
    print("Missing Data Analysis")
    print("-" * 40)

    feature_cols = get_feature_columns(include_race=True)
    missing_cols = ['hiv_tested', 'race'] + feature_cols

    for col in missing_cols:
        if col in df.columns:
            missing_pct = df[col].isna().mean() * 100
            print(f"{col}: {missing_pct:.1f}% missing")

    # Drop records with missing values
    print("\nDropping records with missing values...")
    df_complete = df.dropna(subset=['hiv_tested', 'race'] + feature_cols)
    print(f"Records remaining: {len(df_complete):,} ({len(df_complete)/len(df)*100:.1f}%)")

    # Final dataset statistics
    print("\n" + "-" * 40)
    print("Final Dataset Summary")
    print("-" * 40)
    print(f"Total complete records: {len(df_complete):,}")

    print(f"\nHIV Testing rates:")
    tested_rate = df_complete['hiv_tested'].mean()
    print(f"Ever tested: {df_complete['hiv_tested'].sum():,.0f} ({tested_rate*100:.1f}%)")
    print(f"Never tested: {(df_complete['hiv_tested'] == 0).sum():,.0f} ({(1-tested_rate)*100:.1f}%)")

    print(f"\nRace/Ethnicity distribution (complete cases):")
    race_dist = df_complete['race'].value_counts()
    for race, count in race_dist.items():
        pct = count / len(df_complete) * 100
        print(f"  {race}: {count:,} ({pct:.1f}%)")

    print(f"\nHIV Testing rates by race:")
    for race in race_dist.index:
        race_data = df_complete[df_complete['race'] == race]
        rate = race_data['hiv_tested'].mean()
        print(f"  {race}: {rate*100:.1f}%")

    print(f"\nFeature distributions:")
    binary_features = ['male', 'low_income', 'depression_dx', 'cost_barrier', 'poor_health', 'region_south']
    for feat in binary_features:
        if feat in df_complete.columns:
            rate = df_complete[feat].mean()
            print(f"  {feat}: {rate*100:.1f}%")

    print(f"\nAge statistics:")
    print(f"  Mean: {df_complete['age'].mean():.1f}")
    print(f"  Std: {df_complete['age'].std():.1f}")
    print(f"  Range: {df_complete['age'].min():.0f} - {df_complete['age'].max():.0f}")

    # Save processed dataset
    print(f"\nSaving processed dataset to {output_path}...")
    os.makedirs(output_path.parent, exist_ok=True)
    df_complete.to_csv(output_path, index=False)
    print(f"Saved {len(df_complete):,} records")

    # Verification
    print("\n" + "=" * 60)
    print("Data Loading Complete")
    print("=" * 60)
    print(f"\nExpected ~386,775 records (manuscript)")
    print(f"Actual: {len(df_complete):,} records")

    return df_complete


if __name__ == "__main__":
    main()
