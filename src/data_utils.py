"""
BRFSS Data Loading and Preprocessing Utilities

Functions for parsing BRFSS ASCII data files and creating analysis features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


# BRFSS 2024 column positions (start, end) - 1-indexed from codebook
BRFSS_COLUMNS = {
    'HIVTST7': (167, 167),      # Ever tested for HIV
    '_AGEG5YR': (2128, 2129),   # Age category
    'SEXVAR': (143, 143),       # Sex at birth
    '_RACE': (2131, 2131),      # Race/ethnicity
    'INCOME3': (130, 131),      # Income category
    'ADDEPEV3': (118, 118),     # Ever told had depressive disorder
    'MEDCOST1': (104, 104),     # Could not see doctor due to cost
    'GENHLTH': (91, 91),        # General health status
    '_STATE': (1, 2),           # State FIPS code
}

# Age category midpoints (BRFSS _AGEG5YR values 1-13)
AGE_MIDPOINTS = {
    1: 21,   # 18-24
    2: 27,   # 25-29
    3: 32,   # 30-34
    4: 37,   # 35-39
    5: 42,   # 40-44
    6: 47,   # 45-49
    7: 52,   # 50-54
    8: 57,   # 55-59
    9: 62,   # 60-64
    10: 67,  # 65-69
    11: 72,  # 70-74
    12: 77,  # 75-79
    13: 82,  # 80+
    14: np.nan,  # Don't know/refused
}

# Race/ethnicity labels (BRFSS _RACE values)
RACE_LABELS = {
    1: 'White',
    2: 'Black',
    3: 'AIAN',        # American Indian/Alaska Native
    4: 'Asian',
    5: 'NHPI',        # Native Hawaiian/Pacific Islander
    6: 'Other',
    7: 'Multiracial',
    8: 'Hispanic',
    9: np.nan,        # Don't know/refused
}

# Southern state FIPS codes
SOUTH_FIPS = [
    1,   # Alabama
    5,   # Arkansas
    10,  # Delaware
    11,  # District of Columbia
    12,  # Florida
    13,  # Georgia
    21,  # Kentucky
    22,  # Louisiana
    24,  # Maryland
    28,  # Mississippi
    37,  # North Carolina
    40,  # Oklahoma
    45,  # South Carolina
    47,  # Tennessee
    48,  # Texas
    51,  # Virginia
    54,  # West Virginia
]


def load_brfss(filepath: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load BRFSS ASCII data file using fixed-width column positions.

    Args:
        filepath: Path to BRFSS .ASC file
        nrows: Optional number of rows to read (for testing)

    Returns:
        DataFrame with extracted BRFSS variables
    """
    # Convert 1-indexed positions to 0-indexed tuples for pandas
    colspecs = [(v[0]-1, v[1]) for v in BRFSS_COLUMNS.values()]
    names = list(BRFSS_COLUMNS.keys())

    df = pd.read_fwf(
        filepath,
        colspecs=colspecs,
        names=names,
        nrows=nrows,
        dtype=str
    )

    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create analysis features from raw BRFSS variables.

    Args:
        df: DataFrame with raw BRFSS variables

    Returns:
        DataFrame with engineered features
    """
    result = pd.DataFrame()

    # Target variable: HIV testing (1=Yes, 2=No, 7/9=Missing)
    result['hiv_tested'] = df['HIVTST7'].apply(
        lambda x: 1 if x == 1 else (0 if x == 2 else np.nan)
    )

    # Age: Convert category to continuous midpoint
    result['age'] = df['_AGEG5YR'].map(AGE_MIDPOINTS)

    # Sex: Male indicator (1=Male, 2=Female)
    result['male'] = (df['SEXVAR'] == 1).astype(float)
    result.loc[df['SEXVAR'].isna(), 'male'] = np.nan

    # Race/ethnicity: Keep original categories and create race indicators
    result['race'] = df['_RACE'].map(RACE_LABELS)
    result['race_white'] = (df['_RACE'] == 1).astype(float)
    result['race_black'] = (df['_RACE'] == 2).astype(float)
    result['race_hispanic'] = (df['_RACE'] == 8).astype(float)
    result['race_asian'] = (df['_RACE'] == 4).astype(float)
    result['race_aian'] = (df['_RACE'] == 3).astype(float)
    result['race_nhpi'] = (df['_RACE'] == 5).astype(float)
    result['race_multiracial'] = (df['_RACE'] == 7).astype(float)
    result['race_other'] = (df['_RACE'] == 6).astype(float)

    # Income: Low income indicator (<$25K, INCOME3 values 1-4)
    result['low_income'] = df['INCOME3'].apply(
        lambda x: 1 if x in [1, 2, 3, 4] else (0 if x in range(5, 12) else np.nan)
    )

    # Depression diagnosis (1=Yes, 2=No, 7/9=Missing)
    result['depression_dx'] = df['ADDEPEV3'].apply(
        lambda x: 1 if x == 1 else (0 if x == 2 else np.nan)
    )

    # Cost barrier to care (1=Yes, 2=No, 7/9=Missing)
    result['cost_barrier'] = df['MEDCOST1'].apply(
        lambda x: 1 if x == 1 else (0 if x == 2 else np.nan)
    )

    # Poor health: Poor/Fair vs Good+ (GENHLTH: 1-2=Excellent/VeryGood, 3=Good, 4-5=Fair/Poor)
    result['poor_health'] = df['GENHLTH'].apply(
        lambda x: 1 if x in [4, 5] else (0 if x in [1, 2, 3] else np.nan)
    )

    # Geographic: Southern state indicator
    result['state_fips'] = df['_STATE']
    result['region_south'] = df['_STATE'].isin(SOUTH_FIPS).astype(float)
    result.loc[df['_STATE'].isna(), 'region_south'] = np.nan

    return result


def define_south_states() -> List[int]:
    """Return list of Southern US state FIPS codes."""
    return SOUTH_FIPS.copy()


def get_feature_columns(include_race: bool = True) -> List[str]:
    """
    Get list of feature columns for modeling.

    Args:
        include_race: Whether to include race indicator variables

    Returns:
        List of feature column names
    """
    base_features = [
        'age', 'male', 'low_income', 'depression_dx',
        'cost_barrier', 'poor_health', 'region_south'
    ]

    race_features = [
        'race_black', 'race_hispanic', 'race_asian',
        'race_aian', 'race_nhpi', 'race_multiracial', 'race_other'
    ]

    if include_race:
        return base_features + race_features
    return base_features


def preprocess_for_modeling(
    df: pd.DataFrame,
    include_race: bool = True,
    drop_missing: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare dataset for modeling.

    Args:
        df: DataFrame with engineered features
        include_race: Whether to include race features
        drop_missing: Whether to drop records with missing values

    Returns:
        Tuple of (X features, y target, race sensitive attribute)
    """
    feature_cols = get_feature_columns(include_race)

    # Select relevant columns
    analysis_cols = feature_cols + ['hiv_tested', 'race']
    subset = df[analysis_cols].copy()

    if drop_missing:
        subset = subset.dropna()

    X = subset[feature_cols]
    y = subset['hiv_tested']
    sensitive = subset['race']

    return X, y, sensitive


def create_intersectional_groups(df: pd.DataFrame) -> pd.Series:
    """
    Create intersectional race x sex groups.

    Args:
        df: DataFrame with 'race' and 'male' columns

    Returns:
        Series with intersectional group labels
    """
    sex_label = df['male'].apply(lambda x: 'Male' if x == 1 else 'Female')
    return df['race'] + '_' + sex_label
