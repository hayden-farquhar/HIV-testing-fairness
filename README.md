# HIV Testing Fairness Audit

Reproducible code repository for the HIV testing fairness audit manuscript. This repository implements a comprehensive algorithmic fairness analysis of machine learning models for predicting HIV testing behavior using BRFSS data.

## Overview

This project audits machine learning models that predict HIV testing behavior for racial/ethnic fairness. Key contributions include:

1. **Fairness Audit**: Comprehensive metrics (DPD, EOD, calibration) across 8 racial/ethnic groups
2. **Mitigation Methods**: ThresholdOptimizer and ExponentiatedGradient implementations
3. **Proxy Discrimination Analysis**: Testing whether removing race features eliminates disparities
4. **Clinical Utility**: Decision curve analysis and Number Needed to Screen metrics
5. **External Validation**: Comparison with Ryan White Program and CDC AtlasPlus data

## Repository Structure

```
HIV-testing-fairness/
├── scripts/                        # Analysis pipeline scripts
│   ├── 01_data_loading.py          # BRFSS data extraction
│   ├── 02_baseline_models.py       # Train ML classifiers
│   ├── 03_fairness_audit.py        # Calculate fairness metrics
│   ├── 04_mitigation.py            # Apply fairness mitigation
│   ├── 05_race_blind_analysis.py   # Proxy discrimination test
│   ├── 06_statistical_tests.py     # Bootstrap CIs, McNemar
│   ├── 07_clinical_utility.py      # Decision curve analysis
│   ├── 08_external_validation.py   # External data validation
│   └── 09_generate_figures.py      # Publication figures
├── src/                            # Source modules
│   ├── data_utils.py               # Data loading utilities
│   ├── fairness_metrics.py         # Fairness calculations
│   ├── mitigation.py               # Mitigation wrappers
│   └── visualization.py            # Figure generation
├── data/
│   ├── raw/                        # BRFSS data (download separately)
│   ├── processed/                  # Generated datasets
│   └── external/                   # External validation data
├── figures/                        # Generated figures
├── results/                        # Analysis results
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/HIV-testing-fairness.git
cd HIV-testing-fairness

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Requirements

### BRFSS 2024 Data

Download the Behavioral Risk Factor Surveillance System (BRFSS) 2024 data from the CDC:

1. Visit: https://www.cdc.gov/brfss/annual_data/annual_2024.html
2. Download the ASCII file (LLCP2024.ASC)
3. Place in `data/raw/LLCP2024.ASC`

### External Validation Data (Optional)

- Ryan White HIV/AIDS Program data
- CDC AtlasPlus state-level HIV data

## Running the Analysis

Execute scripts in order:

```bash
# Full pipeline
python scripts/01_data_loading.py
python scripts/02_baseline_models.py
python scripts/03_fairness_audit.py
python scripts/04_mitigation.py
python scripts/05_race_blind_analysis.py
python scripts/06_statistical_tests.py
python scripts/07_clinical_utility.py
python scripts/08_external_validation.py
python scripts/09_generate_figures.py
```

Or run all at once:

```bash
for script in scripts/0*.py; do python "$script"; done
```

## Key Results

Expected results from the analysis (manuscript values):

| Metric | Value |
|--------|-------|
| Baseline XGBoost DPD | ~0.539 |
| Black-White selection gap | ~49.7 pp |
| ThresholdOptimizer DPD reduction | 86-97% |
| Race-blind model DPD | ~0.154-0.169 |
| CV DPD reduction | 90.9% ± 3.1% |

## Methods

### Models

- Logistic Regression
- Random Forest (200 trees, max_depth=10)
- XGBoost (200 estimators, max_depth=6, lr=0.1)
- Gradient Boosting (200 estimators, max_depth=6, lr=0.1)

### Fairness Metrics

- **Demographic Parity Difference (DPD)**: Maximum difference in selection rates across groups
- **Equalized Odds Difference (EOD)**: Maximum of TPR and FPR differences
- **Cohen's h**: Effect size for proportion differences
- **Calibration by group**: Brier score and reliability diagrams

### Mitigation Methods

- **ThresholdOptimizer**: Post-processing method that finds group-specific thresholds
- **ExponentiatedGradient**: In-processing method with constraint optimization

## BRFSS Variables

| BRFSS Code | Variable | Processing |
|------------|----------|------------|
| HIVTST7 | HIV testing | 1=Yes → 1, 2=No → 0 |
| _AGEG5YR | Age category | Midpoint conversion |
| SEXVAR | Sex | 1=Male, 2=Female |
| _RACE | Race/ethnicity | 8 categories |
| INCOME3 | Income | <$25K → low_income |
| ADDEPEV3 | Depression | Ever diagnosed |
| MEDCOST1 | Cost barrier | Past 12 months |
| GENHLTH | Health status | Poor/Fair vs Good+ |
| _STATE | State FIPS | Southern region |

## Citation

If you use this code, please cite:

```bibtex
@article{hiv_testing_fairness_2024,
  title={Algorithmic Fairness Audit of HIV Testing Prediction Models},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.

## Contact

For questions about this repository, please open an issue on GitHub.
