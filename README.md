# HIV Drug Resistance Prediction with ESM-2

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains the analysis code for predicting HIV drug resistance from protein sequences using ESM-2 protein language model embeddings. The approach achieves state-of-the-art performance across 18 antiretroviral drugs.

**Author:** Hayden Farquhar
**Contact:** hayden.farquhar@icloud.com

## Key Results

| Metric | Value |
|--------|-------|
| ESM-2 Mean AUC | 0.968 |
| Baseline (XGBoost) AUC | 0.955 |
| Improvement | +0.013 (p=0.0017) |
| DRM Enrichment | 2.48x |
| Novel Positions | 228 |
| Drugs Improved | 15/18 |

## Method Summary

Our approach combines:
1. **ESM-2 protein language model embeddings** (1,280 dimensions) - captures evolutionary and structural information from HIV protease and reverse transcriptase sequences
2. **Attention-weighted pooling** - leverages model attention to focus on resistance-relevant positions
3. **Per-drug classifiers** - logistic regression with 5-fold stratified cross-validation

The method demonstrates strong enrichment (2.48x) for known drug resistance mutations in high-attention positions, validating biological relevance.

## Repository Structure

```
HIV-ESM-2/
├── README.md                      # This file
├── LICENSE                        # MIT License
├── CITATION.cff                   # Citation metadata
├── requirements.txt               # pip dependencies
├── environment.yml                # Conda environment
├── src/
│   ├── __init__.py
│   ├── data_processing.py        # HIVDB data parsing
│   ├── feature_engineering.py    # ESM-2 embeddings, attention pooling
│   ├── models.py                 # Classifiers (LogReg, XGBoost, etc.)
│   ├── evaluation.py             # Metrics, CV, calibration
│   ├── visualization.py          # Plotting utilities
│   └── interpretability.py       # DRM enrichment, SHAP
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_baseline_development.ipynb
│   ├── 03_esm2_embedding_extraction.ipynb
│   ├── 04_classification_evaluation.ipynb
│   ├── 05_interpretability_analysis.ipynb
│   └── 06_external_validation.ipynb
├── data/
│   └── README.md                 # Data access instructions
├── figures/
├── results/
└── docs/
    └── METHODS.md                # Detailed methodology
```

## Requirements

### Python Dependencies

```
torch>=1.12.0
fair-esm>=2.0.0
scikit-learn>=1.3.0
xgboost>=1.7.0
shap>=0.42.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
biopython>=1.81
tqdm>=4.65.0
jupyter>=1.0.0
```

### Hardware

- **ESM-2 embedding extraction:** GPU with 16+ GB VRAM recommended (e.g., NVIDIA T4, V100, A100)
- **Model training:** CPU sufficient, GPU optional

## Installation

### Option 1: pip

```bash
git clone https://github.com/hayden-farquhar/HIV-ESM-2.git
cd HIV-ESM-2
pip install -r requirements.txt
```

### Option 2: conda

```bash
git clone https://github.com/hayden-farquhar/HIV-ESM-2.git
cd HIV-ESM-2
conda env create -f environment.yml
conda activate hiv-esm2
```

## Data Access

The analysis uses publicly available data from Stanford HIVDB:

1. **Stanford HIV Drug Resistance Database**
   - Website: https://hivdb.stanford.edu/
   - Genotype-phenotype datasets: https://hivdb.stanford.edu/pages/genopheno.dataset.html

2. **Required files:**
   - PI_DataSet.txt (Protease Inhibitors)
   - NRTI_DataSet.txt (NRTIs)
   - NNRTI_DataSet.txt (NNRTIs)

3. **Download and place in `data/raw/`**

See `data/README.md` for detailed instructions.

## Quick Start

### 1. Download HIVDB data

Follow instructions in `data/README.md` to download and place datasets.

### 2. Run notebooks in sequence

```bash
# Data acquisition and preprocessing
jupyter notebook notebooks/01_data_acquisition.ipynb

# Baseline development
jupyter notebook notebooks/02_baseline_development.ipynb

# ESM-2 embedding extraction (requires GPU)
jupyter notebook notebooks/03_esm2_embedding_extraction.ipynb

# Classification evaluation
jupyter notebook notebooks/04_classification_evaluation.ipynb

# Interpretability analysis
jupyter notebook notebooks/05_interpretability_analysis.ipynb

# External validation
jupyter notebook notebooks/06_external_validation.ipynb
```

## Notebooks Overview

| Notebook | Description |
|----------|-------------|
| 01 | Data acquisition from Stanford HIVDB |
| 02 | Baseline XGBoost with binary mutation encoding |
| 03 | ESM-2 embedding extraction (requires GPU) |
| 04 | Classifier comparison and evaluation |
| 05 | DRM enrichment and interpretability analysis |
| 06 | Holdout validation and calibration |

## Drug Coverage

### Protease Inhibitors (PI) - 8 drugs
ATV, DRV, FPV, IDV, LPV, NFV, SQV, TPV

### NRTIs - 6 drugs
ABC, AZT, D4T, DDI, 3TC, TDF

### NNRTIs - 4 drugs
EFV, ETR, NVP, RPV

## Citation

If you use this code, please cite:

```bibtex
@software{farquhar2026hiv,
  author = {Farquhar, Hayden},
  title = {HIV Drug Resistance Prediction with ESM-2 Protein Language Model},
  year = {2026},
  url = {https://github.com/hayden-farquhar/HIV-ESM-2}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [ESM-2](https://github.com/facebookresearch/esm) protein language models from Meta AI
- [Stanford HIVDB](https://hivdb.stanford.edu/) for genotype-phenotype data
- [IAS-USA](https://www.iasusa.org/) for drug resistance mutation guidelines
