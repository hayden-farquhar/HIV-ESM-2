# Data Directory

This directory contains data for the HIV-ESM-2 drug resistance prediction project.

## Data Source

**Stanford HIV Drug Resistance Database (HIVDB)**
- Website: https://hivdb.stanford.edu/
- Genotype-phenotype datasets: https://hivdb.stanford.edu/pages/genopheno.dataset.html

## Download Instructions

1. Visit https://hivdb.stanford.edu/pages/genopheno.dataset.html

2. Download the following datasets:
   - **Protease Inhibitors (PI)**: PI genotype-phenotype dataset
   - **NRTIs**: NRTI genotype-phenotype dataset
   - **NNRTIs**: NNRTI genotype-phenotype dataset

3. Place the downloaded files in `data/raw/`:
   ```
   data/raw/PI_DataSet.txt
   data/raw/NRTI_DataSet.txt
   data/raw/NNRTI_DataSet.txt
   ```

4. Run notebook `01_data_acquisition.ipynb` to process the data

## Directory Structure

After running the pipeline, this directory will contain:

```
data/
├── README.md                     # This file
├── raw/                          # Raw downloaded data (not tracked)
│   ├── PI_DataSet.txt
│   ├── NRTI_DataSet.txt
│   └── NNRTI_DataSet.txt
├── processed/                    # Processed data (not tracked)
│   ├── PI_sequences.fasta
│   ├── PI_phenotypes.csv
│   ├── NRTI_sequences.fasta
│   ├── NRTI_phenotypes.csv
│   ├── NNRTI_sequences.fasta
│   ├── NNRTI_phenotypes.csv
│   └── metadata.json
└── embeddings/                   # ESM-2 embeddings (not tracked)
    ├── PI_pooled_mean.npy
    ├── PI_pooled_max.npy
    ├── PI_pooled_mean_max.npy
    ├── NRTI_pooled_mean.npy
    ├── NRTI_pooled_max.npy
    ├── NRTI_pooled_mean_max.npy
    ├── NNRTI_pooled_mean.npy
    ├── NNRTI_pooled_max.npy
    └── NNRTI_pooled_mean_max.npy
```

## Dataset Statistics

| Drug Class | Sequences | Drugs | Avg Length |
|------------|-----------|-------|------------|
| PI | 2,171 | 8 | 99 aa |
| NRTI | 1,867 | 6 | ~240 aa |
| NNRTI | 2,270 | 4 | ~240 aa |
| **Total** | **6,308** | **18** | - |

## Drug Distribution

### Protease Inhibitors (8 drugs)
- ATV (Atazanavir)
- DRV (Darunavir)
- FPV (Fosamprenavir)
- IDV (Indinavir)
- LPV (Lopinavir)
- NFV (Nelfinavir)
- SQV (Saquinavir)
- TPV (Tipranavir)

### NRTIs (6 drugs)
- ABC (Abacavir)
- AZT (Zidovudine)
- D4T (Stavudine)
- DDI (Didanosine)
- 3TC (Lamivudine)
- TDF (Tenofovir)

### NNRTIs (4 drugs)
- EFV (Efavirenz)
- ETR (Etravirine)
- NVP (Nevirapine)
- RPV (Rilpivirine)

## File Sizes (Approximate)

| File | Size |
|------|------|
| Raw datasets | ~5 MB total |
| Processed sequences | ~1.5 MB |
| Processed phenotypes | ~0.5 MB |
| ESM-2 embeddings (mean pooled) | ~32 MB |
| All embeddings | ~100 MB |

## What's NOT Tracked in Git

The following are excluded via `.gitignore`:
- `raw/` - Raw downloaded data
- `processed/` - Processed sequences and phenotypes
- `embeddings/` - ESM-2 embedding files
- `*.npy`, `*.npz` - Numpy arrays
- `*.csv`, `*.tsv`, `*.gz` - Data files
- `*.fasta` - Sequence files

This ensures the repository stays lightweight. Users regenerate these files by running the notebooks.

## Phenotype Encoding

The HIVDB uses fold-change (FC) values and class labels:

- **FC**: Continuous fold-change resistance value
- **class2**: Binary classification (0 = susceptible, 1 = resistant)
- **class3**: Three-class (0 = susceptible, 1 = intermediate, 2 = resistant)

Our models use `class2` (binary) labels.

## Reference Sequences

- **HIV-1 Protease**: 99 amino acids (HXB2 reference)
- **HIV-1 Reverse Transcriptase**: 560 amino acids (active site region ~240 aa)

## Citation

If using this data, please cite:

```
Stanford University HIV Drug Resistance Database
https://hivdb.stanford.edu/
```
