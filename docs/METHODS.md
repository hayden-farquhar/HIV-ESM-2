# Methods

Detailed methodology for HIV drug resistance prediction using ESM-2 protein language model embeddings.

## 1. Data Sources and Preprocessing

### 1.1 Stanford HIVDB Dataset

We used genotype-phenotype datasets from the Stanford HIV Drug Resistance Database (HIVDB), which contains HIV-1 protease and reverse transcriptase sequences with associated drug susceptibility measurements.

**Dataset composition:**
- Protease Inhibitors (PI): 2,171 sequences, 8 drugs
- NRTIs: 1,867 sequences, 6 drugs
- NNRTIs: 2,270 sequences, 4 drugs
- Total: 6,308 sequences, 18 drugs

### 1.2 Resistance Labels

Phenotypic drug resistance was defined using HIVDB's binary classification:
- **Susceptible (0)**: Normal susceptibility to the drug
- **Resistant (1)**: Reduced susceptibility (fold-change ≥ clinical cutoff)

### 1.3 Quality Filtering

Sequences were filtered to remove:
- Sequences with ambiguous amino acids (X)
- Sequences with excessive gaps (>10%)
- Sequences outside expected length range

## 2. Baseline Model: Binary Mutation Encoding

### 2.1 Feature Representation

Each sequence was encoded as a binary vector indicating mutations relative to the HXB2 reference:

```
Position i: 1 if sequence[i] != reference[i], else 0
```

This creates a sparse feature vector of length equal to protein length (99 for protease, ~240 for RT).

### 2.2 Classifier

XGBoost classifier with the following parameters:
- n_estimators: 300
- max_depth: 6
- learning_rate: 0.05
- subsample: 0.8
- colsample_bytree: 0.8
- scale_pos_weight: balanced based on class frequencies

### 2.3 Evaluation

5-fold stratified cross-validation with ROC-AUC as the primary metric.

**Baseline results:** Mean AUC = 0.955

## 3. ESM-2 Protein Language Model

### 3.1 Model Selection

We used ESM-2 (esm2_t33_650M_UR50D):
- 33 transformer layers
- 650 million parameters
- 1,280-dimensional embeddings
- Pre-trained on UniRef50 database

### 3.2 Embedding Extraction

For each sequence:
1. Tokenize using ESM-2 alphabet
2. Forward pass through model
3. Extract representations from layer 33 (final layer)
4. Remove BOS/EOS token representations

This yields a (sequence_length × 1280) embedding matrix per sequence.

### 3.3 Pooling Strategies

We evaluated multiple pooling methods to reduce variable-length embeddings to fixed-length vectors:

**Mean Pooling:**
```python
pooled = embeddings.mean(axis=0)  # Shape: (1280,)
```

**Max Pooling:**
```python
pooled = embeddings.max(axis=0)  # Shape: (1280,)
```

**Mean + Max Concatenation:**
```python
pooled = concat(mean, max)  # Shape: (2560,)
```

**Attention-Weighted Pooling (novel):**
```python
weights = attention_scores / attention_scores.sum()
pooled = (embeddings * weights[:, None]).sum(axis=0)
```

Mean pooling showed the best balance of performance and simplicity.

### 3.4 Classifier

Logistic regression with L2 regularization:
- StandardScaler for feature normalization
- max_iter: 1000
- class_weight: balanced
- C: 1.0 (default regularization)

Logistic regression was preferred over XGBoost for ESM-2 embeddings as the dense, continuous features are better suited to linear models.

## 4. Interpretability Analysis

### 4.1 Attention Weight Extraction

ESM-2 attention weights were extracted to identify positions the model focuses on:

1. Extract attention matrices from penultimate layer (layer 32)
2. Average across all attention heads
3. Compute column sums (attention received per position)
4. Compare attention patterns between resistant and susceptible sequences

**Attention differential:**
```python
differential = resistant_attention.mean(axis=0) - susceptible_attention.mean(axis=0)
```

### 4.2 DRM Enrichment Analysis

We validated that high-attention positions correspond to known Drug Resistance Mutations (DRMs) from the IAS-USA 2022 guidelines.

**Enrichment calculation:**
```python
observed = count(top_k_positions ∩ DRM_positions)
expected = top_k × (DRM_positions / sequence_length)
enrichment_ratio = observed / expected
```

**Statistical testing:**
- Fisher's exact test (one-sided, greater)
- H0: No enrichment of DRMs in top-k positions
- Significance threshold: p < 0.05

**Results:** Mean enrichment = 2.48x (top-20 positions)

### 4.3 Novel Position Discovery

Positions with high attention differential but NOT in current DRM lists were identified as candidates for experimental validation:

1. Rank positions by |attention_differential|
2. Select top-30 positions
3. Exclude known DRM positions
4. Record remaining as "novel"

**Results:** 228 unique novel positions identified across all drugs

## 5. Validation Strategy

### 5.1 Cross-Validation

5-fold stratified cross-validation:
- Stratification by resistance label
- Same folds used for baseline and ESM-2 comparison
- Enables paired statistical tests

### 5.2 Statistical Comparison

**DeLong test** for comparing AUC values:
- Tests H0: AUC1 = AUC2
- Accounts for correlation between predictions on same samples
- p < 0.05 considered significant

**Wilcoxon signed-rank test** for overall comparison:
- Paired test across all drugs
- Tests whether ESM-2 systematically improves over baseline

### 5.3 Holdout Validation

80/20 train/test split:
- Stratified by resistance label
- Evaluated generalization (train vs test AUC)
- Target: <1% AUC drop from train to test

### 5.4 Calibration Analysis

Probability calibration was assessed using:

**Expected Calibration Error (ECE):**
```python
ECE = Σ (n_bin / n_total) × |accuracy_bin - confidence_bin|
```

**Platt scaling** for post-hoc calibration:
- Logistic regression on validation predictions
- Reduces ECE from 0.071 to 0.040

### 5.5 Bootstrap Confidence Intervals

1000 bootstrap iterations:
- Sample with replacement
- Compute AUC for each sample
- Report 95% CI as [2.5th percentile, 97.5th percentile]

## 6. Limitations

1. **Dataset bias**: HIVDB contains predominantly subtype B sequences
2. **Temporal validity**: Training data may not reflect emerging resistance patterns
3. **Computational cost**: ESM-2 embedding extraction requires GPU
4. **Single-mutation focus**: Model may miss complex mutation interactions
5. **No structural validation**: Novel positions require experimental confirmation

## 7. Reproducibility

All random operations use `random_state=42` for reproducibility:
- Train/test splits
- Cross-validation folds
- XGBoost training
- Bootstrap sampling

Code and parameters are provided in the `src/` modules and Jupyter notebooks.

## References

1. Lin Z, et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. Science.

2. Rhee SY, et al. (2003). Human immunodeficiency virus reverse transcriptase and protease sequence database. Nucleic Acids Research.

3. Wensing AM, et al. (2022). Update of the drug resistance mutations in HIV-1. Topics in Antiviral Medicine.
