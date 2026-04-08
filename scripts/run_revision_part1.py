#!/usr/bin/env python3
"""
Revision Part 1: Subtype Analysis + Temporal Holdout Evaluation
================================================================
This script runs the subtype-stratified and temporal holdout analyses
required for manuscript revision.

Steps:
1. Reconstruct AA sequences from HIVDB position columns
2. Assign subtypes via Hamming distance to HXB2
3. Extract ESM-2 embeddings (with caching to data/embeddings/)
4. Binarize fold-change at FC >= 2.5
5. Run subtype-stratified AUC evaluation
6. Run temporal holdout evaluation
7. Save results to results/revision/
8. Generate figures to figures/revision/

Note: Uses HuggingFace transformers for ESM-2 loading because the installed
`esm` v3 package shadows `fair-esm` and lacks the ESM-2 pretrained API.
"""

import sys
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Project paths
PROJECT_DIR = Path("/Users/haydenfarquhar/Documents/Research Projects/06 HIV-ESM-2/00 Repository/HIV-ESM-2")
DATA_DIR = Path("/Users/haydenfarquhar/Documents/Research Projects/06 HIV-ESM-2")
EMBEDDING_DIR = PROJECT_DIR / "data" / "embeddings"
RESULTS_DIR = PROJECT_DIR / "results" / "revision"
FIGURES_DIR = PROJECT_DIR / "figures" / "revision"

# Add project to path
sys.path.insert(0, str(PROJECT_DIR))

from src.data_processing import PI_DRUGS, NRTI_DRUGS, NNRTI_DRUGS
from src.subtype_analysis import (
    reconstruct_all_datasets,
    assign_subtypes_via_sequence_similarity,
    subtype_stratified_evaluation,
    create_temporal_split,
    temporal_holdout_evaluation,
)

# Create output directories
EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── ESM-2 embedding extraction via HuggingFace transformers ──────────────

def load_esm2_transformers(model_name="facebook/esm2_t33_650M_UR50D"):
    """Load ESM-2 model and tokenizer via HuggingFace transformers."""
    from transformers import EsmModel, EsmTokenizer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Loading tokenizer from {model_name}...")
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    print(f"  Loading model from {model_name}...")
    model = EsmModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    print(f"  Model loaded on {device}")
    return model, tokenizer, device


def extract_mean_pooled_embeddings(
    sequences, model, tokenizer, device, batch_size=4
):
    """
    Extract mean-pooled ESM-2 embeddings using HuggingFace transformers.

    Returns array of shape (n_sequences, hidden_size) where hidden_size=1280
    for esm2_t33_650M_UR50D.
    """
    all_pooled = []

    for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting ESM-2 embeddings"):
        batch_seqs = sequences[i:i + batch_size]

        # Tokenize
        encoded = tokenizer(
            batch_seqs, return_tensors='pt', padding=True, truncation=True,
            max_length=1024
        )
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Last hidden state: (batch, seq_len, hidden_size)
        last_hidden = outputs.last_hidden_state

        # Mean pooling over non-padding tokens (excluding BOS/EOS)
        for j in range(len(batch_seqs)):
            seq_len = len(batch_seqs[j])
            # tokens: [CLS] + AA tokens + [EOS] + [PAD]...
            # Take positions 1 to seq_len+1 (the actual AA tokens)
            seq_emb = last_hidden[j, 1:seq_len+1, :].cpu().numpy()
            pooled = seq_emb.mean(axis=0)
            all_pooled.append(pooled)

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return np.array(all_pooled)


# ── Phenotype binarization ───────────────────────────────────────────────

def binarize_phenotypes(df, drugs, threshold=2.5):
    """
    Binarize fold-change values into resistant (1) / susceptible (0).
    Values >= threshold are resistant; missing values remain NaN.
    """
    pheno = pd.DataFrame(index=df.index)
    for drug in drugs:
        if drug in df.columns:
            vals = pd.to_numeric(df[drug], errors='coerce')
            pheno[drug] = np.where(vals.isna(), np.nan, (vals >= threshold).astype(float))
        else:
            print(f"  Warning: {drug} not found in columns")
    return pheno


# ── Main pipeline ────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # ── Step 1: Reconstruct sequences ────────────────────────────────────
    print("=" * 70)
    print("STEP 1: Reconstructing sequences from HIVDB position columns")
    print("=" * 70)
    datasets = reconstruct_all_datasets(DATA_DIR)

    for dc, df in datasets.items():
        print(f"  {dc}: {len(df)} sequences, seq length = {df['sequence'].str.len().iloc[0]}")

    # ── Step 2: Assign subtypes ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: Assigning subtypes via Hamming distance to HXB2")
    print("=" * 70)
    subtype_dfs = {}
    for dc, df in datasets.items():
        st_df = assign_subtypes_via_sequence_similarity(
            df['sequence'].tolist(),
            df['SeqID'].tolist(),
            drug_class=dc
        )
        subtype_dfs[dc] = st_df
        print(f"\n  {dc} subtype distribution:")
        print(st_df['subtype'].value_counts().to_string())

    # Save subtype distributions
    all_subtypes = []
    for dc, st_df in subtype_dfs.items():
        st_df = st_df.copy()
        st_df['drug_class'] = dc
        all_subtypes.append(st_df)
    all_subtypes_df = pd.concat(all_subtypes, ignore_index=True)
    all_subtypes_df.to_csv(RESULTS_DIR / "subtype_assignments.csv", index=False)
    print(f"\n  Saved subtype assignments to {RESULTS_DIR / 'subtype_assignments.csv'}")

    # ── Step 3: Extract ESM-2 embeddings (with caching) ─────────────────
    print("\n" + "=" * 70)
    print("STEP 3: Extracting ESM-2 embeddings (batch_size=4, CPU)")
    print("=" * 70)

    # Check for cached embeddings
    all_cached = True
    for dc in datasets:
        cache_path = EMBEDDING_DIR / f"{dc}_esm2_embeddings.npy"
        if not cache_path.exists():
            all_cached = False
            break

    if all_cached:
        print("  All embeddings found in cache, loading...")
        embeddings = {}
        for dc in datasets:
            cache_path = EMBEDDING_DIR / f"{dc}_esm2_embeddings.npy"
            embeddings[dc] = np.load(cache_path)
            print(f"  {dc}: loaded {embeddings[dc].shape}")
    else:
        # Load ESM-2 via HuggingFace transformers (avoids esm v3/fair-esm conflict)
        model, tokenizer, device = load_esm2_transformers()

        embeddings = {}
        for dc, df in datasets.items():
            cache_path = EMBEDDING_DIR / f"{dc}_esm2_embeddings.npy"
            if cache_path.exists():
                print(f"  {dc}: loading cached embeddings")
                embeddings[dc] = np.load(cache_path)
                print(f"    shape: {embeddings[dc].shape}")
            else:
                print(f"  {dc}: extracting embeddings for {len(df)} sequences...")
                t_start = time.time()
                emb = extract_mean_pooled_embeddings(
                    df['sequence'].tolist(),
                    model, tokenizer, device,
                    batch_size=4
                )
                elapsed = time.time() - t_start
                print(f"    Done in {elapsed:.1f}s, shape: {emb.shape}")
                np.save(cache_path, emb)
                print(f"    Cached to {cache_path}")
                embeddings[dc] = emb

        # Free model memory
        del model, tokenizer
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ── Step 4: Prepare phenotype DataFrames ─────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4: Binarizing phenotypes at FC >= 2.5")
    print("=" * 70)

    drug_lists = {'PI': PI_DRUGS, 'NRTI': NRTI_DRUGS, 'NNRTI': NNRTI_DRUGS}
    phenotypes = {}
    for dc, df in datasets.items():
        drugs = drug_lists[dc]
        pheno = binarize_phenotypes(df, drugs, threshold=2.5)
        phenotypes[dc] = pheno
        print(f"\n  {dc} phenotype summary (FC >= 2.5):")
        for drug in drugs:
            if drug in pheno.columns:
                valid = pheno[drug].dropna()
                n_res = int(valid.sum())
                n_total = len(valid)
                prev = n_res / n_total if n_total > 0 else 0
                print(f"    {drug}: {n_res}/{n_total} resistant ({prev:.1%})")

    # ── Step 5: Subtype-stratified AUC evaluation ────────────────────────
    print("\n" + "=" * 70)
    print("STEP 5: Subtype-stratified AUC evaluation (logistic regression)")
    print("=" * 70)

    all_subtype_results = []
    for dc in datasets:
        drugs = drug_lists[dc]
        X = embeddings[dc]
        pheno = phenotypes[dc]
        subtypes = subtype_dfs[dc]['subtype']

        print(f"\n  --- {dc} ({len(drugs)} drugs) ---")
        sub_results = subtype_stratified_evaluation(
            X, pheno, subtypes, drugs,
            model_type='logistic',
            n_splits=5,
            random_state=42
        )
        sub_results['drug_class'] = dc
        all_subtype_results.append(sub_results)

    subtype_results_df = pd.concat(all_subtype_results, ignore_index=True)
    subtype_results_df.to_csv(RESULTS_DIR / "subtype_stratified_auc.csv", index=False)
    print(f"\n  Saved to {RESULTS_DIR / 'subtype_stratified_auc.csv'}")
    print("\n  Summary:")
    print(subtype_results_df.to_string(index=False))

    # ── Step 6: Temporal holdout evaluation ──────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 6: Temporal holdout evaluation (80/20 split by SeqID)")
    print("=" * 70)

    all_temporal_results = []
    for dc in datasets:
        drugs = drug_lists[dc]
        X = embeddings[dc]
        pheno = phenotypes[dc]
        df = datasets[dc]

        print(f"\n  --- {dc} ---")
        train_idx, test_idx = create_temporal_split(df, seq_id_col='SeqID', cutoff_quantile=0.8)

        temp_results = temporal_holdout_evaluation(
            X, pheno, drugs,
            train_idx, test_idx,
            model_type='logistic',
            random_state=42
        )
        temp_results['drug_class'] = dc
        all_temporal_results.append(temp_results)

    temporal_results_df = pd.concat(all_temporal_results, ignore_index=True)
    temporal_results_df.to_csv(RESULTS_DIR / "temporal_holdout_auc.csv", index=False)
    print(f"\n  Saved to {RESULTS_DIR / 'temporal_holdout_auc.csv'}")
    print("\n  Summary:")
    print(temporal_results_df.to_string(index=False))

    # ── Step 7: Generate figures ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 7: Generating figures")
    print("=" * 70)

    # Figure 1: Subtype distribution bar chart
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for i, dc in enumerate(['PI', 'NRTI', 'NNRTI']):
        st_df = subtype_dfs[dc]
        counts = st_df['subtype'].value_counts()
        colors = {'B': '#2196F3', 'B_divergent': '#FF9800', 'non-B': '#F44336'}
        bar_colors = [colors.get(s, '#999999') for s in counts.index]
        axes[i].bar(counts.index, counts.values, color=bar_colors)
        axes[i].set_title(f'{dc} (n={len(st_df)})', fontsize=13, fontweight='bold')
        axes[i].set_ylabel('Count')
        axes[i].set_xlabel('Subtype Group')
        for j, (idx, val) in enumerate(counts.items()):
            axes[i].text(j, val + 5, str(val), ha='center', fontsize=10)
    plt.suptitle('HIV-1 Subtype Distribution by Drug Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "subtype_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'subtype_distribution.png'}")

    # Figure 2: Subtype-stratified AUC heatmap
    if not subtype_results_df.empty:
        pivot = subtype_results_df.pivot_table(
            index='drug', columns='subtype', values='auc'
        )
        fig, ax = plt.subplots(figsize=(8, max(6, len(pivot) * 0.45)))
        sns.heatmap(
            pivot, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0,
            ax=ax, cbar_kws={'label': 'AUC'}
        )
        ax.set_title('Subtype-Stratified AUC by Drug', fontsize=13, fontweight='bold')
        ax.set_ylabel('Drug')
        ax.set_xlabel('Subtype Group')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "subtype_stratified_auc_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {FIGURES_DIR / 'subtype_stratified_auc_heatmap.png'}")

    # Figure 3: Temporal holdout AUC comparison
    if not temporal_results_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        colors_map = {'PI': '#2196F3', 'NRTI': '#4CAF50', 'NNRTI': '#FF9800'}
        bar_colors = [colors_map.get(r['drug_class'], '#999') for _, r in temporal_results_df.iterrows()]

        bars = ax.bar(
            range(len(temporal_results_df)),
            temporal_results_df['auc'].values,
            color=bar_colors, edgecolor='white', linewidth=0.5
        )
        ax.set_xticks(range(len(temporal_results_df)))
        ax.set_xticklabels(temporal_results_df['drug'].values, rotation=45, ha='right')
        ax.set_ylabel('AUC', fontsize=12)
        ax.set_title('Temporal Holdout AUC (train: SeqID <= Q80, test: SeqID > Q80)',
                      fontsize=13, fontweight='bold')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
        ax.set_ylim(0, 1.05)

        # Add value labels
        for bar, val in zip(bars, temporal_results_df['auc'].values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', fontsize=9)

        # Legend for drug classes
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2196F3', label='PI'),
            Patch(facecolor='#4CAF50', label='NRTI'),
            Patch(facecolor='#FF9800', label='NNRTI'),
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "temporal_holdout_auc.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {FIGURES_DIR / 'temporal_holdout_auc.png'}")

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed_total = time.time() - t0
    print("\n" + "=" * 70)
    print(f"REVISION PART 1 COMPLETE ({elapsed_total:.1f}s)")
    print("=" * 70)

    print("\n--- SUBTYPE DISTRIBUTIONS ---")
    for dc in datasets:
        print(f"\n  {dc}:")
        print(subtype_dfs[dc]['subtype'].value_counts().to_string())

    print("\n--- SUBTYPE-STRATIFIED AUC (mean across drugs) ---")
    if not subtype_results_df.empty:
        summary = subtype_results_df.groupby(['drug_class', 'subtype'])['auc'].agg(['mean', 'std', 'count'])
        print(summary.to_string())

    print("\n--- TEMPORAL HOLDOUT AUC ---")
    if not temporal_results_df.empty:
        print(temporal_results_df[['drug_class', 'drug', 'auc', 'n_train', 'n_test']].to_string(index=False))
        print(f"\n  Overall mean AUC: {temporal_results_df['auc'].mean():.4f}")

    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == '__main__':
    main()
