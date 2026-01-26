"""
Model training and prediction for HIV drug resistance.

This module provides functions for:
- Training logistic regression and XGBoost classifiers
- Per-drug model training
- Cross-validation evaluation
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score


def get_default_xgb_params(class_weight: float = 1.0) -> Dict:
    """
    Get default XGBoost parameters for resistance prediction.

    Args:
        class_weight: Weight for positive class (resistant)

    Returns:
        Dictionary of XGBoost parameters
    """
    return {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': class_weight,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'auc',
        'use_label_encoder': False
    }


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42
) -> Tuple[LogisticRegression, StandardScaler]:
    """
    Train logistic regression classifier with standardization.

    Args:
        X_train: Training features
        y_train: Training labels
        C: Regularization strength
        max_iter: Maximum iterations
        random_state: Random seed

    Returns:
        Tuple of (trained model, fitted scaler)
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Train model
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        solver='lbfgs',
        class_weight='balanced'
    )
    model.fit(X_scaled, y_train)

    return model, scaler


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    params: Optional[Dict] = None,
    early_stopping_rounds: int = 50,
    verbose: int = 0
) -> xgb.XGBClassifier:
    """
    Train XGBoost classifier with optional early stopping.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (for early stopping)
        y_val: Validation labels
        params: XGBoost parameters (uses defaults if None)
        early_stopping_rounds: Early stopping patience
        verbose: Verbosity level

    Returns:
        Trained XGBClassifier
    """
    if params is None:
        # Compute class weight
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        class_weight = n_neg / n_pos if n_pos > 0 else 1.0
        params = get_default_xgb_params(class_weight)

    model = xgb.XGBClassifier(**params)

    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=verbose
        )
    else:
        model.fit(X_train, y_train, verbose=verbose)

    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 300,
    max_depth: int = 10,
    random_state: int = 42
) -> RandomForestClassifier:
    """
    Train Random Forest classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        random_state: Random seed

    Returns:
        Trained RandomForestClassifier
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    return model


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    kernel: str = 'rbf',
    random_state: int = 42
) -> Tuple[SVC, StandardScaler]:
    """
    Train SVM classifier with standardization.

    Args:
        X_train: Training features
        y_train: Training labels
        C: Regularization parameter
        kernel: Kernel type
        random_state: Random seed

    Returns:
        Tuple of (trained model, fitted scaler)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = SVC(
        C=C,
        kernel=kernel,
        probability=True,
        random_state=random_state,
        class_weight='balanced'
    )
    model.fit(X_scaled, y_train)

    return model, scaler


def per_drug_training(
    X: np.ndarray,
    phenotypes: pd.DataFrame,
    drugs: List[str],
    model_type: str = 'logistic',
    n_splits: int = 5,
    random_state: int = 42
) -> Dict:
    """
    Train models for each drug and evaluate with cross-validation.

    Args:
        X: Feature matrix (n_samples, n_features)
        phenotypes: DataFrame with drug resistance labels
        drugs: List of drug names to train models for
        model_type: 'logistic', 'xgboost', 'rf', or 'svm'
        n_splits: Number of CV folds
        random_state: Random seed

    Returns:
        Dictionary with results per drug
    """
    results = {}

    for drug in drugs:
        # Get binary labels (using class2 suffix)
        label_col = f"{drug}_class2" if f"{drug}_class2" in phenotypes.columns else drug

        if label_col not in phenotypes.columns:
            print(f"  Skipping {drug}: no label column found")
            continue

        y = phenotypes[label_col].values

        # Filter valid samples
        valid_mask = ~np.isnan(y)
        X_valid = X[valid_mask]
        y_valid = y[valid_mask].astype(int)

        if len(np.unique(y_valid)) < 2:
            print(f"  Skipping {drug}: single class only")
            continue

        n_resistant = y_valid.sum()
        n_susceptible = len(y_valid) - n_resistant

        # Cross-validation predictions
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        if model_type == 'logistic':
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_valid)
            model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=random_state)
            y_pred = cross_val_predict(model, X_scaled, y_valid, cv=cv, method='predict_proba')[:, 1]
        elif model_type == 'xgboost':
            n_pos = y_valid.sum()
            n_neg = len(y_valid) - n_pos
            scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
            model = xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                scale_pos_weight=scale_pos_weight, random_state=random_state,
                use_label_encoder=False, eval_metric='auc', n_jobs=-1
            )
            y_pred = cross_val_predict(model, X_valid, y_valid, cv=cv, method='predict_proba')[:, 1]
        elif model_type == 'rf':
            model = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=random_state, n_jobs=-1)
            y_pred = cross_val_predict(model, X_valid, y_valid, cv=cv, method='predict_proba')[:, 1]
        elif model_type == 'svm':
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_valid)
            model = SVC(probability=True, class_weight='balanced', random_state=random_state)
            y_pred = cross_val_predict(model, X_scaled, y_valid, cv=cv, method='predict_proba')[:, 1]
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Compute AUC
        auc = roc_auc_score(y_valid, y_pred)

        results[drug] = {
            'auc': auc,
            'n_samples': len(y_valid),
            'n_resistant': int(n_resistant),
            'n_susceptible': int(n_susceptible),
            'y_true': y_valid,
            'y_pred': y_pred
        }

        print(f"  {drug}: AUC = {auc:.4f} (n={len(y_valid)}, R={n_resistant}, S={n_susceptible})")

    return results


def aggregate_drug_results(results: Dict) -> pd.DataFrame:
    """
    Aggregate per-drug results into a summary DataFrame.

    Args:
        results: Dictionary from per_drug_training()

    Returns:
        DataFrame with summary statistics
    """
    summary = []

    for drug, res in results.items():
        summary.append({
            'drug': drug,
            'auc': res['auc'],
            'n_samples': res['n_samples'],
            'n_resistant': res['n_resistant'],
            'n_susceptible': res['n_susceptible'],
            'prevalence': res['n_resistant'] / res['n_samples']
        })

    df = pd.DataFrame(summary)

    # Add summary row
    mean_row = pd.DataFrame([{
        'drug': 'MEAN',
        'auc': df['auc'].mean(),
        'n_samples': df['n_samples'].sum(),
        'n_resistant': df['n_resistant'].sum(),
        'n_susceptible': df['n_susceptible'].sum(),
        'prevalence': df['n_resistant'].sum() / df['n_samples'].sum()
    }])

    df = pd.concat([df, mean_row], ignore_index=True)

    return df


def compare_models(
    X: np.ndarray,
    phenotypes: pd.DataFrame,
    drugs: List[str],
    model_types: List[str] = ['logistic', 'xgboost', 'rf'],
    n_splits: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compare multiple model types across all drugs.

    Args:
        X: Feature matrix
        phenotypes: DataFrame with resistance labels
        drugs: List of drugs to evaluate
        model_types: List of model types to compare
        n_splits: Number of CV folds
        random_state: Random seed

    Returns:
        DataFrame with comparison results
    """
    comparison = []

    for model_type in model_types:
        print(f"\n{model_type.upper()}:")
        results = per_drug_training(
            X, phenotypes, drugs,
            model_type=model_type,
            n_splits=n_splits,
            random_state=random_state
        )

        for drug, res in results.items():
            comparison.append({
                'model': model_type,
                'drug': drug,
                'auc': res['auc'],
                'n_samples': res['n_samples']
            })

    return pd.DataFrame(comparison)
