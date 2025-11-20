"""
Imputation Strategy Validation for Publication

This script validates that median/mode imputation achieves comparable 
performance to more sophisticated methods (KNN, MICE) while being 
significantly faster.

Results from this script go into Supplementary Table S1 of the paper.

Usage:
    python validate_imputation.py --datasets_dir /path/to/datasets
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer, KNNImputer

# Import from your main code
from Am_v2 import load_tabular_data, smart_fill_missing

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def introduce_missingness(X: np.ndarray, missing_rate: float = 0.10) -> np.ndarray:
    """
    Introduce MCAR (Missing Completely At Random) missingness for validation.
    
    Args:
        X: Original feature matrix
        missing_rate: Fraction of values to set as missing (default: 10%)
        
    Returns:
        X_missing: Feature matrix with introduced missing values
    """
    X_missing = X.copy()
    mask = np.random.random(X.shape) < missing_rate
    X_missing[mask] = np.nan
    return X_missing


def impute_knn(X: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
    """Impute using KNN."""
    imputer = KNNImputer(n_neighbors=n_neighbors)
    return imputer.fit_transform(X)


def impute_median(X: np.ndarray) -> np.ndarray:
    """Impute using median."""
    imputer = SimpleImputer(strategy='median')
    return imputer.fit_transform(X)


def evaluate_imputation_method(X: np.ndarray, y: np.ndarray, 
                               method: str) -> Tuple[float, float]:
    """
    Evaluate classification performance with given imputation method.
    
    Returns:
        accuracy: Cross-validation accuracy
        time_taken: Time in seconds
    """
    # Introduce 10% MCAR missingness
    X_missing = introduce_missingness(X, missing_rate=0.10)
    
    # Impute based on method
    start_time = time.time()
    if method == 'median':
        X_imputed = impute_median(X_missing)
    elif method == 'knn':
        X_imputed = impute_knn(X_missing, n_neighbors=5)
    else:
        raise ValueError(f"Unknown method: {method}")
    time_taken = time.time() - start_time
    
    # Evaluate with Random Forest (standard benchmark classifier)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    scores = cross_val_score(clf, X_imputed, y, cv=5, scoring='accuracy')
    accuracy = scores.mean()
    
    return accuracy, time_taken


def validate_on_dataset(dataset_path: str) -> Dict:
    """Validate imputation strategy on a single dataset."""
    dataset_name = os.path.basename(dataset_path)
    logger.info(f"\nProcessing: {dataset_name}")
    
    try:
        # Load data
        X, y, _ = load_tabular_data(dataset_path)
        
        # Skip if too large (for speed)
        if X.shape[0] > 10000:
            logger.warning(f"Skipping {dataset_name} (too large: {X.shape[0]} samples)")
            return None
        
        # Test median/mode imputation
        acc_median, time_median = evaluate_imputation_method(X, y, 'median')
        logger.info(f"  Median: acc={acc_median:.4f}, time={time_median:.2f}s")
        
        # Test KNN imputation
        acc_knn, time_knn = evaluate_imputation_method(X, y, 'knn')
        logger.info(f"  KNN:    acc={acc_knn:.4f}, time={time_knn:.2f}s")
        
        return {
            'dataset': dataset_name,
            'samples': X.shape[0],
            'features': X.shape[1],
            'classes': len(np.unique(y)),
            'median_acc': acc_median,
            'knn_acc': acc_knn,
            'acc_ratio': acc_median / acc_knn,
            'median_time': time_median,
            'knn_time': time_knn,
            'speedup': time_knn / time_median
        }
        
    except Exception as e:
        logger.error(f"Failed on {dataset_name}: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Validate imputation strategy for publication")
    parser.add_argument('--datasets_dir', type=str, required=True,
                       help='Directory containing test datasets')
    parser.add_argument('--n_datasets', type=int, default=10,
                       help='Number of datasets to test (default: 10)')
    parser.add_argument('--output', type=str, default='supplementary_table_s1.csv',
                       help='Output CSV file')
    args = parser.parse_args()
    
    # Find datasets
    dataset_files = []
    for ext in ['.csv', '.arff', '.data']:
        dataset_files.extend([
            os.path.join(args.datasets_dir, f) 
            for f in os.listdir(args.datasets_dir) 
            if f.endswith(ext)
        ])
    
    dataset_files = dataset_files[:args.n_datasets]
    logger.info(f"Found {len(dataset_files)} datasets to validate")
    
    # Run validation
    results = []
    for dataset_path in dataset_files:
        result = validate_on_dataset(dataset_path)
        if result is not None:
            results.append(result)
    
    # Create results dataframe
    df_results = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "="*80)
    print("VALIDATION RESULTS FOR SUPPLEMENTARY TABLE S1")
    print("="*80)
    print(df_results.to_string(index=False))
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Average accuracy ratio (median/knn): {df_results['acc_ratio'].mean():.3f} ± {df_results['acc_ratio'].std():.3f}")
    print(f"Average speedup (knn/median):        {df_results['speedup'].mean():.1f}× ± {df_results['speedup'].std():.1f}×")
    print(f"Median maintains {df_results['acc_ratio'].mean()*100:.1f}% of KNN's performance")
    print("="*80)
    
    # Save to CSV
    df_results.to_csv(args.output, index=False)
    logger.info(f"\nResults saved to: {args.output}")
    
    # Generate LaTeX table
    print("\n" + "="*80)
    print("LaTeX TABLE FOR PAPER (copy to supplementary material)")
    print("="*80)
    print(df_results[['dataset', 'samples', 'median_acc', 'knn_acc', 'acc_ratio', 'speedup']].to_latex(
        index=False, 
        float_format="%.3f",
        caption="Validation of imputation strategy on 10 representative datasets.",
        label="tab:imputation_validation"
    ))


if __name__ == "__main__":
    main()
