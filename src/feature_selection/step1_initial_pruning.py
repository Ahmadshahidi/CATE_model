"""
Step 1: Initial Pruning — Variance Thresholding + Correlation Filter
=====================================================================
Removes features that are:
  1. Near-constant  (variance < VARIANCE_THRESHOLD)
  2. Redundant      (Pearson correlation > CORRELATION_THRESHOLD with another feature)

This step reduces ~1,800 Epsilon features to ~500-700 informative ones
before the more expensive Boruta-SHAP step.

Usage:
    from src.feature_selection.step1_initial_pruning import run_initial_pruning
    X_pruned, pruner = run_initial_pruning(X, y, treatment)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import config


class InitialPruning:
    """
    Two-stage filter: variance thresholding → correlation pruning.

    Attributes
    ----------
    removed_variance   : list[str]  Columns dropped by variance filter.
    removed_correlation: list[str]  Columns dropped by correlation filter.
    selected_features  : list[str]  Columns that survived both filters.
    """

    def __init__(self,
                 variance_threshold: float    = None,
                 correlation_threshold: float = None):
        self.variance_threshold    = variance_threshold    or config.VARIANCE_THRESHOLD
        self.correlation_threshold = correlation_threshold or config.CORRELATION_THRESHOLD

        self.removed_variance    = []
        self.removed_correlation = []
        self.selected_features   = []
        self._vt                 = None   # fitted VarianceThreshold

    # ------------------------------------------------------------------
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the pruner on X and return the pruned DataFrame.

        Parameters
        ----------
        X : pd.DataFrame  (numeric features only)

        Returns
        -------
        pd.DataFrame  (subset of columns that survived pruning)
        """
        print(f"\n{'─'*60}")
        print("STEP 1: INITIAL PRUNING")
        print(f"  Input : {X.shape[0]:,} rows × {X.shape[1]} features")
        print(f"  Variance threshold    : {self.variance_threshold}")
        print(f"  Correlation threshold : {self.correlation_threshold}")
        print(f"{'─'*60}")

        # ── Stage 1: Variance filter ──────────────────────────────
        t0 = time.time()
        self._vt = VarianceThreshold(threshold=self.variance_threshold)
        X_var    = self._vt.fit_transform(X)
        kept_var = X.columns[self._vt.get_support()].tolist()
        dropped  = [c for c in X.columns if c not in kept_var]
        self.removed_variance = dropped
        X_var_df = pd.DataFrame(X_var, columns=kept_var, index=X.index)

        print(f"\n  [Stage 1] Variance filter:")
        print(f"    Removed {len(dropped):>5} near-constant features  "
              f"({len(dropped)/X.shape[1]*100:.1f}%)")
        print(f"    Retained {len(kept_var):>4} features  "
              f"({len(kept_var)/X.shape[1]*100:.1f}%)")

        # ── Stage 2: Correlation filter ───────────────────────────
        print(f"\n  [Stage 2] Correlation filter  (threshold={self.correlation_threshold}):")
        print(f"    Computing {len(kept_var)} × {len(kept_var)} correlation matrix ...")

        # Compute pairwise Pearson correlation
        corr_matrix = X_var_df.corr().abs()

        # Upper-triangle mask
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find columns with any correlation > threshold
        to_drop_corr = [col for col in upper.columns
                        if any(upper[col] > self.correlation_threshold)]
        self.removed_correlation = to_drop_corr

        X_pruned = X_var_df.drop(columns=to_drop_corr)
        self.selected_features = X_pruned.columns.tolist()

        elapsed = time.time() - t0
        print(f"    Removed {len(to_drop_corr):>5} correlated features  "
              f"({len(to_drop_corr)/len(kept_var)*100:.1f}%)")
        print(f"    Retained {len(self.selected_features):>4} features")
        print(f"  ✓ Stage 2 complete  ({elapsed:.1f}s)")

        print(f"\n  STEP 1 SUMMARY")
        print(f"  {'─'*40}")
        print(f"  Original features         : {X.shape[1]:>5}")
        print(f"  After variance filter     : {len(kept_var):>5}  "
              f"(removed {len(dropped)})")
        print(f"  After correlation filter  : {len(self.selected_features):>5}  "
              f"(removed {len(to_drop_corr)})")
        print(f"  Total removed             : {X.shape[1] - len(self.selected_features):>5}  "
              f"({(X.shape[1]-len(self.selected_features))/X.shape[1]*100:.1f}%)")
        print(f"{'─'*60}")

        return X_pruned

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply saved selected features to a new DataFrame (inference mode)."""
        missing = [c for c in self.selected_features if c not in X.columns]
        if missing:
            for c in missing:
                X = X.copy()
                X[c] = 0
        return X[self.selected_features].copy()

    def save_report(self, path: str) -> None:
        """Write a human-readable report of what was removed and why."""
        lines = [
            "=" * 70,
            "STEP 1: INITIAL PRUNING REPORT",
            "=" * 70,
            "",
            f"Variance threshold    : {self.variance_threshold}",
            f"Correlation threshold : {self.correlation_threshold}",
            "",
            f"Total features removed: {len(self.removed_variance) + len(self.removed_correlation)}",
            f"  Via variance filter     : {len(self.removed_variance)}",
            f"  Via correlation filter  : {len(self.removed_correlation)}",
            f"Features retained         : {len(self.selected_features)}",
            "",
            "─" * 70,
            f"REMOVED — Near-constant (variance < {self.variance_threshold}):",
            "─" * 70,
        ]
        for f in sorted(self.removed_variance):
            lines.append(f"  {f}")
        lines += [
            "",
            "─" * 70,
            f"REMOVED — Highly correlated (|r| > {self.correlation_threshold}):",
            "─" * 70,
        ]
        for f in sorted(self.removed_correlation):
            lines.append(f"  {f}")
        lines += [
            "",
            "─" * 70,
            "RETAINED FEATURES:",
            "─" * 70,
        ]
        for f in sorted(self.selected_features):
            lines.append(f"  {f}")
        lines.append("=" * 70)

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        print(f"\n  Step-1 report saved to: {path}")


def run_initial_pruning(X: pd.DataFrame,
                         y=None,
                         treatment=None,
                         save_report_path: str = None) -> tuple:
    """
    Convenience wrapper: fit InitialPruning and return (X_pruned, pruner).

    Parameters
    ----------
    X                : pd.DataFrame  Full feature matrix (1,800 columns)
    y                : ignored (kept for API symmetry)
    treatment        : ignored (kept for API symmetry)
    save_report_path : str  If provided, save a text pruning report here.

    Returns
    -------
    (X_pruned, pruner)
        X_pruned : pd.DataFrame  with variance + corr-filtered features
        pruner   : InitialPruning  (fitted; can be serialised for inference)
    """
    pruner  = InitialPruning()
    X_pruned = pruner.fit_transform(X)

    if save_report_path:
        pruner.save_report(save_report_path)

    return X_pruned, pruner


if __name__ == "__main__":
    # Quick smoke test
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.data_generation import generate_epsilon_data
    df = generate_epsilon_data()
    outcome_cols = ['treatment', 'treatment_name', 'opening_balance',
                    'on_book_month9', 'offer']
    X = df[[c for c in df.columns if c not in outcome_cols]]
    X_p, pruner = run_initial_pruning(X)
    print(f"\nFinal: {X.shape[1]} → {X_p.shape[1]} features")
