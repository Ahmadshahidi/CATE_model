"""
Step 1: Initial Pruning — Variance Thresholding + Correlation Filter
=====================================================================
Removes features that are:
  1. Near-constant  (variance < VARIANCE_THRESHOLD)
  2. Redundant      (Pearson correlation > CORRELATION_THRESHOLD with another feature)

Categorical handling
--------------------
Columns whose dtype is ``'category'`` or ``object`` are **passed through
unchanged** — the variance and correlation filters are applied only to
numeric columns.  The categorical columns are re-appended to the output
DataFrame after filtering.

NaN handling
------------
The variance filter uses ``Series.var(skipna=True)`` (pandas) instead of
sklearn's ``VarianceThreshold`` so that columns with missing values are not
dropped solely because sklearn cannot compute their variance.  The
correlation matrix also uses ``DataFrame.corr()`` which is NaN-safe.

This step reduces ~1,800 Epsilon numeric features to ~500-700 informative
ones before the more expensive Leshy (Boruta-variant) step.

Usage:
    from src.feature_selection.step1_initial_pruning import run_initial_pruning
    X_pruned, pruner = run_initial_pruning(X, y, treatment)
"""

import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import config


class InitialPruning:
    """
    Two-stage filter: variance thresholding -> correlation pruning.

    Categorical columns (dtype 'category' or 'object') are split off before
    filtering and merged back into the final output unchanged.

    Attributes
    ----------
    removed_variance    : list[str]  Numeric columns dropped by variance filter.
    removed_correlation : list[str]  Numeric columns dropped by correlation filter.
    selected_features   : list[str]  Columns that survived (numeric + categorical).
    categorical_features: list[str]  Categorical columns passed through.
    """

    def __init__(self,
                 variance_threshold: float    = None,
                 correlation_threshold: float = None):
        self.variance_threshold    = (variance_threshold
                                      or config.VARIANCE_THRESHOLD)
        self.correlation_threshold = (correlation_threshold
                                      or config.CORRELATION_THRESHOLD)

        self.removed_variance     = []
        self.removed_correlation  = []
        self.selected_features    = []
        self.categorical_features = []

    # ------------------------------------------------------------------
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the pruner on X and return the pruned DataFrame.

        Parameters
        ----------
        X : pd.DataFrame  Feature matrix (numeric + optional categorical columns).

        Returns
        -------
        pd.DataFrame  with only the surviving numeric columns PLUS all
        categorical columns unchanged.
        """
        print(f"\n{'─'*60}")
        print("STEP 1: INITIAL PRUNING")
        print(f"  Input : {X.shape[0]:,} rows x {X.shape[1]} features")
        print(f"  Variance threshold    : {self.variance_threshold}")
        print(f"  Correlation threshold : {self.correlation_threshold}")

        # ── Split numeric vs categorical ─────────────────────────────
        cat_cols = [c for c in X.columns
                    if pd.api.types.is_categorical_dtype(X[c])
                    or pd.api.types.is_object_dtype(X[c])]
        num_cols = [c for c in X.columns if c not in cat_cols]
        self.categorical_features = cat_cols

        if cat_cols:
            print(f"  Categorical columns   : {len(cat_cols)} (passed through)")
        print(f"  Numeric columns       : {len(num_cols)}")
        print(f"{'─'*60}")

        X_num = X[num_cols].copy()
        X_cat = X[cat_cols].copy() if cat_cols else pd.DataFrame(index=X.index)

        # ── Stage 1: NaN-safe variance filter ────────────────────────
        t0 = time.time()

        # pandas .var(skipna=True) computes variance ignoring NaN
        variances = X_num.var(skipna=True)
        kept_var  = variances[variances >= self.variance_threshold].index.tolist()
        dropped   = [c for c in num_cols if c not in kept_var]
        self.removed_variance = dropped
        X_var_df  = X_num[kept_var]

        print(f"\n  [Stage 1] Variance filter  (NaN-safe):")
        print(f"    Removed {len(dropped):>5} near-constant numeric features  "
              f"({len(dropped)/len(num_cols)*100:.1f}%)")
        print(f"    Retained {len(kept_var):>4} numeric features  "
              f"({len(kept_var)/len(num_cols)*100:.1f}%)")

        # ── Stage 2: Correlation filter ───────────────────────────────
        print(f"\n  [Stage 2] Correlation filter  "
              f"(threshold={self.correlation_threshold}):")
        print(f"    Computing {len(kept_var)} x {len(kept_var)} "
              f"correlation matrix ...")

        corr_matrix = X_var_df.corr(numeric_only=True).abs()

        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop_corr = [col for col in upper.columns
                        if any(upper[col] > self.correlation_threshold)]
        self.removed_correlation = to_drop_corr

        X_pruned_num = X_var_df.drop(columns=to_drop_corr)

        elapsed = time.time() - t0
        print(f"    Removed {len(to_drop_corr):>5} correlated numeric features  "
              f"({len(to_drop_corr)/len(kept_var)*100:.1f}%)")
        print(f"    Retained {len(X_pruned_num.columns):>4} numeric features")
        print(f"  Step 1 complete  ({elapsed:.1f}s)")

        # ── Recombine numeric + categorical ───────────────────────────
        X_out = pd.concat([X_pruned_num, X_cat], axis=1)
        self.selected_features = X_out.columns.tolist()

        print(f"\n  STEP 1 SUMMARY")
        print(f"  {'─'*40}")
        print(f"  Original numeric features     : {len(num_cols):>5}")
        print(f"  After variance filter         : {len(kept_var):>5}  "
              f"(removed {len(dropped)})")
        print(f"  After correlation filter      : {len(X_pruned_num.columns):>5}  "
              f"(removed {len(to_drop_corr)})")
        if cat_cols:
            print(f"  Categorical features (pass-through): {len(cat_cols):>3}")
        print(f"  Total output features         : {len(self.selected_features):>5}")
        print(f"  Total numeric removed         : "
              f"{len(num_cols) - len(X_pruned_num.columns):>5}  "
              f"({(len(num_cols)-len(X_pruned_num.columns))/len(num_cols)*100:.1f}%)")
        print(f"{'─'*60}")

        return X_out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply saved selected features to a new DataFrame (inference mode)."""
        X = X.copy()
        missing = [c for c in self.selected_features if c not in X.columns]
        if missing:
            for c in missing:
                print(f"  WARNING  transform(): column '{c}' not found — zero-filled.")
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
            f"Total numeric features removed : "
            f"{len(self.removed_variance) + len(self.removed_correlation)}",
            f"  Via variance filter     : {len(self.removed_variance)}",
            f"  Via correlation filter  : {len(self.removed_correlation)}",
            f"Categorical features (pass-through): {len(self.categorical_features)}",
            f"Total features retained : {len(self.selected_features)}",
            "",
            "─" * 70,
            f"REMOVED — Near-constant numeric (variance < {self.variance_threshold}):",
            "─" * 70,
        ]
        for f in sorted(self.removed_variance):
            lines.append(f"  {f}")
        lines += [
            "",
            "─" * 70,
            f"REMOVED — Highly correlated numeric (|r| > {self.correlation_threshold}):",
            "─" * 70,
        ]
        for f in sorted(self.removed_correlation):
            lines.append(f"  {f}")
        if self.categorical_features:
            lines += [
                "",
                "─" * 70,
                "CATEGORICAL FEATURES PASSED THROUGH UNCHANGED:",
                "─" * 70,
            ]
            for f in sorted(self.categorical_features):
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
    X                : pd.DataFrame  Full feature matrix (numeric + categorical).
    y                : ignored (kept for API symmetry)
    treatment        : ignored (kept for API symmetry)
    save_report_path : str  If provided, save a text pruning report here.

    Returns
    -------
    (X_pruned, pruner)
        X_pruned : pd.DataFrame  with variance + corr-filtered numeric features
                   PLUS all categorical columns unchanged.
        pruner   : InitialPruning  (fitted; can be serialised for inference)
    """
    pruner   = InitialPruning()
    X_pruned = pruner.fit_transform(X)

    if save_report_path:
        pruner.save_report(save_report_path)

    return X_pruned, pruner


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.data_generation import generate_epsilon_data
    df = generate_epsilon_data()
    outcome_cols = ['treatment', 'treatment_name', 'opening_balance',
                    'on_book_month9', 'offer']
    X = df[[c for c in df.columns if c not in outcome_cols]]
    X_p, pruner = run_initial_pruning(X)
    print(f"\nFinal: {X.shape[1]} -> {X_p.shape[1]} features")
    print(f"  Numeric survived: "
          f"{len([c for c in X_p.columns if c not in pruner.categorical_features])}")
    print(f"  Categorical passed through: {len(pruner.categorical_features)}")
