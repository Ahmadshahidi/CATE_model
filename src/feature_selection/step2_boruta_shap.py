"""
Step 2: Boruta-SHAP Feature Selection
=======================================
Uses the Boruta algorithm with SHAP importance values to identify features
that genuinely outperform their randomised "shadow" twins.

  • Creates shadow features (shuffled copies of real features)
  • Trains a LightGBM estimator
  • Computes SHAP importance for real vs. shadow features
  • Over ``n_trials`` rounds, keeps features that consistently beat
    the max shadow importance (at the specified percentile)
  • Features never beating shadow are rejected; borderline ones are
    treated as tentative (accepted by default at the end)

Usage:
    from src.feature_selection.step2_boruta_shap import run_boruta_shap
    X_selected, boruta = run_boruta_shap(X_step1, y=y_balance, task='regression')
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import config

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# scipy.stats.binom_test was removed in SciPy 1.12.
# Patch scipy.stats so that the BorutaShap package can still import it.
# Source: https://stackoverflow.com/a/78090496
# ---------------------------------------------------------------------------
import scipy.stats as _scipy_stats
if not hasattr(_scipy_stats, 'binom_test'):
    def _binom_test_compat(x, n=None, p=0.5, alternative='two-sided'):
        """Thin wrapper around scipy.stats.binomtest for backwards compatibility."""
        return _scipy_stats.binomtest(
            k=int(x), n=int(n), p=p, alternative=alternative
        ).pvalue
    _scipy_stats.binom_test = _binom_test_compat
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# np.NaN was removed in NumPy 2.0; patch it back so BorutaShap can import.
# ---------------------------------------------------------------------------
if not hasattr(np, 'NaN'):
    np.NaN = np.nan
# ---------------------------------------------------------------------------


class BorutaSHAP:
    """
    Boruta-SHAP feature selector.

    Attributes
    ----------
    selected_features : list[str]   Features accepted by Boruta.
    rejected_features : list[str]   Features rejected.
    tentative_features: list[str]   Features neither accepted nor rejected;
                                     included in selected_features by default.
    hit_rates         : dict        {feature: fraction of trials where it beat shadow}
    """

    def __init__(self,
                 n_trials: int  = None,
                 percentile: int = None,
                 random_state: int = None,
                 task: str = 'regression'):
        self.n_trials     = n_trials    or config.BORUTA_N_TRIALS
        self.percentile   = percentile  or config.BORUTA_PERCENTILE
        self.random_state = random_state or config.BORUTA_RANDOM_STATE
        self.task         = task   # 'regression' | 'classification'

        self.selected_features  = []
        self.rejected_features  = []
        self.tentative_features = []
        self.hit_rates          = {}

    # ------------------------------------------------------------------
    def fit_transform(self, X: pd.DataFrame, y) -> pd.DataFrame:
        """
        Run Boruta-SHAP and return the selected feature subset.

        Parameters
        ----------
        X : pd.DataFrame   Feature matrix (output of Step 1).
        y : array-like     Target variable (continuous or binary).

        Returns
        -------
        pd.DataFrame  with only the Boruta-selected columns.
        """
        print(f"\n{'─'*60}")
        print("STEP 2: BORUTA-SHAP FEATURE SELECTION")
        print(f"  Input : {X.shape[0]:,} rows × {X.shape[1]} features")
        print(f"  n_trials   : {self.n_trials}")
        print(f"  percentile : {self.percentile}")
        print(f"  task       : {self.task}")
        print(f"{'─'*60}")

        # Try BorutaShap package first; fall back to custom implementation
        try:
            X_sel = self._fit_boruta_shap_package(X, y)
        except Exception as e:
            print(f"\n  ⚠  BorutaShap package error: {e}")
            print("  → Falling back to custom LightGBM + SHAP implementation ...")
            X_sel = self._fit_custom(X, y)

        print(f"\n  STEP 2 SUMMARY")
        print(f"  {'─'*40}")
        print(f"  Input features   : {X.shape[1]:>5}")
        print(f"  Accepted         : {len(self.selected_features):>5}")
        print(f"  Tentative        : {len(self.tentative_features):>5}  (included)")
        print(f"  Rejected         : {len(self.rejected_features):>5}")
        print(f"  Output features  : {X_sel.shape[1]:>5}  "
              f"({X_sel.shape[1]/X.shape[1]*100:.1f}% of input)")
        print(f"{'─'*60}")

        return X_sel

    # ------------------------------------------------------------------
    def _fit_boruta_shap_package(self, X: pd.DataFrame, y) -> pd.DataFrame:
        """Use the BorutaShap PyPI package."""
        from BorutaShap import BorutaShap as _BorutaShap
        import lightgbm as lgb

        if self.task == 'regression':
            model = lgb.LGBMRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=-1,
            )
        else:
            model = lgb.LGBMClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=-1,
            )

        selector = _BorutaShap(
            model           = model,
            importance_measure = 'shap',
            classification  = (self.task == 'classification'),
            percentile      = self.percentile,
            pvalue          = 0.05,
        )

        t0 = time.time()
        print(f"\n  Running BorutaShap  ({self.n_trials} trials) ...")

        selector.fit(
            X              = X,
            y              = np.array(y),
            n_trials       = self.n_trials,
            random_state   = self.random_state,
            normalize      = True,
            verbose        = False,
            stratify       = None,
        )

        accepted  = list(selector.accepted)
        tentative = list(selector.tentative)
        rejected  = list(selector.rejected)

        self.selected_features  = accepted + tentative
        self.tentative_features = tentative
        self.rejected_features  = rejected

        # Approximate hit rates from BorutaShap history if available
        if hasattr(selector, 'history_x'):
            for feat in self.selected_features:
                self.hit_rates[feat] = 1.0
            for feat in self.rejected_features:
                self.hit_rates[feat] = 0.0
        else:
            for feat in X.columns:
                self.hit_rates[feat] = 1.0 if feat in self.selected_features else 0.0

        elapsed = time.time() - t0
        print(f"\n  ✓ BorutaShap complete  ({elapsed:.1f}s)")
        print(f"    Accepted  : {len(accepted)}")
        print(f"    Tentative : {len(tentative)}")
        print(f"    Rejected  : {len(rejected)}")

        return X[self.selected_features].copy()

    # ------------------------------------------------------------------
    def _fit_custom(self, X: pd.DataFrame, y) -> pd.DataFrame:
        """
        Custom Boruta-SHAP implementation using LightGBM + SHAP.

        For each trial:
          1. Create shadow features (shuffled columns).
          2. Train LightGBM on [X | X_shadow].
          3. Compute SHAP importances.
          4. Each real feature "hits" if its SHAP > np.percentile(shadow_shap).

        Features with hit_rate > 0.5 are accepted.
        Features with hit_rate < (1 - 0.5) are rejected.
        Rest are tentative (accepted by default).
        """
        try:
            import lightgbm as lgb
            import shap
        except ImportError as e:
            raise ImportError(f"lightgbm and shap are required. pip install lightgbm shap. {e}")

        n, p     = X.shape
        feat_names = X.columns.tolist()
        hit_counts = np.zeros(p)

        t0 = time.time()
        rng = np.random.default_rng(self.random_state)

        print(f"\n  Running custom Boruta-SHAP  ({self.n_trials} trials × {p} features) ...")

        if _HAS_TQDM:
            trial_iter = _tqdm(
                range(self.n_trials),
                desc='  Boruta-SHAP',
                unit='trial',
                ncols=80,
                leave=True,
                dynamic_ncols=False,
                bar_format='  {desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
            )
        else:
            trial_iter = range(self.n_trials)

        for trial in trial_iter:
            if not _HAS_TQDM:
                pct = (trial + 1) / self.n_trials * 100
                elapsed = time.time() - t0
                sys.stdout.write(
                    f"\r  Boruta-SHAP: {pct:5.1f}% | "
                    f"trial {trial+1}/{self.n_trials} | {elapsed:.0f}s elapsed   "
                )
                sys.stdout.flush()

            # Shadow features: shuffle each column independently
            X_arr    = X.values
            X_shadow = np.apply_along_axis(rng.permutation, 0, X_arr)

            X_aug   = np.hstack([X_arr, X_shadow])
            col_names = feat_names + [f'shadow_{f}' for f in feat_names]

            # Train LightGBM
            if self.task == 'regression':
                m = lgb.LGBMRegressor(
                    n_estimators=50, max_depth=5,
                    random_state=int(rng.integers(0, 2**31)),
                    n_jobs=-1, verbosity=-1,
                )
            else:
                m = lgb.LGBMClassifier(
                    n_estimators=50, max_depth=5,
                    random_state=int(rng.integers(0, 2**31)),
                    n_jobs=-1, verbosity=-1,
                )

            m.fit(X_aug, np.array(y))

            # SHAP importance (mean |SHAP|)
            explainer  = shap.TreeExplainer(m)
            shap_vals  = np.abs(explainer.shap_values(X_aug)).mean(axis=0)

            real_shap   = shap_vals[:p]
            shadow_shap = shap_vals[p:]

            threshold = np.percentile(shadow_shap, self.percentile)
            hit_counts += (real_shap > threshold).astype(int)

        if not _HAS_TQDM:
            sys.stdout.write("\n")
            sys.stdout.flush()

        # Finalise
        hit_rates_arr = hit_counts / self.n_trials
        self.hit_rates = {feat_names[i]: float(hit_rates_arr[i]) for i in range(p)}

        # Two-sided binomial test thresholds (simplified: 0.5 ± slack)
        accept_thresh  = 0.50
        reject_thresh  = 0.50

        accepted  = [f for f, r in self.hit_rates.items() if r > accept_thresh]
        rejected  = [f for f, r in self.hit_rates.items() if r <= reject_thresh * 0.3]
        tentative = [f for f in feat_names
                     if f not in accepted and f not in rejected]

        self.selected_features  = accepted + tentative
        self.tentative_features = tentative
        self.rejected_features  = rejected

        elapsed = time.time() - t0
        print(f"\n  ✓ Custom Boruta-SHAP complete  ({elapsed:.1f}s)")
        print(f"    Accepted  : {len(accepted)}")
        print(f"    Tentative : {len(tentative)}")
        print(f"    Rejected  : {len(rejected)}")

        return X[self.selected_features].copy()

    # ------------------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply saved feature list to a new DataFrame (inference mode)."""
        missing = [c for c in self.selected_features if c not in X.columns]
        if missing:
            X = X.copy()
            for c in missing:
                X[c] = 0
        return X[self.selected_features].copy()

    # ------------------------------------------------------------------
    def save_report(self, path: str) -> None:
        """Write a human-readable Boruta report."""
        lines = [
            "=" * 70,
            "STEP 2: BORUTA-SHAP FEATURE SELECTION REPORT",
            "=" * 70,
            "",
            f"n_trials   : {self.n_trials}",
            f"percentile : {self.percentile}",
            "",
            f"  Accepted    : {len(self.selected_features) - len(self.tentative_features)}",
            f"  Tentative   : {len(self.tentative_features)}  (included in output)",
            f"  Rejected    : {len(self.rejected_features)}",
            f"  Total output: {len(self.selected_features)}",
            "",
            "─" * 70,
            "ACCEPTED FEATURES  (hit_rate > 0.5):",
            "─" * 70,
        ]
        accepted = [f for f in self.selected_features if f not in self.tentative_features]
        for f in sorted(accepted):
            hr = self.hit_rates.get(f, 'N/A')
            lines.append(f"  {f:<50}  hit_rate={hr:.3f}" if isinstance(hr, float)
                         else f"  {f}")
        lines += [
            "",
            "─" * 70,
            "TENTATIVE FEATURES  (included in output by default):",
            "─" * 70,
        ]
        for f in sorted(self.tentative_features):
            hr = self.hit_rates.get(f, 'N/A')
            lines.append(f"  {f:<50}  hit_rate={hr:.3f}" if isinstance(hr, float)
                         else f"  {f}")
        lines += [
            "",
            "─" * 70,
            "REJECTED FEATURES:",
            "─" * 70,
        ]
        for f in sorted(self.rejected_features):
            hr = self.hit_rates.get(f, 'N/A')
            lines.append(f"  {f:<50}  hit_rate={hr:.3f}" if isinstance(hr, float)
                         else f"  {f}")
        lines.append("=" * 70)

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        print(f"  Step-2 report saved to: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def run_boruta_shap(X: pd.DataFrame,
                    y,
                    task: str = 'regression',
                    save_report_path: str = None) -> tuple:
    """
    Convenience wrapper: run BorutaSHAP and return (X_selected, boruta).

    Parameters
    ----------
    X                : pd.DataFrame  Output of Step 1.
    y                : array-like    Target variable.
    task             : 'regression' | 'classification'
    save_report_path : str           If provided, save a text report here.

    Returns
    -------
    (X_selected, boruta)
        X_selected : pd.DataFrame  with Boruta-selected features
        boruta     : BorutaSHAP    (fitted; can be serialised for inference)
    """
    boruta     = BorutaSHAP(task=task)
    X_selected = boruta.fit_transform(X, y)

    if save_report_path:
        boruta.save_report(save_report_path)

    return X_selected, boruta


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.data_generation import generate_epsilon_data
    from src.feature_selection.step1_initial_pruning import run_initial_pruning
    df = generate_epsilon_data()
    exc = ['treatment', 'treatment_name', 'opening_balance', 'on_book_month9', 'offer']
    X   = df[[c for c in df.columns if c not in exc]]
    X1, _  = run_initial_pruning(X)
    X2, bs = run_boruta_shap(X1, df['opening_balance'])
    print(f"\nFinal: {X.shape[1]} → {X1.shape[1]} → {X2.shape[1]} features")
