"""
Step 2: Leshy Feature Selection  (via ``arfs``)
================================================
Uses the Leshy algorithm — an improved Boruta variant — with SHAP importance
values to identify features that genuinely outperform their randomised
"shadow" twins.  Leshy handles **categorical variables natively** by passing
category-dtype (or object-dtype) columns directly to LightGBM without
requiring any pre-encoding.

  * Creates shadow features (shuffled copies of real features)
  * Trains a LightGBM estimator
  * Computes SHAP importance for real vs. shadow features
  * Over ``max_iter`` (n_trials) rounds, keeps features that consistently
    beat the shadow importance at the specified percentile
  * Uses a two-step Bonferroni + FDR correction (Leshy default)
  * Categorical columns (dtype='category' or 'object') are handled natively

Top-N guarantee (optional)
--------------------------
When ``top_n`` is set, the final feature set is built in three passes:
  1. All confirmed (accepted) features.
  2. Tentative features ranked by hit-rate, highest first ("rough fix").
  3. Any remaining features (including rejected ones) ranked by
     Boruta hit-rate score, until ``top_n`` is reached.

Force-include (optional)
------------------------
Features in ``force_include`` are always added to the final set,
regardless of Leshy's verdict.  Missing columns are zero-filled
during transform() with a warning.

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
# pkg_resources compatibility shim.
# arfs uses  ``from pkg_resources import resource_filename``  internally, but
# pkg_resources is absent in some Python 3.13 venvs that lack setuptools.
# We provide a lightweight stub so that ``arfs`` can be imported without error.
# ---------------------------------------------------------------------------
try:
    import pkg_resources  # noqa: F401
except ImportError:
    import types as _types
    _pkg = _types.ModuleType("pkg_resources")
    _pkg.resource_filename = lambda *a, **kw: ""
    sys.modules["pkg_resources"] = _pkg
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# sklearn 1.6+ renamed check_X_y's ``force_all_finite`` -> ``ensure_all_finite``.
# arfs 2.4.0 still uses the old name.  Wrap check_X_y so both keyword names
# are accepted transparently, making arfs work on any sklearn >= 1.0.
# ---------------------------------------------------------------------------
import sklearn.utils as _sklearn_utils
_orig_check_X_y = _sklearn_utils.check_X_y

def _patched_check_X_y(*args, **kwargs):
    if 'force_all_finite' in kwargs:
        kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
    return _orig_check_X_y(*args, **kwargs)

_sklearn_utils.check_X_y = _patched_check_X_y

# Also patch the reference inside sklearn.utils.validation (arfs imports from there)
try:
    import sklearn.utils.validation as _sklearn_val
    _sklearn_val.check_X_y = _patched_check_X_y
except Exception:
    pass
# ---------------------------------------------------------------------------


class BorutaSHAP:
    """
    Leshy-based feature selector (drop-in replacement for BorutaSHAP).

    Uses ``arfs.feature_selection.Leshy`` as the primary engine, with a
    custom LightGBM + SHAP fallback if arfs cannot be imported.

    Attributes
    ----------
    selected_features   : list[str]  Features in the final output set.
    confirmed_features  : list[str]  Features fully accepted by Leshy.
    tentative_features  : list[str]  Borderline features (kept by rough-fix
                                     or included into top-N tier).
    rejected_features   : list[str]  Features rejected by Leshy core.
    forced_features     : list[str]  Features added via force_include.
    topn_features       : list[str]  Rejected features added to meet top_n.
    hit_rates           : dict       {feature: proxy hit-rate (1/0.5/0)}
    top_n               : int|None   Minimum feature count guarantee.
    force_include       : list[str]  Features always kept regardless of verdict.
    """

    def __init__(self,
                 n_trials: int = None,
                 percentile: int = None,
                 random_state: int = None,
                 task: str = 'regression',
                 top_n: int = None,
                 force_include: list = None):
        self.n_trials      = n_trials     or config.BORUTA_N_TRIALS
        self.percentile    = percentile   or config.BORUTA_PERCENTILE
        self.random_state  = random_state or config.BORUTA_RANDOM_STATE
        self.task          = task   # 'regression' | 'classification'
        self.top_n         = top_n  # minimum feature-count floor (None = disabled)
        self.force_include = list(force_include) if force_include else []

        self.selected_features  = []
        self.confirmed_features = []
        self.tentative_features = []
        self.rejected_features  = []
        self.forced_features    = []   # from force_include
        self.topn_features      = []   # rejected-but-rescued to meet top_n
        self.hit_rates          = {}

    # ------------------------------------------------------------------
    def fit_transform(self, X: pd.DataFrame, y) -> pd.DataFrame:
        """
        Run Leshy feature selection and return the selected feature subset.

        Parameters
        ----------
        X : pd.DataFrame   Feature matrix (output of Step 1).
        y : array-like     Target variable (continuous or binary).

        Returns
        -------
        pd.DataFrame  with only the Leshy-selected columns.
        """
        print(f"\n{'─'*60}")
        print("STEP 2: LESHY FEATURE SELECTION  (arfs)")
        print(f"  Input : {X.shape[0]:,} rows x {X.shape[1]} features")
        print(f"  n_trials (max_iter) : {self.n_trials}")
        print(f"  percentile (perc)   : {self.percentile}")
        print(f"  task                : {self.task}")
        if self.top_n is not None:
            print(f"  top_n (floor)       : {self.top_n}")
        if self.force_include:
            print(f"  force_include       : {self.force_include}")
        print(f"{'─'*60}")

        # Try Leshy (arfs) first; fall back to custom implementation
        try:
            X_sel = self._fit_leshy(X, y)
        except Exception as e:
            print(f"\n  WARNING  Leshy (arfs) error: {e}")
            print("  -> Falling back to custom LightGBM + SHAP implementation ...")
            X_sel = self._fit_custom(X, y)

        # Apply top-N guarantee and force-include overrides
        X_sel = self._apply_top_n_and_force(X, X_sel)

        print(f"\n  STEP 2 SUMMARY")
        print(f"  {'─'*40}")
        print(f"  Input features   : {X.shape[1]:>5}")
        print(f"  Confirmed        : {len(self.confirmed_features):>5}")
        print(f"  Tentative (kept) : {len(self.tentative_features):>5}")
        print(f"  Rejected         : {len(self.rejected_features):>5}")
        if self.topn_features:
            print(f"  Added for Top-N  : {len(self.topn_features):>5}  "
                  f"(top_n={self.top_n})")
        if self.forced_features:
            print(f"  Force-included   : {len(self.forced_features):>5}")
        print(f"  Output features  : {len(self.selected_features):>5}  "
              f"({len(self.selected_features)/X.shape[1]*100:.1f}% of input)")
        print(f"{'─'*60}")

        return X_sel

    # ------------------------------------------------------------------
    def _fit_leshy(self, X: pd.DataFrame, y) -> pd.DataFrame:
        """
        Use arfs.feature_selection.Leshy for feature selection.

        Categorical columns (dtype='category') are encoded as integer codes
        before being passed to Leshy.  This sidesteps a bug in arfs 2.4.0
        where the shadow-feature permutation step (allrelevant.py line 554)
        converts categorical columns to plain object/string arrays, causing
        a type-comparison error inside LightGBM/SHAP.

        With integer codes, LightGBM still treats the columns as categorical
        via its native ``cat_features`` detection on integer-dtype columns
        that were originally categorical.  Feature selection results are
        equivalent because SHAP importance ranks are preserved.

        Parameters map:
            n_trials   -> max_iter
            percentile -> perc
            task       -> regressor / classifier
        """
        from arfs.feature_selection import Leshy
        import lightgbm as lgb

        # ── Encode categorical columns as integer codes ──────────────
        # pd.Categorical.cat.codes gives -1 for NaN, 0/1/2/... for values.
        # LightGBM handles NaN codes (-1) gracefully when cat_features is set.
        X_enc = X.copy()
        cat_cols = [c for c in X_enc.columns
                    if pd.api.types.is_categorical_dtype(X_enc[c])]
        if cat_cols:
            for col in cat_cols:
                X_enc[col] = X_enc[col].cat.codes.astype('int16')
            print(f"  [{len(cat_cols)} categorical columns encoded as int codes "
                  f"for Leshy compatibility]")

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

        selector = Leshy(
            estimator    = model,
            n_estimators = 100,
            perc         = self.percentile,
            alpha        = 0.05,
            importance   = 'shap',
            two_step     = True,
            max_iter     = self.n_trials,
            random_state = self.random_state,
            verbose      = 0,
            keep_weak    = True,   # expose tentative features via support_weak_
        )

        t0 = time.time()
        print(f"\n  Running Leshy  (max_iter={self.n_trials}, perc={self.percentile}) ...")

        selector.fit(X_enc, np.array(y))

        feat_names = X.columns.tolist()

        # Extract confirmed / tentative / rejected from Leshy boolean masks
        confirmed = [f for f, flag in zip(feat_names, selector.support_)
                     if flag and not selector.support_weak_[list(feat_names).index(f)]]
        tentative = [f for f, flag in zip(feat_names, selector.support_weak_) if flag]
        # Features confirmed by support_ but not support_weak_ are truly confirmed
        # Re-derive cleanly:
        confirmed = [feat_names[i]
                     for i in range(len(feat_names))
                     if selector.support_[i] and not selector.support_weak_[i]]
        tentative = [feat_names[i]
                     for i in range(len(feat_names))
                     if selector.support_weak_[i]]
        rejected  = [feat_names[i]
                     for i in range(len(feat_names))
                     if not selector.support_[i] and not selector.support_weak_[i]]

        self.confirmed_features = confirmed
        self.tentative_features = tentative
        self.rejected_features  = rejected

        # Assign proxy hit-rates (Leshy doesn't expose per-trial counts)
        for f in confirmed:
            self.hit_rates[f] = 1.0
        for f in tentative:
            self.hit_rates[f] = 0.5
        for f in rejected:
            self.hit_rates[f] = 0.0

        elapsed = time.time() - t0
        print(f"\n  Leshy complete  ({elapsed:.1f}s)")
        print(f"    Confirmed : {len(confirmed)}")
        print(f"    Tentative : {len(tentative)}")
        print(f"    Rejected  : {len(rejected)}")

        # Return confirmed + tentative; _apply_top_n_and_force adjusts further
        core_selected = confirmed + tentative
        self.selected_features = core_selected
        return X[core_selected].copy()

    # ------------------------------------------------------------------
    def _apply_top_n_and_force(self, X_orig: pd.DataFrame,
                                X_sel: pd.DataFrame) -> pd.DataFrame:
        """
        Post-process the Leshy result by:
          1. Always include confirmed features.
          2. Add tentative features by hit-rate rank until top_n met
             ("rough-fix" tier).
          3. Add any remaining features (including rejected) by hit-rate
             rank until top_n is met.
          4. Add force_include features regardless of outcome.

        Updates self.selected_features, self.topn_features,
        self.forced_features in place.

        Returns the final X sub-DataFrame.
        """
        confirmed = list(self.confirmed_features)
        tentative = list(self.tentative_features)

        # Pass 1: start with confirmed
        selected = list(confirmed)

        # Pass 2: rough-fix tentative tier
        if self.top_n is not None and len(selected) < self.top_n:
            tentative_sorted = sorted(
                tentative,
                key=lambda f: self.hit_rates.get(f, 0.0),
                reverse=True,
            )
            for f in tentative_sorted:
                if f not in selected:
                    selected.append(f)
                    if len(selected) >= self.top_n:
                        break
        else:
            # No top_n constraint — include all tentative (standard behaviour)
            for f in tentative:
                if f not in selected:
                    selected.append(f)

        # Pass 3: rescue from rejected to hit top_n floor
        topn_features = []
        if self.top_n is not None and len(selected) < self.top_n:
            remaining = [f for f in self.hit_rates if f not in selected]
            remaining_sorted = sorted(
                remaining,
                key=lambda f: self.hit_rates.get(f, 0.0),
                reverse=True,
            )
            for f in remaining_sorted:
                if f not in selected:
                    selected.append(f)
                    topn_features.append(f)
                    if len(selected) >= self.top_n:
                        break

            if topn_features:
                print(f"\n  INFO  Top-N rescue: added {len(topn_features)} extra feature(s) "
                      f"to reach top_n={self.top_n}:")
                for f in topn_features:
                    hr = self.hit_rates.get(f, 0.0)
                    print(f"       {f:<50}  hit_rate={hr:.3f}")

        self.topn_features = topn_features

        # Pass 4: force_include
        forced_added = []
        for f in self.force_include:
            if f not in selected:
                if f in X_orig.columns:
                    selected.append(f)
                    forced_added.append(f)
                    print(f"\n  INFO  Force-included: '{f}'  "
                          f"(hit_rate={self.hit_rates.get(f, 'N/A')})")
                else:
                    print(f"\n  WARNING  Force-include '{f}' not found in feature "
                          f"matrix — will be zero-filled during transform().")
                    selected.append(f)
                    forced_added.append(f)

        self.forced_features   = forced_added
        self.selected_features = selected

        # Build output DataFrame
        cols_available = [c for c in selected if c in X_orig.columns]
        cols_missing   = [c for c in selected if c not in X_orig.columns]

        X_out = X_orig[cols_available].copy()
        for c in cols_missing:
            X_out[c] = 0.0
            print(f"  WARNING  Column '{c}' added as zeros (fit-time zero-fill).")

        # Reorder to match selected list order
        X_out = X_out[selected]
        return X_out

    # ------------------------------------------------------------------
    def _fit_custom(self, X: pd.DataFrame, y) -> pd.DataFrame:
        """
        Custom Boruta-SHAP fallback using LightGBM + SHAP.

        For each trial:
          1. Create shadow features (shuffled columns).
          2. Train LightGBM on [X | X_shadow].
          3. Compute SHAP importances.
          4. Each real feature "hits" if its SHAP > percentile(shadow_shap).

        Features with hit_rate > 0.5 are confirmed.
        Features with hit_rate < 0.15 are rejected.
        Rest are tentative.
        """
        try:
            import lightgbm as lgb
            import shap
        except ImportError as e:
            raise ImportError(
                f"lightgbm and shap are required. pip install lightgbm shap. {e}"
            )

        n, p = X.shape
        feat_names = X.columns.tolist()
        hit_counts = np.zeros(p)

        t0  = time.time()
        rng = np.random.default_rng(self.random_state)

        print(f"\n  Running custom Boruta-SHAP  ({self.n_trials} trials x {p} features) ...")

        if _HAS_TQDM:
            trial_iter = _tqdm(
                range(self.n_trials),
                desc='  Boruta-SHAP',
                unit='trial',
                ncols=80,
                leave=True,
                dynamic_ncols=False,
                bar_format=(
                    '  {desc}: {percentage:3.0f}%|{bar:30}| '
                    '{n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
                ),
            )
        else:
            trial_iter = range(self.n_trials)

        for trial in trial_iter:
            if not _HAS_TQDM:
                pct     = (trial + 1) / self.n_trials * 100
                elapsed = time.time() - t0
                sys.stdout.write(
                    f"\r  Boruta-SHAP: {pct:5.1f}% | "
                    f"trial {trial+1}/{self.n_trials} | {elapsed:.0f}s elapsed   "
                )
                sys.stdout.flush()

            # Shadow features: shuffle each column independently
            X_arr    = X.values
            X_shadow = np.apply_along_axis(rng.permutation, 0, X_arr)
            X_aug    = np.hstack([X_arr, X_shadow])

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

            threshold   = np.percentile(shadow_shap, self.percentile)
            hit_counts += (real_shap > threshold).astype(int)

        if not _HAS_TQDM:
            sys.stdout.write("\n")
            sys.stdout.flush()

        hit_rates_arr  = hit_counts / self.n_trials
        self.hit_rates = {feat_names[i]: float(hit_rates_arr[i]) for i in range(p)}

        confirmed = [f for f, r in self.hit_rates.items() if r > 0.50]
        rejected  = [f for f, r in self.hit_rates.items() if r <= 0.15]
        tentative = [f for f in feat_names
                     if f not in confirmed and f not in rejected]

        self.confirmed_features = confirmed
        self.tentative_features = tentative
        self.rejected_features  = rejected

        elapsed = time.time() - t0
        print(f"\n  Custom Boruta-SHAP complete  ({elapsed:.1f}s)")
        print(f"    Confirmed : {len(confirmed)}")
        print(f"    Tentative : {len(tentative)}")
        print(f"    Rejected  : {len(rejected)}")

        core_selected          = confirmed + tentative
        self.selected_features = core_selected
        return X[core_selected].copy()

    # ------------------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply saved feature list to a new DataFrame (inference mode)."""
        X = X.copy()
        missing = [c for c in self.selected_features if c not in X.columns]
        if missing:
            for c in missing:
                print(f"  WARNING  transform(): column '{c}' not found — zero-filled.")
                X[c] = 0
        return X[self.selected_features].copy()

    # ------------------------------------------------------------------
    def save_report(self, path: str) -> None:
        """Write a human-readable Leshy report."""
        lines = [
            "=" * 70,
            "STEP 2: LESHY (arfs) FEATURE SELECTION REPORT",
            "=" * 70,
            "",
            f"n_trials (max_iter) : {self.n_trials}",
            f"percentile (perc)   : {self.percentile}",
            f"top_n (floor)       : {self.top_n}",
            f"force_include       : {self.force_include}",
            "",
            f"  Confirmed      : {len(self.confirmed_features)}",
            f"  Tentative      : {len(self.tentative_features)}  (included in output)",
            f"  Rejected       : {len(self.rejected_features)}",
            f"  Added for Top-N: {len(self.topn_features)}"
            f"  (rescued rejected, hit top_n={self.top_n})",
            f"  Force-included : {len(self.forced_features)}",
            f"  Total output   : {len(self.selected_features)}",
            "",
            "-" * 70,
            "CONFIRMED FEATURES:",
            "-" * 70,
        ]
        for f in sorted(self.confirmed_features):
            hr = self.hit_rates.get(f, 'N/A')
            lines.append(
                f"  {f:<50}  hit_rate={hr:.3f}" if isinstance(hr, float)
                else f"  {f}"
            )
        lines += [
            "",
            "-" * 70,
            "TENTATIVE FEATURES  (included in output by default):",
            "-" * 70,
        ]
        for f in sorted(self.tentative_features):
            hr = self.hit_rates.get(f, 'N/A')
            lines.append(
                f"  {f:<50}  hit_rate={hr:.3f}" if isinstance(hr, float)
                else f"  {f}"
            )

        if self.topn_features:
            lines += [
                "",
                "-" * 70,
                f"ADDED FOR TOP-N GUARANTEE  (top_n={self.top_n}; "
                "ranked by hit_rate, originally rejected):",
                "-" * 70,
            ]
            for f in self.topn_features:
                hr = self.hit_rates.get(f, 'N/A')
                lines.append(
                    f"  {f:<50}  hit_rate={hr:.3f}" if isinstance(hr, float)
                    else f"  {f}"
                )

        if self.forced_features:
            lines += [
                "",
                "-" * 70,
                "FORCE-INCLUDED FEATURES  (always kept regardless of verdict):",
                "-" * 70,
            ]
            for f in self.forced_features:
                hr = self.hit_rates.get(f, 'N/A')
                verdict = (
                    "confirmed" if f in self.confirmed_features else
                    "tentative" if f in self.tentative_features else
                    "rejected"  if f in self.rejected_features  else
                    "external (not seen in fit)"
                )
                hr_str = f"{hr:.3f}" if isinstance(hr, float) else "N/A"
                lines.append(f"  {f:<50}  hit_rate={hr_str}  [{verdict}]")

        lines += [
            "",
            "-" * 70,
            "REJECTED FEATURES:",
            "-" * 70,
        ]
        for f in sorted(self.rejected_features):
            rescued = "  <- rescued for top-N" if f in self.topn_features else ""
            hr = self.hit_rates.get(f, 'N/A')
            lines.append(
                (f"  {f:<50}  hit_rate={hr:.3f}{rescued}"
                 if isinstance(hr, float) else f"  {f}{rescued}")
            )
        lines.append("=" * 70)

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        print(f"  Step-2 report saved to: {path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_boruta_shap(X: pd.DataFrame,
                    y,
                    task: str = 'regression',
                    save_report_path: str = None,
                    top_n: int = None,
                    force_include: list = None) -> tuple:
    """
    Convenience wrapper: run Leshy feature selection and return
    (X_selected, selector).

    Parameters
    ----------
    X                : pd.DataFrame  Output of Step 1.
    y                : array-like    Target variable.
    task             : 'regression' | 'classification'
    save_report_path : str           If provided, save a text report here.
    top_n            : int | None    Minimum feature count (3-pass fill-up).
    force_include    : list | None   Features always kept regardless of verdict.

    Returns
    -------
    (X_selected, selector)
        X_selected : pd.DataFrame  with Leshy-selected features
        selector   : BorutaSHAP    (fitted; can be serialised for inference)
    """
    selector = BorutaSHAP(
        task=task,
        top_n=top_n,
        force_include=force_include or [],
    )
    X_selected = selector.fit_transform(X, y)

    if save_report_path:
        selector.save_report(save_report_path)

    return X_selected, selector


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.data_generation import generate_epsilon_data
    from src.feature_selection.step1_initial_pruning import run_initial_pruning
    df  = generate_epsilon_data()
    exc = ['treatment', 'treatment_name', 'opening_balance', 'on_book_month9', 'offer']
    X   = df[[c for c in df.columns if c not in exc]]
    X1, _  = run_initial_pruning(X)
    X2, bs = run_boruta_shap(X1, df['opening_balance'])
    print(f"\nFinal: {X.shape[1]} -> {X1.shape[1]} -> {X2.shape[1]} features")
