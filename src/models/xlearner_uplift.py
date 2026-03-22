"""
X-Learner for Multi-Arm Uplift Modeling
=========================================
Estimates Conditional Average Treatment Effects (CATEs) for each offer arm
vs. control using the X-Learner meta-learner (Künzel et al., 2019).

X-Learner algorithm (per arm k):
  Stage 1 — Outcome models:
    µ̂₀(x)  = E[Y | X=x, T=0]        (response surface for control)
    µ̂ₖ(x)  = E[Y | X=x, T=k]        (response surface for arm k)

  Stage 2 — Pseudo-outcomes:
    D̃ₖᵢ = Yᵢ − µ̂₀(xᵢ)   for treated  (i ∈ treated)
    D̃₀ⱼ = µ̂ₖ(xⱼ) − Yⱼ  for control  (j ∈ control)

  Stage 3 — CATE models:
    τ̂ₜ(x)  = model fitted on (X_treated, D̃ₖ)
    τ̂_c(x) = model fitted on (X_control, D̃₀)

  Prediction:
    τ̂ₖ(x) = ê(x) · τ̂ₜ(x) + (1 − ê(x)) · τ̂_c(x)
    where ê(x) = propensity score = P(T=k | X=x)

Usage:
    from src.models.xlearner_uplift import train_xlearner
    xlearner, auuc_df = train_xlearner(X, y, treatment, save_results_dir)
    cates = xlearner.predict_all_cates(X_new)
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import config

warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_xgb_regressor(**overrides) -> XGBRegressor:
    """Return an XGBRegressor using config.XGBOOST_PARAMS, with optional overrides."""
    params = dict(config.XGBOOST_PARAMS)
    params.update(overrides)

    # Apply monotone constraints if all referenced features are present — handled at fit time
    mc = config.MONOTONE_CONSTRAINTS
    if mc:
        # Store separately; applied at fit time when feature names are known
        params['_monotone_constraints'] = mc
    return XGBRegressor(**{k: v for k, v in params.items()
                            if not k.startswith('_')})


def _make_monotone_tuple(feature_names: list, constraints: dict) -> tuple:
    """
    Build the XGBoost ``monotone_constraints`` tuple aligned to feature order.
    Only apply constraints for features that exist in feature_names.
    """
    return tuple(
        constraints.get(f, 0) for f in feature_names
    )


# ──────────────────────────────────────────────────────────────────────────────
# X-Learner class
# ──────────────────────────────────────────────────────────────────────────────

class XLearnerUplift:
    """
    Multi-arm X-Learner with XGBoost base learners.

    Attributes
    ----------
    models        : dict {arm_id: {'mu0','mu1','tau_t','tau_c','ps'}}
    feature_names : list[str]
    arm_ids       : list[int]
    """

    def __init__(self):
        self.models        = {}
        self.feature_names = []
        self.arm_ids       = []

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y, treatment,
            sample_weight=None, test_size: float = None):
        """
        Fit the X-Learner for all treatment arms vs. control.

        Parameters
        ----------
        X             : pd.DataFrame   Boruta-SHAP selected features.
        y             : array-like     Continuous outcome (opening_balance).
        treatment     : array-like     Multi-arm treatment ID (0 = control).
        sample_weight : array-like or None
            Per-observation weights (e.g. IPTW weights).  When provided,
            weights are passed to every XGBoost ``fit()`` call.
            The control-arm weight array is re-used across all arms.
            If None, all weights default to 1.0 (standard, unweighted fit).
        test_size     : float          Held-out fraction for evaluation curves.
        """
        test_size = test_size or config.TEST_SIZE
        self.feature_names = X.columns.tolist()
        self.arm_ids       = sorted(k for k in config.TREATMENT_COMPONENTS if k != 0)

        X_arr   = np.array(X)
        y_arr   = np.array(y, dtype=float)
        t_arr   = np.array(treatment, dtype=int)

        # Normalise sample_weight to a numpy array (or None)
        if sample_weight is not None:
            sw_arr = np.array(sample_weight, dtype=float)
            assert len(sw_arr) == len(y_arr), (
                f"sample_weight length ({len(sw_arr)}) must match dataset "
                f"length ({len(y_arr)})"
            )
            print(f"\n  IPTW sample weights supplied — "
                  f"min={sw_arr.min():.4f}  max={sw_arr.max():.4f}  "
                  f"mean={sw_arr.mean():.4f}")
        else:
            sw_arr = None

        if config.LOG_TRANSFORM_TARGET:
            y_arr = np.log1p(y_arr.clip(0))

        # ── Control outcome model (shared across arms) ────────────
        ctrl_mask = t_arr == 0
        sw_ctrl   = sw_arr[ctrl_mask] if sw_arr is not None else None
        print(f"\n  Fitting control outcome model  (n={ctrl_mask.sum():,}) ...")
        mu0 = self._fit_outcome_model(X_arr[ctrl_mask], y_arr[ctrl_mask],
                                       self.feature_names,
                                       sample_weight=sw_ctrl)
        mu0_pred_all = self._predict_outcome(mu0, X_arr, self.feature_names)

        # ── Per-arm X-Learner ─────────────────────────────────────
        arm_iter = (
            _tqdm(self.arm_ids, desc='  Training arms', unit='arm', ncols=80,
                  bar_format='  {desc}: {percentage:3.0f}%|{bar}| '
                              '{n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            if _HAS_TQDM else self.arm_ids
        )

        for arm_id in arm_iter:
            t_arm_start = time.time()
            arm_mask = t_arr == arm_id
            print(f"\n  Arm {arm_id} ({config.TREATMENTS[arm_id]})  "
                  f"n_treated={arm_mask.sum():,}  n_control={ctrl_mask.sum():,}")

            X_t  = X_arr[arm_mask]
            y_t  = y_arr[arm_mask]
            X_c  = X_arr[ctrl_mask]
            y_c  = y_arr[ctrl_mask]

            sw_t = sw_arr[arm_mask] if sw_arr is not None else None
            sw_c = sw_ctrl          # already sliced above

            # Stage 1 — treated outcome model
            print(f"    [1/4] Fitting treated outcome model (µ₁) ...")
            mu1 = self._fit_outcome_model(X_t, y_t, self.feature_names,
                                           sample_weight=sw_t)
            mu1_pred_ctrl = self._predict_outcome(mu1, X_c, self.feature_names)

            # Stage 2 — pseudo-outcomes
            print(f"    [2/4] Computing CATE pseudo-outcomes ...")
            mu0_pred_treated = mu0_pred_all[arm_mask]
            D_tilde_treated  = y_t - mu0_pred_treated          # for treated
            D_tilde_control  = mu1_pred_ctrl - y_c             # for control

            print(f"    CATE pseudo-outcomes: "
                  f"treated mean={D_tilde_treated.mean():+.2f}  "
                  f"control mean={D_tilde_control.mean():+.2f}")

            # Stage 3 — CATE models
            print(f"    [3/4] Fitting CATE models (τ_t, τ_c) ...")
            tau_t = self._fit_cate_model(X_t, D_tilde_treated, self.feature_names,
                                          sample_weight=sw_t)
            tau_c = self._fit_cate_model(X_c, D_tilde_control, self.feature_names,
                                          sample_weight=sw_c)

            # Propensity score (used to blend τ_t and τ_c)
            print(f"    [4/4] Fitting propensity score model ...")
            ps_model, ps_scaler = self._fit_propensity(X_arr, (t_arr == arm_id).astype(int))

            self.models[arm_id] = {
                'mu0':      mu0,
                'mu1':      mu1,
                'tau_t':    tau_t,
                'tau_c':    tau_c,
                'ps_model': ps_model,
                'ps_scaler':ps_scaler,
            }

            # Quick diagnostic
            cate_all = self._predict_cate_for_arm(arm_id, X_arr, self.feature_names)
            arm_elapsed = time.time() - t_arm_start
            print(f"    CATE diagnostics: mean={cate_all.mean():+.2f}  "
                  f"median={np.median(cate_all):+.2f}  "
                  f"std={cate_all.std():.2f}  "
                  f"pct_positive={100*(cate_all>0).mean():.1f}%  "
                  f"[{arm_elapsed:.1f}s]")

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict_all_cates(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict CATEs for all arms for a given feature matrix.

        Parameters
        ----------
        X : pd.DataFrame  (same feature columns as used in training)

        Returns
        -------
        pd.DataFrame  columns: cate_treatment_{arm_id} for each arm
        """
        X_arr  = X[self.feature_names].values
        result = {}
        for arm_id in self.arm_ids:
            cate = self._predict_cate_for_arm(arm_id, X_arr, self.feature_names)
            if config.LOG_TRANSFORM_TARGET:
                cate = np.expm1(cate)
            result[f'cate_treatment_{arm_id}'] = cate
        return pd.DataFrame(result, index=X.index)

    def predict_cate_scenario(self, X: pd.DataFrame, scenario: dict) -> pd.DataFrame:
        """
        Predict CATEs after overriding remail / stipulation columns
        with the values in ``scenario``.

        This produces *counterfactual* CATEs, i.e. what the uplift
        would be if remail / stipulation were set as in the scenario
        for all prospects.

        Parameters
        ----------
        X        : pd.DataFrame  Feature matrix.
        scenario : dict          e.g. {'remail': 1, 'stipulation': 0}

        Returns
        -------
        pd.DataFrame  (same structure as predict_all_cates)
        """
        X_scen = X[self.feature_names].copy()
        for col, val in scenario.items():
            if col in X_scen.columns:
                X_scen[col] = val
        return self.predict_all_cates(X_scen)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def compute_auuc(self, X: pd.DataFrame, y, treatment) -> pd.DataFrame:
        """
        Compute AUUC (Area Under the Uplift Curve) for each arm.

        Returns
        -------
        pd.DataFrame  columns: arm_id, arm_name, auuc, auuc_random, auuc_lift
        """
        cates = self.predict_all_cates(X)
        y_arr = np.array(y, dtype=float)
        t_arr = np.array(treatment, dtype=int)
        rows  = []

        for arm_id in self.arm_ids:
            cate_col  = cates[f'cate_treatment_{arm_id}'].values
            arm_mask  = (t_arr == arm_id)
            ctrl_mask = (t_arr == 0)
            sub_mask  = arm_mask | ctrl_mask

            cate_sub  = cate_col[sub_mask]
            y_sub     = y_arr[sub_mask]
            t_sub     = arm_mask[sub_mask].astype(int)

            auuc, auuc_rand, auuc_lift = self._compute_auuc_arm(y_sub, cate_sub, t_sub)
            rows.append({
                'arm_id':    arm_id,
                'arm_name':  config.TREATMENTS[arm_id],
                'auuc':      auuc,
                'auuc_rand': auuc_rand,
                'auuc_lift': auuc_lift,
                'auuc_norm': auuc_lift / auuc_rand if auuc_rand else 0,
            })

        return pd.DataFrame(rows)

    @staticmethod
    def _compute_auuc_arm(y, cate_score, t_binary) -> tuple:
        """Compute scalar AUUC for one arm."""
        order      = np.argsort(-cate_score)
        n          = len(order)
        n_t        = t_binary.sum()
        n_c        = n - n_t

        cum_uplift, cum_random = [0.0], [0.0]
        cum_t_resp = cum_c_resp = 0.0
        cum_t_n    = cum_c_n   = 0.0

        for i, idx in enumerate(order):
            if t_binary[idx] == 1:
                cum_t_resp += y[idx]; cum_t_n += 1
            else:
                cum_c_resp += y[idx]; cum_c_n += 1
            uplift = (cum_t_resp / cum_t_n if cum_t_n else 0) - \
                     (cum_c_resp / cum_c_n if cum_c_n else 0)
            cum_uplift.append(uplift * (i + 1))
            cum_random.append((y[t_binary == 1].mean() - y[t_binary == 0].mean())
                               * (i + 1) if n_t and n_c else 0)

        pcts  = np.linspace(0, 100, len(cum_uplift))
        auuc  = integrate.trapezoid(cum_uplift, pcts)
        randc = integrate.trapezoid(cum_random, pcts)
        return auuc, randc, auuc - randc

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_uplift_curves(self, X: pd.DataFrame, y, treatment,
                            save_path: str = None):
        """Uplift curves and Qini curves for each arm."""
        cates  = self.predict_all_cates(X)
        y_arr  = np.array(y, dtype=float)
        t_arr  = np.array(treatment, dtype=int)
        arm_ids = self.arm_ids
        n_arms  = len(arm_ids)
        cmap    = plt.get_cmap('tab10')

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for i, arm_id in enumerate(arm_ids):
            arm_name = config.TREATMENTS[arm_id]
            color    = cmap(i)
            cate_col = cates[f'cate_treatment_{arm_id}'].values
            arm_mask  = (t_arr == arm_id)
            ctrl_mask = (t_arr == 0)
            sub_mask  = arm_mask | ctrl_mask

            cate_sub  = cate_col[sub_mask]
            y_sub     = y_arr[sub_mask]
            t_sub     = arm_mask[sub_mask].astype(int)

            # Uplift curve
            order = np.argsort(-cate_sub)
            n     = len(order)
            pcts  = []
            gains = []
            cum_t = cum_c = cum_tn = cum_cn = 0.0

            for j, idx in enumerate(order):
                if t_sub[idx] == 1:
                    cum_t += y_sub[idx]; cum_tn += 1
                else:
                    cum_c += y_sub[idx]; cum_cn += 1
                pcts.append(100 * (j + 1) / n)
                gains.append((cum_t / cum_tn if cum_tn else 0) -
                              (cum_c / cum_cn if cum_cn else 0))

            axes[0].plot(pcts, gains, color=color, lw=2, label=arm_name)

            # Qini curve
            qini_vals = [0.0]
            for j, idx in enumerate(order):
                prev = qini_vals[-1]
                if t_sub[idx] == 1:
                    cum_c2 = sum(y_sub[order[:j]] * (1 - t_sub[order[:j]]))
                    cum_cn2 = (1 - t_sub[order[:j+1]]).sum()
                    qini_vals.append(cum_t - (cum_c2 + y_sub[idx] * 0) *
                                     (sum(t_sub[order[:j+1]]) /
                                      max(1, cum_cn2)))
                else:
                    qini_vals.append(prev)
            qini_pcts = np.linspace(0, 100, len(qini_vals))
            axes[1].plot(qini_pcts, qini_vals, color=color, lw=2, label=arm_name)

        baseline_mean = (y_arr[t_arr > 0].mean() -
                         y_arr[t_arr == 0].mean() if (t_arr == 0).sum() else 0)
        axes[0].axhline(y=baseline_mean, color='grey', linestyle='--',
                        label='Overall mean uplift', alpha=0.7)
        axes[0].set_xlabel('% Population Targeted (by CATE)', fontsize=11)
        axes[0].set_ylabel('Average Uplift vs. Control', fontsize=11)
        axes[0].set_title('Uplift Curves by Arm', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('% Population Targeted', fontsize=11)
        axes[1].set_ylabel('Qini Value', fontsize=11)
        axes[1].set_title('Qini Curves by Arm', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

        plt.suptitle('X-Learner Uplift Model Performance',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Uplift curves saved to: {save_path}")
        plt.close()

    def plot_auuc_comparison(self, auuc_df: pd.DataFrame,
                              net_value_auuc: dict = None,
                              save_path: str = None):
        """Bar chart of AUUC lift by arm, optionally including net-value AUUC."""
        fig, ax = plt.subplots(figsize=(max(8, len(auuc_df) * 2), 5))
        cmap   = plt.get_cmap('tab10')
        arms   = auuc_df['arm_name'].tolist()
        colors = [cmap(i) for i in range(len(arms))]

        bars = ax.bar(arms, auuc_df['auuc_lift'], color=colors, alpha=0.8)

        if net_value_auuc and 'AUUC_lift' in net_value_auuc:
            ax.axhline(y=net_value_auuc['AUUC_lift'], color='red',
                       linestyle='--', linewidth=2,
                       label=f"Personalized strategy AUUC lift: "
                             f"${net_value_auuc['AUUC_lift']:,.0f}")
            ax.legend(fontsize=10)

        ax.set_ylabel('AUUC Lift vs. Random', fontsize=11)
        ax.set_title('AUUC Comparison by Treatment Arm',
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, auuc in zip(bars, auuc_df['auuc_lift']):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f'{auuc:,.0f}',
                    ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  AUUC comparison chart saved to: {save_path}")
        plt.close()

    def plot_cumulative_gain(self, X: pd.DataFrame, y, treatment,
                              net_value_qini_data: dict = None,
                              save_path: str = None):
        """Cumulative gain chart per arm + optional net-value overlay."""
        cates  = self.predict_all_cates(X)
        y_arr  = np.array(y, dtype=float)
        t_arr  = np.array(treatment, dtype=int)
        cmap   = plt.get_cmap('tab10')

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, arm_id in enumerate(self.arm_ids):
            cate_col = cates[f'cate_treatment_{arm_id}'].values
            arm_mask  = (t_arr == arm_id)
            ctrl_mask = (t_arr == 0)
            sub_mask  = arm_mask | ctrl_mask
            cate_sub  = cate_col[sub_mask]
            y_sub     = y_arr[sub_mask]

            order   = np.argsort(-cate_sub)
            pcts    = [100 * (j + 1) / len(order) for j in range(len(order))]
            cum_gain = np.cumsum(y_sub[order]) / y_sub[order].sum() * 100

            ax.plot(pcts, cum_gain, lw=2, color=cmap(i),
                    label=config.TREATMENTS[arm_id])

        # Random baseline (diagonal)
        ax.plot([0, 100], [0, 100], '--', color='grey', alpha=0.6,
                label='Random baseline')

        if net_value_qini_data:
            ax.plot(net_value_qini_data['percentiles'],
                    [100 * v / max(net_value_qini_data['cumulative_net_value'][-1], 1)
                     for v in net_value_qini_data['cumulative_net_value']],
                    lw=2.5, color='black', linestyle='-.',
                    label='Personalized strategy')

        ax.set_xlabel('% Population Targeted (by CATE score)', fontsize=11)
        ax.set_ylabel('% Cumulative Response Captured', fontsize=11)
        ax.set_title('Cumulative Gain Chart by Treatment Arm',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Cumulative gain chart saved to: {save_path}")
        plt.close()

    # ------------------------------------------------------------------
    # Internal model helpers
    # ------------------------------------------------------------------
    def _fit_outcome_model(self, X_arr: np.ndarray, y_arr: np.ndarray,
                            feature_names: list,
                            sample_weight=None) -> XGBRegressor:
        """Fit Stage-1 outcome model, optionally with IPTW sample weights."""
        model = XGBRegressor(**{k: v for k, v in config.XGBOOST_PARAMS.items()})

        mc = config.MONOTONE_CONSTRAINTS
        if mc:
            mono_tuple = _make_monotone_tuple(feature_names, mc)
            if any(v != 0 for v in mono_tuple):
                model = XGBRegressor(
                    **{k: v for k, v in config.XGBOOST_PARAMS.items()},
                    monotone_constraints=mono_tuple,
                )

        model.fit(X_arr, y_arr, sample_weight=sample_weight, verbose=False)
        return model

    def _predict_outcome(self, model: XGBRegressor,
                          X_arr: np.ndarray,
                          feature_names: list) -> np.ndarray:
        return model.predict(X_arr)

    def _fit_cate_model(self, X_arr: np.ndarray, D_tilde: np.ndarray,
                         feature_names: list,
                         sample_weight=None) -> XGBRegressor:
        """Fit Stage-3 CATE model on pseudo-outcomes, optionally with IPTW weights."""
        model = XGBRegressor(**{k: v for k, v in config.XGBOOST_PARAMS.items()})
        model.fit(X_arr, D_tilde, sample_weight=sample_weight, verbose=False)
        return model

    def _fit_propensity(self, X_arr: np.ndarray,
                         t_binary: np.ndarray) -> tuple:
        """Logistic regression propensity score model."""
        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X_arr)
        lr     = LogisticRegression(
            max_iter=500, C=1.0, random_state=config.RANDOM_SEED,
            solver='lbfgs', n_jobs=-1,
        )
        lr.fit(X_sc, t_binary)
        return lr, scaler

    def _predict_cate_for_arm(self, arm_id: int,
                               X_arr: np.ndarray,
                               feature_names: list) -> np.ndarray:
        """Blend τ_t and τ_c using propensity score."""
        m      = self.models[arm_id]
        ps_scaler = m['ps_scaler']
        ps_model  = m['ps_model']
        tau_t     = m['tau_t']
        tau_c     = m['tau_c']

        X_sc = ps_scaler.transform(X_arr)
        e_x  = ps_model.predict_proba(X_sc)[:, 1].clip(0.05, 0.95)

        tau_t_pred = tau_t.predict(X_arr)
        tau_c_pred = tau_c.predict(X_arr)

        return e_x * tau_t_pred + (1 - e_x) * tau_c_pred


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def train_xlearner(X: pd.DataFrame,
                   y,
                   treatment,
                   sample_weight=None,
                   save_results_dir: str = None) -> tuple:
    """
    Train the X-Learner uplift model and optionally save outputs.

    Parameters
    ----------
    X                : pd.DataFrame   Boruta-SHAP selected features.
    y                : array-like     Continuous outcome (opening_balance).
    treatment        : array-like     Multi-arm treatment IDs.
    sample_weight    : array-like or None
        Per-observation IPTW weights.  Passed directly to XGBoost
        ``fit()`` at every stage.  None → unweighted (default).
    save_results_dir : str            If provided, save charts and CSVs here.

    Returns
    -------
    (xlearner_model, auuc_df)
        xlearner_model : XLearnerUplift  (fitted)
        auuc_df        : pd.DataFrame    AUUC metrics per arm
    """
    print("\n" + "="*60)
    print("X-LEARNER UPLIFT MODEL TRAINING")
    print("="*60)
    bias_method = getattr(config, 'BIAS_CORRECTION_METHOD', 'none')
    print(f"\n  Bias correction  : {bias_method.upper()}")
    print(f"  Weighted fit     : {'Yes (IPTW)' if sample_weight is not None else 'No'}")
    print(f"\n  Samples  : {len(X):,}")
    print(f"  Features : {X.shape[1]}")
    arm_counts = pd.Series(treatment).value_counts().sort_index()
    for arm_id, cnt in arm_counts.items():
        print(f"  Arm {arm_id} ({config.TREATMENTS.get(arm_id, arm_id):<10}): {cnt:>6,}")

    xlearner = XLearnerUplift()
    xlearner.fit(X, y, treatment, sample_weight=sample_weight)

    print(f"\n  ✓ X-Learner training complete")

    # AUUC evaluation
    print(f"\n  Computing AUUC metrics ...")
    auuc_df = xlearner.compute_auuc(X, y, treatment)

    print(f"\n  AUUC Summary:")
    for _, row in auuc_df.iterrows():
        print(f"    Arm {int(row['arm_id'])} ({row['arm_name']:<8}):  "
              f"AUUC={row['auuc']:>10,.2f}  "
              f"Lift={row['auuc_lift']:>10,.2f}  "
              f"Norm={row['auuc_norm']:.3f}x")

    if save_results_dir:
        os.makedirs(save_results_dir, exist_ok=True)

        # AUUC CSV
        auuc_path = os.path.join(save_results_dir, 'auuc_metrics.csv')
        auuc_df.to_csv(auuc_path, index=False)
        print(f"\n  AUUC metrics saved to: {auuc_path}")

        # Plots
        xlearner.plot_uplift_curves(
            X, y, treatment,
            save_path=os.path.join(save_results_dir, 'uplift_curves.png'),
        )
        xlearner.plot_auuc_comparison(
            auuc_df,
            save_path=os.path.join(save_results_dir, 'auuc_comparison.png'),
        )
        xlearner.plot_cumulative_gain(
            X, y, treatment,
            save_path=os.path.join(save_results_dir, 'cumulative_gain.png'),
        )

    print("="*60 + "\n")
    return xlearner, auuc_df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.data_generation import generate_epsilon_data
    from src.feature_selection.step1_initial_pruning import run_initial_pruning
    from src.feature_selection.step2_boruta_shap import run_boruta_shap
    df  = generate_epsilon_data()
    exc = ['treatment', 'treatment_name', 'opening_balance', 'on_book_month9', 'offer']
    X   = df[[c for c in df.columns if c not in exc]]
    X1, _ = run_initial_pruning(X)
    X2, _ = run_boruta_shap(X1, df['opening_balance'])
    model, auuc = train_xlearner(X2, df['opening_balance'],
                                  df['treatment'],
                                  save_results_dir=config.RESULTS_DIR)
    cates = model.predict_all_cates(X2)
    print(f"\nCATEs shape: {cates.shape}")
    print(cates.describe())
