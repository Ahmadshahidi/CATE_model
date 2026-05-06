"""
Optuna Tuning — X-Learner CatBoost Regressors
===============================================
Optimises the CatBoost hyper-parameters used for the *outcome models*
(µ₀, µ₁) and *CATE models* (τ̂ₜ, τ̂_c) inside the X-Learner.

Objective
---------
Maximise AUUC lift on a held-out validation fold.
Each trial trains a single-arm X-Learner (arm 1 vs. control) with the
suggested parameters; the remaining arms share the same parameter set
so one trial is a reliable proxy for the full model.

Storage
-------
Study is persisted to  tuning/studies/xlearner_study.db  (SQLite) so it
can be paused and resumed with  --resume.

Output
------
tuning/best_params/xlearner_params.json   — best CatBoost regressor params
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Directories ─────────────────────────────────────────────────────────────
_STUDY_DIR   = os.path.join(os.path.dirname(__file__), 'studies')
_PARAMS_DIR  = os.path.join(os.path.dirname(__file__), 'best_params')
_STORAGE     = f"sqlite:///{os.path.join(_STUDY_DIR, 'xlearner_study.db')}"
_OUT_JSON    = os.path.join(_PARAMS_DIR, 'xlearner_params.json')
_STUDY_NAME  = 'xlearner_catboost_regressors'


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_data():
    """
    Load the bias-corrected dataset saved by pipeline.py after bias correction.

    Returns (X, y, treatment, sample_weight) where sample_weight is None for
    PSM-matched or unweighted runs, and a float array for IPTW/overlap runs.
    """
    tuning_path = os.path.join(config.DATA_DIR, 'xlearner_tuning_data.csv')
    if not os.path.exists(tuning_path):
        raise FileNotFoundError(
            f"Tuning dataset not found: {tuning_path}\n"
            "Run pipeline.py first to generate the bias-corrected tuning dataset."
        )

    print(f"  Loading bias-corrected tuning dataset: {tuning_path}")
    df = pd.read_csv(tuning_path)
    reserved = {'__treatment__', '__opening_balance__', '__sample_weight__'}
    X = df[[c for c in df.columns if c not in reserved]]
    y = df['__opening_balance__'].values
    t = df['__treatment__'].values.astype(int)
    w_col = df['__sample_weight__'].values.astype(float)
    # Uniform weights (all 1.0) means PSM-matched or no correction — pass None
    w = None if np.allclose(w_col, 1.0) else w_col
    print(f"  Rows: {len(X):,}  |  Features: {X.shape[1]}  |  "
          f"Weights: {'custom' if w is not None else 'uniform'}")
    return X, y, t, w


def _compute_auuc(y, cate_scores, t_binary):
    """Scalar AUUC lift for one arm."""
    from scipy import integrate
    order  = np.argsort(-cate_scores)
    n_t    = t_binary.sum()
    n_c    = len(t_binary) - n_t
    if n_t == 0 or n_c == 0:
        return 0.0

    cum_uplift, cum_rand = [0.0], [0.0]
    ct = cc = ctn = ccn = 0.0
    mean_uplift = y[t_binary == 1].mean() - y[t_binary == 0].mean()

    for i, idx in enumerate(order):
        if t_binary[idx]:
            ct += y[idx]; ctn += 1
        else:
            cc += y[idx]; ccn += 1
        up   = (ct / ctn if ctn else 0) - (cc / ccn if ccn else 0)
        cum_uplift.append(up * (i + 1))
        cum_rand.append(mean_uplift * (i + 1))

    pcts = np.linspace(0, 100, len(cum_uplift))
    auuc = integrate.trapezoid(cum_uplift, pcts)
    rand = integrate.trapezoid(cum_rand,   pcts)
    return auuc - rand


def _build_params(trial: optuna.Trial) -> dict:
    """Sample CatBoost regressor hyperparameters from the trial."""
    return {
        'iterations':       trial.suggest_int('iterations',       200, 1000, step=100),
        'depth':            trial.suggest_int('depth',             4,   8),
        'learning_rate':    trial.suggest_float('learning_rate',   0.01, 0.20, log=True),
        'l2_leaf_reg':      trial.suggest_float('l2_leaf_reg',     1.0,  20.0, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',  1,    50),
        'border_count':     trial.suggest_int('border_count',      32,   255),
        # Fixed keys — not tuned
        'eval_metric':         'RMSE',
        'od_type':             'Iter',
        'od_wait':             40,
        'random_seed':         config.RANDOM_SEED,
        'thread_count':        -1,
        'verbose':             False,
        'allow_writing_files': False,
        'nan_mode':            'Min',
    }


# ── Objective ────────────────────────────────────────────────────────────────

def objective(trial: optuna.Trial, X: pd.DataFrame, y, treatment,
              sample_weight=None) -> float:
    """
    Train a single-arm (arm=1 vs. control) X-Learner on the training split
    and return -AUUC_lift on the validation split.

    sample_weight — per-row IPTW/overlap weights from pipeline.py (or None for
    PSM-matched / unweighted data).  Weights are applied to the Stage-1 outcome
    models and Stage-2 CATE models so the objective reflects the same weighting
    scheme used during full training.
    """
    from catboost import CatBoostRegressor, CatBoostClassifier, Pool

    params = _build_params(trial)

    # ── Data split ─────────────────────────────────────────────────────
    t_arr = np.array(treatment, dtype=int)
    y_arr = np.array(y, dtype=float)
    w_arr = np.array(sample_weight, dtype=float) if sample_weight is not None else None

    # Keep only arm-1 and control for a fast single-arm proxy
    mask_proxy = (t_arr == 0) | (t_arr == 1)
    X_p = X[mask_proxy].reset_index(drop=True)
    y_p = y_arr[mask_proxy]
    t_p = (t_arr[mask_proxy] == 1).astype(int)
    w_p = w_arr[mask_proxy] if w_arr is not None else None

    split_args = [X_p, y_p, t_p]
    if w_p is not None:
        split_args.append(w_p)

    splits = train_test_split(
        *split_args,
        test_size    = 0.25,
        stratify     = t_p,
        random_state = config.RANDOM_SEED,
    )

    if w_p is not None:
        X_tr, X_val, y_tr, y_val, t_tr, t_val, w_tr, w_val = splits
    else:
        X_tr, X_val, y_tr, y_val, t_tr, t_val = splits
        w_tr = w_val = None

    feat_names  = X_p.columns.tolist()
    cat_indices = [i for i, c in enumerate(feat_names)
                   if X_p[c].dtype.name in ('object', 'category')]

    def _pool(X_arr, y_arr=None, w=None):
        df = pd.DataFrame(X_arr, columns=feat_names)
        return Pool(df, label=y_arr, cat_features=cat_indices, weight=w)

    ctrl_tr = t_tr == 0
    arm_tr  = t_tr == 1

    w_ctrl = w_tr[ctrl_tr] if w_tr is not None else None
    w_arm  = w_tr[arm_tr]  if w_tr is not None else None

    # Stage 1 — outcome models
    mu0 = CatBoostRegressor(**params)
    mu0.fit(_pool(X_tr[ctrl_tr], y_tr[ctrl_tr], w=w_ctrl))

    mu1 = CatBoostRegressor(**params)
    mu1.fit(_pool(X_tr[arm_tr], y_tr[arm_tr], w=w_arm))

    # Stage 2 — pseudo-outcomes
    mu0_pred_all_tr = mu0.predict(_pool(X_tr))
    mu1_pred_ctrl   = mu1.predict(_pool(X_tr[ctrl_tr]))

    D_treated = y_tr[arm_tr]  - mu0_pred_all_tr[arm_tr]
    D_control = mu1_pred_ctrl - y_tr[ctrl_tr]

    # Stage 3 — CATE models
    tau_t = CatBoostRegressor(**params)
    tau_t.fit(_pool(X_tr[arm_tr], D_treated, w=w_arm))

    tau_c = CatBoostRegressor(**params)
    tau_c.fit(_pool(X_tr[ctrl_tr], D_control, w=w_ctrl))

    # Propensity (simple logistic for speed; fixed params)
    ps_model = CatBoostClassifier(
        iterations=200, depth=4, learning_rate=0.05, verbose=False,
        allow_writing_files=False, random_seed=config.RANDOM_SEED,
    )
    ps_model.fit(_pool(X_tr, t_tr))

    # ── Validation AUUC ────────────────────────────────────────────────
    X_val_arr = np.array(X_val)
    pool_val  = _pool(X_val_arr)

    e_x      = ps_model.predict_proba(pool_val)[:, 1].clip(0.05, 0.95)
    tau_t_v  = tau_t.predict(pool_val)
    tau_c_v  = tau_c.predict(pool_val)
    cate_val = e_x * tau_t_v + (1 - e_x) * tau_c_v

    auuc_lift = _compute_auuc(y_val, cate_val, t_val)
    return -auuc_lift     # Optuna minimises


# ── Main ─────────────────────────────────────────────────────────────────────

def run_study(n_trials: int = 50, resume: bool = False) -> dict:
    os.makedirs(_STUDY_DIR,  exist_ok=True)
    os.makedirs(_PARAMS_DIR, exist_ok=True)

    print("\n" + "="*65)
    print("OPTUNA TUNING — X-LEARNER (CatBoost Regressors)")
    print("="*65)
    print(f"  Trials    : {n_trials}")
    print(f"  Storage   : {_STORAGE}")
    print(f"  Study name: {_STUDY_NAME}")
    print("="*65 + "\n")

    print("Loading data ...")
    X, y, treatment, sample_weight = _load_data()
    print(f"  Features : {X.shape[1]}   Rows : {len(X):,}\n")

    study = optuna.create_study(
        study_name       = _STUDY_NAME,
        storage          = _STORAGE,
        sampler          = TPESampler(seed=config.RANDOM_SEED),
        direction        = 'minimize',
        load_if_exists   = resume,
    )

    study.optimize(
        lambda trial: objective(trial, X, y, treatment, sample_weight=sample_weight),
        n_trials  = n_trials,
        show_progress_bar = True,
    )

    best = study.best_trial
    print(f"\n  Best trial : #{best.number}  "
          f"AUUC lift = {-best.value:,.2f}")
    print("  Best params:")
    for k, v in best.params.items():
        print(f"    {k:<22} : {v}")

    # Build full param dict (merge tuned values with fixed keys)
    full_params = {
        'iterations':          best.params['iterations'],
        'depth':               best.params['depth'],
        'learning_rate':       round(best.params['learning_rate'], 6),
        'l2_leaf_reg':         round(best.params['l2_leaf_reg'], 4),
        'min_data_in_leaf':    best.params['min_data_in_leaf'],
        'border_count':        best.params['border_count'],
        'eval_metric':         'RMSE',
        'od_type':             'Iter',
        'od_wait':             50,
        'random_seed':         config.RANDOM_SEED,
        'thread_count':        -1,
        'verbose':             False,
        'allow_writing_files': False,
        'nan_mode':            'Min',
    }

    with open(_OUT_JSON, 'w') as fh:
        json.dump(full_params, fh, indent=2)
    print(f"\n  Best params saved to: {_OUT_JSON}")

    # Param importance
    try:
        importance = optuna.importance.get_param_importances(study)
        print("\n  Parameter importances:")
        for name, imp in sorted(importance.items(), key=lambda x: -x[1])[:8]:
            print(f"    {name:<22} : {imp:.3f}")
    except Exception:
        pass

    return full_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Optuna tuning for X-Learner CatBoost regressors')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of Optuna trials (default 50)')
    parser.add_argument('--resume',   action='store_true',
                        help='Load existing study from storage and continue')
    args = parser.parse_args()
    run_study(n_trials=args.n_trials, resume=args.resume)
