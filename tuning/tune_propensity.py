"""
Optuna Tuning — Propensity Score CatBoost Classifier
=====================================================
Optimises the CatBoost hyper-parameters shared by the PSM and IPTW
propensity models (CATBOOST_PROPENSITY_PARAMS in config.py).

For each trial we fit a binary propensity model for *each* non-control arm
(arm vs. control) using 5-fold stratified cross-validation and report the
mean log-loss (cross-entropy) across all arms and folds.

Lower log-loss → sharper probability separation → better matching/weighting.

Storage
-------
Study is persisted to  tuning/studies/propensity_study.db  (SQLite).

Output
------
tuning/best_params/propensity_params.json  — best CatBoost classifier params
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

_STUDY_DIR  = os.path.join(os.path.dirname(__file__), 'studies')
_PARAMS_DIR = os.path.join(os.path.dirname(__file__), 'best_params')
_STORAGE    = f"sqlite:///{os.path.join(_STUDY_DIR, 'propensity_study.db')}"
_OUT_JSON   = os.path.join(_PARAMS_DIR, 'propensity_params.json')
_STUDY_NAME = 'propensity_catboost_classifier'


def _load_data():
    """Load data and return the Boruta-selected balance feature matrix."""
    data_path = os.path.join(config.DATA_DIR, config.RAW_DATA_FILE)
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            f"Set RAW_DATA_FILE in config.py and place your data in {config.DATA_DIR}."
        )
    df = pd.read_csv(data_path)

    outcome_cols = ['treatment', 'treatment_name', 'opening_balance',
                    'on_book_month9', 'offer']
    X_raw = df[[c for c in df.columns if c not in outcome_cols]]
    t     = df['treatment']

    # Prefer saved balance features (same features used in PSM/IPTW)
    feat_path = os.path.join(config.MODELS_DIR, 'balance_feature_names.json')
    if os.path.exists(feat_path):
        with open(feat_path) as fh:
            feats = json.load(fh)
        feats = [f for f in feats if f in X_raw.columns]
        if feats:
            print(f"  Using saved balance features ({len(feats)}) for tuning.")
            return X_raw[feats], t

    # Fallback: use all pruned features
    feat_path2 = os.path.join(config.MODELS_DIR, 'feature_names.json')
    if os.path.exists(feat_path2):
        with open(feat_path2) as fh:
            feats = json.load(fh)
        feats = [f for f in feats if f in X_raw.columns]
        if feats:
            print(f"  Using saved step-2 features ({len(feats)}) for tuning.")
            return X_raw[feats], t

    from src.feature_selection.step1_initial_pruning import run_initial_pruning
    print("  Running step-1 pruning for propensity tuning ...")
    X1, _ = run_initial_pruning(X_raw)
    return X1, t


def _build_params(trial: optuna.Trial) -> dict:
    return {
        'iterations':       trial.suggest_int('iterations',      100, 600, step=50),
        'depth':            trial.suggest_int('depth',             3,   7),
        'learning_rate':    trial.suggest_float('learning_rate',   0.01, 0.15, log=True),
        'l2_leaf_reg':      trial.suggest_float('l2_leaf_reg',     1.0, 15.0, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',  1,   30),
        'border_count':     trial.suggest_int('border_count',      32,  255),
        # Fixed keys
        'eval_metric':         'AUC',
        'od_type':             'Iter',
        'od_wait':             40,
        'random_seed':         config.RANDOM_SEED,
        'thread_count':        -1,
        'verbose':             False,
        'allow_writing_files': False,
        'nan_mode':            'Min',
    }


def objective(trial: optuna.Trial, X: pd.DataFrame, treatment) -> float:
    """
    For each non-control arm, run 5-fold stratified CV (arm vs. control)
    and accumulate binary log-loss.  Return mean log-loss.
    """
    from catboost import CatBoostClassifier, Pool

    params      = _build_params(trial)
    t_arr       = np.array(treatment, dtype=int)
    arm_ids     = sorted(k for k in config.TREATMENT_COMPONENTS if k != 0)
    cat_indices = [i for i, c in enumerate(X.columns)
                   if X[c].dtype.name in ('object', 'category')]

    ctrl_mask = t_arr == 0
    all_scores: list[float] = []
    step = 0

    for arm_id in arm_ids:
        arm_mask = t_arr == arm_id
        sub_mask = ctrl_mask | arm_mask

        X_sub = X[sub_mask].reset_index(drop=True)
        t_sub = (t_arr[sub_mask] == arm_id).astype(int)

        skf = StratifiedKFold(n_splits=5, shuffle=True,
                               random_state=config.RANDOM_SEED)

        for tr_idx, val_idx in skf.split(X_sub, t_sub):
            X_tr, X_val = X_sub.iloc[tr_idx], X_sub.iloc[val_idx]
            t_tr, t_val = t_sub[tr_idx],       t_sub[val_idx]

            tr_pool  = Pool(X_tr, label=t_tr,  cat_features=cat_indices)
            val_pool = Pool(X_val, label=t_val, cat_features=cat_indices)

            model = CatBoostClassifier(**params)
            model.fit(tr_pool, eval_set=val_pool, use_best_model=True)

            proba = model.predict_proba(val_pool)[:, 1]
            ll = log_loss(t_val, proba)
            all_scores.append(ll)

            # Intermediate pruning signal
            trial.report(np.mean(all_scores), step=step)
            step += 1
            if trial.should_prune():
                raise optuna.TrialPruned()

    return float(np.mean(all_scores))   # minimise log-loss


def run_study(n_trials: int = 40, resume: bool = False) -> dict:
    os.makedirs(_STUDY_DIR,  exist_ok=True)
    os.makedirs(_PARAMS_DIR, exist_ok=True)

    print("\n" + "="*65)
    print("OPTUNA TUNING — PROPENSITY MODEL (CatBoost Classifier)")
    print("="*65)
    print(f"  Trials    : {n_trials}")
    print(f"  Storage   : {_STORAGE}")
    print(f"  Study name: {_STUDY_NAME}")
    print("="*65 + "\n")

    print("Loading data ...")
    X, treatment = _load_data()
    print(f"  Features : {X.shape[1]}   Rows : {len(X):,}\n")

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study  = optuna.create_study(
        study_name     = _STUDY_NAME,
        storage        = _STORAGE,
        sampler        = TPESampler(seed=config.RANDOM_SEED),
        pruner         = pruner,
        direction      = 'minimize',
        load_if_exists = resume,
    )

    study.optimize(
        lambda trial: objective(trial, X, treatment),
        n_trials          = n_trials,
        show_progress_bar = True,
    )

    best = study.best_trial
    print(f"\n  Best trial : #{best.number}  Log-loss = {best.value:.4f}")
    print("  Best params:")
    for k, v in best.params.items():
        print(f"    {k:<24} : {v}")

    full_params = {
        'iterations':          best.params['iterations'],
        'depth':               best.params['depth'],
        'learning_rate':       round(best.params['learning_rate'], 6),
        'l2_leaf_reg':         round(best.params['l2_leaf_reg'], 4),
        'min_data_in_leaf':    best.params['min_data_in_leaf'],
        'border_count':        best.params['border_count'],
        'eval_metric':         'AUC',
        'od_type':             'Iter',
        'od_wait':             40,
        'random_seed':         config.RANDOM_SEED,
        'thread_count':        -1,
        'verbose':             False,
        'allow_writing_files': False,
        'nan_mode':            'Min',
    }

    with open(_OUT_JSON, 'w') as fh:
        json.dump(full_params, fh, indent=2)
    print(f"\n  Best params saved to: {_OUT_JSON}")

    try:
        importance = optuna.importance.get_param_importances(study)
        print("\n  Parameter importances:")
        for name, imp in sorted(importance.items(), key=lambda x: -x[1])[:8]:
            print(f"    {name:<24} : {imp:.3f}")
    except Exception:
        pass

    return full_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Optuna tuning for Propensity CatBoost classifier')
    parser.add_argument('--n-trials', type=int, default=40)
    parser.add_argument('--resume',   action='store_true')
    args = parser.parse_args()
    run_study(n_trials=args.n_trials, resume=args.resume)
