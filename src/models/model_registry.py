"""
Model Registry — Save & Load Pipeline Artefacts for Handoff / Deployment

Saves the complete trained pipeline to ``config.MODELS_DIR`` so that scoring
new prospect files requires no re-training.

Two independent feature-selection paths are now persisted:
  • Balance path  — Boruta-SHAP run against opening_balance  → feeds X-Learner
  • Attrition path — Boruta-SHAP run against on_book_month9  → feeds AttritionModel

Artefacts written
-----------------
  step1_pruner.joblib            — InitialPruning  (variance + correlation filter)
  step2_boruta_balance.joblib    — BorutaSHAP  (balance target → X-Learner features)
  step2_boruta_attrition.joblib  — BorutaSHAP  (attrition target → retention features)
  xlearner_uplift.joblib         — XLearnerUplift  (CatBoost CATE models, N arms)
  attrition_model.joblib         — AttritionModel  (CatBoostClassifier wrapper)
  balance_feature_names.json     — ordered list of features for X-Learner
  attrition_feature_names.json   — ordered list of features for AttritionModel
  pipeline_config.json           — cost params, arm map, decile settings (snapshot)
  MANIFEST.txt                   — human-readable package summary

Usage
-----
  # At end of training pipeline:
  from src.models.model_registry import save_pipeline
  save_pipeline(pruner, boruta_balance, boruta_attrition,
                xlearner_model, attrition_model,
                balance_feature_names, attrition_feature_names)

  # At inference time:
  from src.models.model_registry import load_pipeline
  pkg = load_pipeline()
  # pkg keys: 'pruner', 'boruta_balance', 'boruta_attrition',
  #           'xlearner', 'attrition',
  #           'balance_feature_names', 'attrition_feature_names', 'config'
"""

import os
import sys
import json
import datetime
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


# ------------------------------------------------------------------
# File-name constants (single source of truth)
# ------------------------------------------------------------------
_PRUNER_FILE            = 'step1_pruner.joblib'
_BORUTA_BALANCE_FILE    = 'step2_boruta_balance.joblib'
_BORUTA_ATTRITION_FILE  = 'step2_boruta_attrition.joblib'
_XLEARNER_FILE          = 'xlearner_uplift.joblib'
_ATTRITION_FILE         = 'attrition_model.joblib'
_BALANCE_FEAT_FILE      = 'balance_feature_names.json'
_ATTRITION_FEAT_FILE    = 'attrition_feature_names.json'
_CONFIG_FILE            = 'pipeline_config.json'
_MANIFEST_FILE          = 'MANIFEST.txt'

# Backwards-compatibility alias (scoring scripts that loaded 'feature_names.json')
_LEGACY_FEATURE_FILE    = 'feature_names.json'


# ------------------------------------------------------------------
# SAVE
# ------------------------------------------------------------------
def save_pipeline(pruner,
                  boruta_balance,
                  boruta_attrition,
                  xlearner_model,
                  attrition_model,
                  balance_feature_names=None,
                  attrition_feature_names=None,
                  save_dir=None):
    """
    Serialize all pipeline artefacts to ``save_dir``.

    Parameters
    ----------
    pruner                  : InitialPruning
        Fitted Step-1 pruner (variance + correlation filter).
    boruta_balance          : BorutaSHAP
        Fitted Step-2 selector run against the BALANCE target.
        Selected features are the input to the X-Learner.
    boruta_attrition        : BorutaSHAP
        Fitted Step-2 selector run against the ATTRITION target.
        Selected features are the input to the AttritionModel.
    xlearner_model          : XLearnerUplift
        Fitted X-Learner uplift model (all arms, CatBoost base learners).
    attrition_model         : AttritionModel
        Fitted binary retention classifier (CatBoostClassifier).
    balance_feature_names   : list[str], optional
        Feature columns that feed the X-Learner.
        Falls back to ``boruta_balance.selected_features``.
    attrition_feature_names : list[str], optional
        Feature columns that feed the AttritionModel.
        Falls back to ``boruta_attrition.selected_features``.
    save_dir                : str, optional
        Target directory.  Defaults to ``config.MODELS_DIR``.

    Returns
    -------
    str  — absolute path of the directory where artefacts were saved.
    """
    save_dir = save_dir or config.MODELS_DIR
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "="*70)
    print("STEP 5: SAVING MODEL PACKAGE  (for handoff / deployment)")
    print("="*70)

    # ── 1. Feature selection objects ───────────────────────────────
    pruner_path = os.path.join(save_dir, _PRUNER_FILE)
    joblib.dump(pruner, pruner_path)
    print(f"\n  ✓ Step-1 pruner               → {pruner_path}")

    boruta_bal_path = os.path.join(save_dir, _BORUTA_BALANCE_FILE)
    joblib.dump(boruta_balance, boruta_bal_path)
    print(f"  ✓ Step-2 Boruta (balance)     → {boruta_bal_path}")

    boruta_att_path = os.path.join(save_dir, _BORUTA_ATTRITION_FILE)
    joblib.dump(boruta_attrition, boruta_att_path)
    print(f"  ✓ Step-2 Boruta (attrition)   → {boruta_att_path}")

    # ── 2. Predictive models ───────────────────────────────────────
    xlearner_path = os.path.join(save_dir, _XLEARNER_FILE)
    joblib.dump(xlearner_model, xlearner_path)
    print(f"  ✓ X-Learner uplift            → {xlearner_path}")

    attrition_path = os.path.join(save_dir, _ATTRITION_FILE)
    joblib.dump(attrition_model, attrition_path)
    print(f"  ✓ Attrition model             → {attrition_path}")

    # ── 3. Feature name lists ──────────────────────────────────────
    if balance_feature_names is None:
        balance_feature_names = (
            boruta_balance.selected_features
            if hasattr(boruta_balance, 'selected_features') and boruta_balance.selected_features
            else xlearner_model.feature_names
        )
    bal_feat_path = os.path.join(save_dir, _BALANCE_FEAT_FILE)
    with open(bal_feat_path, 'w') as fh:
        json.dump(balance_feature_names, fh, indent=2)
    print(f"  ✓ Balance features ({len(balance_feature_names)})    → {bal_feat_path}")

    if attrition_feature_names is None:
        attrition_feature_names = (
            boruta_attrition.selected_features
            if hasattr(boruta_attrition, 'selected_features') and boruta_attrition.selected_features
            else attrition_model.feature_names
        )
    att_feat_path = os.path.join(save_dir, _ATTRITION_FEAT_FILE)
    with open(att_feat_path, 'w') as fh:
        json.dump(attrition_feature_names, fh, indent=2)
    print(f"  ✓ Attrition features ({len(attrition_feature_names)})  → {att_feat_path}")

    # Write a legacy alias 'feature_names.json' pointing to balance features
    # so any existing scoring scripts don't break immediately.
    legacy_path = os.path.join(save_dir, _LEGACY_FEATURE_FILE)
    with open(legacy_path, 'w') as fh:
        json.dump(balance_feature_names, fh, indent=2)

    # ── 4. Config snapshot ─────────────────────────────────────────
    cfg_snapshot = {
        'saved_at':                datetime.datetime.now().isoformat(),
        'random_seed':             config.RANDOM_SEED,
        'treatment_components':    {str(k): v for k, v in config.TREATMENT_COMPONENTS.items()},
        'treatments':              {str(k): v for k, v in config.TREATMENTS.items()},
        'offer_amounts':           config.OFFER_AMOUNTS,
        'offer_cost_rate':         config.OFFER_COST_RATE,
        'remail_cost':             config.REMAIL_COST,
        'n_deciles':               config.N_DECILES,
        'top_n_deciles':           config.TOP_N_DECILES,
        'log_transform_target':    config.LOG_TRANSFORM_TARGET,
        'catboost_classifier':     config.CATBOOST_CLASSIFIER_PARAMS,
        'catboost_regressor':      config.CATBOOST_REGRESSOR_PARAMS,
        'catboost_propensity':     config.CATBOOST_PROPENSITY_PARAMS,
        'variance_threshold':      config.VARIANCE_THRESHOLD,
        'correlation_threshold':   config.CORRELATION_THRESHOLD,
        'boruta_n_trials':                config.BORUTA_N_TRIALS,
        'boruta_percentile':              config.BORUTA_PERCENTILE,
        'boruta_balance_top_n':           getattr(config, 'BORUTA_BALANCE_TOP_N', None),
        'boruta_attrition_top_n':         getattr(config, 'BORUTA_ATTRITION_TOP_N', None),
        'boruta_balance_force_include':   getattr(config, 'BORUTA_BALANCE_FORCE_INCLUDE', []),
        'boruta_attrition_force_include': getattr(config, 'BORUTA_ATTRITION_FORCE_INCLUDE', []),
        'n_balance_features':             len(balance_feature_names),
        'n_attrition_features':           len(attrition_feature_names),
    }
    cfg_path = os.path.join(save_dir, _CONFIG_FILE)
    with open(cfg_path, 'w') as fh:
        json.dump(cfg_snapshot, fh, indent=2)
    print(f"  ✓ Config snapshot             → {cfg_path}")

    # ── 5. Human-readable manifest ─────────────────────────────────
    _write_manifest(save_dir, pruner, boruta_balance, boruta_attrition,
                    xlearner_model, attrition_model,
                    balance_feature_names, attrition_feature_names,
                    cfg_snapshot)

    print(f"\n{'='*70}")
    print(f"  Model package saved to  : {save_dir}")
    print(f"  To score new data run   : python src/scoring/score_new_data.py "
          f"--input <prospects.csv> --output <scored.csv>")
    print(f"{'='*70}\n")

    return save_dir


# ------------------------------------------------------------------
# LOAD
# ------------------------------------------------------------------
def load_pipeline(save_dir=None):
    """
    Deserialize all pipeline artefacts from ``save_dir``.

    Parameters
    ----------
    save_dir : str, optional
        Directory created by ``save_pipeline()``.
        Defaults to ``config.MODELS_DIR``.

    Returns
    -------
    dict with keys:
        'pruner'                  : InitialPruning
        'boruta_balance'          : BorutaSHAP  (balance target)
        'boruta_attrition'        : BorutaSHAP  (attrition target)
        'xlearner'                : XLearnerUplift
        'attrition'               : AttritionModel
        'balance_feature_names'   : list[str]
        'attrition_feature_names' : list[str]
        'config'                  : dict  (pipeline_config.json snapshot)
    """
    save_dir = save_dir or config.MODELS_DIR

    print("\n" + "="*70)
    print("LOADING SAVED MODEL PACKAGE")
    print("="*70)
    print(f"\n  Package directory: {save_dir}\n")

    def _load(filename, label):
        path = os.path.join(save_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Artefact not found: {path}\n"
                f"Run pipeline.py first to train and save the models."
            )
        obj = joblib.load(path)
        print(f"  ✓ Loaded {label:<35} ← {path}")
        return obj

    pruner           = _load(_PRUNER_FILE,           'Step-1 pruner')
    boruta_balance   = _load(_BORUTA_BALANCE_FILE,   'Step-2 Boruta (balance)')
    boruta_attrition = _load(_BORUTA_ATTRITION_FILE, 'Step-2 Boruta (attrition)')
    xlearner         = _load(_XLEARNER_FILE,         'X-Learner uplift')
    attrition        = _load(_ATTRITION_FILE,        'Attrition model')

    bal_feat_path = os.path.join(save_dir, _BALANCE_FEAT_FILE)
    with open(bal_feat_path) as fh:
        balance_feature_names = json.load(fh)
    print(f"  ✓ Balance features ({len(balance_feature_names)} cols)")

    att_feat_path = os.path.join(save_dir, _ATTRITION_FEAT_FILE)
    with open(att_feat_path) as fh:
        attrition_feature_names = json.load(fh)
    print(f"  ✓ Attrition features ({len(attrition_feature_names)} cols)")

    cfg_path = os.path.join(save_dir, _CONFIG_FILE)
    with open(cfg_path) as fh:
        cfg_snapshot = json.load(fh)
    print(f"  ✓ Config snapshot (saved {cfg_snapshot.get('saved_at', 'unknown')})")

    print(f"\n{'='*70}\n")

    return {
        'pruner':                  pruner,
        'boruta_balance':          boruta_balance,
        'boruta_attrition':        boruta_attrition,
        'xlearner':                xlearner,
        'attrition':               attrition,
        'balance_feature_names':   balance_feature_names,
        'attrition_feature_names': attrition_feature_names,
        'config':                  cfg_snapshot,
    }


# ------------------------------------------------------------------
# Internal helper
# ------------------------------------------------------------------
def _write_manifest(save_dir, pruner, boruta_balance, boruta_attrition,
                    xlearner_model, attrition_model,
                    balance_feature_names, attrition_feature_names, cfg):
    """Write a human-readable MANIFEST.txt to save_dir."""
    lines = [
        "=" * 70,
        "UPLIFT MODELING PIPELINE — MODEL PACKAGE MANIFEST",
        "=" * 70,
        f"Generated  : {cfg['saved_at']}",
        f"Directory  : {save_dir}",
        "",
        "─" * 70,
        "DUAL FEATURE SELECTION PATHS",
        "─" * 70,
        "  The pipeline runs two independent Boruta-SHAP selections after the",
        "  shared Step-1 variance/correlation pruning:",
        "",
        "  Balance path  → features selected against opening_balance",
        f"                  {len(balance_feature_names)} features → X-Learner (CATE models + propensity)",
        "",
        "  Attrition path → features selected against on_book_month9",
        f"                  {len(attrition_feature_names)} features → AttritionModel (retention prob)",
        "",
        "─" * 70,
        "ARTEFACTS",
        "─" * 70,
    ]

    n_bal_accepted  = (len(boruta_balance.selected_features)
                       if hasattr(boruta_balance, 'selected_features')
                       and boruta_balance.selected_features else len(balance_feature_names))
    n_att_accepted  = (len(boruta_attrition.selected_features)
                       if hasattr(boruta_attrition, 'selected_features')
                       and boruta_attrition.selected_features else len(attrition_feature_names))

    artefacts = [
        (_PRUNER_FILE,
         'Step-1 Pruner',
         f"Variance threshold={cfg['variance_threshold']}, "
         f"Correlation threshold={cfg['correlation_threshold']}",
         f"Removed (variance): {len(pruner.removed_variance)}, "
         f"(correlation): {len(pruner.removed_correlation)}"),

        (_BORUTA_BALANCE_FILE,
         'Step-2 Boruta (Balance target)',
         f"n_trials={cfg['boruta_n_trials']}, percentile={cfg['boruta_percentile']}",
         f"Features selected: {n_bal_accepted}  → {_BALANCE_FEAT_FILE}"),

        (_BORUTA_ATTRITION_FILE,
         'Step-2 Boruta (Attrition target)',
         f"n_trials={cfg['boruta_n_trials']}, percentile={cfg['boruta_percentile']}",
         f"Features selected: {n_att_accepted}  → {_ATTRITION_FEAT_FILE}"),

        (_XLEARNER_FILE,
         'X-Learner Uplift (CatBoostRegressor)',
         f"Arms: {list(xlearner_model.models.keys())}",
         f"Balance feature set: {len(balance_feature_names)} features | "
         f"Propensity: CatBoostClassifier"),

        (_ATTRITION_FILE,
         'Attrition Model (CatBoostClassifier)',
         'Predicts P(on-book at month 9)',
         f"Attrition feature set: {len(attrition_feature_names)} features"),

        (_BALANCE_FEAT_FILE,
         'Balance Feature Name List',
         f'{len(balance_feature_names)} features for X-Learner', ''),

        (_ATTRITION_FEAT_FILE,
         'Attrition Feature Name List',
         f'{len(attrition_feature_names)} features for AttritionModel', ''),

        (_CONFIG_FILE,
         'Config Snapshot',
         'Cost params, arm map, decile settings at training time', ''),
    ]

    for fname, label, desc, detail in artefacts:
        lines.append(f"\n  {label}")
        lines.append(f"    File    : {fname}")
        lines.append(f"    Details : {desc}")
        if detail:
            lines.append(f"            : {detail}")

    lines += [
        "",
        "─" * 70,
        "TREATMENT / OFFER DESIGN",
        "─" * 70,
    ]
    for arm_id, offer in cfg['treatment_components'].items():
        label = cfg['treatments'][arm_id]
        lines.append(f"  Arm {arm_id}: {label:<10}  offer=${offer}")

    lines += [
        "",
        "─" * 70,
        "COST PARAMETERS",
        "─" * 70,
        f"  Offer cost rate  : {cfg['offer_cost_rate']} (× offer $)",
        f"  Remail cost      : ${cfg['remail_cost']} per prospect",
        "  Stipulation cost : $0 (zero)",
        "",
        "─" * 70,
        "DECILE TARGETING",
        "─" * 70,
        f"  Total deciles    : {cfg['n_deciles']}",
        f"  Top deciles mailed: {cfg['top_n_deciles']}  "
        f"(~{100 * cfg['top_n_deciles'] / cfg['n_deciles']:.0f}% of population)",
        "",
        "─" * 70,
        "HOW TO SCORE NEW DATA",
        "─" * 70,
        "  python src/scoring/score_new_data.py \\",
        "      --input  path/to/new_prospects.csv \\",
        "      --output path/to/scored_prospects.csv",
        "",
        "  Output columns:",
        "    cate_treatment_1/2/3  — CATE (uplift) per offer arm",
        "    retention_probability — P(on-book at month 9)",
        "    net_value_arm_0/1/2/3 — net value per arm",
        "    optimal_offer_arm     — best arm ID",
        "    optimal_offer_name    — e.g. '$400'",
        "    optimal_net_value     — net value of best arm",
        "    decile                — 1 (lowest) to 10 (highest)",
        "    mail_flag             — 1 = top-3 decile → send letter",
        "=" * 70,
    ]

    manifest_path = os.path.join(save_dir, _MANIFEST_FILE)
    with open(manifest_path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f"  ✓ Manifest                    → {manifest_path}")
