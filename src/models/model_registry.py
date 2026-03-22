"""
Model Registry — Save & Load Pipeline Artefacts for Handoff / Deployment

Saves the complete trained pipeline to ``config.MODELS_DIR`` so that scoring
new prospect files requires no re-training.

Artefacts written
-----------------
  step1_pruner.joblib       — InitialPruning  (variance + correlation filter)
  step2_boruta.joblib       — BorutaSHAP      (Boruta-SHAP feature selection)
  xlearner_uplift.joblib    — XLearnerUplift  (CATE models, 3 arms)
  attrition_model.joblib    — AttritionModel  (XGBClassifier wrapper)
  feature_names.json        — ordered list of features that survive Steps 1+2
  pipeline_config.json      — cost params, arm map, decile settings (snapshot)
  MANIFEST.txt              — human-readable summary of saved package

Usage
-----
  # At end of training pipeline:
  from src.models.model_registry import save_pipeline
  save_pipeline(pruner, boruta, xlearner_model, attrition_model)

  # At inference time:
  from src.models.model_registry import load_pipeline
  pkg = load_pipeline()
  # pkg keys: 'pruner', 'boruta', 'xlearner', 'attrition', 'feature_names', 'config'
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
_PRUNER_FILE       = 'step1_pruner.joblib'
_BORUTA_FILE       = 'step2_boruta.joblib'
_XLEARNER_FILE     = 'xlearner_uplift.joblib'
_ATTRITION_FILE    = 'attrition_model.joblib'
_FEATURE_FILE      = 'feature_names.json'
_CONFIG_FILE       = 'pipeline_config.json'
_MANIFEST_FILE     = 'MANIFEST.txt'


# ------------------------------------------------------------------
# SAVE
# ------------------------------------------------------------------
def save_pipeline(pruner, boruta, xlearner_model, attrition_model,
                  save_dir=None, feature_names=None):
    """
    Serialize all pipeline artefacts to ``save_dir`` (default: config.MODELS_DIR).

    Parameters
    ----------
    pruner : InitialPruning
        Fitted Step-1 pruner (variance + correlation filter).
    boruta : BorutaSHAP
        Fitted Step-2 Boruta-SHAP selector.
    xlearner_model : XLearnerUplift
        Fitted X-Learner uplift model (all arms).
    attrition_model : AttritionModel
        Fitted binary retention classifier.
    save_dir : str, optional
        Target directory.  Defaults to ``config.MODELS_DIR``.
    feature_names : list[str], optional
        Column names that survive Steps 1 & 2.  If None, read from
        ``boruta.selected_features``.

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
    print(f"\n  ✓ Step-1 pruner         → {pruner_path}")

    boruta_path = os.path.join(save_dir, _BORUTA_FILE)
    joblib.dump(boruta, boruta_path)
    print(f"  ✓ Step-2 Boruta-SHAP    → {boruta_path}")

    # ── 2. Predictive models ───────────────────────────────────────
    xlearner_path = os.path.join(save_dir, _XLEARNER_FILE)
    joblib.dump(xlearner_model, xlearner_path)
    print(f"  ✓ X-Learner uplift      → {xlearner_path}")

    attrition_path = os.path.join(save_dir, _ATTRITION_FILE)
    joblib.dump(attrition_model, attrition_path)
    print(f"  ✓ Attrition model       → {attrition_path}")

    # ── 3. Feature name list ───────────────────────────────────────
    if feature_names is None:
        feature_names = (
            boruta.selected_features
            if hasattr(boruta, 'selected_features') and boruta.selected_features
            else xlearner_model.feature_names
        )
    feat_path = os.path.join(save_dir, _FEATURE_FILE)
    with open(feat_path, 'w') as fh:
        json.dump(feature_names, fh, indent=2)
    print(f"  ✓ Feature names ({len(feature_names)})   → {feat_path}")

    # ── 4. Config snapshot ─────────────────────────────────────────
    cfg_snapshot = {
        'saved_at':              datetime.datetime.now().isoformat(),
        'random_seed':           config.RANDOM_SEED,
        # Arm / offer definitions
        'treatment_components':  {str(k): v for k, v in config.TREATMENT_COMPONENTS.items()},
        'treatments':            {str(k): v for k, v in config.TREATMENTS.items()},
        'offer_amounts':         config.OFFER_AMOUNTS,
        # Cost parameters
        'offer_cost_rate':       config.OFFER_COST_RATE,
        'stipulation_cost':      config.STIPULATION_COST,
        'remail_cost':           config.REMAIL_COST,
        # Decile targeting
        'n_deciles':             config.N_DECILES,
        'top_n_deciles':         config.TOP_N_DECILES,
        # Model meta
        'log_transform_target':  config.LOG_TRANSFORM_TARGET,
        'xgboost_params':        config.XGBOOST_PARAMS,
        # Feature selection thresholds
        'variance_threshold':    config.VARIANCE_THRESHOLD,
        'correlation_threshold': config.CORRELATION_THRESHOLD,
        'boruta_n_trials':       config.BORUTA_N_TRIALS,
        'boruta_percentile':     config.BORUTA_PERCENTILE,
    }
    cfg_path = os.path.join(save_dir, _CONFIG_FILE)
    with open(cfg_path, 'w') as fh:
        json.dump(cfg_snapshot, fh, indent=2)
    print(f"  ✓ Config snapshot       → {cfg_path}")

    # ── 5. Human-readable manifest ─────────────────────────────────
    _write_manifest(save_dir, pruner, boruta, xlearner_model,
                    attrition_model, feature_names, cfg_snapshot)

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
        'pruner'        : InitialPruning
        'boruta'        : BorutaSHAP
        'xlearner'      : XLearnerUplift
        'attrition'     : AttritionModel
        'feature_names' : list[str]
        'config'        : dict  (pipeline_config.json snapshot)
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
        print(f"  ✓ Loaded {label:<25} ← {path}")
        return obj

    pruner        = _load(_PRUNER_FILE,    'Step-1 pruner')
    boruta        = _load(_BORUTA_FILE,    'Step-2 Boruta-SHAP')
    xlearner      = _load(_XLEARNER_FILE,  'X-Learner uplift')
    attrition     = _load(_ATTRITION_FILE, 'Attrition model')

    feat_path = os.path.join(save_dir, _FEATURE_FILE)
    with open(feat_path) as fh:
        feature_names = json.load(fh)
    print(f"  ✓ Feature names ({len(feature_names)} cols)")

    cfg_path = os.path.join(save_dir, _CONFIG_FILE)
    with open(cfg_path) as fh:
        cfg_snapshot = json.load(fh)
    print(f"  ✓ Config snapshot (saved {cfg_snapshot.get('saved_at', 'unknown')})")

    print(f"\n{'='*70}\n")

    return {
        'pruner':        pruner,
        'boruta':        boruta,
        'xlearner':      xlearner,
        'attrition':     attrition,
        'feature_names': feature_names,
        'config':        cfg_snapshot,
    }


# ------------------------------------------------------------------
# Internal helper
# ------------------------------------------------------------------
def _write_manifest(save_dir, pruner, boruta, xlearner_model,
                    attrition_model, feature_names, cfg):
    """Write a human-readable MANIFEST.txt to save_dir."""
    lines = [
        "=" * 70,
        "UPLIFT MODELING PIPELINE — MODEL PACKAGE MANIFEST",
        "=" * 70,
        f"Generated  : {cfg['saved_at']}",
        f"Directory  : {save_dir}",
        "",
        "─" * 70,
        "ARTEFACTS",
        "─" * 70,
    ]

    artefacts = [
        (_PRUNER_FILE,    'Step-1 Pruner',
         f"Variance threshold={cfg['variance_threshold']}, "
         f"Correlation threshold={cfg['correlation_threshold']}",
         f"Features removed (variance): {len(pruner.removed_variance)}, "
         f"(correlation): {len(pruner.removed_correlation)}"),

        (_BORUTA_FILE,    'Step-2 Boruta-SHAP',
         f"n_trials={cfg['boruta_n_trials']}, "
         f"percentile={cfg['boruta_percentile']}",
         f"Features selected: {len(boruta.selected_features) if hasattr(boruta, 'selected_features') and boruta.selected_features else 'see feature_names.json'}"),

        (_XLEARNER_FILE,  'X-Learner Uplift',
         f"Arms: {list(xlearner_model.models.keys())}",
         f"Feature set size: {len(feature_names)}"),

        (_ATTRITION_FILE, 'Attrition Model',
         'XGBClassifier — predicts P(on-book at month 9)',
         f"Feature set size: {len(attrition_model.feature_names)}"),

        (_FEATURE_FILE,   'Feature Name List',
         f'{len(feature_names)} features survive Steps 1+2', ''),

        (_CONFIG_FILE,    'Config Snapshot',
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
        f"  Stipulation cost : ${cfg['stipulation_cost']} per prospect",
        f"  Remail cost      : ${cfg['remail_cost']} per prospect",
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
    print(f"  ✓ Manifest              → {manifest_path}")
