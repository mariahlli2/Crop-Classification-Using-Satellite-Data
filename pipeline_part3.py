import json
import os
import time

import torch

from config import CFG, DEVICE, set_seed, ensure_output_dir, get_covariate_features
from dataset import (load_area_csvs, parse_features, normalise,
                     normalise_covariates, stratified_split, CropDataset)
from model_part3    import AttentionCNN
from training_part3 import (train_model, evaluate, print_metrics,
                             compute_metrics_dict)
from visualization  import plot_confusion_matrix, plot_training_history


def run_experiment_part3(state: str, cfg: dict, out_dir: str,
                          config_name: str = "AttentionCNN"):

    print(f"\n{'#'*70}")
    print(f"#  PART 3 — AttentionCNN v3 (+ covariates)  |  {state}")
    print(f"{'#'*70}")

    set_seed(cfg["SEED"])

    
    config_dir  = os.path.join(out_dir, config_name)
    models_dir  = os.path.join(config_dir, "models")
    metrics_dir = os.path.join(config_dir, "metrics")
    viz_dir     = os.path.join(config_dir, "visualizations")
    for d in [models_dir, metrics_dir, viz_dir]:
        ensure_output_dir(d)

    
    print("[1/6] Loading data...")
    df = load_area_csvs(state, cfg["DATA_DIR"])
    print(f"      Raw rows: {len(df)}")

    X, mask, covariates, labels, cov_names = parse_features(df, state, cfg)
    print(f"      Spectral shape  : {X.shape}")
    print(f"      Covariates shape: {covariates.shape}")

    
    use_covariates  = cfg.get("USE_COVARIATES", True)
    cov_subset      = cfg.get("COVARIATE_SUBSET", "all")

    if use_covariates:
        selected_covs, n_cov = get_covariate_features(cfg, cov_subset)
        selected_indices = [i for i, name in enumerate(cov_names)
                            if name in selected_covs]
        if selected_indices:
            covariates = covariates[:, selected_indices]
            print(f"      Using {n_cov} covariates: {selected_covs}")
        else:
            
            n_cov = covariates.shape[1]
            print(f"      WARNING: named covariates not found — "
                  f"using all {n_cov} columns.")
    else:
        n_cov = 0
        print("      Covariates disabled (spectral only).")

    n_classes        = cfg["N_CLASSES"][state]
    class_names_dict = cfg["CLASS_NAMES"][state]
    class_names      = [class_names_dict[i] for i in range(1, n_classes + 1)]

    
    print("[2/6] Normalising...")
    X = normalise(X)
    if use_covariates and n_cov > 0:
        covariates = normalise_covariates(covariates)

    
    print("[3/6] Splitting...")
    (X_tr, m_tr, cov_tr, y_tr), \
    (X_val, m_val, cov_val, y_val), \
    (X_te, m_te, cov_te, y_te) = stratified_split(
        X, mask, covariates, labels,
        n_per_class=cfg["N_PER_CLASS_TV"],
        train_ratio=cfg["TRAIN_RATIO"],
        seed=cfg["SEED"],
    )
    print(f"      Train={len(y_tr)}, Val={len(y_val)}, Test={len(y_te)}")

    
    tr_loader  = _build_loader(X_tr,  m_tr,  cov_tr,  y_tr,
                                cfg["BATCH_SIZE"], shuffle=True,
                                use_covariates=use_covariates)
    val_loader = _build_loader(X_val, m_val, cov_val, y_val, 512,
                               use_covariates=use_covariates)
    te_loader  = _build_loader(X_te,  m_te,  cov_te,  y_te,  512,
                               use_covariates=use_covariates)

    
    print("[4/6] Building AttentionCNN v3 (spectral + covariate fusion)...")
    model = AttentionCNN(
        n_classes       = n_classes,
        n_bands         = cfg["N_BANDS"],
        n_timesteps     = cfg["N_TIMESTEPS"],
        n_head          = cfg.get("PART3_N_HEAD",          4),
        cnn_kernel      = cfg.get("CNN_KERNEL",            3),
        mlp_hidden      = cfg.get("MLP_HIDDEN",           256),
        d_model         = cfg.get("PART3_D_MODEL",        64),
        cbam_reduction  = cfg.get("PART3_CBAM_REDUCTION",  8),
        drop_path_rate  = cfg.get("PART3_DROP_PATH",    0.0),
        aug_mask_ratio  = cfg.get("PART3_AUG_MASK",     0.0),
        attn_drop       = cfg.get("PART3_ATTN_DROP",    0.05),
        n_covariates    = n_cov if use_covariates else 0,
        cov_mlp_hidden  = cfg.get("PART3_COV_HIDDEN",    48),
        cov_out_dim     = cfg.get("PART3_COV_OUT",       32),
        use_covariates  = use_covariates and n_cov > 0,
    ).to(DEVICE)

    n_params = model.get_n_params()
    print(f"      Parameters  : {n_params:,}")
    print(f"      Device      : {DEVICE}")
    print(f"      Covariates  : {'YES (' + str(n_cov) + ' features)' if use_covariates and n_cov > 0 else 'NO'}")

    
    print("[5/6] Training (CosineAnnealing + Weight Decay)...")
    t0 = time.time()
    model, history = train_model(
        model, tr_loader, val_loader, cfg, DEVICE,
        use_covariates=use_covariates and n_cov > 0,
    )
    train_time = time.time() - t0
    print(f"      Training time: {train_time:.1f}s")

    ckpt = os.path.join(models_dir, f"attentioncnn_{state}.pt")
    torch.save(model.state_dict(), ckpt)


    print("[6/6] Evaluating on test set...")
    t1 = time.time()
    oa, kappa, f1, f1_w, preds, true_labels, _ = evaluate(
        model, te_loader, device=DEVICE,
        use_covariates=use_covariates and n_cov > 0,
    )
    inf_time = time.time() - t1

    print_metrics(oa, kappa, f1, f1_w, preds, true_labels, class_names, state)


    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, preds, labels=list(range(n_classes)))
    plot_confusion_matrix(cm, class_names, state, viz_dir, config_name)
    plot_training_history(history, state, viz_dir, config_name)

    metrics = compute_metrics_dict(oa, kappa, f1, f1_w,
                                   n_params, train_time, inf_time)
    metrics.update({
        "state":           state,
        "config":          config_name,
        "model":           "AttentionCNN_v3",
        "covariates_used": cov_subset if use_covariates else "none",
        "n_covariates":    n_cov,
    })

    out_json = os.path.join(metrics_dir, f"metrics_{state}.json")
    with open(out_json, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"      Metrics saved → {out_json}")

    return metrics


def _build_loader(X, mask, covariates, labels, batch_size,
                  shuffle=False, use_covariates=True):
    ds = CropDataset(X, mask, covariates, labels, use_covariates=use_covariates)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=0, pin_memory=False,
    )