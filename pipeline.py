import json
import os
import time
import numpy as np
import torch

from config import CFG, DEVICE, set_seed, ensure_output_dir, get_covariate_features
from dataset import (load_area_csvs, parse_features, normalise, normalise_covariates,
                     stratified_split, CropDataset)
from model import MCTNet
from visualization import plot_confusion_matrix, plot_training_history, plot_ndvi_profiles
from training import train_model, evaluate, print_metrics, compute_metrics_dict


def compute_ndvi(X: np.ndarray) -> np.ndarray:
    
    red = X[:, :, 2] 
    nir = X[:, :, 6]  
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir - red) / (nir + red + 1e-8)
    
    
    ndvi = np.where(np.isfinite(ndvi), ndvi, np.nan)
    
    return ndvi


def run_experiment(state: str, cfg: dict, out_dir: str, config_name: str = "",
                   covariate_subset: str = "all"):
    
    print(f"\n{'#'*70}")
    print(f"#  {state} - Configuration: {config_name or covariate_subset}")
    print(f"{'#'*70}")

    set_seed(cfg["SEED"])
    

    config_dir = os.path.join(out_dir, config_name) if config_name else out_dir
    models_dir = os.path.join(config_dir, "models")
    metrics_dir = os.path.join(config_dir, "metrics")
    viz_dir = os.path.join(config_dir, "visualizations")
    for d in [models_dir, metrics_dir, viz_dir]:
        ensure_output_dir(d)

    print("[1/6] Loading data...")
    df = load_area_csvs(state, cfg["DATA_DIR"])
    print(f"      Raw rows: {len(df)}")

   
    X, mask, covariates, labels, cov_names = parse_features(df, state, cfg)
    print(f"      Spectral shape: {X.shape}")
    print(f"      Covariates shape: {covariates.shape}")
    print(f"      Covariates: {cov_names}")
    
 
    use_covariates = covariate_subset != "none"
    if use_covariates:
        selected_covs, n_cov = get_covariate_features(cfg, covariate_subset)
        
        selected_indices = [i for i, name in enumerate(cov_names) if name in selected_covs]
        covariates = covariates[:, selected_indices]
        print(f"      Using {n_cov} covariates: {selected_covs}")
    else:
        n_cov = 0
        print("      Using spectral data only (no covariates)")

    n_classes = cfg["N_CLASSES"][state]
    class_names_dict = cfg["CLASS_NAMES"][state]
    class_names = [class_names_dict[i] for i in range(1, n_classes + 1)]

    
    print("[2/6] Data preparation...")
    

    
    print("[3/6] Normalizing...")
    X = normalise(X)
    if use_covariates:
        covariates = normalise_covariates(covariates)


    print("[4/6] Splitting data...")
    (X_tr, m_tr, cov_tr, y_tr), (X_val, m_val, cov_val, y_val), (X_te, m_te, cov_te, y_te) = \
        stratified_split(X, mask, covariates, labels,
                        n_per_class=cfg["N_PER_CLASS_TV"],
                        train_ratio=cfg["TRAIN_RATIO"],
                        seed=cfg["SEED"])
    
    print(f"      Train={len(y_tr)}, Val={len(y_val)}, Test={len(y_te)}")


    tr_loader = _build_loader(X_tr, m_tr, cov_tr, y_tr, cfg["BATCH_SIZE"], 
                              shuffle=True, use_covariates=use_covariates)
    val_loader = _build_loader(X_val, m_val, cov_val, y_val, 512, 
                               shuffle=False, use_covariates=use_covariates)
    te_loader = _build_loader(X_te, m_te, cov_te, y_te, 512, 
                              shuffle=False, use_covariates=use_covariates)


    print("[5/6] Building MCTNet...")
    model = MCTNet(
        n_classes=n_classes,
        n_bands=cfg["N_BANDS"],
        n_timesteps=cfg["N_TIMESTEPS"],
        n_stages=cfg["N_STAGES"],
        n_head=cfg["N_HEAD"],
        cnn_kernel=cfg["CNN_KERNEL"],
        mlp_hidden=cfg["MLP_HIDDEN"],
        n_covariates=n_cov,
        cov_mlp_hidden=cfg["COV_MLP_HIDDEN"],
        use_covariates=use_covariates,
    ).to(DEVICE)

    n_params = model.get_n_params()
    print(f"      Parameters: {n_params:,}")
    print(f"      Device: {DEVICE}")

   
    print("      Training...")
    t0 = time.time()
    model, history = train_model(model, tr_loader, val_loader, cfg, DEVICE, use_covariates)
    train_time = time.time() - t0
    print(f"      Training time: {train_time:.1f}s")

  
    ckpt = os.path.join(models_dir, f"mctnet_{state}.pt")
    torch.save(model.state_dict(), ckpt)

    print("[6/6] Evaluating...")
    t1 = time.time()
    oa, kappa, f1, f1_w, preds, true_labels, _ = evaluate(model, te_loader, device=DEVICE, 
                                                            use_covariates=use_covariates)
    inf_time = time.time() - t1

    print_metrics(oa, kappa, f1, f1_w, preds, true_labels, class_names, state)

   
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, preds, labels=list(range(n_classes)))
    plot_confusion_matrix(cm, class_names, state, viz_dir, config_name)
    plot_training_history(history, state, viz_dir, config_name)
    
   
    if config_name == "S2_only":
        print("      Computing and plotting NDVI profiles...")
        ndvi = compute_ndvi(X_te)
        plot_ndvi_profiles(ndvi, y_te, class_names, state, viz_dir)

  
    metrics = compute_metrics_dict(oa, kappa, f1, f1_w, n_params, train_time, inf_time)
    metrics["state"] = state
    metrics["config"] = config_name or covariate_subset
    metrics["covariates_used"] = covariate_subset
    
    out_json = os.path.join(metrics_dir, f"metrics_{state}.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"      Metrics saved: {out_json}")
    
    return metrics


def _build_loader(X, mask, covariates, labels, batch_size, shuffle=False, use_covariates=True):
    
    ds = CropDataset(X, mask, covariates, labels, use_covariates=use_covariates)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=0, pin_memory=False
    )