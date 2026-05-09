import os
import torch

CFG = dict(
    DATA_DIR="data",
    OUTPUT_DIR="results_part2",
    N_BANDS=10,
    N_TIMESTEPS=36,
    N_PER_CLASS_TV=300,
    TRAIN_RATIO=0.8,
    N_STAGES=3,
    N_HEAD=5,
    CNN_KERNEL=3,
    D_MODEL=10,
    MLP_HIDDEN=128,
    USE_COVARIATES=True,
    COVARIATE_SUBSET="all",  
    TOPO_FEATURES=["topo_elevation", "topo_landforms"],
    N_TOPO=2,
    CLIM_FEATURES=["clim_temp_mean", "clim_precip_annual", "clim_srad_annual"],
    N_CLIM=3,
    SOIL_FEATURES=["soil_ph", "soil_oc", "soil_texture"],
    N_SOIL=3,
    N_COVARIATES=8,  
    COV_MLP_HIDDEN=32,
    BATCH_SIZE=32,
    N_EPOCHS=200,
    LR=0.001,
    WEIGHT_DECAY=0.0001,
    SEED=42,
    EARLY_STOPPING=True,
    PATIENCE=20,
    MIN_DELTA=1e-4,
    LR_FACTOR=0.7,
    LR_MIN=1e-6,
    LR_PATIENCE=5,
    
    AREAS={
        "Arkansas": ["arkansas_unified_2021.csv"],
        "California": ["california_unified_2021.csv"],
    },
    

    CLASS_NAMES={
        "Arkansas": {1: "Corn", 2: "Cotton", 3: "Rice", 4: "Soybeans", 5: "Others"},
        "California": {1: "Grapes", 2: "Rice", 3: "Alfalfa", 4: "Almonds", 5: "Pistachios", 6: "Others"},
    },
    
    N_CLASSES={
        "Arkansas": 5,
        "California": 6,
    },
    
    ABLATION_CONFIGS=[
        {"name": "S2_only", "covariates": "none", "description": "Sentinel-2 only"},
        {"name": "S2_climate", "covariates": "clim", "description": "Sentinel-2 + Climate variables"},
        {"name": "S2_soil", "covariates": "soil", "description": "Sentinel-2 + Soil variables"},
        {"name": "S2_topo", "covariates": "topo", "description": "Sentinel-2 + Topography"},
        {"name": "S2_all", "covariates": "all", "description": "Sentinel-2 + All covariates"},
    ],
)


def get_covariate_features(cfg, subset=None):
    if subset is None:
        subset = cfg.get("COVARIATE_SUBSET", "all")
    
    if subset == "none":
        return [], 0
    elif subset == "topo":
        return cfg["TOPO_FEATURES"], cfg["N_TOPO"]
    elif subset == "clim":
        return cfg["CLIM_FEATURES"], cfg["N_CLIM"]
    elif subset == "soil":
        return cfg["SOIL_FEATURES"], cfg["N_SOIL"]
    elif subset == "all":
        all_features = cfg["TOPO_FEATURES"] + cfg["CLIM_FEATURES"] + cfg["SOIL_FEATURES"]
        return all_features, cfg["N_TOPO"] + cfg["N_CLIM"] + cfg["N_SOIL"]
    else:
        raise ValueError(f"Unknown COVARIATE_SUBSET: {subset}")



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    import numpy as np
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)