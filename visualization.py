
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def plot_confusion_matrix(cm, class_names, state, out_dir, config_name=""):
    
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=1, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    title = f"Confusion Matrix - {state}"
    if config_name:
        title += f" ({config_name})"
    ax.set_title(title)
    plt.tight_layout()
    
    suffix = f"_{config_name}" if config_name else ""
    fpath = os.path.join(out_dir, f"confusion_{state}{suffix}.png")
    plt.savefig(fpath, dpi=150)
    plt.close()


def plot_training_history(history, state, out_dir, config_name=""):
   
    epochs = range(1, len(history["val_oa"]) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    axes[0].plot(epochs, history["train_loss"], label="Train", color="tab:blue")
    axes[0].plot(epochs, history["val_loss"], label="Val", color="tab:orange")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=.3)

    axes[1].plot(epochs, history["train_acc"], label="Train OA", color="tab:blue")
    axes[1].plot(epochs, history["val_oa"], label="Val OA", color="tab:orange")
    axes[1].set_title("Overall Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=.3)

    title = f"Training History - {state}"
    if config_name:
        title += f" ({config_name})"
    fig.suptitle(title)
    plt.tight_layout()
    
    suffix = f"_{config_name}" if config_name else ""
    fpath = os.path.join(out_dir, f"training_{state}{suffix}.png")
    plt.savefig(fpath, dpi=150)
    plt.close()


def plot_ablation_comparison(results, state, out_dir):
    configs = list(results.keys())
    oas = [results[c]["OA"] for c in configs]
    
    x = np.arange(len(configs))
    width = 0.5
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, oas, width, label='OA', color='steelblue')
    
    ax.set_ylabel('Overall Accuracy')
    ax.set_title(f'Ablation Study OA Comparison - {state}')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    min_oa = min(oas)
    max_oa = max(oas)
    padding = max(0.001, (max_oa - min_oa) * 0.1)
    ax.set_ylim(max(0, min_oa - padding), min(1, max_oa + padding))
    for i, oa in enumerate(oas):
        ax.text(i, oa + padding / 2, f'{oa:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    fpath = os.path.join(out_dir, f"ablation_comparison_{state}.png")
    plt.savefig(fpath, dpi=150)
    plt.close()


def plot_covariate_correlation(covariates, labels, cov_names, class_names, state, out_dir):

    import pandas as pd
    
    
    df = pd.DataFrame(covariates, columns=cov_names)
    df['label'] = labels
    

    corr = df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, vmin=-1, vmax=1)
    plt.title(f'Covariate Correlation Matrix - {state}')
    plt.tight_layout()
    fpath = os.path.join(out_dir, f"covariate_correlation_{state}.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    

    n_cov = len(cov_names)
    n_cols = 3
    n_rows = (n_cov + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_cov > 1 else [axes]
    
    for i, cov in enumerate(cov_names):
        ax = axes[i]
        data = [df[df['label'] == cls][cov].values for cls in range(len(class_names))]
        bp = ax.boxplot(data, labels=class_names, patch_artist=True)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title(cov)
        ax.set_ylabel('Normalized Value')
        ax.tick_params(axis='x', rotation=45)
    
    
    for i in range(n_cov, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Covariate Distribution by Class - {state}')
    plt.tight_layout()
    fpath = os.path.join(out_dir, f"covariate_boxplots_{state}.png")
    plt.savefig(fpath, dpi=150)
    plt.close()


def plot_feature_importance(importance_scores, feature_names, state, out_dir):
    
    sorted_idx = np.argsort(importance_scores)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance_scores)), importance_scores[sorted_idx], color='steelblue')
    plt.yticks(range(len(importance_scores)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Importance Score')
    plt.title(f'Feature Importance - {state}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    fpath = os.path.join(out_dir, f"feature_importance_{state}.png")
    plt.savefig(fpath, dpi=150)
    plt.close()


def plot_ndvi_profiles(ndvi: np.ndarray, labels: np.ndarray, class_names, state: str, out_dir: str):
    colors_by_state = {
        "Arkansas": {
            "Corn": "tab:blue",
            "Cotton": "tab:orange",
            "Rice": "tab:green",
            "Soybeans": "tab:red",
            "Others": "tab:purple",
        },
        "California": {
            "Grapes": "tab:blue",
            "Alfalfa": "tab:orange",
            "Rice": "tab:green",
            "Almonds": "tab:red",
            "Pistachios": "tab:purple",
            "Others": "saddlebrown",
        },
    }
    color_map = colors_by_state.get(state, {})

    doy = np.arange(ndvi.shape[1]) * 10 + 5
    plt.figure(figsize=(10, 5))
    for cls_id, name in enumerate(class_names):
        idx = (labels == cls_id)
        if idx.sum() == 0:
            continue
        with np.errstate(invalid='ignore'):
            mean_ndvi = np.nanmean(ndvi[idx], axis=0)
        valid = ~np.isnan(mean_ndvi)
        if not np.any(valid):
            continue
        color = color_map.get(name, None)
        plt.plot(doy[valid], mean_ndvi[valid], marker="o", markersize=4,
                 linewidth=1.75, label=name, color=color)
    plt.xlabel("Day of Year")
    plt.ylabel("Mean NDVI Value")
    plt.title(f"NDVI Time-Series Profiles - {state}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fpath = os.path.join(out_dir, f"ndvi_profiles_{state}.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  [EDA] Saved: {fpath}")