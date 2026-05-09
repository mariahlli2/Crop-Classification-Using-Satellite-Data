
import json
import os
from pathlib import Path

from config import CFG, ensure_output_dir
from pipeline import run_experiment
from visualization import plot_ablation_comparison


def run_ablation_study(state: str, cfg: dict, out_dir: str):
    print(f"\n{'='*70}")
    print(f"  ABLATION STUDY: {state}")
    print(f"{'='*70}")
    
    results = {}
    
    for config in cfg["ABLATION_CONFIGS"]:
        name = config["name"]
        cov_subset = config["covariates"]
        desc = config["description"]
        
        print(f"\n{'-'*70}")
        print(f"  Running: {desc}")
        print(f"{'-'*70}")
        
        try:
            metrics = run_experiment(
                state=state,
                cfg=cfg,
                out_dir=out_dir,
                config_name=name,
                covariate_subset=cov_subset
            )
            results[name] = metrics
            
        except Exception as e:
            print(f"  ERROR in {name}: {str(e)}")
            results[name] = {"error": str(e)}
    
  
    ablation_dir = os.path.join(out_dir, "ablation_summary")
    ensure_output_dir(ablation_dir)
    
    summary_file = os.path.join(ablation_dir, f"ablation_{state}.json")
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    
    plot_ablation_comparison(results, state, ablation_dir)
    
    print(f"\n{'='*70}")
    print(f"  ABLATION STUDY SUMMARY - {state}")
    print(f"{'='*70}")
    print(f"{'Configuration':<20} {'OA':>8} {'Kappa':>8} {'F1':>8} {'Params':>12}")
    print(f"{'-'*70}")
    
    for name, metrics in results.items():
        if "error" not in metrics:
            print(f"{name:<20} {metrics['OA']:>8.4f} {metrics['Kappa']:>8.4f} "
                  f"{metrics['F1_macro']:>8.4f} {metrics['n_params']:>12,}")
    
    print(f"{'='*70}")
    
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    if valid_results:
        best_oa = max(valid_results.items(), key=lambda x: x[1]['OA'])
        best_kappa = max(valid_results.items(), key=lambda x: x[1]['Kappa'])
        best_f1 = max(valid_results.items(), key=lambda x: x[1]['F1_macro'])
        
        print(f"\n  Best OA:      {best_oa[0]} ({best_oa[1]['OA']:.4f})")
        print(f"  Best Kappa:   {best_kappa[0]} ({best_kappa[1]['Kappa']:.4f})")
        print(f"  Best F1:      {best_f1[0]} ({best_f1[1]['F1_macro']:.4f})")
    
    return results


def run_full_ablation(cfg: dict, out_dir: str):
    all_results = {}
    
    for state in ["Arkansas", "California"]:
        try:
            results = run_ablation_study(state, cfg, out_dir)
            all_results[state] = results
        except Exception as e:
            print(f"\nFailed to process {state}: {e}")
    
    
    summary_file = os.path.join(out_dir, "ablation_summary", "all_results.json")
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("  FULL ABLATION STUDY COMPLETE")
    print(f"{'='*70}")
    print(f"  Results saved to: {out_dir}")
    
    return all_results