import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import CFG, DEVICE, set_seed, ensure_output_dir
from ablation_study import run_ablation_study, run_full_ablation





def main():
    parser = argparse.ArgumentParser(description='MCTNet Part 2: Covariate Integration')
    parser.add_argument('--mode', choices=['full', 'ablation', 'single'], 
                       default='full', help='Execution mode')
    parser.add_argument('--state', choices=['Arkansas', 'California', 'both'],
                       default='both', help='State to process')
    parser.add_argument('--config', choices=['none', 'topo', 'clim', 'soil', 'all'],
                       default='all', help='Covariate configuration for single mode')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--output-dir', default='results_part2', help='Output directory')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
   
    CFG["DATA_DIR"] = args.data_dir
    CFG["OUTPUT_DIR"] = args.output_dir
    CFG["N_EPOCHS"] = args.epochs
    CFG["SEED"] = args.seed
    
    
    print(f"Device: {DEVICE}")
    print(f"Output: {args.output_dir}")
    

    ensure_output_dir(args.output_dir)
    
   
    set_seed(args.seed)
    

    if args.mode == 'ablation':
      
        states = ["Arkansas", "California"] if args.state == 'both' else [args.state]
        for state in states:
            run_ablation_study(state, CFG, args.output_dir)
            
    elif args.mode == 'single':
    
        from MCTNet_Part2.pipeline import run_experiment
        
        states = ["Arkansas", "California"] if args.state == 'both' else [args.state]
        for state in states:
            run_experiment(state, CFG, args.output_dir, 
                          config_name=f"S2_{args.config}", 
                          covariate_subset=args.config)
    else:
        
        print(f"\n{'='*70}")
        print("  RUNNING FULL PIPELINE")
        print(f"{'='*70}")
        
        print("\n[Phase 1/1] Ablation Study...")
        run_full_ablation(CFG, args.output_dir)
    
    print(f"\n{'='*70}")
    print("  PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()