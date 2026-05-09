import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import CFG, DEVICE, set_seed, ensure_output_dir
from pipeline_part3 import run_experiment_part3


def main():
    parser = argparse.ArgumentParser(
        description="Part 3  AttentionCNN "
    )
    parser.add_argument("--state",
                        choices=["Arkansas", "California", "both"],
                        default="both")
    parser.add_argument("--data-dir",    default="data")
    parser.add_argument("--output-dir",  default="results_part3")
    parser.add_argument("--epochs",      type=int,   default=200)
    parser.add_argument("--seed",        type=int,   default=42)
    # covariates
    parser.add_argument("--cov-subset",
                        choices=["none", "topo", "clim", "soil", "all"],
                        default="all",
                        help="Which covariate group to use (default: all)")
    # model
    parser.add_argument("--d-model",     type=int,   default=32)
    parser.add_argument("--n-head",      type=int,   default=4)
    parser.add_argument("--mlp-hidden",  type=int,   default=96)
    parser.add_argument("--cov-hidden",  type=int,   default=48,
                        help="CovariateMLP hidden dim")
    parser.add_argument("--cov-out",     type=int,   default=32,
                        help="CovariateMLP output dim")
    parser.add_argument("--drop-path",   type=float, default=0.10)
    parser.add_argument("--aug-mask",    type=float, default=0.10)
    parser.add_argument("--attn-drop",   type=float, default=0.10)
    # training
    parser.add_argument("--lr",          type=float, default=2e-3)
    parser.add_argument("--weight-decay",type=float, default=3e-4)
    parser.add_argument("--mixup-alpha", type=float, default=0.10)
    parser.add_argument("--label-smooth",type=float, default=0.05)
    parser.add_argument("--cosine-t0",   type=int,   default=50)
    parser.add_argument("--patience",    type=int,   default=40)
    args = parser.parse_args()

    use_cov = args.cov_subset != "none"

    CFG.update({
        "DATA_DIR":               args.data_dir,
        "OUTPUT_DIR":             args.output_dir,
        "N_EPOCHS":               args.epochs,
        "SEED":                   args.seed,
        "LR":                     args.lr,
        "WEIGHT_DECAY":           args.weight_decay,
        "MIXUP_ALPHA":            args.mixup_alpha,
        "LABEL_SMOOTHING":        args.label_smooth,
        "COSINE_T0":              args.cosine_t0,
        "COSINE_TMULT":           2,
        "PATIENCE":               args.patience,
        "EARLY_STOPPING":         True,
        "MIN_DELTA":              1e-4,
        "GRAD_CLIP":              1.0,
        "USE_COVARIATES":         use_cov,
        "COVARIATE_SUBSET":       args.cov_subset,
        "PART3_D_MODEL":          args.d_model,
        "PART3_N_HEAD":           args.n_head,
        "MLP_HIDDEN":             args.mlp_hidden,
        "PART3_DROP_PATH":        args.drop_path,
        "PART3_AUG_MASK":         args.aug_mask,
        "PART3_ATTN_DROP":        args.attn_drop,
        "PART3_CBAM_REDUCTION":   8,
        "PART3_COV_HIDDEN":       args.cov_hidden,
        "PART3_COV_OUT":          args.cov_out,
    })

    print(f"\n{'='*70}")
    print("  PART 3 — AttentionCNN ")
    print(f"{'='*70}")
    print(f"  Device        : {DEVICE}")
    print(f"  Covariates    : {args.cov_subset}")
    print(f"  d_model       : {args.d_model}  |  n_head: {args.n_head}")
    print(f"  cov_hidden    : {args.cov_hidden}  |  cov_out: {args.cov_out}")
    print(f"  DropPath rate : {args.drop_path}")
    print(f"  AugMask ratio : {args.aug_mask}")
    print(f"  MixUp alpha   : {args.mixup_alpha}")
    print(f"  LabelSmoothing: {args.label_smooth}")
    print(f"  Cosine T0     : {args.cosine_t0}  patience: {args.patience}")
    print(f"  Epochs        : {args.epochs}  LR: {args.lr}  WD: {args.weight_decay}")
    print(f"{'='*70}")

    ensure_output_dir(args.output_dir)
    set_seed(args.seed)

    states      = (["Arkansas", "California"]
                   if args.state == "both" else [args.state])
    all_results = {}

    for state in states:
        try:
            metrics = run_experiment_part3(
                state=state, cfg=CFG,
                out_dir=args.output_dir,
                config_name="AttentionCNN",
            )
            all_results[state] = metrics
        except Exception as exc:
            print(f"\n[ERROR] {state}: {exc}")
            raise


    print(f"\n{'='*70}")
    print("  FINAL RESULTS SUMMARY — PART 3")
    print(f"{'='*70}")
    print(f"  {'State':<15} {'OA':>8} {'Kappa':>8} {'F1':>8} {'Params':>12}")
    print(f"  {'-'*55}")
    # Part 2 baselines for quick comparison
    baselines = {
        "California": {"OA": 0.8779, "Kappa": 0.8516, "F1_macro": 0.8906},
        "Arkansas":   {"OA": 0.9626, "Kappa": 0.9532, "F1_macro": 0.9624},
    }
    for state, m in all_results.items():
        b = baselines.get(state, {})
        d_oa = m["OA"] - b.get("OA", 0)
        sign = "+" if d_oa >= 0 else ""
        print(f"  {state:<15} {m['OA']:>8.4f} {m['Kappa']:>8.4f} "
              f"{m['F1_macro']:>8.4f} {m['n_params']:>12,}"
              f"  (ΔOA vs P2: {sign}{d_oa*100:.2f}%)")
    print(f"{'='*70}")
    print(f"  Results → {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()