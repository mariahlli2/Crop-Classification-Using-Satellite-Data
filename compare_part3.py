import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_metrics(json_path: str) -> dict | None:
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        return json.load(f)


def load_part2_results(part2_dir: str, state: str) -> dict:
    
    results = {}
    ablation_summary = os.path.join(
        part2_dir, "ablation_summary", f"ablation_{state}.json"
    )
    if os.path.exists(ablation_summary):
        with open(ablation_summary) as f:
            data = json.load(f)
        for cfg_name, m in data.items():
            if "error" not in m:
                results[cfg_name] = m
        return results

    
    for cfg_name in ["S2_only", "S2_climate", "S2_soil", "S2_topo", "S2_all"]:
        p = os.path.join(part2_dir, cfg_name, "metrics", f"metrics_{state}.json")
        m = _load_metrics(p)
        if m:
            results[cfg_name] = m
    return results


def load_part3_result(part3_dir: str, state: str) -> dict | None:
    p = os.path.join(part3_dir, "AttentionCNN", "metrics",
                     f"metrics_{state}.json")
    return _load_metrics(p)



def print_comparison(state: str, part2: dict, part3: dict | None):
    print(f"\n{'='*72}")
    print(f"  COMPARISON — {state}")
    print(f"{'='*72}")
    print(f"  {'Model/Config':<22} {'OA':>8} {'Kappa':>8} "
          f"{'F1-macro':>10} {'Params':>12}")
    print(f"  {'-'*68}")

    for name, m in part2.items():
        print(f"  {name:<22} {m['OA']:>8.4f} {m['Kappa']:>8.4f} "
              f"{m['F1_macro']:>10.4f} {m.get('n_params', 0):>12,}")

    if part3:
        print(f"  {'-'*68}")
        name = "AttentionCNN (P3)"
        print(f"  {name:<22} {part3['OA']:>8.4f} {part3['Kappa']:>8.4f} "
              f"{part3['F1_macro']:>10.4f} {part3.get('n_params', 0):>12,}")

    print(f"{'='*72}")

    if part3 and part2:
        best_p2_oa = max(m["OA"] for m in part2.values())
        delta = part3["OA"] - best_p2_oa
        sign  = "+" if delta >= 0 else ""
        print(f"  AttentionCNN vs best Part-2 config: "
              f"ΔOA = {sign}{delta*100:.2f}%")


def plot_comparison(state: str, part2: dict, part3: dict | None,
                    out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    configs = list(part2.keys())
    oas     = [part2[c]["OA"] for c in configs]
    colors  = ["steelblue"] * len(configs)

    if part3:
        configs.append("AttentionCNN\n(Part 3)")
        oas.append(part3["OA"])
        colors.append("darkorange")

    x = np.arange(len(configs))
    fig, ax = plt.subplots(figsize=(max(10, len(configs) * 1.4), 5))
    bars = ax.bar(x, oas, color=colors, edgecolor="white", linewidth=0.5)

    for bar, oa in zip(bars, oas):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{oa:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Overall Accuracy (OA)")
    ax.set_title(f"Part 2 vs Part 3 — {state}", fontweight="bold")

    lo, hi = min(oas), max(oas)
    pad = max(0.005, (hi - lo) * 0.15)
    ax.set_ylim(max(0, lo - pad), min(1.0, hi + pad * 3))
    ax.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="steelblue",  label="Part 2 (covariate configs)"),
        Patch(color="darkorange", label="Part 3 — AttentionCNN"),
    ], loc="lower right", fontsize=9)

    plt.tight_layout()
    fpath = os.path.join(out_dir, f"comparison_{state}.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Figure saved → {fpath}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part2-dir", default="results_part2")
    parser.add_argument("--part3-dir", default="results_part3")
    parser.add_argument("--out-dir",   default="results_part3/comparison")
    args = parser.parse_args()

    for state in ["Arkansas", "California"]:
        part2 = load_part2_results(args.part2_dir, state)
        part3 = load_part3_result(args.part3_dir, state)

        if not part2 and not part3:
            print(f"[{state}] No results found — skipping.")
            continue

        print_comparison(state, part2, part3)
        plot_comparison(state, part2, part3, args.out_dir)


if __name__ == "__main__":
    main()
