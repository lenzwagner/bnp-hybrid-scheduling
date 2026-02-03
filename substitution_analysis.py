
import pandas as pd
import numpy as np
import logging
import math
import sys
import os

# Adjust path to find modules
sys.path.append(os.getcwd())

from loop_main_substitution import solve_instance

# Setup logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)
logging.getLogger('loop_main_substitution').setLevel(logging.ERROR)

def calc_theta_base_from_start(target_start, k, infl):
    # C = sigmoid value at Y=0 => 1 / (1 + exp(k * infl))
    C = 1.0 / (1.0 + math.exp(k * infl))
    if target_start < C:
        # Cannot achieve target_start with base >= 0
        pass
    # Formula: start = base + (1-base)*C => start - C = base(1-C) => base = (start-C)/(1-C)
    theta_base = (target_start - C) / (1.0 - C)
    return theta_base

def calculate_mlos(instance_data):
    if not instance_data or 'focus_los' not in instance_data:
        return float('inf')
    los_dict = instance_data['focus_los']
    if not los_dict:
        return float('inf')
    return sum(los_dict.values()) / len(los_dict)

def run_substitution_analysis():
    print("=" * 100)
    print(" TECHNOLOGICAL SUBSTITUTION ANALYSIS (Efficiency x Steepness) ".center(100, "="))
    print("=" * 100)
    
    # Configuration
    SEED = 42
    D_FOCUS = 10
    PTTR = 'medium'
    BASELINE_T = 4
    CHALLENGER_T = 3
    INFL_POINT = 5
    
    print(f"\nConfiguration:")
    print(f"  - Baseline T: {BASELINE_T}, Challenger T: {CHALLENGER_T}")
    print(f"  - Infl Point: {INFL_POINT}")
    
    # 1. Run Baseline
    print(f"\nRunning Baseline (T={BASELINE_T})...")
    try:
        baseline_data = solve_instance(
            seed=SEED, D_focus=D_FOCUS, pttr=PTTR, T=BASELINE_T, 
            learn_type=0, theta_base=0.3
        )
        baseline_mlos = calculate_mlos(baseline_data)
        print(f"✓ Baseline MLOS: {baseline_mlos:.4f} days")
    except Exception as e:
        print(f"✗ Baseline Failed: {e}")
        return

    # 2. Run Sweep
    print(f"\nRunning Sweep (varying k_learn and theta_start)...")
    
    k_values = [0.3, 0.5, 0.8, 1.0, 1.5]
    theta_starts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    results_table = []
    
    print("\n" + "-" * 85)
    print(f"{'k_learn':<8} {'Theta_Start':<12} {'Theta_Base':<12} {'MLOS':<10} {'Diff':<10} {'Status':<10}")
    print("-" * 85)
    
    thresholds = {} # Map k -> min_theta_start
    
    for k in k_values:
        threshold_found_for_k = False
        
        for t_start in theta_starts:
            t_base = calc_theta_base_from_start(t_start, k, INFL_POINT)
            
            try:
                challenger_data = solve_instance(
                    seed=SEED, D_focus=D_FOCUS, pttr=PTTR, T=CHALLENGER_T,
                    learn_type='sigmoid',
                    theta_base=t_base,
                    k_learn=k
                )
                challenger_mlos = calculate_mlos(challenger_data)
                diff = challenger_mlos - baseline_mlos
                
                # Check for stability/better performance
                is_better = challenger_mlos <= baseline_mlos + 1e-6
                
                status = "BETTER" if is_better else "WORSE"
                threshold_mark = ""
                
                if is_better and not threshold_found_for_k:
                    threshold_found_for_k = True
                    thresholds[k] = t_start
                    threshold_mark = "(*)"
                
                print(f"{k:<8.2f} {t_start:<12.2f} {t_base:<12.4f} {challenger_mlos:<10.4f} {diff:<10.4f} {status} {threshold_mark}")
                
            except Exception as e:
                print(f"{k:<8.2f} {t_start:<12.2f} {'ERROR':<10} {str(e)[:20]}")
        
        print("-" * 85)

    print("\nSummary of Thresholds (Min Initial Efficiency required):")
    for k in k_values:
        if k in thresholds:
            print(f"  k={k}: Initial Efficiency >= {thresholds[k]}")
        else:
            print(f"  k={k}: No threshold found (need > 1.0)")

if __name__ == "__main__":
    run_substitution_analysis()
