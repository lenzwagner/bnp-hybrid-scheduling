"""
Sensitivity Analysis 2D Grid Search
===================================

This script performs a 2D grid search over:
- Initial Effectiveness (theta_base)
- Learning Speed (k_learn)

Context:
Sensitivity analysis for "Computational Experiments" section.
Scenario: Reducing therapists from 10 (baseline) to 9.
Goal: Find the "Break-even-Point" or "Isoquant" where reduced capacity is feasible/efficient.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from loop_main import solve_instance
from logging_config import setup_multi_level_logging, get_logger

# Setup logging
setup_multi_level_logging(base_log_dir='logs/sensitivity', enable_console=True, print_all_logs=False)
logger = get_logger(__name__)

def run_sensitivity_analysis():
    print("\n" + "=" * 100)
    print(" SENSITIVITY ANALYSIS: 2D GRID SEARCH ".center(100, "="))
    print("=" * 100)

    # --- Parameters ---
    # Baseline Scenario: Original T (Full Capacity) with No AI
    T_baseline = 5 # Example: If testing T=5, maybe baseline is 6. Adjust as needed.
    # Target Scenario: Reduced T with AI
    T_target = 3
    
    # Check if we should override T_baseline based on T_target
    # Typically baseline is +1 or based on "Original Team Size"
    # For this specific user request: "Original T -> Reduced T"
    # Let's assume T_baseline is T_target + 1 for now, or use a fixed value if known.
    # The user said "original T... dann reduzieren". 
    # Let's use T_baseline = 10 (standard) and T_target = 9, OR use the values in the script.
    # The script currently has T_target = 5 (from previous small run). 
    # Let's set T_baseline = T_target + 1 for general logic.
    # T_baseline = T_target + 1  # REMOVED: Respect manual setting above

    # Fixed parameters
    seed = 421
    D_focus = 10
    pttr = 'medium'
    
    # Grid Definition
    # Theta (Start Efficiency): 0.2 to 0.8
    theta_range = np.arange(0.2, 0.80, 0.1)
    # K_learn (Learning Speed): 0.1 to 1.0
    k_learn_range = np.arange(0.1, 1.1, 0.2)
    
    # For testing purposes, uncomment below to run a smaller grid first
    # theta_range = [0.3, 0.7] # Small grid for verification
    # k_learn_range = [0.2]

    print(f"\nConfiguration:")
    print(f"  - Baseline T (Human Only): {T_baseline}")
    print(f"  - Target T (AI Supported): {T_target}")
    print(f"  - Seed: {seed}")
    print(f"  - PTTR: {pttr}")
    print(f"  - Theta Range: {min(theta_range):.2f} - {max(theta_range):.2f} (Steps: {len(theta_range)})")
    print(f"  - K_learn Range: {min(k_learn_range):.2f} - {max(k_learn_range):.2f} (Steps: {len(k_learn_range)})")
    
    # --- 1. CALCULATE BASELINE (Original T, No AI) ---
    print("\n" + "=" * 60)
    print(" CALCULATING BASELINE (Original T, No AI) ".center(60, "="))
    print("=" * 60)
    
    try:
        baseline_res = solve_instance(
            seed=seed,
            D_focus=D_focus,
            pttr=pttr,
            T=T_baseline,
            allow_gaps=False,
            use_warmstart=True,
            learn_type=0, # 0 = Human Only / No Learning
        )
        
        # Calculate Baseline Average LOS
        b_focus_los = baseline_res.get('focus_los', {})
        if b_focus_los:
             b_values = list(b_focus_los.values())
             baseline_mean_los = np.mean(b_values)
             baseline_median_los = np.median(b_values)
        else:
             baseline_mean_los = float('inf')
             baseline_median_los = float('inf')
             
        print(f" -> Baseline Mean LOS (Target): {baseline_mean_los:.4f}")
        print(f" -> Baseline Median LOS:        {baseline_median_los:.4f}")
        
    except Exception as e:
        logger.error(f"Baseline calculation failed: {e}")
        baseline_mean_los = 7.0 # Fallback
        print(f" -> FAILED to calculate baseline. Using default fallback: {baseline_mean_los}")

    print("=" * 60 + "\n")

    results = []
    total_steps = len(theta_range) * len(k_learn_range)
    counter = 0

    for theta in theta_range:
        for k_learn in k_learn_range:
            counter += 1
            theta_val = float(round(theta, 2))
            k_val = float(round(k_learn, 2))
            
            print(f"\n[{counter}/{total_steps}] Testing: theta={theta_val}, k={k_val} ...")
            
            app_data_overrides = {
                'theta_base': theta_val,
                'k_learn': k_val
            }
            
            try:
                # Solve instance with TARGET T
                # Pass T_demand=T_baseline to ensure patient load matches baseline
                instance_res = solve_instance(
                    seed=seed,
                    D_focus=D_focus,
                    pttr=pttr,
                    T=T_target, # Reduced Capacity
                    allow_gaps=False,
                    use_warmstart=True,
                    learn_type='sigmoid',
                    app_data_overrides=app_data_overrides,
                    T_demand=T_baseline # FREEZE PATIENT SET: Use Baseline T for Demand
                )
                
                is_optimal = instance_res.get('is_optimal', False)
                final_ub = instance_res.get('final_ub', None)
                final_gap = instance_res.get('final_gap', None)
                
                focus_los = instance_res.get('focus_los', {})
                if focus_los:
                    los_values = list(focus_los.values())
                    median_los = np.median(los_values)
                    mean_los = np.mean(los_values)
                else:
                    median_los = None
                    mean_los = None

                results.append({
                    'theta_base': theta_val,
                    'k_learn': k_val,
                    'is_feasible': 1 if is_optimal and final_ub is not None else 0,
                    'final_obj': final_ub,
                    'gap': final_gap,
                    'median_los': median_los,
                    'mean_los': mean_los,
                    'status': 'OK'
                })
                
                print(f"   -> Feasible: {is_optimal}, Mean LOS: {mean_los:.4f} (Base: {baseline_mean_los:.4f})")

            except Exception as e:
                logger.error(f"Failed for theta={theta_val}, k={k_val}: {e}")
                print(f"   -> FAILED: {str(e)}")
                results.append({
                    'theta_base': theta_val,
                    'k_learn': k_val,
                    'status': 'FAILED',
                    'error': str(e)
                })

    # --- Save Results ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(results)
    
    output_dir = 'results/sensitivity'
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{output_dir}/sensitivity_dynamic_{timestamp}.xlsx'
    
    df.to_excel(filename, index=False)
    print(f"\nResults saved to: {filename}")
    
    # Create a Pivot Table for quick view
    if 'mean_los' in df.columns:
        try:
            pivot = df.pivot_table(index='k_learn', columns='theta_base', values='mean_los')
            print("\nHeatmap Preview (Mean LOS):")
            print(pivot)
            
            # --- CALCULATE BREAK-EVEN FRONTIER ---
            # Compare Mean LOS against Baseline Mean LOS
            
            frontier = []
            
            print(f"\n" + "=" * 80)
            print(f" BREAK-EVEN FRONTIER (Target Mean LOS <= {baseline_mean_los:.4f}) ".center(80, "="))
            print("=" * 80)
            print(f"{'k_learn':<10} | {'Break-Even Theta':<20} | {'Mean LOS':<15} | {'Diff to Base':<15}")
            print("-" * 80)
            
            unique_ks = sorted(df['k_learn'].unique())
            
            for k in unique_ks:
                # Filter rows for this k
                k_subset = df[df['k_learn'] == k].sort_values('theta_base')
                
                found = False
                for _, row in k_subset.iterrows():
                    if row['mean_los'] is not None and row['mean_los'] <= baseline_mean_los + 1e-5: # Epsilon for tolerance
                        diff = row['mean_los'] - baseline_mean_los
                        print(f"{k:<10} | {row['theta_base']:<20} | {row['mean_los']:<15.4f} | {diff:<15.4f}")
                        frontier.append({
                            'k_learn': k,
                            'break_even_theta': row['theta_base'],
                            'achieved_mean_los': row['mean_los'],
                            'baseline_mean_los': baseline_mean_los,
                            'diff': diff
                        })
                        found = True
                        break # Found the first one (frontier)
                
                if not found:
                    print(f"{k:<10} | {'> Max Tested':<20} | {'N/A':<15} | {'N/A':<15}")
            
            print("=" * 80 + "\n")
            
            if frontier:
                frontier_df = pd.DataFrame(frontier)
                frontier_file = f'{output_dir}/break_even_dynamic_{timestamp}.xlsx'
                frontier_df.to_excel(frontier_file, index=False)
                print(f"Break-Even Frontier saved to: {frontier_file}")

        except Exception as e:
            print(f"Could not calculate break-even frontier: {e}")


if __name__ == "__main__":
    run_sensitivity_analysis()
