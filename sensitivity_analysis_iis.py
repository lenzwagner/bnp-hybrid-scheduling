"""
Sensitivity Analysis 2D Grid Search with IIS Pre-check
======================================================

This script optimizes the 2D grid search by:
1. Building a Compact Model (MIP) first.
2. Checking feasibility.
3. If INFEASIBLE: Computing IIS (Irreducible Inconsistent Subsystem) and saving .ilp file.
4. If FEASIBLE: Running the full Branch-and-Price solver.
"""

import pandas as pd
import numpy as np
import os
import sys
import gurobipy as gu
from datetime import datetime
from loop_main import solve_instance
from logging_config import setup_multi_level_logging, get_logger
from CG import ColumnGeneration

# Setup logging
setup_multi_level_logging(base_log_dir='logs/sensitivity_iis', enable_console=True, print_all_logs=False)
logger = get_logger(__name__)

def check_feasibility_compact(seed, D_focus, pttr, T, app_data_overrides, T_demand, output_dir, pre_generated_data=None):
    """
    Checks feasibility using the Compact Model.
    Returns: (is_feasible, message)
    """
    # Reconstruct app_data (defaults from solve_instance)
    app_data = {
        'learn_type': ['sigmoid'],
        'theta_base': [0.3],
        'lin_increase': [0.05],
        'k_learn': [0.3],
        'infl_point': [5],
        'MS': [5],
        'MS_min': [2],
        'W_on': [5],
        'W_off': [2],
        'daily': [4]
    }
    
    if app_data_overrides:
        for k, v in app_data_overrides.items():
            if k in app_data:
                app_data[k] = [v] if not isinstance(v, list) else v
    
    try:
        # Create Solver to build model
        # Note: Suppress output for speed
        cg_solver = ColumnGeneration(
            seed=seed,
            app_data=app_data,
            T=T,
            D_focus=D_focus,
            max_itr=10, # Minimal iterations as we just want setup
            threshold=1e-5,
            pttr=pttr,
            show_plots=False,
            pricing_filtering=True,
            therapist_agg=False,
            learn_method='pwl',
            verbose=False,
            T_demand=T_demand,
            pre_generated_data=pre_generated_data
        )
        
        # Build Model
        cg_solver.setup()
        
        # Access Compact Model (Gurobi)
        if not hasattr(cg_solver, 'problem') or not hasattr(cg_solver.problem, 'Model'):
            return True, "Could not access Compact Model (Fallback to BnP)"
            
        model = cg_solver.problem.Model
        model.setParam('OutputFlag', 0)
        
        # Optimize for feasibility only
        model.setParam('SolutionLimit', 1)  # Stop after 1 feasible solution
        model.setParam('MIPFocus', 1)       # Focus on finding feasible solutions
        model.setParam('TimeLimit', 300)    # Time limit: 5 minutes
        
        # Solve
        model.optimize()
        
        # Check status
        # OPTIMAL or SOLUTION_LIMIT means feasible
        if model.Status in [gu.GRB.OPTIMAL, gu.GRB.SOLUTION_LIMIT, gu.GRB.SUBOPTIMAL]:
            return True, "Feasible"
        elif model.Status == gu.GRB.TIME_LIMIT:
             return True, "Time Limit Reached (Assume Feasible / Proceed to BnP)"
        elif model.Status == gu.GRB.INFEASIBLE:
            # OPTIMIZATION: Calc IIS ONLY for Theta=0.6 and k=0.7
            theta = app_data_overrides.get('theta_base', 0)
            k = app_data_overrides.get('k_learn', 0)
            
            # Check for float equality with tolerance
            if abs(theta - 0.6) < 1e-5 and abs(k - 0.7) < 1e-5:
                 print(f"   -> TARGET DEBUG ({theta}, {k}): Computing IIS...")
                 model.computeIIS()
                 ilp_filename = f"{output_dir}/DEBUG_IIS_theta{theta:.2f}_k{k:.2f}.ilp"
                 model.write(ilp_filename)
                 return False, f"Infeasible (DEBUG IIS saved to {ilp_filename})"
            
            return False, "Infeasible (Compact Model)"
        else:
            return False, f"Solver Status: {model.Status}"
            
    except Exception as e:
        logger.error(f"Compact check failed: {e}")
        return True, f"Check Failed ({e}) - Proceeding to BnP"

def run_sensitivity_analysis():
    print("\n" + "=" * 100)
    print(" SENSITIVITY ANALYSIS (IIS OPTIMIZED) ".center(100, "="))
    print("=" * 100)

    # --- Parameters ---
    T_baseline = 4
    T_target = 3
    seed = 421
    D_focus = 5
    pttr = 'medium'
    
    # Grid Definition
    theta_range = np.arange(0.2, 0.80, 0.1)
    k_learn_range = np.arange(0.1, 1.1, 0.2)
    
    # Output Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'results/sensitivity_iis_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nConfiguration:")
    print(f"  - Baseline T: {T_baseline}")
    print(f"  - Target T: {T_target}")
    print(f"  - Seed: {seed}")
    print(f"  - Theta Range: {min(theta_range):.2f} - {max(theta_range):.2f}")
    print(f"  - K_learn Range: {min(k_learn_range):.2f} - {max(k_learn_range):.2f}")
    print(f"  - Output Dir: {output_dir}")
    
    # --- 1. PRE-GENERATE PATIENT DATA (Optimization) ---
    print("\n" + "=" * 60)
    print(" PRE-GENERATING PATIENT DATA ".center(60, "="))
    print(f" (Seed={seed}, T_demand={T_baseline}, Target T={T_target})".center(60))
    print("=" * 60)
    
    from Utils.Generell.instance_setup import generate_patient_data_log
    
    # We generate data for the TARGET scenario (T=3) but with DEMAND from BASELINE (T=4)
    # This ensures correct Max_t for T=3, but correct Patient Count for T=4.
    Req, Entry, Max_t, P, D, D_Ext, D_Full, T_list, M_p, W_coeff, DRG = generate_patient_data_log(
        seed=seed,
        D_focus=D_focus,
        pttr_scenario=pttr,
        T=T_target,        # Solver will see 3 therapists
        T_demand=T_baseline, # Generator creates patients for 4 therapists
        verbose=False
    )
    
    # Pack into dictionary for injection
    pre_generated_data = {
        'Req': Req, 'Entry': Entry, 'Max_t': Max_t, 'P': P, 'D': D, 
        'D_Ext': D_Ext, 'D_Full': D_Full, 'T': T_list, 'M_p': M_p, 
        'W_coeff': W_coeff, 'DRG': DRG
    }
    
    print(f" -> Generated {len(P)} patients (Optimized for reuse).")
    print("=" * 60 + "\n")

    # --- 2. CALCULATE BASELINE (Original T, No AI) ---
    print("\n" + "=" * 60)
    print(" CALCULATING BASELINE ".center(60, "="))
    print("=" * 60)
    
    try:
        baseline_res = solve_instance(
            seed=seed, D_focus=D_focus, pttr=pttr, T=T_baseline,
            allow_gaps=False, use_warmstart=True, learn_type=0
        )
        b_values = list(baseline_res.get('focus_los', {}).values())
        baseline_mean_los = np.mean(b_values) if b_values else 7.0
        print(f" -> Baseline Mean LOS: {baseline_mean_los:.4f}")
    except Exception as e:
        print(f" -> Baseline Failed: {e}")
        baseline_mean_los = 7.0

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
            
            # --- CHECK FEASIBILITY (Compact Model) ---
            # Pass pre_generated_data to avoid re-generation
            is_feasible, msg = check_feasibility_compact(
                seed, D_focus, pttr, T_target, app_data_overrides, T_baseline, output_dir, pre_generated_data
            )
            
            if not is_feasible:
                print(f"   -> SKIP BnP: {msg}")
                results.append({
                    'theta_base': theta_val,
                    'k_learn': k_val,
                    'is_feasible': 0,
                    'final_obj': None,
                    'mean_los': None,
                    'status': 'INFEASIBLE (MIP)',
                    'notes': msg
                })
                continue # SKIP BnP
            
            # --- RUN BnP ---
            try:
                # INJECT PRE-GENERATED DATA HERE
                instance_res = solve_instance(
                    seed=seed, D_focus=D_focus, pttr=pttr, T=T_target,
                    allow_gaps=False, use_warmstart=True, learn_type='sigmoid',
                    app_data_overrides=app_data_overrides, T_demand=T_baseline,
                    pre_generated_data=pre_generated_data 
                )
                
                focus_los = instance_res.get('focus_los', {})
                mean_los = np.mean(list(focus_los.values())) if focus_los else None
                is_optimal = instance_res.get('is_optimal', False)
                
                results.append({
                    'theta_base': theta_val,
                    'k_learn': k_val,
                    'is_feasible': 1 if is_optimal else 0,
                    'final_obj': instance_res.get('final_ub'),
                    'mean_los': mean_los,
                    'baseline_los': baseline_mean_los,
                    'los_diff': mean_los - baseline_mean_los if mean_los else None,
                    'is_rentable': 1 if mean_los and mean_los <= baseline_mean_los else 0,
                    'status': 'OK'
                })
                
                savings = baseline_mean_los - mean_los if mean_los else 0
                rentable_str = "✅ RENTABLE" if savings >= 0 else "❌ NOT YET"
                print(f"   -> Feasible. Mean LOS: {mean_los:.4f} (Base: {baseline_mean_los:.4f}) -> {rentable_str}")

            except Exception as e:
                logger.error(f"BnP Failed: {e}")
                results.append({
                    'theta_base': theta_val,
                    'k_learn': k_val,
                    'status': 'FAILED',
                    'baseline_los': baseline_mean_los
                })
                print(f"   -> FAILED: {e}")

    # --- SUMMARY OF PROFITABLE SCENARIOS ---
    print("\n" + "=" * 80)
    print(" ✅ RENTABLE SCENARIOS (Better or Equal to Baseline) ".center(80, "="))
    print("=" * 80)
    print(f"{'Theta':<10} | {'k_learn':<10} | {'Mean LOS':<15} | {'Baseline':<15} | {'Improvement':<15}")
    print("-" * 80)
    
    profitable = [r for r in results if r.get('is_rentable') == 1]
    
    if not profitable:
        print("No profitable scenarios found.")
    else:
        # Sort by improvement (descending)
        profitable.sort(key=lambda x: x.get('los_diff', 0)) # Diff is negative for improvement
        
        for p in profitable:
            theta = p['theta_base']
            k = p['k_learn']
            los = p['mean_los']
            base = p['baseline_los']
            diff = p['los_diff'] # e.g. -0.5 days
            
            print(f"{theta:<10.2f} | {k:<10.2f} | {los:<15.4f} | {base:<15.4f} | {abs(diff):<15.4f} days")
            
    print("=" * 80 + "\n")

    # Save Results
    df = pd.DataFrame(results)
    filename = f'{output_dir}/sensitivity_iis_results.xlsx'
    df.to_excel(filename, index=False)
    print(f"\nResults saved to: {filename}")

if __name__ == "__main__":
    run_sensitivity_analysis()
