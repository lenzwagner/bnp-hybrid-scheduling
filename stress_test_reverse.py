"""
Reverse Stress Test - Technological Break-Even Point Determination

This script performs a reverse stress test to determine the minimum AI proficiency 
(theta_req) required to compensate for workforce reduction from 10 to 9 therapists 
without degrading system stability.

Methodology (Option A - Pre-Patient Filtering):
1. Generate patient data once per instance
2. Run baseline with full workforce (T=10, human-only)
3. Identify and remove pre-patients assigned to removed therapist
4. Run stress test with reduced workforce (T=9) and filtered patients
5. Iterate over theta values to find break-even point

Author: Auto-generated from implementation plan
Date: 2026-02-17
"""

import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from loop_main_learning import solve_instance
from Utils.Generell.instance_setup import generate_patient_data_log
from logging_config import setup_multi_level_logging, get_logger

logger = get_logger(__name__)


def filter_pre_patients(
    base_data_dict: Dict,
    baseline_pre_x: Dict,
    therapists_to_remove: List[int],
    verbose: bool = True
) -> Tuple[Dict, set]:
    """
    Filter out pre-patients assigned to the removed therapists.
    
    Args:
        base_data_dict: Base patient and therapist data
        baseline_pre_x: Pre-patient assignments from baseline run {(p, t, d): value}
        therapists_to_remove: List of therapist IDs to remove
        verbose: Whether to print filtering details
        
    Returns:
        Tuple of (filtered_data_dict, pre_patients_removed_set)
    """
    # Identify pre-patients assigned to any therapist being removed
    pre_patients_of_removed = set()
    for (p, t, d), val in baseline_pre_x.items():
        if t in therapists_to_remove:
            pre_patients_of_removed.add(p)
    
    if verbose:
        logger.info(f"Pre-patients assigned to therapist(s) {therapists_to_remove}: {len(pre_patients_of_removed)}")
        logger.info(f"Pre-patient IDs: {sorted(pre_patients_of_removed)}")
    
    # Filter patient data
    filtered_data = {
        'Req': {p: v for p, v in base_data_dict['Req'].items() if p not in pre_patients_of_removed},
        'Entry': {p: v for p, v in base_data_dict['Entry'].items() if p not  in pre_patients_of_removed},
        'P': [p for p in base_data_dict['P'] if p not in pre_patients_of_removed],
        'DRG': {p: v for p, v in base_data_dict['DRG'].items() if p not in pre_patients_of_removed},
        'M_p': {p: v for p, v in base_data_dict['M_p'].items() if p not in pre_patients_of_removed},
        # Keep unchanged (patient-independent)
        'D': base_data_dict['D'],
        'D_Ext': base_data_dict['D_Ext'],
        'D_Full': base_data_dict['D_Full'],
        'W_coeff': base_data_dict['W_coeff'],
        # Remove therapists from T and Max_t
        'T': [t for t in base_data_dict['T'] if t not in therapists_to_remove],
        'Max_t': {k: v for k, v in base_data_dict['Max_t'].items() if k[0] not in therapists_to_remove}
    }
    
    if verbose:
        logger.info(f"Baseline patients: {len(base_data_dict['P'])}")
        logger.info(f"Filtered patients: {len(filtered_data['P'])}")
        logger.info(f"Patient reduction ratio: {len(filtered_data['P']) / len(base_data_dict['P']):.3f}")
    
    return filtered_data, pre_patients_of_removed


def run_reverse_stress_test(
    seed: int,
    D_focus: int,
    pttr: str,
    T_count: int,
    instance_id: str,
    num_therapists_to_remove: int = 1,
    theta_range: np.ndarray = None,
    learning_curve_type: str = 'sigmoid',
    k_learn: float = None,
    timeout_per_solve: int = 1200,
    verbose: bool = True
) -> Dict:
    """
    Run reverse stress test for a single instance.
    
    Args:
        seed: Random seed
        D_focus: Focus horizon days
        pttr: Patient-to-therapist ratio scenario ('light', 'medium', 'heavy')
        T_count: Number of therapists (typically 10)
        instance_id: Unique instance identifier
        num_therapists_to_remove: Number of therapists to remove (default: 1)
        theta_range: Array of theta values to test (default: linspace(0.3, 1.0, 15))
        learning_curve_type: Learning curve type ('sigmoid', 'exp', 'lin')
        k_learn: Learning rate parameter (default: None = use curve-specific default)
        timeout_per_solve: Time limit per solve in seconds
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary with stress test results
    """
    logger.info("=" * 100)
    logger.info(f"REVERSE STRESS TEST: {instance_id}")
    logger.info(f"seed={seed}, D_focus={D_focus}, pttr={pttr}, T={T_count}, remove={num_therapists_to_remove}")
    logger.info("=" * 100)
    
    if theta_range is None:
        theta_range = np.linspace(0.3, 1.0, 15)
    
    # =====================================================================
    # STEP 1: Generate Master Data (ONCE per instance)
    # =====================================================================
    logger.info("\n[Step 1/7] Generating master patient data...")
    
    Req, Entry, Max_t, P, D, D_Ext, D_Full, T, M_p, W_coeff, DRG = generate_patient_data_log(
        T=T_count,
        D_focus=D_focus,
        W_on=5,
        W_off=2,
        daily=4,
        pttr_scenario=pttr,
        seed=seed,
        plot_show=False,
        verbose=verbose
    )
    
    base_data_dict = {
        'Req': Req,
        'Entry': Entry,
        'Max_t': Max_t,
        'P': P,
        'D': D,
        'D_Ext': D_Ext,
        'D_Full': D_Full,
        'T': T,
        'M_p': M_p,
        'W_coeff': W_coeff,
        'DRG': DRG
    }
    
    logger.info(f"Generated {len(P)} patients, {len(T)} therapists")
    
    # =====================================================================
    # STEP 2: Baseline Solve (Human-Only, Full Workforce)
    # =====================================================================
    logger.info(f"\n[Step 2/7] Running BASELINE solve (T={T_count}, Human-only)...")
    
    try:
        baseline_results = solve_instance(
            seed=seed,
            D_focus=D_focus,
            pttr=pttr,
            T=T_count,
            allow_gaps=False,
            use_warmstart=True,
            dual_smoothing_alpha=None,
            learn_type=0,  # Human-only
            app_data_overrides={'learn_type': 0},
            pre_generated_data=base_data_dict
        )
    except Exception as e:
        logger.error(f"Baseline solve FAILED: {str(e)}", exc_info=True)
        return {
            'instance_id': instance_id,
            'seed': seed,
            'D_focus': D_focus,
            'pttr': pttr,
            'status': 'BASELINE_FAILED',
            'error': str(e)
        }
    
    UB_baseline = baseline_results.get('final_ub')
    baseline_pre_x = baseline_results.get('pre_x', {})
    
    logger.info(f"Baseline UB: {UB_baseline:.2f}")
    logger.info(f"Baseline optimal: {baseline_results.get('is_optimal', False)}")
    
    if not baseline_results.get('is_optimal', False):
        logger.warning("Baseline is NOT optimal - results may be unreliable")
    
    # =====================================================================
    # STEP 3: Pre-Patient Filtering (Option A)
    # =====================================================================
    logger.info("\n[Step 3/7] Filtering pre-patients...")
    
    # Choose therapists to remove (last N in list)
    therapists_to_remove = T[-num_therapists_to_remove:]
    logger.info(f"Removing {num_therapists_to_remove} therapist(s): {therapists_to_remove}")
    
    stress_data_dict, pre_patients_removed = filter_pre_patients(
        base_data_dict,
        baseline_pre_x,
        therapists_to_remove,
        verbose=verbose
    )
    
    # =====================================================================
    # STEP 4: Feasibility Check (Optional - Human-only with reduced workforce)
    # =====================================================================
    logger.info(f"\n[Step 4/7] Running FEASIBILITY CHECK (T={T_count - num_therapists_to_remove}, Human-only)...")
    
    try:
        feasibility_results = solve_instance(
            seed=seed,
            D_focus=D_focus,
            pttr=pttr,
            T=T_count - num_therapists_to_remove,
            allow_gaps=False,
            use_warmstart=True,
            dual_smoothing_alpha=None,
            learn_type=0,  # Human-only
            app_data_overrides={'learn_type': 0},
            pre_generated_data=stress_data_dict
        )
        
        UB_stress_humanonly = feasibility_results.get('final_ub')
        feasible_with_reduction = feasibility_results.get('is_optimal') is not None
        
        logger.info(f"Feasibility check UB: {UB_stress_humanonly:.2f if UB_stress_humanonly else 'N/A'}")
        logger.info(f"Feasible with reduction: {feasible_with_reduction}")
        
    except Exception as e:
        logger.error(f"Feasibility check FAILED: {str(e)}")
        UB_stress_humanonly = None
        feasible_with_reduction = False
    
    if not feasible_with_reduction:
        logger.warning("System is INFEASIBLE with reduced workforce - no break-even possible")
        return {
            'instance_id': instance_id,
            'seed': seed,
            'D_focus': D_focus,
            'pttr': pttr,
            'T_original': T_count,
            'T_reduced': T_count - num_therapists_to_remove,
            'therapists_removed': therapists_to_remove,
            'num_patients_baseline': len(base_data_dict['P']),
            'num_patients_stress': len(stress_data_dict['P']),
            'num_pre_patients_removed': len(pre_patients_removed),
            'patient_reduction_ratio': len(stress_data_dict['P']) / len(base_data_dict['P']),
            'UB_baseline': UB_baseline,
            'UB_stress_humanonly': UB_stress_humanonly,
            'feasible_with_reduction': False,
            'break_even_found': False,
            'status': 'INFEASIBLE'
        }
    
    # =====================================================================
    # STEP 5: Compensation Loop - Break-Even Search
    # =====================================================================
    logger.info("\n[Step 5/7] Starting COMPENSATION LOOP for break-even search...")
    logger.info(f"Testing theta values: {theta_range}")
    logger.info(f"Learning curve type: {learning_curve_type}")
    logger.info(f"Target: UB_stress <= UB_baseline ({UB_baseline:.2f})")
    
    break_even_found = False
    theta_req = None
    UB_at_breakeven = None
    
    # Learning curve parameters (defaults)
    learning_params = {
        'sigmoid': {'k_learn': 1.5, 'infl_point': 4},
        'exp': {'k_learn': 0.732},
        'lin': {'lin_increase': 0.088}
    }
    
    # Override k_learn if provided by user
    if k_learn is not None:
        if learning_curve_type in ['sigmoid', 'exp']:
            learning_params[learning_curve_type]['k_learn'] = k_learn
            logger.info(f"Using custom k_learn: {k_learn}")
    
    for theta_base in theta_range:
        logger.info(f"\n--- Testing theta_base = {theta_base:.3f} ---")
        
        # Prepare app_data overrides
        app_data_overrides = {'learn_type': learning_curve_type, 'theta_base': theta_base}
        app_data_overrides.update(learning_params.get(learning_curve_type, {}))
        
        try:
            stress_results = solve_instance(
                seed=seed,
                D_focus=D_focus,
                pttr=pttr,
                T=T_count - num_therapists_to_remove,
                allow_gaps=False,
                use_warmstart=True,
                dual_smoothing_alpha=None,
                learn_type=learning_curve_type,
                app_data_overrides=app_data_overrides,
                pre_generated_data=stress_data_dict
            )
            
            UB_stress = stress_results.get('final_ub')
            
            if UB_stress is None:
                logger.warning(f"Theta {theta_base:.3f}: No solution found")
                continue
            
            logger.info(f"Theta {theta_base:.3f}: UB_stress = {UB_stress:.2f} (Baseline = {UB_baseline:.2f})")
            
            # Check break-even condition
            if UB_stress <= UB_baseline:
                logger.info(f"✓ BREAK-EVEN FOUND at theta = {theta_base:.3f}")
                theta_req = theta_base
                UB_at_breakeven = UB_stress
                break_even_found = True
                break
            else:
                improvement = ((UB_baseline - UB_stress) / UB_baseline) * 100
                logger.info(f"  Not yet compensated (improvement: {improvement:.2f}%)")
                
        except Exception as e:
            logger.error(f"Theta {theta_base:.3f}: Solve FAILED - {str(e)}")
            continue
    
    if not break_even_found:
        logger.warning(f"No break-even found in theta range {theta_range[0]:.2f}-{theta_range[-1]:.2f}")
    
    # =====================================================================
    # STEP 6: Calculate Theoretical Break-Even (Analytical Formula)
    # =====================================================================
    logger.info("\n[Step 6/7] Calculating theoretical break-even...")
    
    rho = (T_count - num_therapists_to_remove) / T_count  # e.g., 0.9 for T=10→9, 0.8 for T=10→8
    theta_0 = 0.0  # Baseline is human-only
    theoretical_theta_req = 1 - rho * (1 - theta_0)
    
    logger.info(f"Theoretical theta_req (analytical): {theoretical_theta_req:.3f}")
    if theta_req is not None:
        logger.info(f"Simulated theta_req: {theta_req:.3f}")
        logger.info(f"Difference: {abs(theta_req - theoretical_theta_req):.3f}")
    
    # =====================================================================
    # STEP 7: Results Collection
    # =====================================================================
    logger.info("\n[Step 7/7] Collecting results...")
    
    results = {
        'instance_id': instance_id,
        'seed': seed,
        'D_focus': D_focus,
        'pttr': pttr,
        'T_original': T_count,
        'T_reduced': T_count - num_therapists_to_remove,
        'therapists_removed': therapists_to_remove,
        'num_therapists_removed': num_therapists_to_remove,
        'num_patients_baseline': len(base_data_dict['P']),
        'num_patients_stress': len(stress_data_dict['P']),
        'num_pre_patients_removed': len(pre_patients_removed),
        'patient_reduction_ratio': len(stress_data_dict['P']) / len(base_data_dict['P']),
        'UB_baseline': UB_baseline,
        'UB_stress_humanonly': UB_stress_humanonly,
        'feasible_with_reduction': feasible_with_reduction,
        'break_even_found': break_even_found,
        'theta_req': theta_req,
        'UB_at_breakeven': UB_at_breakeven,
        'learning_curve_type': learning_curve_type,
        'k_learn_used': k_learn if k_learn is not None else learning_params.get(learning_curve_type, {}).get('k_learn', 'N/A'),
        'reduction_ratio_rho': rho,
        'theoretical_theta_req': theoretical_theta_req,
        'theta_range_min': theta_range[0],
        'theta_range_max': theta_range[-1],
        'theta_range_steps': len(theta_range),
        'status': 'SUCCESS' if break_even_found else 'NO_BREAKEVEN'
    }
    
    logger.info("=" * 100)
    logger.info(f"REVERSE STRESS TEST COMPLETE: {instance_id}")
    logger.info(f"Break-even found: {break_even_found}")
    if break_even_found:
        logger.info(f"Required theta: {theta_req:.3f}")
    logger.info("=" * 100)
    
    return results


def main():
    """
    Main loop to run reverse stress tests on all instances from newest Excel file.
    Tests multiple k_learn values for each instance.
    """
    # ===========================
    # CONFIGURATION - EDIT HERE
    # ===========================
    
    # Theta range
    THETA_MIN = 0.3
    THETA_MAX = 1.0
    THETA_STEPS = 15
    
    # k_learn range
    K_LEARN_VALUES = np.linspace(1.0, 1.0, 1).tolist()  # 11 values: [1.0, 1.2, ..., 3.0]
    # Alternative: K_LEARN_VALUES = [1.5, 2.0, 2.5]  # Fixed list
    
    # Other settings
    LEARNING_CURVE_TYPE = 'sigmoid'
    TIMEOUT_PER_SOLVE = 1200
    
    # ===========================
    # LOGGING CONFIGURATION
    # ===========================
    print_all_logs = False
    setup_multi_level_logging(base_log_dir='logs', enable_console=True, print_all_logs=print_all_logs)
    
    # ===========================
    # LOAD INSTANCES FROM EXCEL
    # ===========================
    instances_dir = 'results/instances'
    excel_files = glob.glob(os.path.join(instances_dir, '*.xlsx'))
    
    if not excel_files:
        raise FileNotFoundError(f"No Excel files found in {instances_dir}")
    
    newest_excel = max(excel_files, key=os.path.getmtime)
    
    # Calculate theta range
    theta_range = np.linspace(THETA_MIN, THETA_MAX, THETA_STEPS)
    
    print("\n" + "=" * 100)
    print(f" REVERSE STRESS TEST - LOADING INSTANCES ".center(100, "="))
    print(f" File: {newest_excel} ".center(100, "="))
    print("=" * 100)
    
    print(f"\n📊 Configuration:")
    print(f"  Theta range: {THETA_MIN} to {THETA_MAX} ({THETA_STEPS} steps)")
    print(f"  k_learn values: {K_LEARN_VALUES}")
    print(f"  Total k_learn values: {len(K_LEARN_VALUES)}")
    print(f"  Combinations per instance: {THETA_STEPS} θ × {len(K_LEARN_VALUES)} k_learn = {THETA_STEPS * len(K_LEARN_VALUES)}")
    print(f"  Learning curve: {LEARNING_CURVE_TYPE}")
    print(f"  Timeout: {TIMEOUT_PER_SOLVE}s per solve")
    print("=" * 100 + "\n")
    
    scenarios_df = pd.read_excel(newest_excel)
    
    print(f"Loaded {len(scenarios_df)} scenarios from {os.path.basename(newest_excel)}")
    print(f"Columns: {scenarios_df.columns.tolist()}\\n")
    
    # ===========================
    # MAIN LOOP
    # ===========================
    results_list = []
    total_instances = len(scenarios_df)
    
    print("\n" + "=" * 100)
    print(f" BATCH RUN: {total_instances} instances ".center(100, "="))
    print("=" * 100 + "\n")
    
    for idx, row in scenarios_df.iterrows():
        current_instance = idx + 1
        instance_id_base = row.get('instance_id', f'instance_{idx}')
        seed = int(row['seed'])
        D_focus = int(row['D_focus_count'])
        pttr = row.get('pttr', 'medium')
        T = int(row.get('T_count', 10))
        num_to_remove = int(row.get('num_therapists_to_remove', 1))
        
        print(f"\n{'=' * 100}")
        print(f" Instance {current_instance}/{total_instances}: {instance_id_base} ".center(100, "="))
        print(f" Testing {len(K_LEARN_VALUES)} k_learn values ".center(100, "="))
        print(f"{'=' * 100}\n")
        
        # OUTER LOOP: Iterate over k_learn values
        for k_idx, k_learn_val in enumerate(K_LEARN_VALUES, 1):
            instance_id = f"{instance_id_base}_k{k_learn_val:.1f}"
            
            print(f"\n{'─' * 100}")
            print(f" k_learn {k_idx}/{len(K_LEARN_VALUES)}: {k_learn_val:.2f} ".center(100, "─"))
            print(f"{'─' * 100}\n")
            
            try:
                results = run_reverse_stress_test(
                    seed=seed,
                    D_focus=D_focus,
                    pttr=pttr,
                    T_count=T,
                    instance_id=instance_id,
                    num_therapists_to_remove=num_to_remove,
                    theta_range=theta_range,  # From config
                    learning_curve_type=LEARNING_CURVE_TYPE,  # From config
                    k_learn=k_learn_val,  # Current k_learn value
                    timeout_per_solve=TIMEOUT_PER_SOLVE,  # From config
                    verbose=False
                )
                
                results_list.append(results)
                
                # Print combination result
                print(f"\n┌{'─' * 98}┐")
                print(f"│ {'RESULT'.center(98)} │")
                print(f"├{'─' * 98}┤")
                print(f"│ Instance: {instance_id_base:<50} k_learn: {k_learn_val:<10.2f} │")
                print(f"│ Status: {results['status']:<88} │")
                
                if results.get('break_even_found'):
                    theta_req = results.get('theta_req', 0)
                    ub_baseline = results.get('UB_baseline', 0)
                    ub_breakeven = results.get('UB_at_breakeven', 0)
                    
                    print(f"│ {'✓ BREAK-EVEN FOUND'.center(98)} │")
                    print(f"│   theta_req = {theta_req:.3f} ({theta_req*100:.1f}%){'':60} │")
                    print(f"│   UB: {ub_baseline:.2f} → {ub_breakeven:.2f}{'':75} │")
                    print(f"│   Combination: θ={theta_req:.3f} × k_learn={k_learn_val:.2f}{'':50} │")
                else:
                    print(f"│ {'✗ No break-even found'.center(98)} │")
                
                print(f"└{'─' * 98}┘\n")
                
            except Exception as e:
                print(f"\n✗ Instance {current_instance}/{total_instances} (k_learn={k_learn_val:.2f}) FAILED:")
                print(f"  Error: {str(e)}")
                logger.error(f"Instance {instance_id} failed: {str(e)}", exc_info=True)
                
                results_list.append({
                    'instance_id': instance_id,
                    'seed': seed,
                    'D_focus': D_focus,
                    'pttr': pttr,
                    'k_learn_used': k_learn_val,
                    'error': str(e),
                    'status': 'FAILED'
                })
    
    # ===========================
    # SAVE RESULTS
    # ===========================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = 'results/stress_test'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to Excel
    excel_filename = f"{output_dir}/reverse_stress_test_{timestamp}.xlsx"
    results_df = pd.DataFrame(results_list)
    
    try:
        results_df.to_excel(excel_filename, index=False)
        print(f"\n✓ Results saved to {excel_filename}")
    except Exception as e:
        print(f"\n✗ Could not save Excel: {e}")
    
    # ===========================
    # SUMMARY
    # ===========================
    print("\n" + "=" * 100)
    print(" REVERSE STRESS TEST SUMMARY ".center(100, "="))
    print("=" * 100)
    
    successful = results_df[results_df['status'] == 'SUCCESS']
    no_breakeven = results_df[results_df['status'] == 'NO_BREAKEVEN']
    failed = results_df[results_df['status'] == 'FAILED']
    infeasible = results_df[results_df['status'] == 'INFEASIBLE']
    
    print(f"Total instances: {total_instances}")
    print(f"Success (break-even found): {len(successful)}")
    print(f"No break-even in range: {len(no_breakeven)}")
    print(f"Infeasible with reduction: {len(infeasible)}")
    print(f"Failed: {len(failed)}")
    
    if len(successful) > 0:
        print(f"\nOverall break-even statistics:")
        print(f"  Mean theta_req: {successful['theta_req'].mean():.3f}")
        print(f"  Std theta_req: {successful['theta_req'].std():.3f}")
        print(f"  Min theta_req: {successful['theta_req'].min():.3f}")
        print(f"  Max theta_req: {successful['theta_req'].max():.3f}")
        
        print(f"\n\nBreak-even statistics by k_learn:")
        print("=" * 80)
        for k_val in K_LEARN_VALUES:
            k_results = successful[successful['k_learn_used'] == k_val]
            if len(k_results) > 0:
                print(f"\nk_learn = {k_val:.2f}:")
                print(f"  Success rate: {len(k_results)}/{total_instances}")
                print(f"  Mean theta_req: {k_results['theta_req'].mean():.3f}")
                print(f"  Min theta_req: {k_results['theta_req'].min():.3f}")
                print(f"  Max theta_req: {k_results['theta_req'].max():.3f}")
        
        # Best combination
        best_row = successful.loc[successful['theta_req'].idxmin()]
        print(f"\n{'=' * 80}")
        print(f"🏆 Best combination (lowest theta_req):")
        print(f"  Instance: {best_row['instance_id']}")
        print(f"  k_learn: {best_row['k_learn_used']:.2f}")
        print(f"  theta_req: {best_row['theta_req']:.3f}")
    
    print("=" * 100 + "\n")
    
    return results_df


if __name__ == "__main__":
    results_df = main()
