import pandas as pd
import math
import copy
import os
import glob
import pickle
from datetime import datetime
from CG import ColumnGeneration
from logging_config import setup_multi_level_logging, get_logger
from Utils.Generell.instance_setup import generate_patient_data_log
from loop_main import solve_instance

logger = get_logger(__name__)

def generate_master_data(seed, T, D_focus, pttr, app_data_for_gen):
    """
    Generates the master patient data ONCE using the seed.
    Returns the dictionary required for 'pre_generated_data'.
    """
    # Unpack necessary parameters for generation
    W_on = app_data_for_gen['W_on'][0]
    W_off = app_data_for_gen['W_off'][0]
    daily = app_data_for_gen['daily'][0]
    
    # Call the generation function directly
    # Note: We pass T directly. The result contains T as a list of IDs.
    data_tuple = generate_patient_data_log(
        T=T,
        D_focus=D_focus,
        W_on=W_on,
        W_off=W_off,
        daily=daily,
        pttr_scenario=pttr,
        seed=seed,
        plot_show=False,
        verbose=False
    )
    
    # Unpack based on CG.py expectation (11 items)
    # Req, Entry, Max_t, P, D, D_Ext, D_Full, T_ids, M_p, W_coeff, DRG
    
    # Safely handle if it returns 10 or 11 items
    if len(data_tuple) == 11:
        Req, Entry, Max_t, P, D, D_Ext, D_Full, T_ids, M_p, W_coeff, DRG = data_tuple
    else:
        # Fallback if DRG is missing (based on the truncated view I had)
        Req, Entry, Max_t, P, D, D_Ext, D_Full, T_ids, M_p, W_coeff = data_tuple
        DRG = {} # Placeholder if not returned
        logger.warning("DRG data not returned by generate_patient_data_log!")

    master_data = {
        'Req': Req,
        'Entry': Entry,
        'Max_t': Max_t,
        'P': P,
        'D': D,
        'D_Ext': D_Ext,
        'D_Full': D_Full,
        'T': T_ids,
        'M_p': M_p,
        'W_coeff': W_coeff,
        'DRG': DRG
    }
    
    return master_data

def stress_test_instance(instance_params, results_path):
    """
    Runs the stress test for a single instance.
    """
    seed = instance_params['seed']
    T = instance_params['T'] # Original T
    D_focus = instance_params['D_focus']
    pttr = instance_params['pttr']
    instance_id = instance_params['instance_id']
    
    logger.info(f"Processing Instance {instance_id}: Seed={seed}, T={T}, D_focus={D_focus}")

    # Standard app data for generation
    # We use these values for consistency with loop_main
    default_app_data = {
        'W_on': [5],
        'W_off': [2],
        'daily': [4],
        'learn_type': [0], # Baseline
        'theta_base': [0.3],
        'lin_increase': [0.05],
        'k_learn': [1.5],
        'infl_point': [4],
        'MS': [5],
        'MS_min': [2]
    }

    # ==========================================
    # 1. Generate Master Data (Immutable)
    # ==========================================
    master_data = generate_master_data(seed, T, D_focus, pttr, default_app_data)
    
    # ==========================================
    # 2. Baseline Run (Human Only)
    # ==========================================
    print(f"   -> Running Baseline (T={T})...")
    logger.info(f"  -> Running Baseline (Human Only, T={T})...")
    
    # Explicitly set learn_type to 0 for baseline
    baseline_app_data = copy.deepcopy(default_app_data)
    baseline_app_data['learn_type'] = [0]
    
    try:
        baseline_res = solve_instance(
            seed=seed,
            D_focus=D_focus,
            pttr=pttr,
            T=T,
            allow_gaps=False,
            use_warmstart=True,
            learn_type=0, 
            app_data_overrides=None,
            pre_generated_data=master_data # PASS MASTER DATA
        )
        ub_baseline = baseline_res['final_ub']
        los_baseline = baseline_res.get('sum_focus_los', 0)
        baseline_focus_los_dict = baseline_res.get('focus_los', {})
        
        print(f"      Baseline UB: {ub_baseline:.2f}, LOS: {los_baseline}")
        print(f"      Baseline Focus LOS Detail: {baseline_focus_los_dict}")
        logger.info(f"  -> Baseline UB: {ub_baseline}, LOS: {los_baseline}")
        logger.info(f"  -> Baseline Focus LOS: {baseline_focus_los_dict}")
        
    except Exception as e:
        logger.error(f"  -> Baseline Failed: {e}")
        return {'error': 'Baseline Failed', 'details': str(e)}

    # ==========================================
    # 3. Create Stress Data (Remove 1 Therapist)
    # ==========================================
    # Strategy: Remove therapist with MOST pre-assigned sessions.
    
    original_T_list = master_data['T']
    pre_x_baseline = master_data.get('pre_x', {}) # {(p, t, d): count}
    
    therapist_usage = {t: 0 for t in original_T_list}
    for (p, t, d), count in pre_x_baseline.items():
        if t in therapist_usage:
            therapist_usage[t] += count
            
    # Select therapist with MAX usage
    t_removed = max(therapist_usage, key=therapist_usage.get)
    usage_count = therapist_usage[t_removed]
    
    # Create new T list
    new_T_list = [t for t in original_T_list if t != t_removed]
    
    logger.info(f"  -> Selecting Therapist to Remove: ID {t_removed} (Usage: {usage_count} sessions). Remaining T: {len(new_T_list)}")
    
    stress_data = copy.deepcopy(master_data)
    stress_data['T'] = new_T_list
    
    # Remove availability for t_removed from Max_t
    # Max_t keys are (t, d)
    new_Max_t = {k: v for k, v in stress_data['Max_t'].items() if k[0] != t_removed}
    stress_data['Max_t'] = new_Max_t
    
    # ==========================================
    # 3.1 PRE-PATIENT FALLBACK: Reduce Req for removed therapist's sessions
    # ==========================================
    # Goal: If a pre-patient was scheduled with t_removed in baseline,
    # we CANCEL those sessions (reduce demand) so they don't block the new schedule.
    
    pre_x_baseline = master_data.get('pre_x', {}) # {(p, t, d): count}
    
    # Identify sessions with t_removed
    cancelled_sessions_count = {} # {p: count}
    
    for (p, t, d), count in pre_x_baseline.items():
        if t == t_removed:
            if p not in cancelled_sessions_count:
                cancelled_sessions_count[p] = 0
            cancelled_sessions_count[p] += count
            
    if cancelled_sessions_count:
        logger.info(f"  -> Fallback: Reducing demand for {len(cancelled_sessions_count)} pre-patients due to removed therapist {t_removed}")
        for p, reduction in cancelled_sessions_count.items():
            # Reduce Requirement in stress_data
            # stress_data has 'Req' dict {p: req}
            if p in stress_data['Req']:
                old_req = stress_data['Req'][p]
                new_req = max(0, old_req - reduction) # Don't go below 0
                stress_data['Req'][p] = new_req
                print(f"     Patient {p}: Req reduced {old_req} -> {new_req} (removed {reduction} sessions)") 
            else:
                logger.warning(f"     Patient {p} not found in Req dict, cannot reduce demand.")

    # ==========================================
    # 3.2 REMOVE IMPOSSIBLE PATIENTS (Zero Capacity Days)
    # ==========================================
    # If a patient enters on a day where TOTAL capacity is 0 (due to removal),
    # they must be removed entirely, otherwise pre-check fails.
    
    # Calculate daily capacity of REMAINING therapists
    daily_capacity = {}
    for d in master_data['D_Full']: # Check all relevant days
        cap = sum(new_Max_t.get((t, d), 0) for t in new_T_list)
        daily_capacity[d] = cap
        
    impossible_entry_days = [d for d, cap in daily_capacity.items() if cap == 0]
    
    patients_to_remove = []
    
    # Check all patients in stress_data
    # We need to iterate over a copy of the list
    for p in list(stress_data['P']):
        # Check 1: Enters on zero-capacity day?
        p_entry = stress_data['Entry'].get(p)
        if p_entry in impossible_entry_days:
            patients_to_remove.append(p)
            print(f"     Patient {p}: REMOVED (Enters on day {p_entry} with 0 capacity)")
            continue
            
        # Check 2: Req reduced to 0?
        if stress_data['Req'].get(p, 0) == 0:
             patients_to_remove.append(p)
             print(f"     Patient {p}: REMOVED (Requirement reduced to 0)")
             continue
             
    # Perform Removal
    if patients_to_remove:
        logger.info(f"  -> Removing {len(patients_to_remove)} impossible patients from stress data.")
        stress_data['P'] = [p for p in stress_data['P'] if p not in patients_to_remove]
        stress_data['P_Pre'] = [p for p in stress_data['P_Pre'] if p not in patients_to_remove]
        # P_F/Post/Join are usually defined later or not relevant for pre-check unless they invoke pre-check?
        # Pre-check iterates 'P'. 
        
        # Clean up dicts (optional but good practice)
        for p in patients_to_remove:
            stress_data['Entry'].pop(p, None)
            stress_data['Req'].pop(p, None)
            stress_data['Nr_agg'].pop(p, None) # Important for pre-check demand calc!
    
    # ==========================================
    # 4. Stress Test Loop (Find Break-Even for each k)
    # ==========================================
    
    # Define learning parameter search space 
    # We vary k_learn (speed) and theta_base (start competence)
    k_values = [0, 1.0, 1.5, 2.0] # 0 means no learning, higher means faster learning
    theta_values = [0.2]

    
    results_log = []
    
    # Outer Loop: Learning Speed (k)
    for k in k_values:
        
        logger.info(f"\nExample {instance_params['instance_id']}: Testing Learning Speed k={k}...")
        found_break_even_for_k = False
        
        # Inner Loop: Base Competence (Theta)
        for theta in theta_values:
            logger.info(f"  -> Testing k={k}, Theta_base={theta}")
            
            # Update learning params
            stress_app_data_overrides = {
                'theta_base': theta,
                'lin_increase': 0.0, # Disable linear
                'k_learn': k,        # VARY K
                'infl_point': 4
            }
            
            try:
                # Run solver with stress data and new learning params
                res = solve_instance(
                    seed=seed,
                    D_focus=D_focus,
                    pttr=pttr,
                    T=len(new_T_list), # Pass new T count
                    allow_gaps=False,
                    use_warmstart=True,
                    learn_type='sigmoid',
                    app_data_overrides=stress_app_data_overrides,
                    pre_generated_data=stress_data # PASS STRESS DATA
                )
                
                # Check Results
                ub_stress = res.get('final_ub')
                los_stress = res.get('sum_focus_los')
                focus_los_dict = res.get('focus_los', {})
                
                if ub_stress is not None:
                    # Calculate Gap based on LOS (as requested)
                    gap_los = los_stress - los_baseline
                    
                    # Recovered if Stress LOS <= Baseline LOS
                    recovered = los_stress <= los_baseline + 1e-5
                    feasible = True
                    
                    status_str = f"Result: LOS={los_stress:.2f} (Base={los_baseline:.2f}, Gap={gap_los:.2f}). Feasible? {feasible}. Recovered? {recovered}"
                    print(f"      {status_str}")
                    print(f"      Focus LOS Detail: {focus_los_dict}") # Print dict
                    logger.info(f"     -> {status_str}")
                    logger.info(f"     -> Focus LOS: {focus_los_dict}") 
                    
                    # Store Result
                    results_log.append({
                        'instance_id': instance_params['instance_id'],
                        'k_learn': k,
                        'theta_base': theta,
                        'ub_baseline': ub_baseline,
                        'los_baseline': los_baseline,
                        'ub_stress': ub_stress,
                        'los_stress': los_stress,
                        'gap_los': gap_los,
                        'recovered': recovered,
                        'feasible': True,
                        't_removed': t_removed,
                        'focus_los_dict': str(focus_los_dict) # Save as string for Excel
                    })
                    
                    if recovered and not found_break_even_for_k:
                        found_break_even_for_k = True
                        print(f"      >>> BREAK-EVEN FOUND for k={k} at Theta={theta}! <<<")
                        logger.info(f"     -> BREAK-EVEN FOUND for k={k} at Theta={theta}!")
                        # We can stop testing higher thetas for this k if we only want the break-even point
                        break 
                else:
                    # Solver returned no solution (Infeasible in optimization)
                    print(f"      Result: INFEASIBLE (Solver)")
                    logger.info(f"     -> Result: INFEASIBLE (Solver)")
                    results_log.append({
                        'instance_id': instance_params['instance_id'],
                        'k_learn': k,
                        'theta_base': theta,
                        'ub_baseline': ub_baseline,
                        'los_baseline': los_baseline,
                        'ub_stress': None,
                        'los_stress': None,
                        'gap_los': None,
                        'recovered': False,
                        'feasible': False,
                        'error': 'Solver Infeasible',
                        't_removed': t_removed
                    })

            except Exception as e:
                print(f"      Result: FAILED (Error: {e})")
                logger.error(f"     -> Stress Run Failed at k={k}, Theta={theta}: {e}")
                results_log.append({
                        'instance_id': instance_params['instance_id'],
                        'k_learn': k,
                        'theta_base': theta,
                        'ub_baseline': ub_baseline,
                        'los_baseline': los_baseline,
                        'ub_stress': None,
                        'los_stress': None,
                        'gap_los': None,
                        'recovered': False,
                        'feasible': False,
                        'error': str(e),
                        't_removed': t_removed
                    })
    
    return results_log

def main():
    setup_multi_level_logging(base_log_dir='logs_stress', enable_console=True)
    
    # 1. Load latest instances
    import glob
    import os
    
    instances_dir = 'results/instances'
    excel_files = glob.glob(os.path.join(instances_dir, '*.xlsx'))
    
    if not excel_files:
        logger.error(f"No instance files found in {instances_dir}.")
        return

    newest_excel = max(excel_files, key=os.path.getmtime) # Use newest file
    logger.info(f"Loading instances from: {newest_excel}")
    
    df_instances = pd.read_excel(newest_excel)
    
    final_results = []
    
    # 2. Iterate instances
    total_instances = len(df_instances)
    print(f"\nStarting Stress Test on {total_instances} instances...\n" + "="*50)

    for idx, row in df_instances.iterrows():
        print(f"\n[Progress: {idx + 1}/{total_instances}] Processing Instance...")
        
        instance_params = {
            'instance_id': row.get('instance_id', f'inst_{idx}'),
            'seed': int(row['seed']),
            'D_focus': int(row['D_focus_count']),
            'pttr': row.get('pttr', 'medium'),
            'T': int(row.get('T_count', 4)) # Default to 4 if missing
        }
        
        res_list = stress_test_instance(instance_params, 'results/stress_test')
        final_results.extend(res_list) # Flatten results immediately
        
    # 3. Save Summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('results/stress_test', exist_ok=True)
    
    df_summary = pd.DataFrame(final_results)
    out_file = f'results/stress_test/stress_summary_{timestamp}.xlsx'
    
    try:
        df_summary.to_excel(out_file, index=False)
        logger.info(f"Stress test complete. Summary saved to {out_file}")
        print(f"\n[DONE] Stress test complete. Summary saved to {out_file}")
    except Exception as e:
        logger.error(f"Failed to save summary Excel: {e}")
        print(f"\n[ERROR] Failed to save summary Excel: {e}")

if __name__ == "__main__":
    main()
