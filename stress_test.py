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
        logger.info(f"  -> Baseline UB: {ub_baseline}")
        
    except Exception as e:
        logger.error(f"  -> Baseline Failed: {e}")
        return {'error': 'Baseline Failed', 'details': str(e)}

    # ==========================================
    # 3. Create Stress Data (Remove 1 Therapist)
    # ==========================================
    # Remove the last therapist in the list
    original_T_list = master_data['T']
    t_removed = original_T_list[-1]
    new_T_list = original_T_list[:-1] # All except last
    
    logger.info(f"  -> Removing Therapist {t_removed}. New T count: {len(new_T_list)}")
    
    stress_data = copy.deepcopy(master_data)
    stress_data['T'] = new_T_list
    
    # Remove availability for t_removed from Max_t
    # Max_t keys are (t, d)
    new_Max_t = {k: v for k, v in stress_data['Max_t'].items() if k[0] != t_removed}
    stress_data['Max_t'] = new_Max_t
    
    # ==========================================
    # 4. Stress Test Loop (Find Break-Even)
    # ==========================================
    
    # Define learning parameter search space 
    # We will vary theta_base for sigmoid/exp learning
    # theta_base values to test: 0.3 up to 0.9
    theta_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    best_config = None
    found_break_even = False
    
    results_log = []
    
    for theta in theta_values:
        logger.info(f"  -> Testing Compensation: Theta_base = {theta}")
        
        # Setup learning params
        stress_app_data_overrides = {
            'learn_type': ['sigmoid'], # Use sigmoid learning
            'theta_base': [theta],
            'k_learn': [1.5], # Keep constant or vary if needed
            'infl_point': [4]
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
                pre_generated_data=stress_data # PASS STRESS DATA (Reduced T, same patients)
            )
            
            ub_stress = res['final_ub']
            gap = ub_baseline - ub_stress # Positive if Stress is Better (Lower Cost)
            
            log_entry = {
                'theta': theta,
                'ub_stress': ub_stress,
                'ub_baseline': ub_baseline,
                'is_feasible': True,
                'recovered': ub_stress <= ub_baseline
            }
            results_log.append(log_entry)
            
            logger.info(f"     Result: UB_stress={ub_stress} (Base={ub_baseline}). Recovered? {ub_stress <= ub_baseline}")
            
            if ub_stress <= ub_baseline:
                found_break_even = True
                best_config = theta
                logger.info(f"     -> Break-even found at Theta={theta}!")
                break # Stop searching
                
        except Exception as e:
            logger.error(f"     -> Stress Run Failed at Theta={theta}: {e}")
            results_log.append({'theta': theta, 'error': str(e), 'is_feasible': False})
            
    return {
        'instance_id': instance_id,
        'baseline_ub': ub_baseline,
        'removed_therapist': t_removed,
        'break_even_theta': best_config,
        'found_break_even': found_break_even,
        'logs': results_log
    }

def main():
    setup_multi_level_logging(base_log_dir='logs_stress', enable_console=True)
    
    # 1. Load latest instances
    # 1. Load latest instances
    # Using specific file path found in verification
    instances_dir = 'results/comp_study'
    excel_files = glob.glob(os.path.join(instances_dir, 'instances_computational.xlsx'))
    
    if not excel_files:
        logger.error(f"No instance files found in {instances_dir}.")
        return

    newest_excel = excel_files[0] # Use the specific file found
    logger.info(f"Loading instances from: {newest_excel}")
    
    df_instances = pd.read_excel(newest_excel)
    
    final_results = []
    
    # 2. Iterate instances
    for idx, row in df_instances.iterrows():
        instance_params = {
            'instance_id': row.get('instance_id', f'inst_{idx}'),
            'seed': int(row['seed']),
            'D_focus': int(row['D_focus_count']),
            'pttr': row.get('pttr', 'medium'),
            'T': int(row.get('T_count', 4)) # Default to 4 if missing
        }
        
        res = stress_test_instance(instance_params, 'results/stress_test')
        final_results.append(res)
        
    # 3. Save Summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('results/stress_test', exist_ok=True)
    
    summary_data = []
    for r in final_results:
        if 'error' in r:
            summary_data.append(r)
        else:
            summary_data.append({
                'instance_id': r['instance_id'],
                'baseline_ub': r['baseline_ub'],
                'removed_therapist': r['removed_therapist'],
                'break_even_theta': r['break_even_theta'],
                'found_break_even': r['found_break_even']
            })
            
    df_summary = pd.DataFrame(summary_data)
    out_file = f'results/stress_test/stress_summary_{timestamp}.xlsx'
    df_summary.to_excel(out_file, index=False)
    logger.info(f"Stress test complete. Summary saved to {out_file}")

if __name__ == "__main__":
    main()
