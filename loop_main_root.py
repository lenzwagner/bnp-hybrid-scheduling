import pandas as pd
import math
from CG import ColumnGeneration
from branch_and_price import BranchAndPrice
from logging_config import setup_multi_level_logging, get_logger
from Utils.derived_vars import compute_derived_variables
from Utils.extra_values import calculate_extra_metrics
import pickle
from datetime import datetime
import time
import os

logger = get_logger(__name__)

def solve_instance(seed, D_focus, pttr='medium', T=2, allow_gaps=False, use_warmstart=True, dual_smoothing_alpha=None, learn_type=0, instance_id=None):
    """
    Solve a single instance's ROOT NODE with given seed, D_focus, pttr, and T.
    Returns a dictionary with instance parameters and root node results.
    """
    
    logger.info("=" * 100)
    logger.info(f"SOLVING ROOT INSTANCE: seed={seed}, D_focus={D_focus}, learn_type={learn_type}")
    logger.info("=" * 100)
    
    # ===========================
    # CONFIGURATION PARAMETERS
    # ===========================

    # Learning parameters
    app_data = {
        'learn_type': [learn_type],
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

    # Algorithm parameters
    dual_improvement_iter = 20
    dual_stagnation_threshold = 1e-5
    max_itr = 100
    threshold = 1e-5

    # Additional settings
    show_plots = False
    pricing_filtering = True
    therapist_agg = False
    learn_method = 'pwl'

    logger.info(f"Configuration: seed={seed}, T={T}, D_focus={D_focus}, pttr={pttr}")

    # Branch-and-Price settings
    use_branch_and_price = True
    branching_strategy = 'mp'
    search_strategy = 'bfs'
    
    # Parallelization settings
    use_parallel_pricing = True
    n_pricing_workers = min(os.cpu_count(), 4) if use_parallel_pricing else 1
    
    # Output settings
    save_lps = False  # Set to False for batch runs
    verbose_output = False
    
    # Solver settings
    deterministic = False
    
    # Define labeling specs
    labeling_spec = {
        'use_labeling': True, 
        'max_columns_per_iter': 100,
        # Pricing parallelization
        'use_parallel_pricing': use_parallel_pricing,
        'n_pricing_workers': n_pricing_workers,
        'use_parallel_tree': False,
        'n_tree_workers': 1,
        # Other settings
        'debug_mode': False,
        'use_apriori_pruning': False, 
        'use_pure_dp_optimization': True,
        'use_persistent_pool': True,
        'use_heuristic_pricing': False, 
        'heuristic_max_labels': 20, 
        'use_relaxed_history': False,
        'use_numba_labeling': True,
        'allow_gaps': allow_gaps, 
        'use_label_recycling': False
    }

    # ===========================
    # SETUP CG SOLVER
    # ===========================

    cg_solver = ColumnGeneration(
        seed=seed,
        app_data=app_data,
        T=T,
        D_focus=D_focus,
        max_itr=max_itr,
        threshold=threshold,
        pttr=pttr,
        show_plots=show_plots,
        pricing_filtering=pricing_filtering,
        therapist_agg=therapist_agg,
        max_stagnation_itr=dual_improvement_iter,
        stagnation_threshold=dual_stagnation_threshold,
        learn_method=learn_method,
        save_lps=save_lps,
        verbose=verbose_output,
        deterministic=deterministic,
        use_warmstart=use_warmstart,
        dual_smoothing_alpha=dual_smoothing_alpha
    )

    cg_solver.setup()

    # ===========================
    # SOLVE ROOT NODE
    # ===========================

    bnp_solver = BranchAndPrice(
        cg_solver,
        branching_strategy=branching_strategy,
        search_strategy=search_strategy,
        verbose=verbose_output,
        ip_heuristic_frequency=0, # Disable heuristic for root only
        early_incumbent_iteration=None,
        save_lps=save_lps,
        label_dict=labeling_spec
    )
    
    # Manually run root node sequence
    start_time = time.time()
    
    bnp_solver.create_root_node()
    lp_bound, is_integral, frac_info, root_lambdas = bnp_solver.solve_root_node()
    
    root_time = time.time() - start_time
    
    bnp_solver._cleanup_pricing_pool()

    # ===========================
    # BUILD INSTANCE DATA DICTIONARY
    # ===========================
    instance_data = {
        # Instance parameters
        'seed': seed,
        'D_focus': D_focus,
        'pttr': pttr,
        'learn_type': app_data['learn_type'][0],
        'OnlyHuman': 1 if app_data['learn_type'][0] == 0 else 0,
        'T': len(cg_solver.T),
        
        # Root Results
        'root_lp': lp_bound,
        'root_integral': is_integral,
        'root_time': root_time,
        
        # Instance data stats (Optional)
        'total_columns': len(root_lambdas) if root_lambdas else 0,
    }
    
    return instance_data


def main_loop():
    """
    Main loop to solve instances' root nodes from the newest Excel file.
    """
    
    # ===========================
    # LOGGING CONFIGURATION
    # ===========================
    print_all_logs = False
    setup_multi_level_logging(base_log_dir='logs', enable_console=True, print_all_logs=print_all_logs)
    
    # ===========================
    # LOAD SCENARIOS FROM EXCEL
    # ===========================
    import glob
    
    # Find the newest Excel file in results/instances
    instances_dir = 'results/instances'
    excel_files = glob.glob(os.path.join(instances_dir, '*.xlsx'))
    
    if not excel_files:
        raise FileNotFoundError(f"No Excel files found in {instances_dir}")
    
    # Get the newest file by modification time
    newest_excel = max(excel_files, key=os.path.getmtime)
    
    print("\n" + "=" * 100)
    print(f" LOADING SCENARIOS FROM: {newest_excel} ".center(100, "="))
    print("=" * 100 + "\n")
    
    # Load scenarios
    scenarios_df = pd.read_excel(newest_excel)
    
    # Dictionary to store all results
    results_dict = {}
    
    # DataFrame to collect all results
    results_df = pd.DataFrame()
    
    # ===========================
    # MAIN LOOP
    # ===========================
    total_instances = len(scenarios_df)
    
    print("\n" + "=" * 100)
    print(f" BATCH RUN (ROOT ONLY): {total_instances} instances ".center(100, "="))
    print("=" * 100 + "\n")
    
    for idx, row in scenarios_df.iterrows():
        current_instance = idx + 1
        instance_id = row.get('instance_id', f'instance_{idx}')
        seed = int(row['seed'])
        D_focus = int(row['D_focus_count'])
        pttr = row.get('pttr', 'medium')
        T = int(row.get('T_count', 2))
        
        print("\n" + "=" * 100)
        print(f" Instance {current_instance}/{total_instances}: {instance_id} ".center(100, "="))
        print(f" seed={seed}, D_focus={D_focus}, pttr={pttr}, T={T} ".center(100, "="))
        print("=" * 100 + "\n")
        
        # Iterate over learn_types
        learn_types = ['sigmoid']
        
        for lt in learn_types:
            if lt == 'sigmoid':
                lt_suffix = '_sigmoid'
            elif lt == 'linear':
                lt_suffix = '_linear'
            else:
                lt_suffix = '_0'
            current_instance_id = f"{instance_id}{lt_suffix}"
            
            print(f"\n Solving ROOT for learn_type: {lt} (ID: {current_instance_id})")
            
            try:
                # Solve instance ROOT
                instance_data = solve_instance(
                    seed=seed,
                    D_focus=D_focus,
                    pttr=pttr,
                    T=T,
                    allow_gaps=False,
                    use_warmstart=True,
                    dual_smoothing_alpha=None,
                    learn_type=lt,
                    instance_id=current_instance_id
                )
                
                # Add instance_id to the results
                instance_data['instance_id'] = current_instance_id
                instance_data['scenario_nr'] = row.get('scenario_nr', idx)
                instance_data['original_instance_id'] = instance_id
                
                # Store in dictionary with instance_id as key
                results_dict[current_instance_id] = instance_data
                
                # Add to DataFrame
                results_df = pd.concat([results_df, pd.DataFrame([instance_data])], ignore_index=True)
                
                # Print summary
                print(f"\n✓ Instance {current_instance}/{total_instances} [{lt}] ROOT completed:")
                print(f"  - Instance ID: {current_instance_id}")
                print(f"  - Root LP: {instance_data['root_lp']:.6f}")
                print(f"  - Root Time: {instance_data['root_time']:.2f}s")
                print(f"  - Root Integral: {instance_data['root_integral']}")
                
            except Exception as e:
                print(f"\n✗ Instance {current_instance}/{total_instances} [{lt}] FAILED:")
                print(f"  Error: {str(e)}")
                logger.error(f"Instance {current_instance_id} failed: {str(e)}", exc_info=True)
                
                # Store failure indication
                results_dict[current_instance_id] = {
                    'instance_id': current_instance_id,
                    'original_instance_id': instance_id,
                    'error': str(e),
                    'status': 'FAILED'
                }
    
    # ===========================
    # SAVE RESULTS
    # ===========================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save to Excel
    excel_filename = f"results/cg/results_root_only_{timestamp}.xlsx"
    try:
        os.makedirs('results/cg', exist_ok=True)
        results_df.to_excel(excel_filename, index=False)
        print(f"\n✓ Results saved to {excel_filename}")
    except Exception as e:
        print(f"\n✗ Could not save Excel: {e}")
    
    # Print summary table
    print("\nResults Summary Table (Root Only):")
    print(f"{'Instance ID':<30} {'Seed':<8} {'D_focus':<10} {'PTTR':<10} {'Root LP':<15} {'Time (s)':<12}")
    print("-" * 100)
    
    for instance_id, data in results_dict.items():
        if data.get('status') == 'FAILED':
             print(f"{instance_id:<30} FAILED")
        else:
            lp = f"{data.get('root_lp', 'N/A'):.2f}" if data.get('root_lp') else "N/A"
            time_val = f"{data.get('root_time', 0):.2f}" if data.get('root_time') else "N/A"
            seed_val = data.get('seed', 'N/A')
            d_focus_val = data.get('D_focus', 'N/A')
            pttr_val = data.get('pttr', 'N/A')
            print(f"{instance_id:<30} {seed_val:<8} {d_focus_val:<10} {pttr_val:<10} {lp:<15} {time_val:<12}")
            
    print("=" * 100 + "\n")

if __name__ == "__main__":
    main_loop()
