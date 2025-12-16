from CG import ColumnGeneration
from branch_and_price import BranchAndPrice
from logging_config import setup_multi_level_logging, get_logger

logger = get_logger(__name__)

def main():
    """
    Main function to run Column Generation or Branch-and-Price algorithm.

    TODOs:
    1. Bitmasking for History Tracking
       What changed: Instead of using Python tuples (e.g., (0, 1, 0)) to track the recent schedule history for the rolling window constraints, I implemented Integer Bitmasking.
       How it works: The history is now stored as a single integer. We use bitwise operations (left shift <<, bitwise AND &, bitwise OR |) to update the history and check for feasibility.
       Benefit:
       - Speed: Bitwise operations are natively supported by the CPU and are extremely fast compared to creating, slicing, and concatenating tuple objects.
       - Memory: Storing a single integer is much more memory-efficient than storing tuple objects, reducing overhead.
       - Hashing: Integers can be hashed and compared much faster than tuples, speeding up the dominance checks in the dynamic programming buckets.

    2. Transposed Constraint Lookup (Hashing)
       What changed: I optimized how the algorithm checks for "forbidden" moves (branching constraints). Previously, the algorithm scanned the entire list of constraints linearly (O(N)) for every single step. I replaced this with a Hash Map (Dictionary) lookup.
       How it works: Before running the labeling algorithm, we pre-process the constraints into a "transposed" lookup table: map[(worker, time)] -> [list of specific constraints].
       Benefit:
       - O(1) Complexity: Instead of checking every constraint, the algorithm now performs a single hash lookup to find only the constraints that apply to the current worker and time step.
       - Scalability: This guarantees that the algorithm does not slow down significantly even as the number of branching constraints grows large deep in the search tree.

    3. Unified Labeling Configuration
       Problem: The labeling algorithm ('solve_pricing_for_profile_bnp' and 'run_labeling_algorithm') currently accepts 15+ individual arguments. This makes the signature brittle and hard to extend.
       Solution: Encapsulate all static configuration (workers, max_time, ms, theta_lookup, etc.) into a single `LabelingConfig` object or dictionary.
       Implementation:
       - Define a config structure (dataclass or dict).
       - Create this config ONCE in `BranchAndPrice.__init__`.
       - Pass this single object to `_parallel_pricing_worker` and then to the labeling function.
       - Unpack it inside the labeling algorithm.

    4. Heuristic Pricing (Multi-Phase Pricing)
       Problem: The exact labeling algorithm always finds the optimal column (most negative reduced cost), but this is expensive. In Column Generation, ANY column with negative reduced cost is sufficient to improve the master.
       Solution: Implement a two-phase pricing approach:
       - Phase 1 (Heuristic): Run labeling with aggressive pruning (e.g., keep max 10-20 labels per bucket, relaxed dominance). If rc < 0 found, add and stop.
       - Phase 2 (Exact): Only if heuristic fails, run full exact labeling to prove optimality or find hard-to-reach columns.
       Benefit: Massively speeds up early CG iterations. Most columns can be found heuristically.

    5. A* Search with Backward Heuristic
       Problem: Current bound pruning uses a simple "fastest completion" estimate. This can be significantly tightened.
       Solution: Compute a backward heuristic h(state_t) that estimates the MINIMUM possible cost from time t to the end, assuming ideal conditions (max AI efficiency, no capacity conflicts).
       Implementation:
       - Precompute or dynamically calculate h(state_t) based on remaining required progress and theta lookup.
       - In dominance: if Cost(state) + h(state) - gamma >= 0, prune immediately.
       - The tighter h(state), the more labels can be pruned early.
       Benefit: Reduces the number of states explored by orders of magnitude in deep trees.

    6. State-Space Relaxation (ng-route inspired)
       Problem: The rolling window history vector (h_t) causes exponential state space growth (2^(MS-1) states).
       Solution: In heuristic pricing phase, partially or fully relax the rolling window constraint:
       - Solve without tracking h_t (or with reduced MS).
       - If found column happens to satisfy constraint (1p): add it.
       - If violated: discard or attempt local repair.
       Benefit: Dramatically reduces state space in heuristic phase. Most feasible columns will naturally satisfy the constraint.
       Note: This is experimental and requires careful validation.

    7. Bucket Refinement by Progress
       Problem: Current bucketing groups states by (ai_count, hist, zeta). If buckets are too large, pairwise dominance checks become O(N^2) expensive within each bucket.
       Solution: Subdivide buckets by discretized progress omega_t (e.g., round to nearest 0.1).
       - Only check dominance between labels with similar progress.
       - Labels far apart in progress rarely dominate each other anyway.
       Benefit: Reduces dominance check overhead in buckets with many labels.
    """
    # ===========================
    # LOGGING CONFIGURATION
    # ===========================
    # Setup multi-level logging: separate files for DEBUG, INFO, WARNING, ERROR
    # Set print_all_logs=True to also print all log levels to console (not just PRINT level)
    print_all_logs = False # Set to True to see all logger output on console
    setup_multi_level_logging(base_log_dir='logs', enable_console=True, print_all_logs=print_all_logs)




    logger.info("=" * 100)
    logger.info("STARTING BRANCH-AND-PRICE SOLVER")
    logger.info("=" * 100)

    # ===========================
    # CONFIGURATION PARAMETERS
    # ===========================

    # Random seed
    seed = 13

    # Learning parameters
    app_data = {
        'learn_type': ['lin'],  # Learning curve type: 'exp', 'sigmoid', 'lin', or numeric value
        'theta_base': [0.02],  # Base effectiveness
        'lin_increase': [0.01],  # Linear increase rate (for 'lin' type)
        'k_learn': [0.01],  # Learning rate (for 'exp' and 'sigmoid')
        'infl_point': [2],  # Inflection point (for 'sigmoid')
        'MS': [5],  # Maximum session window
        'MS_min': [2],  # Minimum sessions in window
        'W_on': [6],  # Work days per week
        'W_off': [1],  # Days off per week
        'daily': [4]  # Daily capacity per therapist
    }

    # Instance parameters
    T = 3  # Number of therapists
    D_focus = 5  # Number of focus days

    # Algorithm parameters
    dual_improvement_iter = 20  # Max Iterations without dual improvement
    dual_stagnation_threshold = 1e-5
    max_itr = 100  # Maximum CG iterations
    threshold = 1e-5  # Convergence threshold

    # Additional settings
    pttr = 'medium'  # Patient-to-therapist ratio: 'low', 'medium', 'high'
    show_plots = False  # Show visualization plots
    pricing_filtering = True  # Enable pricing filter
    therapist_agg = False  # Enable therapist aggregation
    learn_method = 'pwl'

    # Logger info
    logger.info(f"Configuration: seed={seed}, T={T}, D_focus={D_focus}, pttr={pttr}")

    # Branch-and-Price settings
    use_branch_and_price = True  # Set to False for standard CG
    branching_strategy = 'mp'  # 'mp' for MP variable branching, 'sp' for SP variable branching
    search_strategy = 'bfs' # 'dfs' for Depth-First, 'bfs' for Best-Fit-Search
    
    # Parallelization settings
    use_parallel_pricing = True  # Enable parallel pricing (requires use_labeling=True)
    import os
    n_pricing_workers = min(os.cpu_count(), 4) if use_parallel_pricing else 1  # Auto-detect CPUs, max 4

    # Output settings
    save_lps = True # Set to True to save LP and SOL files
    verbose_output = False # Set to False to suppress all non-final output

    # Solver settings
    deterministic = False  # Set to True for deterministic solver behavior (single-threaded, barrier method)

    # Visualization settings
    visualize_tree = False  # Enable tree visualization
    tree_layout = 'hierarchical'  # 'hierarchical' or 'radial'
    detailed_tree = False  # Show detailed info on nodes
    save_tree_path = 'bnp_tree.png'  # Path to save (None to not save)

    # ===========================
    # CONFIGURATION SUMMARY
    # ===========================

    if verbose_output:
        print("\n" + "=" * 100)
        print(" STARTING SETUP ".center(100, "="))
        print("=" * 100)
        print(f"\nConfiguration:")
        print(f"  - Mode: {'Branch-and-Price' if use_branch_and_price else 'Column Generation'}")
        if use_branch_and_price:
            print(f"  - Branching Strategy: {branching_strategy.upper()}")
            print(f"  - Search Strategy: {'Depth-First (DFS)' if search_strategy == 'dfs' else 'Best-Fit (BFS)'}")
            print(f"  - Parallel Pricing: {'Enabled' if use_parallel_pricing else 'Disabled'}")
            if use_parallel_pricing:
                print(f"  - Pricing Workers: {n_pricing_workers}")
        print(f"  - Seed: {seed}")
        print(f"  - Learning type: {app_data['learn_type'][0]}")
        print(f"  - Learning method: {learn_method}")
        print(f"  - Therapists: {T}")
        print(f"  - Focus days: {D_focus}")
        print(f"  - Max CG iterations: {max_itr}")
        print(f"  - Threshold: {threshold}")
        print(f"  - PTTR scenario: {pttr}")
        print(f"  - Pricing filtering: {pricing_filtering}")
        print(f"  - Save LPs: {save_lps}")
        print()

    # ===========================
    # SETUP CG SOLVER
    # ===========================

    # Create CG solver
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
        deterministic=deterministic
    )

    # Setup
    cg_solver.setup()

    # ===========================
    # SOLVE
    # ===========================

    if use_branch_and_price:
        # Branch-and-Price
        if verbose_output:
            print("\n" + "=" * 100)
            print(" INITIALIZING BRANCH-AND-PRICE ".center(100, "="))
            print("=" * 100 + "\n")

        bnp_solver = BranchAndPrice(cg_solver,
                                    branching_strategy=branching_strategy,
                                    search_strategy=search_strategy,
                                    verbose=verbose_output,
                                    ip_heuristic_frequency=5,
                                    early_incumbent_iteration=1,
                                    save_lps=save_lps,
                                    use_labeling=True,
                                    max_columns_per_iter=50,
                                    use_parallel_pricing=use_parallel_pricing,
                                    n_pricing_workers=n_pricing_workers,
                                    use_apriori_pruning=True,
                                    use_pure_dp_optimization=True,
                                    use_persistent_pool=False)
        results = bnp_solver.solve(time_limit=3600, max_nodes=300)

        # Extract optimal schedules
        if results['incumbent'] is not None:
            if verbose_output:
                print("\n" + "=" * 100)
                print(" EXTRACTING OPTIMAL SCHEDULES ".center(100, "="))
                print("=" * 100)

            optimal_schedules = bnp_solver.extract_optimal_schedules()

            # Print example schedules
            if optimal_schedules and verbose_output:
                p_focus_patients = {
                    patient_id: info
                    for patient_id, info in optimal_schedules['patient_schedules'].items()
                    if info['profile'] in cg_solver.P_F
                }

                # Print first 3 patient schedules as examples
                patient_ids = list(p_focus_patients.keys())[:3]
                for patient_id in patient_ids:
                    bnp_solver.print_detailed_schedule(
                        patient_id,
                        p_focus_patients[patient_id]
                    )

            # Export to CSV
            if verbose_output:
                bnp_solver.export_schedules_to_csv('results/optimal_schedules.csv')

            if verbose_output:
                print("\n" + "=" * 100)
                print(" SCHEDULE EXTRACTION COMPLETE ".center(100, "="))
                print("=" * 100)

        # Print CG statistics (from root node)
        if verbose_output:
            print("\n" + "=" * 100)
            print(" COLUMN GENERATION STATISTICS (ROOT NODE) ".center(100, "="))
            print("=" * 100 + "\n")
            cg_solver.print_statistics()

        # Visualize tree
        if visualize_tree:
            if verbose_output:
                print("\n" + "=" * 100)
                print(" GENERATING TREE VISUALIZATION ".center(100, "="))
                print("=" * 100 + "\n")

            import os
            os.makedirs("Pictures/Tree", exist_ok=True)

            # Academic/Thesis style visualization (publication-ready)
            bnp_solver.visualize_tree(
                academic=True,
                save_path='Pictures/Tree/tree_academic.png',
                dpi=600  # High resolution for papers
            )

            # Other options:
            # Standard hierarchical: layout='hierarchical', save_path='Pictures/Tree/tree_hierarchical.png'
            # Radial layout: layout='radial', save_path='Pictures/Tree/tree_radial.png'
            # Detailed view: detailed=True, save_path='Pictures/Tree/tree_detailed.png'
            # Custom colors: academic=True, node_color='#ADD8E6', fathomed_color='#FFB6C1'

            # Export for LaTeX:
            # bnp_solver.export_tree_tikz('Pictures/Tree/bnp_tree.tex')

    else:
        # Standard Column Generation
        results = cg_solver.solve()

    # ===========================
    # SUMMARY
    # ===========================

    print("\n" + "=" * 100)
    print(" EXECUTION SUMMARY ".center(100, "="))
    print("=" * 100)
    print(f"Completed successfully!")
    print(f"  - Mode: {'Branch-and-Price' if use_branch_and_price else 'Column Generation'}")
    print(f"  - Total time: {results['total_time']:.2f}s")

    if use_branch_and_price:
        print(f"\nBranch-and-Price Results:")
        print(f"  - Branching strategy: {branching_strategy.upper()}")
        print(f"  - Search strategy: {'Depth-First (DFS)' if search_strategy == 'dfs' else 'Best-Fit (BFS)'}")
        print(f"  - Nodes explored: {results['nodes_explored']}")
        print(f"  - Nodes fathomed: {results['nodes_fathomed']}")
        print(f"  - Nodes branched: {results.get('nodes_branched', 0)}")
        print(f"  - LP bound (LB): {results['lp_bound']:.5f}")
        if results['incumbent']:
            print(f"  - Incumbent (UB): {results['incumbent']:.5f}")
            print(f"  - Gap: {results['gap']:.5%}")
        else:
            print(f"  - Incumbent (UB): None")
        print(f"  - Integral: {results['is_integral']}")
        print(f"  - Total CG iterations (all nodes): {results['cg_iterations']}")
        print(f"  - IP solves: {results['ip_solves']}")

        # Convergence and optimal solution status
        print(f"\nAlgorithm Status:")
        if results['incumbent'] is not None:
            # Check if tree is complete (all nodes fathomed)
            if results.get('tree_complete', False):
                print(f"  ✓ TREE COMPLETE - All nodes explored")
                print(f"  ✓ OPTIMAL SOLUTION FOUND: {results['incumbent']:.5f}")
                if results['gap'] > 0:
                    print(f"  (Numerical gap: {results['gap']:.5%}, due to floating-point precision)")
            else:
                gap_threshold = 1e-4  # 0.01% gap threshold
                if results['gap'] < gap_threshold:
                    print(f"  ✓ Algorithm CONVERGED (Gap < {gap_threshold:.2%})")
                    print(f"  ✓ OPTIMAL SOLUTION FOUND: {results['incumbent']:.5f}")
                else:
                    print(f"  ! Algorithm terminated with gap: {results['gap']:.5%}")
                    print(f"  ! Best solution found: {results['incumbent']:.5f}")
                    print(f"  ! Lower bound: {results['lp_bound']:.5f}")
        else:
            print(f"  ✗ No feasible solution found")
    else:
        print(f"\nColumn Generation Results:")
        print(f"  - Iterations: {results['num_iterations']}")
        print(f"  - LP objective: {results['lp_obj']:.5f}")
        print(f"  - IP objective: {results['ip_obj']:.5f}")
        print(f"  - Compact model: {results['comp_obj']:.5f}")
        print(f"  - Gap: {results['gap']:.5%}")
        print(f"  - Integral?: {results['is_integral']}")

        # Convergence and optimal solution status
        print(f"\nAlgorithm Status:")
        gap_threshold = 1e-4  # 0.01% gap threshold
        if results['gap'] < gap_threshold:
            print(f"  ✓ Algorithm CONVERGED (Gap < {gap_threshold:.2%})")
            print(f"  ✓ OPTIMAL SOLUTION FOUND: {results['ip_obj']:.5f}")
        else:
            print(f"  ! Algorithm terminated with gap: {results['gap']:.5%}")
            print(f"  ! Best IP solution: {results['ip_obj']:.5f}")
            print(f"  ! LP relaxation: {results['lp_obj']:.5f}")

    print("=" * 100 + "\n")
    return results

if __name__ == "__main__":
    results = main()