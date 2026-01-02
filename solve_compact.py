"""
Compact Model Solver
====================

This script solves the same instance from main.py using the Compact Model directly
(without Column Generation or Branch-and-Price).

The instance configuration is taken from main.py to ensure we solve the exact same problem.
"""

import gurobipy as gu
import time
from CG import ColumnGeneration
from logging_config import setup_multi_level_logging, get_logger

# Setup logging
print_all_logs = False
setup_multi_level_logging(base_log_dir='logs/compact', enable_console=True, print_all_logs=print_all_logs)
logger = get_logger(__name__)


def main():
    """
    Main function to solve the instance using Compact Model only.
    """
    logger.info("=" * 100)
    logger.info("STARTING COMPACT MODEL SOLVER")
    logger.info("=" * 100)

    # ===========================
    # CONFIGURATION PARAMETERS
    # (Same as main.py)
    # ===========================

    # Random seed
    seed = 12

    # Learning parameters
    app_data = {
        'learn_type': ['sigmoid'],  # Learning curve type: 'exp', 'sigmoid', 'lin', or numeric value
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
    T = 2  # Number of therapists
    D_focus = 4  # Number of focus days

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

    # Solver settings
    deterministic = False  # Set to True for deterministic solver behavior
    save_lps = True  # Set to True to save LP files
    verbose_output = True  # Set to False to suppress output

    # ===========================
    # CONFIGURATION SUMMARY
    # ===========================

    if verbose_output:
        print("\n" + "=" * 100)
        print(" STARTING SETUP ".center(100, "="))
        print("=" * 100)
        print(f"\nConfiguration:")
        print(f"  - Mode: Compact Model (Direct MIP)")
        print(f"  - Seed: {seed}")
        print(f"  - Learning type: {app_data['learn_type'][0]}")
        print(f"  - Learning method: {learn_method}")
        print(f"  - Therapists: {T}")
        print(f"  - Focus days: {D_focus}")
        print(f"  - PTTR scenario: {pttr}")
        print(f"  - Save LPs: {save_lps}")
        print()

    # ===========================
    # SETUP INSTANCE
    # ===========================

    if verbose_output:
        print("\n" + "=" * 100)
        print(" CREATING INSTANCE ".center(100, "="))
        print("=" * 100 + "\n")

    # Create CG solver (only to generate the instance and compact model)
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

    # Setup instance (this creates the compact model in cg_solver.problem)
    cg_solver.setup()

    if verbose_output:
        print("\n" + "=" * 100)
        print(" INSTANCE CREATED ".center(100, "="))
        print("=" * 100)
        print(f"\nInstance Details:")
        print(f"  - Total Patients: {len(cg_solver.P_Pre) + len(cg_solver.P_F) + len(cg_solver.P_Post)}")
        print(f"  - Pre Patients: {len(cg_solver.P_Pre)}")
        print(f"  - Focus Patients: {len(cg_solver.P_F)}")
        print(f"  - Post Patients: {len(cg_solver.P_Post)}")
        print(f"  - Therapists: {len(cg_solver.T)}")
        print(f"  - Planning Horizon (Focus): {len(cg_solver.D)} days")
        print()

    # ===========================
    # SOLVE COMPACT MODEL
    # ===========================

    if verbose_output:
        print("\n" + "=" * 100)
        print(" SOLVING COMPACT MODEL ".center(100, "="))
        print("=" * 100 + "\n")

    # Solve compact model directly
    start_time = time.time()
    cg_solver.problem.solveModel()
    solve_time = time.time() - start_time

    # ===========================
    # RESULTS
    # ===========================

    print("\n" + "=" * 100)
    print(" COMPACT MODEL RESULTS ".center(100, "="))
    print("=" * 100)

    # Check model status
    status = cg_solver.problem.Model.status

    if status == gu.GRB.OPTIMAL:
        obj_value = cg_solver.problem.Model.objVal
        print(f"\n✓ OPTIMAL SOLUTION FOUND")
        print(f"  - Objective Value: {obj_value:.5f}")
        print(f"  - Solve Time: {solve_time:.2f}s")
        print(f"  - Status: OPTIMAL")

        # Extract solution details
        num_vars = cg_solver.problem.Model.NumVars
        num_constrs = cg_solver.problem.Model.NumConstrs
        num_nz = cg_solver.problem.Model.NumNZs

        print(f"\nModel Statistics:")
        print(f"  - Variables: {num_vars}")
        print(f"  - Constraints: {num_constrs}")
        print(f"  - Non-zeros: {num_nz}")

        # MIP statistics (if available)
        if hasattr(cg_solver.problem.Model, 'NodeCount'):
            print(f"\nMIP Statistics:")
            print(f"  - Nodes explored: {cg_solver.problem.Model.NodeCount}")
            print(f"  - Simplex iterations: {cg_solver.problem.Model.IterCount}")
            print(f"  - MIP gap: {cg_solver.problem.Model.MIPGap:.5%}")

    elif status == gu.GRB.INFEASIBLE:
        print(f"\n✗ MODEL IS INFEASIBLE")
        print(f"  - Solve Time: {solve_time:.2f}s")
        print(f"  - Status: INFEASIBLE")

        # Compute IIS for debugging
        if verbose_output:
            print("\nComputing IIS (Irreducible Inconsistent Subsystem)...")
            cg_solver.problem.Model.computeIIS()
            cg_solver.problem.Model.write('compact_iis.ilp')
            print("  IIS written to: compact_iis.ilp")

    elif status == gu.GRB.UNBOUNDED:
        print(f"\n✗ MODEL IS UNBOUNDED")
        print(f"  - Solve Time: {solve_time:.2f}s")
        print(f"  - Status: UNBOUNDED")

    elif status == gu.GRB.TIME_LIMIT:
        print(f"\n! TIME LIMIT REACHED")
        print(f"  - Solve Time: {solve_time:.2f}s")
        print(f"  - Status: TIME_LIMIT")
        if cg_solver.problem.Model.SolCount > 0:
            print(f"  - Best Objective Found: {cg_solver.problem.Model.ObjVal:.5f}")
            print(f"  - MIP Gap: {cg_solver.problem.Model.MIPGap:.5%}")

    else:
        print(f"\n! UNKNOWN STATUS: {status}")
        print(f"  - Solve Time: {solve_time:.2f}s")

    # Save model to file if requested
    if save_lps:
        compact_lp_file = 'results/compact_model.lp'
        compact_sol_file = 'results/compact_model.sol'
        cg_solver.problem.Model.write(compact_lp_file)
        if status == gu.GRB.OPTIMAL:
            cg_solver.problem.Model.write(compact_sol_file)
        print(f"\nModel saved to:")
        print(f"  - LP file: {compact_lp_file}")
        if status == gu.GRB.OPTIMAL:
            print(f"  - Solution file: {compact_sol_file}")

    print("=" * 100 + "\n")

    # Return results dictionary
    results = {
        'status': status,
        'solve_time': solve_time,
        'objective': cg_solver.problem.Model.objVal if status == gu.GRB.OPTIMAL else None,
        'num_vars': cg_solver.problem.Model.NumVars,
        'num_constrs': cg_solver.problem.Model.NumConstrs,
        'num_nz': cg_solver.problem.Model.NumNZs
    }

    if status == gu.GRB.OPTIMAL and hasattr(cg_solver.problem.Model, 'NodeCount'):
        results['nodes'] = cg_solver.problem.Model.NodeCount
        results['iterations'] = cg_solver.problem.Model.IterCount
        results['mip_gap'] = cg_solver.problem.Model.MIPGap

    logger.info("=" * 100)
    logger.info("COMPACT MODEL SOLVER FINISHED")
    logger.info("=" * 100)

    return results


if __name__ == "__main__":
    results = main()
