
import gurobipy as gu
import numpy as np
import os
from CG import ColumnGeneration
from Utils.Generell.instance_setup import generate_patient_data_log

def debug_compact_los():
    # --- Configuration matches sensitivity_analysis_iis.py ---
    seed = 421
    D_focus = 7
    pttr = 'medium'
    T_target = 3
    T_baseline = 4
    
    # "Paradox" Parameters
    theta_val = 0.6
    k_val = 0.7
    
    print(f"\nDEBUGGING COMPACT MODEL")
    print(f"Theta: {theta_val}, k_learn: {k_val}")
    print(f"T_target: {T_target}, T_demand: {T_baseline}")
    print("=" * 60)

    # 1. Generate Data
    print("Generating Patient Data...")
    Req, Entry, Max_t, P, D, D_Ext, D_Full, T_list, M_p, W_coeff, DRG = generate_patient_data_log(
        seed=seed,
        D_focus=D_focus,
        pttr_scenario=pttr,
        T=T_target,        
        T_demand=T_baseline,
        verbose=False
    )
    
    pre_generated_data = {
        'Req': Req, 'Entry': Entry, 'Max_t': Max_t, 'P': P, 'D': D, 
        'D_Ext': D_Ext, 'D_Full': D_Full, 'T': T_list, 'M_p': M_p, 
        'W_coeff': W_coeff, 'DRG': DRG
    }

    # 2. Setup Solver
    app_data = {
        'learn_type': ['sigmoid'],
        'theta_base': [theta_val],
        'lin_increase': [0.05],
        'k_learn': [k_val],
        'infl_point': [5],
        'MS': [5],
        'MS_min': [2],
        'W_on': [5],
        'W_off': [2],
        'daily': [4]
    }

    cg_solver = ColumnGeneration(
        seed=seed,
        app_data=app_data,
        T=T_target,
        D_focus=D_focus,
        pttr=pttr,
        show_plots=False,
        verbose=False,
        T_demand=T_baseline,
        pre_generated_data=pre_generated_data
    )
    
    print("Building Model...")
    cg_solver.setup()
    
    # 3. Solve Compact Model
    if hasattr(cg_solver, 'problem'):
        model = cg_solver.problem.Model
        
        # Enable Output for Debugging
        model.setParam('OutputFlag', 1)
        model.setParam('MIPFocus', 1) # Focus on feasibility first
        model.setParam('TimeLimit', 300)
        
        print("Optimizing...")
        model.optimize()
        
        if model.Status in [gu.GRB.OPTIMAL, gu.GRB.SOLUTION_LIMIT, gu.GRB.SUBOPTIMAL]:
            print("\n✅ FEASIBLE SOLUTION FOUND")
            print("-" * 60)
            print(f"{'Patient':<10} | {'Entry':<6} | {'LOS':<6} | {'AI_Sessions':<12} | {'Efficiency (Last)':<15}")
            print("-" * 60)
            
            total_los = 0
            count = 0
            
            # Access variables directly from the problem instance
            problem = cg_solver.problem
            
            for p in problem.P_Focus:
                los_val = problem.LOS[p].X
                
                # Calculate total AI sessions for this patient
                # S[p, d] is cumulative, so take max value over D
                ai_sessions = 0
                max_efficiency = 0
                
                # Check final S value
                # Note: S is defined for D, so we check the value at the last relevant day
                # Or sum y[p,d] manually
                ai_sum = sum(problem.y[p, d].X for d in problem.D if problem.y[p, d].X > 0.5)
                
                print(f"{p:<10} | {Entry[p]:<6} | {los_val:<6.1f} | {ai_sum:<12.1f} | {'?':<15}")
                
                total_los += los_val
                count += 1
                
            print("-" * 60)
            print(f"Mean LOS (Focus): {total_los/count:.4f}")
            
        else:
            print(f"\n❌ INFEASIBLE or FAILED (Status: {model.Status})")
            if model.Status == gu.GRB.INFEASIBLE:
                print("Computing IIS to find conflict...")
                model.computeIIS()
                model.write("debug_compact.ilp")
                print("IIS written to debug_compact.ilp")

if __name__ == "__main__":
    debug_compact_los()
