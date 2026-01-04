import numpy as np
from collections import defaultdict

def calculate_extra_metrics(cg_solver, inc_sol, patients_list, derived_data, T, start_day=None, end_day=None):
    """
    Calculates detailed metrics (E.2 - E.8) based on the solution and derived variables.
    
    Args:
        cg_solver: The ColumnGeneration object (contains static data, M_p, Max_t, etc.)
        inc_sol: The incumbent solution dict (contains 'x', 'y' raw keys)
        patients_list: List of patients to consider (e.g. cg_solver.P_F)
        derived_data: Dict containing 'e', 'Y', 'theta', 'omega', 'g', 'z', etc. for these patients.
                      (Expected keys matches main.py: 'focus_e', 'focus_theta' etc. 
                       OR simple keys 'e', 'theta' if passed directly)
    
    Returns:
        dict: A dictionary containing all the calculated metrics.
    """
    
    # Unpack derived data (handling potential prefix differences from main.py)
    # We expect derived_data to have keys like 'e', 'Y', 'theta' 
    # corresponding to the patients_list provided.
    
    # If passed data has prefixes (e.g. 'focus_e'), we rely on the caller to unpack or we handle it?
    # Better: caller passes exactly the dicts for the relevant group.
    
    e_dict = derived_data.get('e', {})
    Y_dict = derived_data.get('Y', {})
    theta_dict = derived_data.get('theta', {})
    z_dict = derived_data.get('z', {})
    
    # We need aggregated x and raw y for calculations
    # x is typically (p, t, d, col_id) in inc_sol
    # y is (p, d) in inc_sol['y']
    
    raw_x = inc_sol.get('x', {})
    raw_y = inc_sol.get('y', {})
    
    # Filter x and y for patients in patients_list
    x_agg = defaultdict(int) # (p, t, d) -> value
    patient_x_sum = defaultdict(int) # (p, t) -> count
    
    for k, v in raw_x.items():
        if v > 1e-6:
            p = k[0]
            if p in patients_list:
                # k is (p, t, d, col_id)
                t, d = k[1], k[2]
                x_agg[(p, t, d)] += v
                patient_x_sum[(p, t)] += v # for workload
    
    y_filtered = {} # (p, d) -> val
    for k, v in raw_y.items():
        if v > 1e-6:
            p = k[0]
            if p in patients_list:
                d = k[1]
                y_filtered[(p, d)] = v

    metrics = {}
    
    # ==============================================================================
    # E.2 Resource Utilization Metrics
    # ==============================================================================
    total_human = sum(x_agg.values())
    total_ai = sum(y_filtered.values())
    total_sessions = total_human + total_ai
    
    metrics['total_human_sessions'] = total_human
    metrics['total_ai_sessions'] = total_ai
    metrics['total_sessions'] = total_sessions
    metrics['ai_session_share_pct'] = (total_ai / total_sessions * 100) if total_sessions > 0 else 0
    metrics['human_session_share_pct'] = (total_human / total_sessions * 100) if total_sessions > 0 else 0
    
    # Capacity & Workload
    # Capacity is global (all patients), but here we might only be looking at Focus.
    # However, Max_t is defined for the whole system.
    # E.2 usually implies global system state. If patients_list is only Focus, 
    # we can only compute utilization *by Focus patients*.
    # User likely wants global totals if calling for all, but for now we calculate relative to the provided group.
    
    # Calculates workload per therapist (human sessions)
    therapist_workload = defaultdict(int)
    for (p, t, d), val in x_agg.items():
        therapist_workload[t] += val
        
    metrics['avg_therapist_workload'] = sum(list(therapist_workload.values())) * (1/len(T)) if therapist_workload else 0
    metrics['peak_therapist_workload'] = max(therapist_workload.values()) if therapist_workload else 0
    
    # Total Capacity: C_total = sum_{t in T, d in D} Max_t[t, d]
    # IMPORTANT: Use the ACTUAL date range from patient assignments (x_agg) 
    # so that N_human and C_total are calculated over the same horizon.
    
    # Determine date range from actual assignments
    if x_agg:
        actual_days = set(d for (p, t, d) in x_agg.keys())
        s_day = min(actual_days)
        e_day = max(actual_days)
    else:
        # Fallback to passed parameters or cg_solver.D
        s_day = start_day if start_day is not None else min(cg_solver.D)
        e_day = end_day if end_day is not None else max(cg_solver.D)
    
    date_range = range(s_day, e_day + 1)
    
    if hasattr(cg_solver, 'Max_t'):
         relevant_capacity = 0
         for t_id in T:
             for d in date_range:
                 if (t_id, d) in cg_solver.Max_t:
                     relevant_capacity += cg_solver.Max_t[(t_id, d)]
         metrics['total_capacity'] = relevant_capacity
    else:
         metrics['total_capacity'] = 0
    
    # Store the actual date range used for transparency
    metrics['capacity_start_day'] = s_day
    metrics['capacity_end_day'] = e_day

    # Human Utilization (%): N_human / C_total * 100
    metrics['human_utilization_pct'] = (total_human / metrics['total_capacity'] * 100) if metrics.get('total_capacity', 0) > 0 else 0
    
    # Peak Therapist Workload per Day: max_{j,t} sum_i x_ijt
    therapist_day_workload = defaultdict(int)  # (t, d) -> sum of x_ijt
    for (p, t, d), val in x_agg.items():
        therapist_day_workload[(t, d)] += val
    metrics['peak_therapist_day_workload'] = max(therapist_day_workload.values()) if therapist_day_workload else 0
    
    # Peak Period Utilization (%): max_t (sum_{i,j} x_ijt / sum_j Q_jt) * 100
    period_utilization = {}
    for d in date_range:
        sessions_on_day = sum(val for (p, t, day), val in x_agg.items() if day == d)
        capacity_on_day = sum(cg_solver.Max_t.get((t_id, d), 0) for t_id in T) if hasattr(cg_solver, 'Max_t') else 0
        if capacity_on_day > 0:
            period_utilization[d] = sessions_on_day / capacity_on_day * 100
    
    metrics['peak_period_utilization_pct'] = max(period_utilization.values()) if period_utilization else 0

    # ==============================================================================
    # CONSOLIDATED OUTPUT - All Metrics Summary
    # ==============================================================================
    print(f"\n{'='*70}")
    print(f" METRICS SUMMARY (Patients: {len(patients_list)}, Days: {s_day}-{e_day}) ".center(70, '='))
    print(f"{'='*70}")
    
    print(f"\n--- E.2 Resource Utilization ---")
    print(f"  Total Capacity (C_total):     {metrics['total_capacity']}")
    print(f"  N_human (sessions):           {total_human}")
    print(f"  N_ai (sessions):              {total_ai}")
    print(f"  Human Utilization:            {metrics['human_utilization_pct']:.2f}%")
    print(f"  Peak Therapist Day Workload:  {metrics['peak_therapist_day_workload']}")
    print(f"  Peak Period Utilization:      {metrics['peak_period_utilization_pct']:.2f}%")

    print(f"{'='*70}\n")

    return metrics
