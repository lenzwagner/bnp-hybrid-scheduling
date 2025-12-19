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

    print('2therapist_workload:', therapist_workload, sum(list(therapist_workload.values())))
        
    metrics['avg_therapist_workload'] = sum(list(therapist_workload.values())) * (1/len(T)) if therapist_workload else 0
    metrics['peak_therapist_workload'] = max(therapist_workload.values()) if therapist_workload else 0
    
    # Total Capacity: C_total = sum_{t in T, d in D} Max_t[t, d] (Assuming Max_t is available in cg_solver)
    # T is time horizon passed in, D is therapist list.
    # Max_t is typically (T, D) array or similar.
    # If Max_t is accessible via cg_solver.Max_t:
    if hasattr(cg_solver, 'Max_t'):
         # Max_t shape: [therapist, day] based on analysis
         relevant_capacity = 0
         
         # Determine date range
         s_day = start_day if start_day is not None else min(cg_solver.D)
         e_day = end_day if end_day is not None else max(cg_solver.D)
         date_range = range(s_day, e_day + 1)
         
         for t_id in T: # Iterate over therapist IDs supplied
             for d in date_range:
                 # Check if Day 'd' is in Max_t keys (Max_t has keys (t, d))
                 if (t_id, d) in cg_solver.Max_t:
                     relevant_capacity += cg_solver.Max_t[(t_id, d)]
                     
         metrics['total_capacity'] = relevant_capacity
    else:
         metrics['total_capacity'] = 0

    # ==============================================================================
    # E.3 AI Learning Dynamics Metrics
    # ==============================================================================
    # Initial/Final/Max Theta per patient
    theta_initial = {}
    theta_final = {}
    theta_max = {}
    time_to_proficiency = {} # First day > 0.7
    
    ai_sessions_per_patient = defaultdict(int)
    for (p, d), val in y_filtered.items():
        ai_sessions_per_patient[p] += val

    for p in patients_list:
        # Get patient timeline
        days = sorted([d for (pat, d) in theta_dict.keys() if pat == p])
        if not days:
            continue
            
        first_day = days[0]
        last_day = days[-1]
        
        theta_initial[p] = theta_dict.get((p, first_day), 0)
        theta_final[p] = theta_dict.get((p, last_day), 0)
        
        p_thetas = [theta_dict.get((p, d), 0) for d in days]
        theta_max[p] = max(p_thetas) if p_thetas else 0
        
        # Time to proficiency
        # Use entry day from solver
        entry_day = cg_solver.Entry_agg.get(p, first_day)
        
        prof_days = [d for d in days if theta_dict.get((p, d), 0) > 0.7]
        if prof_days:
            # Time from entry (ensure non-negative)
            first_prof_day = prof_days[0]
            if first_prof_day >= entry_day:
                time_to_proficiency[p] = first_prof_day - entry_day
            else:
                # Should not happen if theta is 0 before entry, but for safety
                time_to_proficiency[p] = 0
        else:
            time_to_proficiency[p] = None

    metrics['theta_initial_avg'] = np.mean(list(theta_initial.values())) if theta_initial else 0
    metrics['theta_final_avg'] = np.mean(list(theta_final.values())) if theta_final else 0
    metrics['theta_max_avg'] = np.mean(list(theta_max.values())) if theta_max else 0
    
    valid_ttp = [v for v in time_to_proficiency.values() if v is not None]
    metrics['avg_time_to_proficiency'] = np.mean(valid_ttp) if valid_ttp else None
    metrics['completion_rate_proficiency'] = len(valid_ttp) / len(patients_list) if patients_list else 0

    # Avg Theta when AI used
    theta_ai_sum = 0
    ai_count = 0
    for (p, d), val in y_filtered.items():
        if val > 0.5:
             theta = theta_dict.get((p, d), 0)
             theta_ai_sum += theta
             ai_count += 1
    metrics['avg_theta_when_ai_used'] = (theta_ai_sum / ai_count) if ai_count > 0 else 0

    # Consecutive AI sessions
    max_consecutive = {}
    for p in patients_list:
        days = sorted([d for (pat, d) in y_filtered.keys() if pat == p and y_filtered[(pat, d)] > 0.5])
        if not days:
            max_consecutive[p] = 0
            continue
            
        current_streak = 1
        max_streak = 1
        for i in range(1, len(days)):
            if days[i] == days[i-1] + 1:
                current_streak += 1
            else:
                max_streak = max(max_streak, current_streak)
                current_streak = 1
        max_streak = max(max_streak, current_streak)
        max_consecutive[p] = max_streak
        
    metrics['avg_max_consecutive_ai'] = np.mean(list(max_consecutive.values())) if max_consecutive else 0
    metrics['global_max_consecutive_ai'] = max(max_consecutive.values()) if max_consecutive else 0

    # ==============================================================================
    # E.5 Treatment Gap Metrics
    # ==============================================================================
    # gap = e_it - sum(x) - y
    gap_days = defaultdict(int)
    total_gaps = 0
    
    # Iterate over all active patient-days
    for (p, d), is_present in e_dict.items():
        if is_present > 0.5 and p in patients_list:
            # Check if treated
            treated_human = any(x_agg.get((p, t, d), 0) > 0.5 for t in range(1, 100)) # naive t range, optimize later?
            # actually we can sum x_agg for (p, *, d)
            # Efficient check:
            is_human = 0
            # A bit slow to iterate all t. Better: compute treated days set first.
            
            is_ai = y_filtered.get((p, d), 0) > 0.5
            
            # Since we don't have a quick (p,d)->human index here, we iterate x_agg for this p?
            # No, let's pre-process x_agg
            pass

    # Pre-process treated days
    treated_human_days = set()
    for (p, t, d), val in x_agg.items():
        if val > 0.5:
            treated_human_days.add((p, d))
            
    for (p, d), is_present in e_dict.items():
         if is_present > 0.5 and p in patients_list:
             has_human = (p, d) in treated_human_days
             has_ai = y_filtered.get((p, d), 0) > 0.5
             
             if not has_human and not has_ai:
                 gap_days[p] += 1
                 total_gaps += 1
                 
    metrics['total_gap_days_all'] = total_gaps
    metrics['avg_gap_days_per_patient'] = total_gaps / len(patients_list) if patients_list else 0

    # ==============================================================================
    # E.6 DRG-Specific Metrics
    # ==============================================================================
    # Infer DRG from Mean LOS (M_p)
    # E65A: ~17.9, E65B: ~8.0, E65C: ~6.1
    drg_map = {}
    drg_stats = defaultdict(lambda: {'count': 0, 'los_sum': 0, 'req_sum': 0, 'ai_sum': 0, 'total_treat_sum': 0})
    
    for p in patients_list:
        mean_los = cg_solver.M_p.get(p, 0)
        # Approximate matching using simple buckets
        if mean_los > 12:
            drg = 'E65A'
        elif mean_los > 7:
            drg = 'E65B'
        else:
            drg = 'E65C'
        
        drg_map[p] = drg
        
        # Gather stats
        # We need actual LOS and Req
        # Req is in cg_solver.Req_agg
        req = cg_solver.Req_agg.get(p, 0)
        
        # Realized LOS? From x/y or just count e_it?
        # derived_vars should normally contain LOS dict, but here we only have e_dict/x/y?
        # Calculating realized LOS from e_dict
        p_days = [d for (pat, d), val in e_dict.items() if pat == p and val > 0.5]
        realized_los = len(p_days)
        
        drg_stats[drg]['count'] += 1
        drg_stats[drg]['los_sum'] += realized_los
        drg_stats[drg]['req_sum'] += req
        
        # AI/Human counts
        p_ai = ai_sessions_per_patient[p]
        p_human = sum(x_agg[(p, t, d)] for t in range(1, 200) for d in p_days if (p,t,d) in x_agg) # approx
        # Better: use the pre-calculated total_human/ai per patient if we had it.
        # Let's sum x_agg properly above
        
        drg_stats[drg]['ai_sum'] += p_ai
        
    # Correct human sum per patient
    pat_human_count = defaultdict(int)
    for (p, t, d), val in x_agg.items():
        pat_human_count[p] += val
        
    for p, drg in drg_map.items():
        drg_stats[drg]['total_treat_sum'] += (pat_human_count[p] + ai_sessions_per_patient[p])

    # Compute DRG averages
    for drg, s in drg_stats.items():
        count = s['count']
        if count > 0:
            metrics[f'DRG_{drg}_count'] = count
            metrics[f'DRG_{drg}_avg_los'] = s['los_sum'] / count
            metrics[f'DRG_{drg}_avg_req'] = s['req_sum'] / count
            metrics[f'DRG_{drg}_ai_share_pct'] = (s['ai_sum'] / s['total_treat_sum'] * 100) if s['total_treat_sum'] > 0 else 0

    # ==============================================================================
    # E.7 Session Pattern Analysis
    # ==============================================================================
    # Trigrams
    trigrams = defaultdict(int)
    
    for p in patients_list:
        # Build sequence string
        days = sorted([d for (pat, d) in e_dict.items() if pat == p and e_dict[(pat, d)] > 0.5])
        if not days:
            continue
            
        seq_str = ""
        for d in days:
            is_h = (p, d) in treated_human_days
            is_a = y_filtered.get((p, d), 0) > 0.5
            
            if is_h:
                seq_str += "H"
            elif is_a:
                seq_str += "A"
            else:
                seq_str += "_" 
        
        # Extract trigrams
        if len(seq_str) >= 3:
            for i in range(len(seq_str) - 2):
                tri = seq_str[i:i+3]
                trigrams[tri] += 1
                
    metrics['top_trigrams'] = dict(sorted(trigrams.items(), key=lambda item: item[1], reverse=True)[:5])

    # ==============================================================================
    # E.8 Therapist Continuity Metrics
    # ==============================================================================
    # z_ij is in derived_data['z'] (p, j) -> 1
    # Count assigned per patient
    therapists_per_patient = defaultdict(int)
    for (p, j), val in z_dict.items():
        if val > 0.5:
            therapists_per_patient[p] += 1
            
    continuity_violations = sum(1 for c in therapists_per_patient.values() if c > 1)
    
    metrics['avg_therapists_per_patient'] = np.mean(list(therapists_per_patient.values())) if therapists_per_patient else 0
    metrics['continuity_violations_count'] = continuity_violations

    return metrics
