
import time

# --- 2. Helper Functions (COPIED FROM USER) ---

def check_strict_feasibility(history, next_val, MS, MIN_MS):
    potential_sequence = history + (next_val,)
    seq_len = len(potential_sequence)

    if seq_len < MS:
        current_sum = sum(potential_sequence)
        remaining_slots = MS - seq_len
        max_possible_sum = current_sum + remaining_slots
        if max_possible_sum < MIN_MS:
            return False
        return True
    else:
        current_window = potential_sequence[-MS:]
        if sum(current_window) < MIN_MS:
            return False
        return True

def add_state_to_buckets(buckets, cost, prog, ai_count, hist, path, recipient_id, pruning_stats, dominance_mode='bucket', zeta=None, epsilon=1e-9):
    if zeta is not None:
        bucket_key = (ai_count, hist, zeta)
    else:
        bucket_key = (ai_count, hist)

    if bucket_key not in buckets:
        buckets[bucket_key] = []

    bucket_list = buckets[bucket_key]

    # --- LOCAL DOMINANCE ---
    is_dominated = False
    for c_old, p_old, _ in bucket_list:
        if c_old <= cost + epsilon and p_old >= prog - epsilon:
            is_dominated = True
            break

    if is_dominated:
        pruning_stats['dominance'] += 1
        return

    new_bucket_list = []
    for c_old, p_old, path_old in bucket_list:
        if cost <= c_old + epsilon and prog >= p_old - epsilon:
            pruning_stats['dominance'] += 1
            continue
        new_bucket_list.append((c_old, p_old, path_old))

    new_bucket_list.append((cost, prog, path))
    buckets[bucket_key] = new_bucket_list

def generate_full_column_vector(worker_id, path_assignments, start_time, end_time, max_time, num_workers):
    vector_length = num_workers * max_time
    full_vector = [0.0] * vector_length
    worker_offset = (worker_id - 1) * max_time
    for t_idx, val in enumerate(path_assignments):
        current_time = start_time + t_idx
        global_idx = worker_offset + (current_time - 1)
        if 0 <= global_idx < vector_length:
            full_vector[global_idx] = float(val)
    return full_vector

def compute_lower_bound(current_cost, start_time, end_time, gamma_k, obj_mode):
    duration = end_time - start_time + 1
    time_cost = duration * obj_mode
    lower_bound = current_cost + time_cost - gamma_k
    return lower_bound

def compute_candidate_workers(workers, r_k, tau_max, pi_dict):
    candidate_workers = []
    for j1 in workers:
        is_dominated = False
        for j2 in workers:
            if j1 == j2: continue
            all_better_or_equal = True
            at_least_one_strictly_better = False
            for t in range(r_k, tau_max + 1):
                pi_j1 = pi_dict.get((j1, t), 0.0)
                pi_j2 = pi_dict.get((j2, t), 0.0)
                if pi_j2 < pi_j1:
                    all_better_or_equal = False
                    break
                elif pi_j2 > pi_j1:
                    at_least_one_strictly_better = True
            if all_better_or_equal and at_least_one_strictly_better:
                is_dominated = True
                break
        if not is_dominated:
            candidate_workers.append(j1)
    return candidate_workers

def solve_pricing_for_recipient(recipient_id, r_k, s_k, gamma_k, obj_mode, pi_dict, workers, max_time, ms, min_ms, theta_lookup, use_bound_pruning=True):
    best_reduced_cost = float('inf')
    best_columns = []
    epsilon = 1e-9
    pruning_stats = {'lb': 0, 'dominance': 0, 'printed_dominance': {}}
    time_until_end = max_time - r_k + 1

    candidate_workers = compute_candidate_workers(workers, r_k, max_time, pi_dict)
    
    for j in candidate_workers:
        effective_min_duration = min(int(s_k), time_until_end)
        start_tau = r_k + effective_min_duration - 1
        
        for tau in range(start_tau, max_time + 1):
            is_timeout_scenario = (tau == max_time)
            start_cost = -pi_dict.get((j, r_k), 0)
            current_states = {}
            initial_history = (1,)
            
            add_state_to_buckets(current_states, start_cost, 1.0, 0, initial_history, [1], recipient_id, pruning_stats, 'bucket', None, epsilon)

            for t in range(r_k + 1, tau):
                next_states = {}
                for bucket_key, bucket_list in current_states.items():
                    ai_count, hist = bucket_key
                    for cost, prog, path in bucket_list:
                        if use_bound_pruning:
                            lb = compute_lower_bound(cost, r_k, tau, gamma_k, obj_mode)
                            if lb >= 0:
                                pruning_stats['lb'] += 1
                                continue

                        remaining_steps = tau - t + 1
                        if not is_timeout_scenario:
                            if prog + remaining_steps * 1.0 < s_k - epsilon:
                                continue

                        # Therapist
                        if check_strict_feasibility(hist, 1, ms, min_ms):
                            cost_ther = cost - pi_dict.get((j, t), 0)
                            prog_ther = prog + 1.0
                            new_hist_ther = (hist + (1,))
                            if len(new_hist_ther) > ms - 1: new_hist_ther = new_hist_ther[-(ms - 1):]
                            add_state_to_buckets(next_states, cost_ther, prog_ther, ai_count, new_hist_ther, path + [1], recipient_id, pruning_stats, 'bucket', None, epsilon)

                        # AI
                        if check_strict_feasibility(hist, 0, ms, min_ms):
                            cost_ai = cost
                            efficiency = theta_lookup[ai_count] if ai_count < len(theta_lookup) else 1.0
                            prog_ai = prog + efficiency
                            ai_count_new = ai_count + 1
                            new_hist_ai = (hist + (0,))
                            if len(new_hist_ai) > ms - 1: new_hist_ai = new_hist_ai[-(ms - 1):]
                            add_state_to_buckets(next_states, cost_ai, prog_ai, ai_count_new, new_hist_ai, path + [0], recipient_id, pruning_stats, 'bucket', None, epsilon)

                current_states = next_states
                if not current_states: break

            # Final Step
            for bucket_key, bucket_list in current_states.items():
                ai_count, hist = bucket_key
                for cost, prog, path in bucket_list:
                    possible_moves = []
                    if check_strict_feasibility(hist, 1, ms, min_ms): possible_moves.append(1)
                    if is_timeout_scenario and check_strict_feasibility(hist, 0, ms, min_ms): possible_moves.append(0)

                    for move in possible_moves:
                        if move == 1:
                            final_cost_accum = cost - pi_dict.get((j, tau), 0)
                            final_prog = prog + 1.0
                        else:
                            final_cost_accum = cost
                            efficiency = theta_lookup[ai_count] if ai_count < len(theta_lookup) else 1.0
                            final_prog = prog + efficiency

                        if final_prog >= s_k - epsilon or (is_timeout_scenario and move==0): # Simplified end check
                            duration = tau - r_k + 1
                            reduced_cost = (obj_mode * duration) + final_cost_accum - gamma_k
                            if reduced_cost < -1e-9:
                                best_columns.append(reduced_cost) # Just store RC for test

    return pruning_stats, best_columns

def run_test(case_name, r_i, s_i, gamma, obj_mode, pi):
    MS = 5
    MIN_MS = 2
    MAX_TIME = 42
    WORKERS = [1, 2, 3]
    theta_lookup = [min(0.2 + 0.01 * k, 1.0) for k in range(50)]

    print(f"\n--- Running {case_name} ---")
    start = time.time()
    
    total_pruning = {'lb': 0, 'dominance': 0}
    total_cols = 0
    
    recipient_keys = list(r_i.keys())
    # Limit to first 10 for quick test if needed, but user said 3s vs 0.7s so full run is fine
    
    for k in recipient_keys:
        stats, cols = solve_pricing_for_recipient(
            k, r_i[k], s_i[k], gamma.get(k, 0), obj_mode.get(k, 0), pi, WORKERS, 
            MAX_TIME, MS, MIN_MS, theta_lookup, use_bound_pruning=True
        )
        total_pruning['lb'] += stats['lb']
        total_pruning['dominance'] += stats['dominance']
        total_cols += len(cols)

    duration = time.time() - start
    print(f"Time: {duration:.4f}s")
    print(f"Pruning: LB={total_pruning['lb']}, Dom={total_pruning['dominance']}")
    print(f"Negative Reduced Cost Cols Found: {total_cols}")
    return duration, total_pruning

# --- DATASETS ---

# CASE 1 (Top Block)
pi_1 = {(w, t): 0.0 for w in [1,2,3] for t in range(1, 43)}
gamma_1 = {2: 25.0, 3: 18.0, 4: 13.0, 9: 18.0, 11: 7.0, 15: 13.0, 16: 9.0, 20: 6.0, 21: 14.0, 22: 13.0, 23: 5.0, 24: 8.0, 25: 5.0, 26: 5.0, 28: 13.0, 29: 41.0, 30: 20.0, 31: 35.0, 36: 15.0, 37: 7.0, 38: 2.0, 39: 7.0, 40: 35.0, 41: 13.0, 43: 8.0, 47: 9.0, 50: 6.0, 53: 41.0, 59: 9.0, 60: 5.0, 62: 6.0, 63: 6.0, 65: 5.0, 66: 13.0, 67: 6.0, 70: 3.0, 71: 8.0, 73: 10.0, 75: 7.0, 77: 3.0, 78: 8.0, 79: 5.0, 81: 4.0, 82: 5.0} # ... incomplete but enough for test?
# Actually need to use the full dict provided by user for accurate comparison
r_i_1 = {2: 17, 3: 17, 4: 30, 9: 22, 11: 5, 15: 30, 16: 11, 20: 32, 21: 27, 22: 27, 23: 29, 24: 22, 25: 38, 26: 26, 28: 30, 29: 2, 30: 23, 31: 8, 36: 28, 37: 26, 38: 41, 39: 18, 40: 8, 41: 23, 43: 17, 47: 5, 50: 11, 53: 2, 59: 15, 60: 38, 62: 20, 63: 37, 65: 27, 66: 30, 67: 14, 70: 32, 71: 33, 73: 31, 75: 20, 77: 40, 78: 11, 79: 20, 81: 12, 82: 9}
s_i_1 = {2: 22, 3: 16, 4: 20, 9: 14, 11: 5, 15: 11, 16: 8, 20: 5, 21: 12, 22: 11, 23: 5, 24: 7, 25: 8, 26: 4, 28: 10, 29: 3, 30: 7, 31: 12, 36: 9, 37: 6, 38: 4, 39: 6, 40: 7, 41: 5, 43: 7, 47: 7, 50: 5, 53: 4, 59: 8, 60: 6, 62: 5, 63: 11, 65: 4, 66: 3, 67: 5, 70: 3, 71: 7, 73: 9, 75: 6, 77: 8, 78: 7, 79: 4, 81: 4, 82: 5}
obj_mode_1 = {2: 0, 3: 0, 4: 0, 9: 0, 11: 1, 15: 0, 16: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 28: 0, 29: 1, 30: 0, 31: 0, 36: 0, 37: 0, 38: 0, 39: 0, 40: 0, 41: 0, 43: 0, 47: 1, 50: 0, 53: 1, 59: 0, 60: 0, 62: 0, 63: 0, 65: 0, 66: 0, 67: 0, 70: 0, 71: 0, 73: 0, 75: 0, 77: 0, 78: 0, 79: 0, 81: 0, 82: 0}

# CASE 2 (Bottom Block) - COPIED EXACTLY
pi_2 = {(1, 1): 0.0, (1, 2): 0.0, (1, 3): 0.0, (1, 4): 0.0, (1, 5): 0.0, (1, 6): -2.0, (1, 7): -2.0, (1, 8): 0.0, (1, 9): 0.0, (1, 10): 0.0, (1, 11): 0.0, (1, 12): 0.0, (1, 13): 0.0, (1, 14): 0.0, (1, 15): 0.0, (1, 16): -27.0, (1, 17): 0.0, (1, 18): 0.0, (1, 19): 0.0, (1, 20): 0.0, (1, 21): -27.0, (1, 22): 0.0, (1, 23): 0.0, (1, 24): 0.0, (1, 25): 0.0, (1, 26): 0.0, (1, 27): 0.0, (1, 28): 0.0, (1, 29): 0.0, (1, 30): 0.0, (1, 31): 0.0, (1, 32): 0.0, (1, 33): 0.0, (1, 34): 0.0, (1, 35): 0.0, (1, 36): 0.0, (1, 37): 0.0, (1, 38): 0.0, (1, 39): 0.0, (1, 40): 0.0, (1, 41): 0.0, (1, 42): 0.0, (2, 1): 0.0, (2, 2): -2.0, (2, 3): 0.0, (2, 4): 0.0, (2, 5): 0.0, (2, 6): 0.0, (2, 7): -1.0, (2, 8): -1.0, (2, 9): -1.0, (2, 10): 0.0, (2, 11): 0.0, (2, 12): 0.0, (2, 13): 0.0, (2, 14): 0.0, (2, 15): 0.0, (2, 16): 0.0, (2, 17): 0.0, (2, 18): 0.0, (2, 19): -16.0, (2, 20): 0.0, (2, 21): 0.0, (2, 22): -27.0, (2, 23): 0.0, (2, 24): 0.0, (2, 25): 0.0, (2, 26): 0.0, (2, 27): 0.0, (2, 28): 0.0, (2, 29): 0.0, (2, 30): 0.0, (2, 31): 0.0, (2, 32): 0.0, (2, 33): 0.0, (2, 34): 0.0, (2, 35): 0.0, (2, 36): 0.0, (2, 37): 0.0, (2, 38): 0.0, (2, 39): 0.0, (2, 40): 0.0, (2, 41): 0.0, (2, 42): 0.0, (3, 1): 0.0, (3, 2): -2.0, (3, 3): 0.0, (3, 4): 0.0, (3, 5): 0.0, (3, 6): -1.0, (3, 7): 0.0, (3, 8): 0.0, (3, 9): -2.0, (3, 10): -1.0, (3, 11): 0.0, (3, 12): 0.0, (3, 13): 0.0, (3, 14): 0.0, (3, 15): -16.0, (3, 16): -27.0, (3, 17): 0.0, (3, 18): 0.0, (3, 19): 0.0, (3, 20): 0.0, (3, 21): 0.0, (3, 22): 0.0, (3, 23): -16.0, (3, 24): 0.0, (3, 25): 0.0, (3, 26): 0.0, (3, 27): 0.0, (3, 28): 0.0, (3, 29): 0.0, (3, 30): 0.0, (3, 31): 0.0, (3, 32): 0.0, (3, 33): 0.0, (3, 34): 0.0, (3, 35): 0.0, (3, 36): 0.0, (3, 37): 0.0, (3, 38): 0.0, (3, 39): 0.0, (3, 40): 0.0, (3, 41): 0.0, (3, 42): 0.0}
gamma_2 = {2: 12.0, 6: 27.0, 8: 16.0, 13: 0.0, 15: 0.0, 16: 0.0, 18: 0.0, 19: 0.0, 20: 27.0, 23: 16.0, 28: 0.0, 31: 27.0, 32: 0.0, 35: 0.0, 36: 0.0, 37: 0.0, 38: 8.0, 44: 0.0, 45: 0.0, 47: 0.0, 48: 9.0, 49: 0.0, 50: 10.0, 51: 0.0, 53: 0.0, 54: 10.0, 60: 2.0, 61: 16.0, 64: 9.0, 65: 0.0, 66: 0.0, 67: 0.0, 68: 0.0, 73: 16.0, 74: 0.0, 76: 0.0, 77: 0.0, 78: 0.0, 80: 16.0, 84: 10.0}
r_i_2 = {1: -8, 2: 2, 3: -4, 4: -2, 5: -4, 6: 16, 7: -16, 8: 14, 9: -35, 10: -12, 11: -10, 12: -25, 13: 22, 14: -12, 15: 25, 16: 39, 17: -17, 18: 33, 19: 27, 20: 16, 21: -34, 22: 0, 23: 12, 24: -13, 25: -18, 26: -12, 27: -16, 28: 25, 29: -9, 30: -3, 31: 15, 32: 38, 33: -13, 34: -33, 35: 40, 36: 29, 37: 25, 38: 5, 39: -18, 40: -24, 41: -12, 42: 0, 43: -31, 44: 18, 45: 21, 46: -35, 47: 30, 48: 2, 49: 36, 50: 1, 51: 26, 52: -1, 53: 12, 54: 3, 55: -34, 56: -25, 57: -18, 58: -4, 59: -7, 60: 6, 61: 16, 62: -5, 63: -11, 64: 4, 65: 38, 66: 25, 67: 15, 68: 28, 69: -18, 70: -28, 71: -29, 72: -26, 73: 12, 74: 36, 75: -27, 76: 18, 77: 38, 78: 24, 79: -24, 80: 14, 81: -16, 82: -23, 83: -35, 84: 5}
s_i_2 = {1: 17, 2: 9, 3: 13, 4: 17, 5: 4, 6: 6, 7: 5, 8: 9, 9: 5, 10: 8, 11: 5, 12: 9, 13: 5, 14: 5, 15: 6, 16: 7, 17: 4, 18: 5, 19: 7, 20: 7, 21: 8, 22: 14, 23: 10, 24: 10, 25: 8, 26: 6, 27: 8, 28: 5, 29: 5, 30: 5, 31: 6, 32: 4, 33: 4, 34: 4, 35: 3, 36: 8, 37: 4, 38: 5, 39: 10, 40: 8, 41: 9, 42: 8, 43: 9, 44: 7, 45: 4, 46: 10, 47: 2, 48: 6, 49: 7, 50: 7, 51: 6, 52: 10, 53: 3, 54: 7, 55: 6, 56: 6, 57: 7, 58: 9, 59: 7, 60: 5, 61: 4, 62: 4, 63: 2, 64: 6, 65: 10, 66: 8, 67: 3, 68: 7, 69: 5, 70: 8, 71: 4, 72: 7, 73: 6, 74: 6, 75: 3, 76: 5, 77: 7, 78: 7, 79: 3, 80: 6, 81: 4, 82: 6, 83: 11, 84: 7}
obj_mode_2 = {2: 1, 6: 0, 8: 0, 13: 0, 15: 0, 16: 0, 18: 0, 19: 0, 20: 0, 23: 0, 28: 0, 31: 0, 32: 0, 35: 0, 36: 0, 37: 0, 38: 1, 44: 0, 45: 0, 47: 0, 48: 1, 49: 0, 50: 1, 51: 0, 53: 0, 54: 1, 60: 0, 61: 0, 64: 1, 65: 0, 66: 0, 67: 0, 68: 0, 73: 0, 74: 0, 76: 0, 77: 0, 78: 0, 80: 0, 84: 1}

run_test("CASE 1 (Zero Duals)", r_i_1, s_i_1, gamma_1, obj_mode_1, pi_1)
run_test("CASE 2 (Mixed Duals)", r_i_2, s_i_2, gamma_2, obj_mode_2, pi_2)
