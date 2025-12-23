
import numpy as np
from numba import njit, int64, float64, types, prange
from numba.typed import List, Dict


# =============================================================================
# NUMBA-OPTIMIZED HELPER FUNCTIONS
# =============================================================================

@njit(cache=True)
def validate_column_history_numba(path_mask, duration, MS, MIN_MS):
    """
    Numba-optimized validation that a complete column satisfies rolling window constraints.
    
    Args:
        path_mask: Bitmask representing the path (bit i = 1 means therapist at position i)
        duration: Length of the schedule
        MS: Rolling window size
        MIN_MS: Minimum human services required in window
    
    Returns:
        bool: True if column satisfies all rolling window constraints
    """
    # Check every position in the schedule
    for i in range(duration):
        if i + 1 < MS:
            # Not enough history yet - check if remaining slots can satisfy MIN_MS
            current_sum = 0
            for j in range(i + 1):
                if (path_mask >> j) & 1:
                    current_sum += 1
            remaining_slots = MS - (i + 1)
            max_possible = current_sum + remaining_slots
            if max_possible < MIN_MS:
                return False
        else:
            # Complete window exists
            window_start = i + 1 - MS
            window_sum = 0
            for j in range(window_start, i + 1):
                if (path_mask >> j) & 1:
                    window_sum += 1
            if window_sum < MIN_MS:
                return False
    
    return True


@njit(cache=True, inline='always')
def compute_lower_bound_numba(current_cost, start_time, end_time, gamma_k, obj_mode):
    """
    Numba-optimized Lower Bound calculation for Bound Pruning.
    
    Inlined for maximum performance in hot loops.
    
    Args:
        current_cost: Accumulated cost so far
        start_time: Column start time
        end_time: Column end time
        gamma_k: Gamma dual value
        obj_mode: Objective mode multiplier
    
    Returns:
        float: Minimum achievable final Reduced Cost (optimistic)
    """
    duration = end_time - start_time + 1
    time_cost = duration * obj_mode
    return current_cost + time_cost - gamma_k

@njit(cache=True)
def compute_candidate_workers_numba(workers, r_k, tau_max, pi_matrix):
    """
    Numba-optimized Worker Dominance Pre-Elimination.
    
    Worker j1 dominates j2 if π_{j1,t} >= π_{j2,t} for all t in [r_k, tau_max]
    AND π_{j1,t} > π_{j2,t} for at least one t (strict dominance).
    Since π values are <= 0 (implicit costs), higher π means lower cost.
    
    Args:
        workers: 1D numpy array of worker IDs
        r_k: Release time (int)
        tau_max: Maximum time horizon (int)
        pi_matrix: 2D numpy array [num_workers, max_time+1] of pi values
    
    Returns:
        Tuple of (candidate_array, count) - numpy array with candidates and actual count
    """
    n_workers = len(workers)
    # Pre-allocate result array (worst case: all workers are candidates)
    result = np.empty(n_workers, dtype=np.int64)
    count = 0
    
    for i1 in range(n_workers):
        j1 = workers[i1]
        is_dominated = False
        
        for i2 in range(n_workers):
            if i1 == i2:
                continue
                
            j2 = workers[i2]
            
            # Check if j2 dominates j1
            all_better_or_equal = True
            at_least_one_strictly_better = False
            
            for t in range(r_k, tau_max + 1):
                pi_j1 = pi_matrix[j1, t]
                pi_j2 = pi_matrix[j2, t]
                
                if pi_j2 < pi_j1:  # j2 is worse in this period
                    all_better_or_equal = False
                    break
                elif pi_j2 > pi_j1:  # j2 is strictly better in this period
                    at_least_one_strictly_better = True
            
            # j2 dominates j1 if it's at least as good everywhere and strictly better somewhere
            if all_better_or_equal and at_least_one_strictly_better:
                is_dominated = True
                break
        
        if not is_dominated:
            result[count] = j1
            count += 1
    
    # Return slice of actual candidates
    return result[:count]


@njit(cache=True)
def generate_full_column_vector_numba(worker_id, path_mask, start_time, end_time, max_time, num_workers):
    """
    Numba-optimized generation of the full column vector for a schedule.
    
    Args:
        worker_id: Worker ID (1-indexed)
        path_mask: Bitmask representing the path (bit i = 1 means therapist at time start_time + i)
        start_time: Start time of schedule
        end_time: End time of schedule
        max_time: Maximum time horizon
        num_workers: Total number of workers
        
    Returns:
        1D numpy array representing the full column vector
    """
    vector_length = num_workers * max_time
    full_vector = np.zeros(vector_length, dtype=np.float64)
    
    worker_offset = (worker_id - 1) * max_time
    duration = end_time - start_time + 1
    
    for t_idx in range(duration):
        # Check if bit t_idx is set (therapist assignment)
        if (path_mask >> t_idx) & 1:
            current_time = start_time + t_idx
            global_idx = worker_offset + (current_time - 1)
            if 0 <= global_idx < vector_length:
                full_vector[global_idx] = 1.0
    
    return full_vector


@njit(cache=True)
def path_mask_to_list(path_mask, duration):
    """
    Convert a bitmask path to a list of 0s and 1s.
    
    Args:
        path_mask: Bitmask representing the path
        duration: Length of the schedule
        
    Returns:
        List of 0s and 1s
    """
    result = List.empty_list(int64)
    for i in range(duration):
        if (path_mask >> i) & 1:
            result.append(int64(1))
        else:
            result.append(int64(0))
    return result


# Type aliases for readability
# State: (cost, progress, path_mask, history_mask, history_len, ai_count)
# We store states in a list.
# Optimisation: We group states by (ai_count, history_mask, history_len) for dominance.

# =============================================================================
# STATE PACKING HELPERS (Optimization A)
# =============================================================================
# Pack (ai_count, hist_mask, hist_len) into single int64 for faster Dict hashing
# Bit layout:
#   Bits 0-7:   hist_len (max 255)
#   Bits 8-23:  hist_mask (max 65535, enough for MS up to 17)
#   Bits 24-55: ai_count (max ~4 billion)

@njit(cache=True, inline='always')
def pack_key(ai_count, hist_mask, hist_len):
    """Pack state key components into single int64."""
    return (ai_count << 24) | (hist_mask << 8) | hist_len

@njit(cache=True, inline='always')
def unpack_key(key):
    """Unpack int64 key back to components."""
    hist_len = key & 0xFF
    hist_mask = (key >> 8) & 0xFFFF
    ai_count = key >> 24
    return ai_count, hist_mask, hist_len


# =============================================================================
# OPTIMIZATION 4: PRECOMPUTED TRANSITION LOOKUPS
# =============================================================================
# Pre-compute popcount (number of 1-bits) for all possible history masks
# This eliminates the inner loop that counts bits one-by-one

# Popcount lookup table for 16-bit values (covers hist_mask up to 2^16)
POPCOUNT_TABLE = np.zeros(65536, dtype=np.int64)
for _i in range(65536):
    POPCOUNT_TABLE[_i] = bin(_i).count('1')


@njit(cache=True, inline='always')
def popcount16(x):
    """Fast popcount for values up to 16 bits using lookup table."""
    return POPCOUNT_TABLE[x & 0xFFFF]


@njit(cache=True, inline='always')
def check_strict_feasibility_fast(hist_mask, hist_len, next_val, MS, MIN_MS):
    """
    Optimized feasibility check using precomputed popcount.
    Replaces inner bit-counting loop with single table lookup.
    """
    new_len = hist_len + 1
    new_mask = (hist_mask << 1) | next_val
    
    if new_len < MS:
        # Partial window: Count current ones with lookup
        masked = new_mask & ((1 << new_len) - 1)
        current_ones = POPCOUNT_TABLE[masked & 0xFFFF]
        remaining_slots = MS - new_len
        if current_ones + remaining_slots < MIN_MS:
            return False, new_mask, new_len
        return True, new_mask, new_len
    else:
        # Full window: Count ones in last MS bits
        ms_mask = (1 << MS) - 1
        window_bits = new_mask & ms_mask
        current_ones = POPCOUNT_TABLE[window_bits & 0xFFFF]
        
        if current_ones < MIN_MS:
            return False, new_mask, MS
        
        # Truncate to MS-1 for storage
        ms_minus_1 = MS - 1
        trunc_mask = new_mask & ((1 << ms_minus_1) - 1)
        return True, trunc_mask, ms_minus_1


@njit(cache=True, inline='always')
def check_strict_feasibility_numba(hist_mask, hist_len, next_val, MS, MIN_MS):
    """
    Check if adding next_val to the history satisfies rolling window constraints.
    OPTIMIZED: Uses precomputed popcount table instead of bit-counting loops.
    """
    new_len = hist_len + 1
    new_mask = (hist_mask << 1) | next_val
    
    if new_len < MS:
        # Partial window: use popcount lookup
        masked = new_mask & ((1 << new_len) - 1)
        current_ones = POPCOUNT_TABLE[masked & 0xFFFF]
        remaining_slots = MS - new_len
        if current_ones + remaining_slots < MIN_MS:
            return False, new_mask, new_len
        return True, new_mask, new_len
    else:
        # Full window: count ones in last MS bits using popcount
        ms_mask = (1 << MS) - 1
        window_bits = new_mask & ms_mask
        current_ones = POPCOUNT_TABLE[window_bits & 0xFFFF]
        
        if current_ones < MIN_MS:
            return False, new_mask, MS
        
        # Truncate to MS-1 for storage
        ms_minus_1 = MS - 1
        trunc_mask = new_mask & ((1 << ms_minus_1) - 1)
        return True, trunc_mask, ms_minus_1

# Type definitions for Dict
# Key: packed int64 (ai_count << 24 | hist_mask << 8 | hist_len)
key_type = types.int64  # OPTIMIZATION A: Packed key for faster hashing
# Value: List of (cost, prog, path_mask)
# Note: We must define the tuple type inside the list
val_tuple_type = types.Tuple((types.float64, types.float64, types.int64))
val_list_type = types.ListType(val_tuple_type)


# Return list type
# (j, rc, start, end, path_mask, prog)
result_tuple_type = types.Tuple((types.float64, types.float64, types.int64, types.int64, types.int64, types.float64))

@njit(cache=True)
def run_fast_path_numba(
    r_k, s_k, gamma_k, obj_mode_float, 
    pi_matrix, # 2D array [worker, time]
    candidate_workers, # Array of worker IDs
    max_time, 
    MS, MIN_MS, 
    theta_lookup, # Array
    epsilon
):
    """
    Optimized DP loop using Numba.
    """
    best_columns = List.empty_list(result_tuple_type)
    
    # Constants
    obj_mode = obj_mode_float
    
    for j in candidate_workers:
        # For each worker
        time_until_end = max_time - r_k + 1
        effective_min_duration = min(int(s_k), time_until_end)
        start_tau = r_k + effective_min_duration - 1
        
        for tau in range(start_tau, max_time + 1):
            is_timeout_scenario = (tau == max_time)
            
            # Initial state setup
            start_cost = -pi_matrix[j, r_k]
            
            # Initialize Dict with explicit types
            current_states = Dict.empty(key_type, val_list_type)
            
            init_ai = 0
            init_hist = 1
            init_hlen = 1
            init_path = 1 # Bit 0 is set
            
            init_key = pack_key(init_ai, init_hist, init_hlen)
            
            # Create list for this bucket
            val_list = List.empty_list(val_tuple_type)
            val_list.append((float64(start_cost), float64(1.0), int64(init_path)))
            current_states[init_key] = val_list
            
            # DP Loop
            for t in range(r_k + 1, tau):
                next_states = Dict.empty(key_type, val_list_type)
                
                # Iterate over current buckets
                for key, bucket in current_states.items():
                    ai_count, hist_mask, hist_len = unpack_key(key)
                    
                    for state in bucket:
                        cost, prog, path_mask = state
                        
                        # Reachability Pruning
                        remaining_steps = tau - t + 1
                        if not is_timeout_scenario:
                            if obj_mode > 0.5:
                                if prog + remaining_steps * 1.0 < s_k - epsilon:
                                    continue
                                    
                        # A: Therapist
                        feasible_ther, new_mask_ther, new_len_ther = check_strict_feasibility_numba(
                            hist_mask, hist_len, 1, MS, MIN_MS
                        )
                        
                        if feasible_ther:
                            cost_ther = cost - pi_matrix[j, t]
                            prog_ther = prog + 1.0
                            
                            new_key_ther = pack_key(ai_count, new_mask_ther, new_len_ther)
                            new_val_ther = (cost_ther, prog_ther, (path_mask | (1 << (t - r_k))))
                            
                            if new_key_ther not in next_states:
                                l = List.empty_list(val_tuple_type)
                                l.append(new_val_ther)
                                next_states[new_key_ther] = l
                            else:
                                bucket_t = next_states[new_key_ther]
                                is_dominated = False
                                for i in range(len(bucket_t)):
                                    c_old, p_old, _ = bucket_t[i]
                                    if c_old <= cost_ther + epsilon and p_old >= prog_ther - epsilon:
                                        is_dominated = True
                                        break
                                
                                if not is_dominated:
                                    clean_bucket = List.empty_list(val_tuple_type)
                                    for i in range(len(bucket_t)):
                                        c_old, p_old, path_old = bucket_t[i]
                                        if cost_ther <= c_old + epsilon and prog_ther >= p_old - epsilon:
                                            pass
                                        else:
                                            clean_bucket.append((c_old, p_old, path_old))
                                    clean_bucket.append(new_val_ther)
                                    next_states[new_key_ther] = clean_bucket

                        # B: AI
                        feasible_ai, new_mask_ai, new_len_ai = check_strict_feasibility_numba(
                            hist_mask, hist_len, 0, MS, MIN_MS
                        )
                        
                        if feasible_ai:
                            cost_ai = cost
                            eff = 1.0
                            if ai_count < len(theta_lookup):
                                eff = theta_lookup[ai_count]
                                
                            prog_ai = prog + eff
                            new_ai_count = ai_count + 1
                            
                            new_key_ai = pack_key(new_ai_count, new_mask_ai, new_len_ai)
                            new_val_ai = (cost_ai, prog_ai, (path_mask))
                            
                            if new_key_ai not in next_states:
                                l = List.empty_list(val_tuple_type)
                                l.append(new_val_ai)
                                next_states[new_key_ai] = l
                            else:
                                bucket_a = next_states[new_key_ai]
                                is_dominated = False
                                for i in range(len(bucket_a)):
                                    c_old, p_old, _ = bucket_a[i]
                                    if c_old <= cost_ai + epsilon and p_old >= prog_ai - epsilon:
                                        is_dominated = True
                                        break
                                
                                if not is_dominated:
                                    clean_bucket = List.empty_list(val_tuple_type)
                                    for i in range(len(bucket_a)):
                                        c_old, p_old, path_old = bucket_a[i]
                                        if cost_ai <= c_old + epsilon and prog_ai >= p_old - epsilon:
                                            pass
                                        else:
                                            clean_bucket.append((c_old, p_old, path_old))
                                    clean_bucket.append(new_val_ai)
                                    next_states[new_key_ai] = clean_bucket

                current_states = next_states
                if len(current_states) == 0:
                    break
            
            # Final Step (Transition to Tau)
            for key, bucket in current_states.items():
                ai_count, hist_mask, hist_len = unpack_key(key)
                
                for state in bucket:
                    cost, prog, path_mask = state
                    
                    # We collect possible end steps
                    # Option 1: End with Therapist (1)
                    feasible_ther, _, _ = check_strict_feasibility_numba(hist_mask, hist_len, 1, MS, MIN_MS)
                    
                    if feasible_ther:
                        final_cost = cost - pi_matrix[j, tau]
                        final_prog = prog + 1.0
                        final_path_mask = path_mask | (1 << (tau - r_k))
                        
                        condition_met = (final_prog >= s_k - epsilon)
                        is_valid = False
                        if obj_mode > 0.5:
                            is_valid = condition_met
                        else:
                            is_valid = condition_met or (tau == max_time)

                        if is_valid:
                            duration_val = (tau - r_k + 1)
                            rc = final_cost + (duration_val * obj_mode) - gamma_k
                            if rc < -1e-6:
                                best_columns.append((float64(j), float64(rc), int64(r_k), int64(tau), int64(final_path_mask), float64(final_prog)))

                    # Option 2: End with App (0) - ONLY if Timeout
                    if is_timeout_scenario:
                        feasible_ai, _, _ = check_strict_feasibility_numba(hist_mask, hist_len, 0, MS, MIN_MS)
                        if feasible_ai:
                            final_cost = cost
                            eff = 1.0
                            if ai_count < len(theta_lookup):
                                eff = theta_lookup[ai_count]
                            final_prog = prog + eff
                            final_path_mask = path_mask # 0 bit implies AI
                            
                            condition_met = (final_prog >= s_k - epsilon)
                            is_valid = False
                            if obj_mode > 0.5:
                                is_valid = condition_met
                            else:
                                is_valid = condition_met or (tau == max_time)
                                
                            if is_valid:
                                duration_val = (tau - r_k + 1)
                                rc = final_cost + (duration_val * obj_mode) - gamma_k
                                if rc < -1e-6:
                                    best_columns.append((float64(j), float64(rc), int64(r_k), int64(tau), int64(final_path_mask), float64(final_prog)))

    return best_columns


# =============================================================================
# OPTIMIZATION 1: ng-PATH LOWER BOUND WITH SUFFIX SUMS
# =============================================================================
# Pre-compute worker-specific suffix sums for aggressive state pruning

@njit(cache=True)
def compute_suffix_sums(pi_matrix, candidate_workers, max_time):
    """
    Compute WORKER-SPECIFIC suffix sums of dual gains.
    
    suffix_sum[worker_idx, t] = sum of -pi[j, tau] for tau from t to max_time
    
    This allows O(1) lookup of the maximum possible remaining dual gain
    for each specific worker from any time t to the end.
    
    Returns:
        2D array of shape (n_workers, max_time + 2)
    """
    n_workers = len(candidate_workers)
    
    # 2D array: suffix_sum[worker_idx, time]
    suffix_sum = np.zeros((n_workers, max_time + 2), dtype=np.float64)
    
    for w_idx in range(n_workers):
        j = candidate_workers[w_idx]
        
        # Compute suffix sums for this worker (from end to start)
        for t in range(max_time, 0, -1):
            gain = -pi_matrix[j, t]  # Dual gain at this time
            suffix_sum[w_idx, t] = suffix_sum[w_idx, t + 1] + gain
    
    return suffix_sum


@njit(cache=True, parallel=True)
def run_fast_path_with_lb_numba(
    r_k, s_k, gamma_k, obj_mode_float, 
    pi_matrix,
    candidate_workers,
    max_time, 
    MS, MIN_MS, 
    theta_lookup,
    epsilon,
    suffix_sum  # Pre-computed suffix sums for LB pruning
):
    """
    Parallel DP with ng-path Lower Bound pruning.
    States are pruned when their optimistic bound cannot yield negative RC.
    
    Pruning condition: cost + suffix_sum[t+1] + remaining_obj - gamma >= 0
    """
    obj_mode = obj_mode_float
    n_workers = len(candidate_workers)
    
    MAX_COLS_PER_WORKER = 200
    result_buffer = np.empty((n_workers, MAX_COLS_PER_WORKER, 6), dtype=np.float64)
    result_counts = np.zeros(n_workers, dtype=np.int64)
    
    for worker_idx in prange(n_workers):
        j = candidate_workers[worker_idx]
        col_count = 0
        
        time_until_end = max_time - r_k + 1
        effective_min_duration = min(int(s_k), time_until_end)
        start_tau = r_k + effective_min_duration - 1
        
        for tau in range(start_tau, max_time + 1):
            if col_count >= MAX_COLS_PER_WORKER:
                break
                
            is_timeout_scenario = (tau == max_time)
            start_cost = -pi_matrix[j, r_k]
            
            current_states = Dict.empty(key_type, val_list_type)
            
            init_key = pack_key(0, 1, 1)
            val_list = List.empty_list(val_tuple_type)
            val_list.append((float64(start_cost), float64(1.0), int64(1)))
            current_states[init_key] = val_list
            
            # DP Loop
            for t in range(r_k + 1, tau):
                next_states = Dict.empty(key_type, val_list_type)
                
                for key, bucket in current_states.items():
                    ai_count, hist_mask, hist_len = unpack_key(key)
                    
                    for state in bucket:
                        cost, prog, path_mask = state
                        
                        # === LOWER BOUND PRUNING (Worker-Specific) ===
                        # Optimistic bound: assume max dual gain for THIS worker at future steps
                        remaining_steps = tau - t + 1
                        optimistic_obj = remaining_steps * obj_mode
                        optimistic_cost = cost + suffix_sum[worker_idx, t + 1]
                        lower_bound = optimistic_cost + optimistic_obj - gamma_k
                        
                        if lower_bound >= -epsilon:
                            continue  # Prune: cannot achieve negative RC
                        
                        # Progress pruning (existing)
                        if not is_timeout_scenario and obj_mode > 0.5:
                            if prog + remaining_steps * 1.0 < s_k - epsilon:
                                continue
                        
                        # A: Therapist
                        feasible_ther, new_mask_ther, new_len_ther = check_strict_feasibility_numba(
                            hist_mask, hist_len, 1, MS, MIN_MS
                        )
                        
                        if feasible_ther:
                            cost_ther = cost - pi_matrix[j, t]
                            prog_ther = prog + 1.0
                            new_key_ther = pack_key(ai_count, new_mask_ther, new_len_ther)
                            new_val_ther = (cost_ther, prog_ther, path_mask | (1 << (t - r_k)))
                            
                            if new_key_ther not in next_states:
                                l = List.empty_list(val_tuple_type)
                                l.append(new_val_ther)
                                next_states[new_key_ther] = l
                            else:
                                bucket_t = next_states[new_key_ther]
                                is_dominated = False
                                for i in range(len(bucket_t)):
                                    c_old, p_old, _ = bucket_t[i]
                                    if c_old <= cost_ther + epsilon and p_old >= prog_ther - epsilon:
                                        is_dominated = True
                                        break
                                if not is_dominated:
                                    clean_bucket = List.empty_list(val_tuple_type)
                                    for i in range(len(bucket_t)):
                                        c_old, p_old, path_old = bucket_t[i]
                                        if not (cost_ther <= c_old + epsilon and prog_ther >= p_old - epsilon):
                                            clean_bucket.append((c_old, p_old, path_old))
                                    clean_bucket.append(new_val_ther)
                                    next_states[new_key_ther] = clean_bucket
                        
                        # B: AI
                        feasible_ai, new_mask_ai, new_len_ai = check_strict_feasibility_numba(
                            hist_mask, hist_len, 0, MS, MIN_MS
                        )
                        
                        if feasible_ai:
                            cost_ai = cost
                            eff = 1.0
                            if ai_count < len(theta_lookup):
                                eff = theta_lookup[ai_count]
                            prog_ai = prog + eff
                            new_ai_count = ai_count + 1
                            new_key_ai = pack_key(new_ai_count, new_mask_ai, new_len_ai)
                            new_val_ai = (cost_ai, prog_ai, path_mask)
                            
                            if new_key_ai not in next_states:
                                l = List.empty_list(val_tuple_type)
                                l.append(new_val_ai)
                                next_states[new_key_ai] = l
                            else:
                                bucket_a = next_states[new_key_ai]
                                is_dominated = False
                                for i in range(len(bucket_a)):
                                    c_old, p_old, _ = bucket_a[i]
                                    if c_old <= cost_ai + epsilon and p_old >= prog_ai - epsilon:
                                        is_dominated = True
                                        break
                                if not is_dominated:
                                    clean_bucket = List.empty_list(val_tuple_type)
                                    for i in range(len(bucket_a)):
                                        c_old, p_old, path_old = bucket_a[i]
                                        if not (cost_ai <= c_old + epsilon and prog_ai >= p_old - epsilon):
                                            clean_bucket.append((c_old, p_old, path_old))
                                    clean_bucket.append(new_val_ai)
                                    next_states[new_key_ai] = clean_bucket
                
                current_states = next_states
                if len(current_states) == 0:
                    break
            
            # Final Step
            for key, bucket in current_states.items():
                ai_count, hist_mask, hist_len = unpack_key(key)
                
                for state in bucket:
                    cost, prog, path_mask = state
                    
                    # End with Therapist
                    feasible_ther, _, _ = check_strict_feasibility_numba(hist_mask, hist_len, 1, MS, MIN_MS)
                    if feasible_ther:
                        final_cost = cost - pi_matrix[j, tau]
                        final_prog = prog + 1.0
                        final_path_mask = path_mask | (1 << (tau - r_k))
                        
                        condition_met = final_prog >= s_k - epsilon
                        is_valid = condition_met if obj_mode > 0.5 else (condition_met or tau == max_time)
                        
                        if is_valid:
                            duration_val = tau - r_k + 1
                            rc = final_cost + duration_val * obj_mode - gamma_k
                            if rc < -1e-6 and col_count < MAX_COLS_PER_WORKER:
                                result_buffer[worker_idx, col_count, 0] = float64(j)
                                result_buffer[worker_idx, col_count, 1] = rc
                                result_buffer[worker_idx, col_count, 2] = float64(r_k)
                                result_buffer[worker_idx, col_count, 3] = float64(tau)
                                result_buffer[worker_idx, col_count, 4] = float64(final_path_mask)
                                result_buffer[worker_idx, col_count, 5] = final_prog
                                col_count += 1
                    
                    # End with AI (timeout only)
                    if is_timeout_scenario:
                        feasible_ai, _, _ = check_strict_feasibility_numba(hist_mask, hist_len, 0, MS, MIN_MS)
                        if feasible_ai:
                            final_cost = cost
                            eff = 1.0
                            if ai_count < len(theta_lookup):
                                eff = theta_lookup[ai_count]
                            final_prog = prog + eff
                            
                            condition_met = final_prog >= s_k - epsilon
                            is_valid = condition_met if obj_mode > 0.5 else (condition_met or tau == max_time)
                            
                            if is_valid:
                                duration_val = tau - r_k + 1
                                rc = final_cost + duration_val * obj_mode - gamma_k
                                if rc < -1e-6 and col_count < MAX_COLS_PER_WORKER:
                                    result_buffer[worker_idx, col_count, 0] = float64(j)
                                    result_buffer[worker_idx, col_count, 1] = rc
                                    result_buffer[worker_idx, col_count, 2] = float64(r_k)
                                    result_buffer[worker_idx, col_count, 3] = float64(tau)
                                    result_buffer[worker_idx, col_count, 4] = float64(path_mask)
                                    result_buffer[worker_idx, col_count, 5] = final_prog
                                    col_count += 1
        
        result_counts[worker_idx] = col_count
    
    # Collect results
    all_columns = List.empty_list(result_tuple_type)
    for worker_idx in range(n_workers):
        for col_idx in range(result_counts[worker_idx]):
            row = result_buffer[worker_idx, col_idx]
            all_columns.append((row[0], row[1], int64(row[2]), int64(row[3]), int64(row[4]), row[5]))
    
    return all_columns


# =============================================================================
# OPTIMIZATION C: PARALLEL VERSION WITH prange
# =============================================================================


@njit(cache=True, parallel=True)
def run_fast_path_parallel_numba(
    r_k, s_k, gamma_k, obj_mode_float, 
    pi_matrix,
    candidate_workers,
    max_time, 
    MS, MIN_MS, 
    theta_lookup,
    epsilon
):
    """
    Parallel DP loop using Numba's prange.
    Each worker is processed in parallel, with results stored in thread-local buffers.
    """
    obj_mode = obj_mode_float
    n_workers = len(candidate_workers)
    
    # Pre-allocate result buffer: max 200 columns per worker
    MAX_COLS_PER_WORKER = 200
    result_buffer = np.empty((n_workers, MAX_COLS_PER_WORKER, 6), dtype=np.float64)
    result_counts = np.zeros(n_workers, dtype=np.int64)
    
    for worker_idx in prange(n_workers):
        j = candidate_workers[worker_idx]
        col_count = 0
        
        time_until_end = max_time - r_k + 1
        effective_min_duration = min(int(s_k), time_until_end)
        start_tau = r_k + effective_min_duration - 1
        
        for tau in range(start_tau, max_time + 1):
            if col_count >= MAX_COLS_PER_WORKER:
                break
                
            is_timeout_scenario = (tau == max_time)
            start_cost = -pi_matrix[j, r_k]
            
            current_states = Dict.empty(key_type, val_list_type)
            
            init_key = pack_key(0, 1, 1)
            val_list = List.empty_list(val_tuple_type)
            val_list.append((float64(start_cost), float64(1.0), int64(1)))
            current_states[init_key] = val_list
            
            # DP Loop
            for t in range(r_k + 1, tau):
                next_states = Dict.empty(key_type, val_list_type)
                
                for key, bucket in current_states.items():
                    ai_count, hist_mask, hist_len = unpack_key(key)
                    
                    for state in bucket:
                        cost, prog, path_mask = state
                        
                        remaining_steps = tau - t + 1
                        if not is_timeout_scenario and obj_mode > 0.5:
                            if prog + remaining_steps * 1.0 < s_k - epsilon:
                                continue
                        
                        # A: Therapist
                        feasible_ther, new_mask_ther, new_len_ther = check_strict_feasibility_numba(
                            hist_mask, hist_len, 1, MS, MIN_MS
                        )
                        
                        if feasible_ther:
                            cost_ther = cost - pi_matrix[j, t]
                            prog_ther = prog + 1.0
                            new_key_ther = pack_key(ai_count, new_mask_ther, new_len_ther)
                            new_val_ther = (cost_ther, prog_ther, path_mask | (1 << (t - r_k)))
                            
                            if new_key_ther not in next_states:
                                l = List.empty_list(val_tuple_type)
                                l.append(new_val_ther)
                                next_states[new_key_ther] = l
                            else:
                                bucket_t = next_states[new_key_ther]
                                is_dominated = False
                                for i in range(len(bucket_t)):
                                    c_old, p_old, _ = bucket_t[i]
                                    if c_old <= cost_ther + epsilon and p_old >= prog_ther - epsilon:
                                        is_dominated = True
                                        break
                                if not is_dominated:
                                    clean_bucket = List.empty_list(val_tuple_type)
                                    for i in range(len(bucket_t)):
                                        c_old, p_old, path_old = bucket_t[i]
                                        if not (cost_ther <= c_old + epsilon and prog_ther >= p_old - epsilon):
                                            clean_bucket.append((c_old, p_old, path_old))
                                    clean_bucket.append(new_val_ther)
                                    next_states[new_key_ther] = clean_bucket
                        
                        # B: AI
                        feasible_ai, new_mask_ai, new_len_ai = check_strict_feasibility_numba(
                            hist_mask, hist_len, 0, MS, MIN_MS
                        )
                        
                        if feasible_ai:
                            cost_ai = cost
                            eff = 1.0
                            if ai_count < len(theta_lookup):
                                eff = theta_lookup[ai_count]
                            prog_ai = prog + eff
                            new_ai_count = ai_count + 1
                            new_key_ai = pack_key(new_ai_count, new_mask_ai, new_len_ai)
                            new_val_ai = (cost_ai, prog_ai, path_mask)
                            
                            if new_key_ai not in next_states:
                                l = List.empty_list(val_tuple_type)
                                l.append(new_val_ai)
                                next_states[new_key_ai] = l
                            else:
                                bucket_a = next_states[new_key_ai]
                                is_dominated = False
                                for i in range(len(bucket_a)):
                                    c_old, p_old, _ = bucket_a[i]
                                    if c_old <= cost_ai + epsilon and p_old >= prog_ai - epsilon:
                                        is_dominated = True
                                        break
                                if not is_dominated:
                                    clean_bucket = List.empty_list(val_tuple_type)
                                    for i in range(len(bucket_a)):
                                        c_old, p_old, path_old = bucket_a[i]
                                        if not (cost_ai <= c_old + epsilon and prog_ai >= p_old - epsilon):
                                            clean_bucket.append((c_old, p_old, path_old))
                                    clean_bucket.append(new_val_ai)
                                    next_states[new_key_ai] = clean_bucket
                
                current_states = next_states
                if len(current_states) == 0:
                    break
            
            # Final Step
            for key, bucket in current_states.items():
                ai_count, hist_mask, hist_len = unpack_key(key)
                
                for state in bucket:
                    cost, prog, path_mask = state
                    
                    # End with Therapist
                    feasible_ther, _, _ = check_strict_feasibility_numba(hist_mask, hist_len, 1, MS, MIN_MS)
                    if feasible_ther:
                        final_cost = cost - pi_matrix[j, tau]
                        final_prog = prog + 1.0
                        final_path_mask = path_mask | (1 << (tau - r_k))
                        
                        condition_met = final_prog >= s_k - epsilon
                        is_valid = condition_met if obj_mode > 0.5 else (condition_met or tau == max_time)
                        
                        if is_valid:
                            duration_val = tau - r_k + 1
                            rc = final_cost + duration_val * obj_mode - gamma_k
                            if rc < -1e-6 and col_count < MAX_COLS_PER_WORKER:
                                result_buffer[worker_idx, col_count, 0] = float64(j)
                                result_buffer[worker_idx, col_count, 1] = rc
                                result_buffer[worker_idx, col_count, 2] = float64(r_k)
                                result_buffer[worker_idx, col_count, 3] = float64(tau)
                                result_buffer[worker_idx, col_count, 4] = float64(final_path_mask)
                                result_buffer[worker_idx, col_count, 5] = final_prog
                                col_count += 1
                    
                    # End with AI (timeout only)
                    if is_timeout_scenario:
                        feasible_ai, _, _ = check_strict_feasibility_numba(hist_mask, hist_len, 0, MS, MIN_MS)
                        if feasible_ai:
                            final_cost = cost
                            eff = 1.0
                            if ai_count < len(theta_lookup):
                                eff = theta_lookup[ai_count]
                            final_prog = prog + eff
                            
                            condition_met = final_prog >= s_k - epsilon
                            is_valid = condition_met if obj_mode > 0.5 else (condition_met or tau == max_time)
                            
                            if is_valid:
                                duration_val = tau - r_k + 1
                                rc = final_cost + duration_val * obj_mode - gamma_k
                                if rc < -1e-6 and col_count < MAX_COLS_PER_WORKER:
                                    result_buffer[worker_idx, col_count, 0] = float64(j)
                                    result_buffer[worker_idx, col_count, 1] = rc
                                    result_buffer[worker_idx, col_count, 2] = float64(r_k)
                                    result_buffer[worker_idx, col_count, 3] = float64(tau)
                                    result_buffer[worker_idx, col_count, 4] = float64(path_mask)
                                    result_buffer[worker_idx, col_count, 5] = final_prog
                                    col_count += 1
        
        result_counts[worker_idx] = col_count
    
    # Collect results into list
    total_cols = int(np.sum(result_counts))
    all_columns = List.empty_list(result_tuple_type)
    for worker_idx in range(n_workers):
        for col_idx in range(result_counts[worker_idx]):
            row = result_buffer[worker_idx, col_idx]
            all_columns.append((row[0], row[1], int64(row[2]), int64(row[3]), int64(row[4]), row[5]))
    
    return all_columns


# =============================================================================
# OPTIMIZATION D: DOUBLE BUFFERING WITH FLAT ARRAYS
# =============================================================================
# Combined A+C+D: State-packing, parallel prange, and double-buffered flat arrays

# State buffer layout: [cost, prog, path_mask, packed_key]
# Using 2D arrays and counters instead of Dict+List

@njit(cache=True, parallel=True)
def run_fast_path_optimized_numba(
    r_k, s_k, gamma_k, obj_mode_float, 
    pi_matrix,
    candidate_workers,
    max_time, 
    MS, MIN_MS, 
    theta_lookup,
    epsilon
):
    """
    Fully optimized DP with:
    - A: State-packing (int64 keys)
    - C: prange parallelization
    - D: Double buffering with flat arrays
    """
    obj_mode = obj_mode_float
    n_workers = len(candidate_workers)
    
    # Constants for buffer sizing
    MAX_STATES = 2000      # Max states per worker per tau iteration
    MAX_PARETO = 8         # Max Pareto-optimal points per bucket
    MAX_COLS_PER_WORKER = 200
    
    # Result buffer
    result_buffer = np.empty((n_workers, MAX_COLS_PER_WORKER, 6), dtype=np.float64)
    result_counts = np.zeros(n_workers, dtype=np.int64)
    
    for worker_idx in prange(n_workers):
        j = candidate_workers[worker_idx]
        col_count = 0
        
        time_until_end = max_time - r_k + 1
        effective_min_duration = min(int(s_k), time_until_end)
        start_tau = r_k + effective_min_duration - 1
        
        # Double buffer: current and next state arrays
        # Each row: [cost, prog, path_mask, packed_key]
        buffer_a = np.empty((MAX_STATES, 4), dtype=np.float64)
        buffer_b = np.empty((MAX_STATES, 4), dtype=np.float64)
        
        for tau in range(start_tau, max_time + 1):
            if col_count >= MAX_COLS_PER_WORKER:
                break
                
            is_timeout_scenario = (tau == max_time)
            start_cost = -pi_matrix[j, r_k]
            
            # Initialize with first state
            current = buffer_a
            current_count = 1
            current[0, 0] = start_cost  # cost
            current[0, 1] = 1.0         # prog
            current[0, 2] = 1.0         # path_mask (bit 0 set)
            current[0, 3] = float64(pack_key(0, 1, 1))  # packed key
            
            # DP Loop
            for t in range(r_k + 1, tau):
                # Swap buffers
                if current is buffer_a:
                    next_buf = buffer_b
                else:
                    next_buf = buffer_a
                next_count = 0
                
                for state_idx in range(current_count):
                    cost = current[state_idx, 0]
                    prog = current[state_idx, 1]
                    path_mask = int64(current[state_idx, 2])
                    packed_key = int64(current[state_idx, 3])
                    ai_count, hist_mask, hist_len = unpack_key(packed_key)
                    
                    # Reachability pruning
                    remaining_steps = tau - t + 1
                    if not is_timeout_scenario and obj_mode > 0.5:
                        if prog + remaining_steps * 1.0 < s_k - epsilon:
                            continue
                    
                    # A: Therapist
                    feasible_ther, new_mask_ther, new_len_ther = check_strict_feasibility_numba(
                        hist_mask, hist_len, 1, MS, MIN_MS
                    )
                    
                    if feasible_ther and next_count < MAX_STATES:
                        cost_ther = cost - pi_matrix[j, t]
                        prog_ther = prog + 1.0
                        new_key = pack_key(ai_count, new_mask_ther, new_len_ther)
                        new_path = path_mask | (1 << (t - r_k))
                        
                        # Simple dominance check against existing states with same key
                        is_dominated = False
                        for k in range(next_count):
                            if int64(next_buf[k, 3]) == new_key:
                                if next_buf[k, 0] <= cost_ther + epsilon and next_buf[k, 1] >= prog_ther - epsilon:
                                    is_dominated = True
                                    break
                        
                        if not is_dominated:
                            next_buf[next_count, 0] = cost_ther
                            next_buf[next_count, 1] = prog_ther
                            next_buf[next_count, 2] = float64(new_path)
                            next_buf[next_count, 3] = float64(new_key)
                            next_count += 1
                    
                    # B: AI
                    feasible_ai, new_mask_ai, new_len_ai = check_strict_feasibility_numba(
                        hist_mask, hist_len, 0, MS, MIN_MS
                    )
                    
                    if feasible_ai and next_count < MAX_STATES:
                        cost_ai = cost
                        eff = 1.0
                        if ai_count < len(theta_lookup):
                            eff = theta_lookup[ai_count]
                        prog_ai = prog + eff
                        new_ai_count = ai_count + 1
                        new_key = pack_key(new_ai_count, new_mask_ai, new_len_ai)
                        
                        is_dominated = False
                        for k in range(next_count):
                            if int64(next_buf[k, 3]) == new_key:
                                if next_buf[k, 0] <= cost_ai + epsilon and next_buf[k, 1] >= prog_ai - epsilon:
                                    is_dominated = True
                                    break
                        
                        if not is_dominated:
                            next_buf[next_count, 0] = cost_ai
                            next_buf[next_count, 1] = prog_ai
                            next_buf[next_count, 2] = float64(path_mask)
                            next_buf[next_count, 3] = float64(new_key)
                            next_count += 1
                
                # Swap
                current = next_buf
                current_count = next_count
                
                if current_count == 0:
                    break
            
            # Final Step
            for state_idx in range(current_count):
                cost = current[state_idx, 0]
                prog = current[state_idx, 1]
                path_mask = int64(current[state_idx, 2])
                packed_key = int64(current[state_idx, 3])
                ai_count, hist_mask, hist_len = unpack_key(packed_key)
                
                # End with Therapist
                feasible_ther, _, _ = check_strict_feasibility_numba(hist_mask, hist_len, 1, MS, MIN_MS)
                if feasible_ther:
                    final_cost = cost - pi_matrix[j, tau]
                    final_prog = prog + 1.0
                    final_path = path_mask | (1 << (tau - r_k))
                    
                    condition_met = final_prog >= s_k - epsilon
                    is_valid = condition_met if obj_mode > 0.5 else (condition_met or tau == max_time)
                    
                    if is_valid:
                        duration_val = tau - r_k + 1
                        rc = final_cost + duration_val * obj_mode - gamma_k
                        if rc < -1e-6 and col_count < MAX_COLS_PER_WORKER:
                            result_buffer[worker_idx, col_count, 0] = float64(j)
                            result_buffer[worker_idx, col_count, 1] = rc
                            result_buffer[worker_idx, col_count, 2] = float64(r_k)
                            result_buffer[worker_idx, col_count, 3] = float64(tau)
                            result_buffer[worker_idx, col_count, 4] = float64(final_path)
                            result_buffer[worker_idx, col_count, 5] = final_prog
                            col_count += 1
                
                # End with AI (timeout only)
                if is_timeout_scenario:
                    feasible_ai, _, _ = check_strict_feasibility_numba(hist_mask, hist_len, 0, MS, MIN_MS)
                    if feasible_ai:
                        final_cost = cost
                        eff = 1.0
                        if ai_count < len(theta_lookup):
                            eff = theta_lookup[ai_count]
                        final_prog = prog + eff
                        
                        condition_met = final_prog >= s_k - epsilon
                        is_valid = condition_met if obj_mode > 0.5 else (condition_met or tau == max_time)
                        
                        if is_valid:
                            duration_val = tau - r_k + 1
                            rc = final_cost + duration_val * obj_mode - gamma_k
                            if rc < -1e-6 and col_count < MAX_COLS_PER_WORKER:
                                result_buffer[worker_idx, col_count, 0] = float64(j)
                                result_buffer[worker_idx, col_count, 1] = rc
                                result_buffer[worker_idx, col_count, 2] = float64(r_k)
                                result_buffer[worker_idx, col_count, 3] = float64(tau)
                                result_buffer[worker_idx, col_count, 4] = float64(path_mask)
                                result_buffer[worker_idx, col_count, 5] = final_prog
                                col_count += 1
        
        result_counts[worker_idx] = col_count
    
    # Collect results
    all_columns = List.empty_list(result_tuple_type)
    for worker_idx in range(n_workers):
        for col_idx in range(result_counts[worker_idx]):
            row = result_buffer[worker_idx, col_idx]
            all_columns.append((row[0], row[1], int64(row[2]), int64(row[3]), int64(row[4]), row[5]))
    
    return all_columns


@njit(cache=True)

def run_fast_path_single_worker_numba(
    j,  # Single worker ID
    r_k, s_k, gamma_k, obj_mode_float, 
    pi_matrix,
    max_time, 
    MS, MIN_MS, 
    theta_lookup,
    epsilon
):
    """
    Optimized DP loop for a SINGLE worker.
    This function can be called in parallel from Python using multiprocessing.
    
    Returns:
        List of columns for this worker
    """
    best_columns = List.empty_list(result_tuple_type)
    obj_mode = obj_mode_float
    
    time_until_end = max_time - r_k + 1
    effective_min_duration = min(int(s_k), time_until_end)
    start_tau = r_k + effective_min_duration - 1
    
    for tau in range(start_tau, max_time + 1):
        is_timeout_scenario = (tau == max_time)
        start_cost = -pi_matrix[j, r_k]
        
        current_states = Dict.empty(key_type, val_list_type)
        
        init_ai = 0
        init_hist = 1
        init_hlen = 1
        init_path = 1
        
        init_key = pack_key(init_ai, init_hist, init_hlen)
        val_list = List.empty_list(val_tuple_type)
        val_list.append((float64(start_cost), float64(1.0), int64(init_path)))
        current_states[init_key] = val_list
        
        # DP Loop
        for t in range(r_k + 1, tau):
            next_states = Dict.empty(key_type, val_list_type)
            
            for key, bucket in current_states.items():
                ai_count, hist_mask, hist_len = unpack_key(key)
                
                for state in bucket:
                    cost, prog, path_mask = state
                    
                    # Reachability Pruning
                    remaining_steps = tau - t + 1
                    if not is_timeout_scenario:
                        if obj_mode > 0.5:
                            if prog + remaining_steps * 1.0 < s_k - epsilon:
                                continue
                    
                    # A: Therapist
                    feasible_ther, new_mask_ther, new_len_ther = check_strict_feasibility_numba(
                        hist_mask, hist_len, 1, MS, MIN_MS
                    )
                    
                    if feasible_ther:
                        cost_ther = cost - pi_matrix[j, t]
                        prog_ther = prog + 1.0
                        
                        new_key_ther = pack_key(ai_count, new_mask_ther, new_len_ther)
                        new_val_ther = (cost_ther, prog_ther, (path_mask | (1 << (t - r_k))))
                        
                        if new_key_ther not in next_states:
                            l = List.empty_list(val_tuple_type)
                            l.append(new_val_ther)
                            next_states[new_key_ther] = l
                        else:
                            bucket_t = next_states[new_key_ther]
                            is_dominated = False
                            for i in range(len(bucket_t)):
                                c_old, p_old, _ = bucket_t[i]
                                if c_old <= cost_ther + epsilon and p_old >= prog_ther - epsilon:
                                    is_dominated = True
                                    break
                            
                            if not is_dominated:
                                clean_bucket = List.empty_list(val_tuple_type)
                                for i in range(len(bucket_t)):
                                    c_old, p_old, path_old = bucket_t[i]
                                    if cost_ther <= c_old + epsilon and prog_ther >= p_old - epsilon:
                                        pass
                                    else:
                                        clean_bucket.append((c_old, p_old, path_old))
                                clean_bucket.append(new_val_ther)
                                next_states[new_key_ther] = clean_bucket
                    
                    # B: AI
                    feasible_ai, new_mask_ai, new_len_ai = check_strict_feasibility_numba(
                        hist_mask, hist_len, 0, MS, MIN_MS
                    )
                    
                    if feasible_ai:
                        cost_ai = cost
                        eff = 1.0
                        if ai_count < len(theta_lookup):
                            eff = theta_lookup[ai_count]
                        prog_ai = prog + eff
                        new_ai_count = ai_count + 1
                        
                        new_key_ai = pack_key(new_ai_count, new_mask_ai, new_len_ai)
                        new_val_ai = (cost_ai, prog_ai, path_mask)
                        
                        if new_key_ai not in next_states:
                            l = List.empty_list(val_tuple_type)
                            l.append(new_val_ai)
                            next_states[new_key_ai] = l
                        else:
                            bucket_a = next_states[new_key_ai]
                            is_dominated = False
                            for i in range(len(bucket_a)):
                                c_old, p_old, _ = bucket_a[i]
                                if c_old <= cost_ai + epsilon and p_old >= prog_ai - epsilon:
                                    is_dominated = True
                                    break
                            
                            if not is_dominated:
                                clean_bucket = List.empty_list(val_tuple_type)
                                for i in range(len(bucket_a)):
                                    c_old, p_old, path_old = bucket_a[i]
                                    if cost_ai <= c_old + epsilon and prog_ai >= p_old - epsilon:
                                        pass
                                    else:
                                        clean_bucket.append((c_old, p_old, path_old))
                                clean_bucket.append(new_val_ai)
                                next_states[new_key_ai] = clean_bucket
            
            current_states = next_states
            if len(current_states) == 0:
                break
        
        # Final Step
        for key, bucket in current_states.items():
            ai_count, hist_mask, hist_len = unpack_key(key)
            
            for state in bucket:
                cost, prog, path_mask = state
                
                # Option 1: End with Therapist
                feasible_ther, _, _ = check_strict_feasibility_numba(hist_mask, hist_len, 1, MS, MIN_MS)
                
                if feasible_ther:
                    final_cost = cost - pi_matrix[j, tau]
                    final_prog = prog + 1.0
                    final_path_mask = path_mask | (1 << (tau - r_k))
                    
                    condition_met = (final_prog >= s_k - epsilon)
                    is_valid = False
                    if obj_mode > 0.5:
                        is_valid = condition_met
                    else:
                        is_valid = condition_met or (tau == max_time)
                    
                    if is_valid:
                        duration_val = tau - r_k + 1
                        rc = final_cost + (duration_val * obj_mode) - gamma_k
                        if rc < -1e-6:
                            best_columns.append((float64(j), float64(rc), int64(r_k), int64(tau), int64(final_path_mask), float64(final_prog)))
                
                # Option 2: End with AI (only on timeout)
                if is_timeout_scenario:
                    feasible_ai, _, _ = check_strict_feasibility_numba(hist_mask, hist_len, 0, MS, MIN_MS)
                    if feasible_ai:
                        final_cost = cost
                        eff = 1.0
                        if ai_count < len(theta_lookup):
                            eff = theta_lookup[ai_count]
                        final_prog = prog + eff
                        final_path_mask = path_mask
                        
                        condition_met = (final_prog >= s_k - epsilon)
                        is_valid = False
                        if obj_mode > 0.5:
                            is_valid = condition_met
                        else:
                            is_valid = condition_met or (tau == max_time)
                        
                        if is_valid:
                            duration_val = tau - r_k + 1
                            rc = final_cost + (duration_val * obj_mode) - gamma_k
                            if rc < -1e-6:
                                best_columns.append((float64(j), float64(rc), int64(r_k), int64(tau), int64(final_path_mask), float64(final_prog)))
    
    return best_columns


# =============================================================================
# BRANCHING CONSTRAINTS SUPPORT
# =============================================================================
# Extended state tuple types for branching constraints
# State now includes: cost, prog, path_mask, zeta_mask (for MP branching)
# Key now includes: ai_count, hist_mask, hist_len, zeta_mask

# Key type WITH zeta and mu: (ai_count, hist_mask, hist_len, zeta_mask, mu_encoded)
key_with_zeta_type = types.Tuple((types.int64, types.int64, types.int64, types.int64, types.int64))
val_with_zeta_type = types.Tuple((types.float64, types.float64, types.int64))  # cost, prog, path_mask
val_list_with_zeta_type = types.ListType(val_with_zeta_type)


@njit(cache=True)
def run_with_branching_constraints_numba(
    r_k, s_k, gamma_k, obj_mode_float,
    pi_matrix,           # 2D array [worker, time]
    candidate_workers,   # Array of worker IDs
    max_time,
    MS, MIN_MS,
    theta_lookup,
    epsilon,
    # === SP Variable Fixing (B.1) ===
    forbidden_mask,      # 2D bool array [worker, time] - True if fixed to 0
    required_mask,       # 2D bool array [worker, time] - True if fixed to 1
    has_sp_fixing,       # bool - whether any SP fixes are active
    # === MP No-Good Cuts (B.2) ===
    nogood_patterns,     # 3D array [cut_idx, worker, time] - 1 if that (w,t) is in forbidden pattern
    num_nogood_cuts,     # int - number of active no-good cuts
    has_nogood_cuts,     # bool - whether any no-good cuts are active
    # === SP Pattern Branching (B.3) ===
    left_pattern_elements,   # 2D array [pattern_idx, flat_idx] containing encoded (w*1000+t) or -1
    left_pattern_limits,     # 1D array [pattern_idx] - max allowed coverage
    num_left_patterns,       # int
    has_left_patterns,       # bool
    # === SP Right Pattern Branching (B.3.2) ===
    right_pattern_elements,  # 2D array [pat_idx, elem_idx] encoded (w*1M+t)
    right_pattern_starts,    # 1D array [pat_idx] start time of pattern
    right_pattern_duals,     # 1D array [pat_idx] dual reward
    right_pattern_counts,    # 1D array [pat_idx] number of elements in pattern
    num_right_patterns,      # int
    has_right_patterns       # bool
):
    """
    Extended DP loop with branching constraint support.
    
    Supports:
    - SP Variable Fixing: forbidden_mask[j,t] = True means x[j,t] must be 0
    - MP No-Good Cuts: Track zeta deviation vector via bitmask
    - SP Left Pattern Branching: Track rho counters, prune if limit exceeded
    - SP Right Pattern Branching: Track mu modes (0=exclude, 1=cover), apply duals
    """
    best_columns = List.empty_list(result_tuple_type)
    obj_mode = obj_mode_float
    
    for j in candidate_workers:
        # === SP Variable Fixing Check at r_k ===
        # First time step MUST be therapist (1), check if it's forbidden
        if has_sp_fixing and forbidden_mask[j, r_k]:
            # This worker/time is fixed to 0, but we need 1 -> skip this worker
            continue
        # If required_mask is set, it's consistent (we need 1, it's required to be 1)
        
        time_until_end = max_time - r_k + 1
        effective_min_duration = min(int(s_k), time_until_end)
        start_tau = r_k + effective_min_duration - 1
        
        for tau in range(start_tau, max_time + 1):
            is_timeout_scenario = (tau == max_time)
            start_cost = -pi_matrix[j, r_k]
            
            # === Initialize Zeta for MP Branching ===
            # zeta_mask: bit i = 1 if we've deviated from no-good cut i
            # At start (time r_k, action=1): check if any cut has forbidden_val != 1 at (j, r_k)
            init_zeta = int64(0)
            if has_nogood_cuts:
                for cut_idx in range(num_nogood_cuts):
                    forbidden_val = nogood_patterns[cut_idx, j, r_k]
                    if forbidden_val != 1:  # We took 1, forbidden was not 1 -> deviated
                        init_zeta = init_zeta | (1 << cut_idx)
            
            # === Initialize Rho for SP Left Patterns ===
            # For simplicity, we encode rho as a single int64 with 8 bits per pattern (max 8 patterns, max count 255)
            init_rho = int64(0)
            if has_left_patterns:
                for pat_idx in range(num_left_patterns):
                    # Check if (j, r_k) is in this pattern's elements
                    for elem_idx in range(left_pattern_elements.shape[1]):
                        encoded = left_pattern_elements[pat_idx, elem_idx]
                        if encoded < 0:
                            break
                        w_pat = encoded // 1000000
                        t_pat = encoded % 1000000
                        if w_pat == j and t_pat == r_k:
                            # Increment rho for this pattern
                            current_rho = (init_rho >> (pat_idx * 8)) & 0xFF
                            current_rho += 1
                            # Check limit
                            if current_rho > left_pattern_limits[pat_idx]:
                                init_rho = int64(-1)  # Signal: pruned
                                break
                            # Update rho
                            init_rho = (init_rho & ~(0xFF << (pat_idx * 8))) | (current_rho << (pat_idx * 8))
                    if init_rho == -1:
                        break
            
            if init_rho == -1:
                continue  # This starting state is already infeasible
            
            # === Initialize Mu for SP Right Patterns ===
            # Encoded in int64, 2 bits per pattern (0=inactive/exclude, 1=cover, 2=violated/prune)
            # Actually, logic is:
            # Mode 0: Inactive / Wait
            # Mode 1: Exclude (entered at start, committed to NOT cover)
            # Mode 2: Cover (entered at start, committed to cover)
            # Wait, let's stick to Python logic:
            # - Before start: implicitly Inactive
            # - At start time: Decision -> Enter Cover (1) or Exclude (0)
            # - After start: Follow mode rules
            # We need to map this to "state". 
            # Let's use 2 bits: 00=inactive/exclude(default), 01=cover, 11=pruned?
            # Better: 
            #   0: Exclude/Inactive (Default) - If we hit a required element, we MUST NOT take it (if in window)
            #   1: Cover - We MUST take all required elements
            
            # Python logic review:
            # if time < t_start: continue (wait)
            # if time == t_start: 
            #    if in_pattern: enter Cover (next_mu=1)
            #    else: enter Exclude (next_mu=0)
            # if time > t_start:
            #    if Exclude (0): if in_pattern -> Prune
            #    if Cover (1): if element is active at t and we don't take it -> Prune
            
            # So state is binary: 0 or 1.
            # But we also need to know if we have "started" yet. 
            # Actually, `time` implicitly tells us if we passed `t_start`.
            # So a single bit per pattern is enough: 0=Exclude, 1=Cover.
            # Default init is 0.
            
            # Initial Check at r_k:
            init_mu = int64(0)
            if has_right_patterns:
                for pat_idx in range(num_right_patterns):
                    t_start = right_pattern_starts[pat_idx]
                    
                    # Is current (j, r_k) in pattern?
                    in_pattern = False
                    for elem_idx in range(right_pattern_counts[pat_idx]):
                        encoded = right_pattern_elements[pat_idx, elem_idx]
                        w_pat = encoded // 1000000
                        t_pat = encoded % 1000000
                        if w_pat == j and t_pat == r_k:
                            in_pattern = True
                            break
                    
                    if r_k == t_start:
                        if in_pattern:
                            # Enter Cover Mode
                            init_mu = init_mu | (1 << pat_idx)
                        else:
                            # Enter Exclude Mode (bit remains 0)
                            pass
                    elif r_k > t_start:
                        # Should have started earlier. Since we start at r_k, we missed the start?
                        # If a pattern started before r_k, and we start a NEW column at r_k, 
                        # technically the "past" is empty.
                        # Wait, columns represent a full worker shift.
                        # If start > t_start, we missed the start trigger.
                        # Implies we can never "Cover" it fully from the start?
                        # Actually, if we start late, we can't have covered the start element.
                        # So we effectively are in Exclude mode (having missed the start).
                        # Checks:
                        # If Exclude (0): if in_pattern -> Prune
                         if in_pattern:
                             # We are picking an element but we are in Exclude mode (implicitly, since we missed start or chose Exclude)
                             init_mu = int64(-1)
                             break
            
            if init_mu == -1:
                continue

            # Initialize state dict
            # Key: (ai_count, hist_mask, hist_len, zeta_mask, mu_encoded)
            current_states = Dict.empty(key_with_zeta_type, val_list_with_zeta_type)
            
            init_ai = int64(0)
            init_hist = int64(1)
            init_hlen = int64(1)
            init_path = int64(1)
            
            # Combine zeta and rho for the key field `zeta_mask` to save space if needed, 
            # BUT we added a new field for mu. Let's keep zeta and rho packed in `zeta_mask` for now 
            # to minimize key size change, OR use the new 5-tuple key properly.
            # Let's use the new 5-tuple key: (ai, hist, hlen, zeta_rho, mu)
            # where zeta_rho = zeta | (rho << 32)
            
            combined_zeta_rho = init_zeta | (init_rho << 32)
            
            init_key = (init_ai, init_hist, init_hlen, combined_zeta_rho, init_mu)
            val_list = List.empty_list(val_with_zeta_type)
            val_list.append((float64(start_cost), float64(1.0), init_path))
            current_states[init_key] = val_list
            
            # DP Loop
            for t in range(r_k + 1, tau):
                next_states = Dict.empty(key_with_zeta_type, val_list_with_zeta_type)
                
                for key, bucket in current_states.items():
                    ai_count, hist_mask, hist_len, combined_zeta_rho, mu_encoded = key
                    zeta_mask = combined_zeta_rho & 0xFFFFFFFF
                    rho_encoded = combined_zeta_rho >> 32
                    
                    for state in bucket:
                        cost, prog, path_mask = state
                        
                        # Reachability Pruning
                        remaining_steps = tau - t + 1
                        if not is_timeout_scenario and obj_mode > 0.5:
                            if prog + remaining_steps * 1.0 < s_k - epsilon:
                                continue
                        
                        # === A: Therapist (action = 1) ===
                        can_take_therapist = True
                        if has_sp_fixing and forbidden_mask[j, t]:
                            can_take_therapist = False
                        
                        if can_take_therapist:
                            feasible_ther, new_mask_ther, new_len_ther = check_strict_feasibility_numba(
                                hist_mask, hist_len, 1, MS, MIN_MS
                            )
                            
                            if feasible_ther:
                                cost_ther = cost - pi_matrix[j, t]
                                prog_ther = prog + 1.0
                                
                                # Update zeta
                                new_zeta = zeta_mask
                                if has_nogood_cuts:
                                    for cut_idx in range(num_nogood_cuts):
                                        if (new_zeta >> cut_idx) & 1 == 0:
                                            forbidden_val = nogood_patterns[cut_idx, j, t]
                                            if forbidden_val != 1:
                                                new_zeta = new_zeta | (1 << cut_idx)
                                
                                # Update rho
                                new_rho = rho_encoded
                                rho_valid = True
                                if has_left_patterns:
                                    for pat_idx in range(num_left_patterns):
                                        for elem_idx in range(left_pattern_elements.shape[1]):
                                            encoded = left_pattern_elements[pat_idx, elem_idx]
                                            if encoded < 0: break
                                            w_pat = encoded // 1000000
                                            t_pat = encoded % 1000000
                                            if w_pat == j and t_pat == t:
                                                current_rho = (new_rho >> (pat_idx * 8)) & 0xFF
                                                current_rho += 1
                                                if current_rho > left_pattern_limits[pat_idx]:
                                                    rho_valid = False; break
                                                new_rho = (new_rho & ~(0xFF << (pat_idx * 8))) | (current_rho << (pat_idx * 8))
                                        if not rho_valid: break
                                
                                # Update mu (Right Pattern)
                                new_mu = mu_encoded
                                mu_valid = True
                                if has_right_patterns and rho_valid:
                                    for pat_idx in range(num_right_patterns):
                                        t_start = right_pattern_starts[pat_idx]
                                        current_mode = (new_mu >> pat_idx) & 1
                                        
                                        in_pattern = False
                                        for elem_idx in range(right_pattern_counts[pat_idx]):
                                            encoded = right_pattern_elements[pat_idx, elem_idx]
                                            w_pat = encoded // 1000000
                                            t_pat = encoded % 1000000
                                            if w_pat == j and t_pat == t:
                                                in_pattern = True
                                                break
                                        
                                        if t == t_start:
                                            if in_pattern:
                                                # Enter Cover (Set bit to 1)
                                                new_mu = new_mu | (1 << pat_idx)
                                            else:
                                                # Enter Exclude (Set bit to 0 - clear it)
                                                new_mu = new_mu & ~(1 << pat_idx)
                                        elif t > t_start:
                                            if current_mode == 0: # Exclude
                                                if in_pattern:
                                                    mu_valid = False; break # Pruned
                                            else: # Cover Mode (1)
                                                # If element is available at t but we took something else?
                                                # Here we TOOK Therapist (1).
                                                # If (j,t) is in pattern, we are good (we took it).
                                                # If (j,t) is NOT in pattern, is that a problem?
                                                # Pattern definition: "Cover" means take ALL elements in P.
                                                # So if (j,t) is in P, we MUST take it. We did.
                                                # If (j,t) is NOT in P, we took Therapist. That's allowed in Cover mode?
                                                # Yes, usually Cover only mandates specific (j,t) to be 1. It doesn't forbid others from being 1.
                                                # So: If InPattern -> Good. If NotInPattern -> Good.
                                                # Wait, logic check: "Cover" means x[j,t]=1 for all (j,t) in P.
                                                # Here we set x[j,t]=1.
                                                # Implementation detail: For Cover, we must ensure we NEVER MISS a required one.
                                                # Since we took 1, we satisfy any requirement at (j,t) if it exists.
                                                pass

                                if rho_valid and mu_valid:
                                    new_combined = new_zeta | (new_rho << 32)
                                    new_key_ther = (ai_count, new_mask_ther, new_len_ther, new_combined, new_mu)
                                    new_val_ther = (cost_ther, prog_ther, path_mask | (1 << (t - r_k)))
                                    
                                    # Add to next_states with dominance
                                    if new_key_ther not in next_states:
                                        l = List.empty_list(val_with_zeta_type)
                                        l.append(new_val_ther)
                                        next_states[new_key_ther] = l
                                    else:
                                        bucket_t = next_states[new_key_ther]
                                        is_dominated = False
                                        for i in range(len(bucket_t)):
                                            c_old, p_old, _ = bucket_t[i]
                                            if c_old <= cost_ther + epsilon and p_old >= prog_ther - epsilon:
                                                is_dominated = True; break
                                        if not is_dominated:
                                            clean = List.empty_list(val_with_zeta_type)
                                            for i in range(len(bucket_t)):
                                                c_old, p_old, path_old = bucket_t[i]
                                                if not (cost_ther <= c_old + epsilon and prog_ther >= p_old - epsilon):
                                                    clean.append((c_old, p_old, path_old))
                                            clean.append(new_val_ther)
                                            next_states[new_key_ther] = clean
                        
                        # === B: AI (action = 0) ===
                        can_take_ai = True
                        if has_sp_fixing and required_mask[j, t]:
                            can_take_ai = False  # Required to be 1, can't take 0
                        
                        if can_take_ai:
                            feasible_ai, new_mask_ai, new_len_ai = check_strict_feasibility_numba(
                                hist_mask, hist_len, 0, MS, MIN_MS
                            )
                            
                            if feasible_ai:
                                cost_ai = cost
                                eff = 1.0
                                if ai_count < len(theta_lookup):
                                    eff = theta_lookup[ai_count]
                                prog_ai = prog + eff
                                new_ai_count = ai_count + 1
                                
                                # Update zeta for AI action
                                new_zeta = zeta_mask
                                if has_nogood_cuts:
                                    for cut_idx in range(num_nogood_cuts):
                                        if (new_zeta >> cut_idx) & 1 == 0:
                                            forbidden_val = nogood_patterns[cut_idx, j, t]
                                            if forbidden_val != 0 and forbidden_val > 0:
                                                new_zeta = new_zeta | (1 << cut_idx)
                                
                                # AI action doesn't match pattern elements (therapist only)
                                # so rho stays the same
                                new_rho = rho_encoded
                                
                                # Update mu (Right Pattern)
                                new_mu = mu_encoded
                                mu_valid = True
                                if has_right_patterns:
                                    for pat_idx in range(num_right_patterns):
                                        t_start = right_pattern_starts[pat_idx]
                                        current_mode = (new_mu >> pat_idx) & 1
                                        
                                        in_pattern = False
                                        # Only Therapist actions are "In Pattern" usually? 
                                        # Yes, pattern elements are (j,t). Taking AI means x[j,t]=0, so effectively NOT in pattern.
                                        
                                        if t == t_start:
                                            # At start, if we take AI (0), we cannot be "In Pattern" (which requires 1 at elements)
                                            # So we Enter Exclude (bit=0)
                                            new_mu = new_mu & ~(1 << pat_idx)
                                        elif t > t_start:
                                            if current_mode == 0: # Exclude
                                                # Allowed, we are not picking pattern elements
                                                pass
                                            else: # Cover Mode (1)
                                                # We must take 1 if (j,t) is in pattern.
                                                # Check if this (j,t) is required
                                                is_required_here = False
                                                for elem_idx in range(right_pattern_counts[pat_idx]):
                                                    encoded = right_pattern_elements[pat_idx, elem_idx]
                                                    w_pat = encoded // 1000000
                                                    t_pat = encoded % 1000000
                                                    if w_pat == j and t_pat == t:
                                                        is_required_here = True
                                                        break
                                                
                                                if is_required_here:
                                                    # We took AI (0), but 1 was required! -> Prune
                                                    mu_valid = False; break
                                
                                if mu_valid:
                                    new_combined = new_zeta | (new_rho << 32)
                                    new_key_ai = (new_ai_count, new_mask_ai, new_len_ai, new_combined, new_mu)
                                    new_val_ai = (cost_ai, prog_ai, path_mask)
                                    
                                    if new_key_ai not in next_states:
                                        l = List.empty_list(val_with_zeta_type)
                                        l.append(new_val_ai)
                                        next_states[new_key_ai] = l
                                    else:
                                        bucket_a = next_states[new_key_ai]
                                        is_dominated = False
                                        for i in range(len(bucket_a)):
                                            c_old, p_old, _ = bucket_a[i]
                                            if c_old <= cost_ai + epsilon and p_old >= prog_ai - epsilon:
                                                is_dominated = True; break
                                        if not is_dominated:
                                            clean = List.empty_list(val_with_zeta_type)
                                            for i in range(len(bucket_a)):
                                                c_old, p_old, path_old = bucket_a[i]
                                                if not (cost_ai <= c_old + epsilon and prog_ai >= p_old - epsilon):
                                                    clean.append((c_old, p_old, path_old))
                                            clean.append(new_val_ai)
                                            next_states[new_key_ai] = clean
                
                current_states = next_states
                if len(current_states) == 0:
                    break
            
            # === Final Step (Transition to Tau) ===
            for key, bucket in current_states.items():
                ai_count, hist_mask, hist_len, combined_zeta_rho, mu_encoded = key
                zeta_mask = combined_zeta_rho & 0xFFFFFFFF
                rho_encoded = combined_zeta_rho >> 32
                
                for state in bucket:
                    cost, prog, path_mask = state
                    
                    # === Option 1: End with Therapist ===
                    can_end_ther = True
                    if has_sp_fixing and forbidden_mask[j, tau]:
                        can_end_ther = False
                    
                    if can_end_ther:
                        feasible_ther, _, _ = check_strict_feasibility_numba(hist_mask, hist_len, 1, MS, MIN_MS)
                        if feasible_ther:
                            final_cost = cost - pi_matrix[j, tau]
                            final_prog = prog + 1.0
                            final_path_mask = path_mask | (1 << (tau - r_k))
                            
                            # Update final zeta
                            final_zeta = zeta_mask
                            if has_nogood_cuts:
                                for cut_idx in range(num_nogood_cuts):
                                    if (final_zeta >> cut_idx) & 1 == 0:
                                        forbidden_val = nogood_patterns[cut_idx, j, tau]
                                        if forbidden_val != 1:
                                            final_zeta = final_zeta | (1 << cut_idx)
                            
                            # Rho
                            final_rho = rho_encoded
                            rho_valid = True
                            if has_left_patterns:
                                for pat_idx in range(num_left_patterns):
                                    for elem_idx in range(left_pattern_elements.shape[1]):
                                        encoded = left_pattern_elements[pat_idx, elem_idx]
                                        if encoded < 0: break
                                        w_pat = encoded // 1000000
                                        t_pat = encoded % 1000000
                                        if w_pat == j and t_pat == tau:
                                            current_rho = (final_rho >> (pat_idx * 8)) & 0xFF
                                            current_rho += 1
                                            if current_rho > left_pattern_limits[pat_idx]:
                                                rho_valid = False; break
                                            final_rho = (final_rho & ~(0xFF << (pat_idx * 8))) | (current_rho << (pat_idx * 8))
                                    if not rho_valid: break
                            
                            if not rho_valid: continue

                            # Mu (Right Pattern)
                            final_mu = mu_encoded
                            mu_valid = True
                            right_reward = 0.0
                            if has_right_patterns:
                                for pat_idx in range(num_right_patterns):
                                    t_start = right_pattern_starts[pat_idx]
                                    current_mode = (final_mu >> pat_idx) & 1
                                    
                                    in_pattern = False
                                    for elem_idx in range(right_pattern_counts[pat_idx]):
                                        encoded = right_pattern_elements[pat_idx, elem_idx]
                                        w_pat = encoded // 1000000
                                        t_pat = encoded % 1000000
                                        if w_pat == j and t_pat == tau:
                                            in_pattern = True
                                            break
                                    
                                    if tau == t_start:
                                        if in_pattern: final_mu = final_mu | (1 << pat_idx)
                                        else: final_mu = final_mu & ~(1 << pat_idx)
                                    elif tau > t_start:
                                        if current_mode == 0 and in_pattern:
                                            mu_valid = False; break
                                        # In Cover mode, taking 1 is always checking for broken chains. 
                                        # Since we took 1, we satisfy requirement if present.
                                    
                                    # CHECK DUAL REWARD
                                    # If we end in Cover Mode (1), we effectively covered the pattern
                                    # (assuming future elements beyond tau don't exist in this column logic, 
                                    # or we "commit" to them which is handled by branching logic).
                                    # Simplified: If we are in Cover Mode at the end, apply dual.
                                    if mu_valid and ((final_mu >> pat_idx) & 1):
                                        right_reward += right_pattern_duals[pat_idx]
                            
                            if not mu_valid: continue
                            
                            # === Terminal Feasibility Check for Zeta ===
                            if has_nogood_cuts:
                                all_deviated = True
                                for cut_idx in range(num_nogood_cuts):
                                    if (final_zeta >> cut_idx) & 1 == 0:
                                        all_deviated = False; break
                                if not all_deviated: continue
                            
                            condition_met = (final_prog >= s_k - epsilon)
                            is_valid = False
                            if obj_mode > 0.5: is_valid = condition_met
                            else: is_valid = condition_met or (tau == max_time)
                            
                            if is_valid:
                                duration_val = tau - r_k + 1
                                # Apply right_reward (subtract from cost because it's a dual "benefit" usually handled as cost reduction)
                                # Wait, duals in reduced cost: RC = Cost - Duals.
                                # If SP branching pattern is "covered", we get the dual value?
                                # Yes, usually: if x covers pattern p, term -mu_p is added.
                                # Standard RC = (c - sum(pi)) - gamma.
                                # With SP right branching (cover pattern P):
                                # Constraint: Sum(x in P) >= 1 (Cover) -> Dual mu >= 0
                                # RC = ... - mu_p * (1 if covered else 0) ??
                                # Actually, standard constraint form: sum x_j >= 1.
                                # Dual mu >= 0.
                                # RC term: - coeff * dual.
                                # Coeff is 1 if covered. So -mu.
                                # So we subtract right_reward.
                                rc = final_cost + (duration_val * obj_mode) - gamma_k - right_reward
                                if rc < -1e-6:
                                    best_columns.append((float64(j), float64(rc), int64(r_k), int64(tau), int64(final_path_mask), float64(final_prog)))
                    
                    # === Option 2: End with AI (only if timeout) ===
                    if is_timeout_scenario:
                        can_end_ai = True
                        if has_sp_fixing and required_mask[j, tau]:
                            can_end_ai = False
                        
                        if can_end_ai:
                            feasible_ai, _, _ = check_strict_feasibility_numba(hist_mask, hist_len, 0, MS, MIN_MS)
                            if feasible_ai:
                                final_cost = cost
                                eff = 1.0
                                if ai_count < len(theta_lookup):
                                    eff = theta_lookup[ai_count]
                                final_prog = prog + eff
                                final_path_mask = path_mask
                                
                                # Zeta
                                final_zeta = zeta_mask
                                if has_nogood_cuts:
                                    for cut_idx in range(num_nogood_cuts):
                                        if (final_zeta >> cut_idx) & 1 == 0:
                                            forbidden_val = nogood_patterns[cut_idx, j, tau]
                                            if forbidden_val != 0 and forbidden_val > 0:
                                                final_zeta = final_zeta | (1 << cut_idx)
                                
                                # Mu
                                final_mu = mu_encoded
                                mu_valid = True
                                right_reward = 0.0
                                if has_right_patterns:
                                    for pat_idx in range(num_right_patterns):
                                        t_start = right_pattern_starts[pat_idx]
                                        current_mode = (final_mu >> pat_idx) & 1
                                        
                                        # AI (0) matches "Not In Pattern"
                                        if tau == t_start:
                                            # Must be Exclude
                                            final_mu = final_mu & ~(1 << pat_idx)
                                        elif tau > t_start:
                                            if current_mode == 1: # Cover Mode
                                                # Required 1?
                                                is_required_here = False
                                                for elem_idx in range(right_pattern_counts[pat_idx]):
                                                    encoded = right_pattern_elements[pat_idx, elem_idx]
                                                    w_pat = encoded // 1000000
                                                    t_pat = encoded % 1000000
                                                    if w_pat == j and t_pat == tau:
                                                        is_required_here = True; break
                                                if is_required_here:
                                                    mu_valid = False; break
                                        
                                        if mu_valid and ((final_mu >> pat_idx) & 1):
                                            right_reward += right_pattern_duals[pat_idx]

                                if not mu_valid: continue

                                # Terminal zeta check
                                if has_nogood_cuts:
                                    all_deviated = True
                                    for cut_idx in range(num_nogood_cuts):
                                        if (final_zeta >> cut_idx) & 1 == 0:
                                            all_deviated = False; break
                                    if not all_deviated: continue
                                
                                condition_met = (final_prog >= s_k - epsilon)
                                is_valid = False
                                if obj_mode > 0.5: is_valid = condition_met
                                else: is_valid = condition_met or (tau == max_time)
                                
                                if is_valid:
                                    duration_val = tau - r_k + 1
                                    rc = final_cost + (duration_val * obj_mode) - gamma_k - right_reward
                                    if rc < -1e-6:
                                        best_columns.append((float64(j), float64(rc), int64(r_k), int64(tau), int64(final_path_mask), float64(final_prog)))
    
    return best_columns
