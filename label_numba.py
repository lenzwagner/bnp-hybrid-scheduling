
import numpy as np
from numba import njit, int64, float64, types
from numba.typed import List, Dict

# Type aliases for readability
# State: (cost, progress, path_mask, history_mask, history_len, ai_count)
# We store states in a list.
# Optimisation: We group states by (ai_count, history_mask, history_len) for dominance.

@njit(cache=True)
def check_strict_feasibility_numba(hist_mask, hist_len, next_val, MS, MIN_MS):
    """
    Check if adding next_val to the history satisfies rolling window constraints.
    Using bitwise operations.
    """
    # New history check
    new_len = hist_len + 1
    new_mask = (hist_mask << 1) | next_val
    
    # If we haven't filled the window yet
    if new_len < MS:
        # Check if it's possible to satisfy MIN_MS
        # Current ones + (MS - new_len) ones (optimistic future)
        current_ones = 0
        temp_mask = new_mask
        for _ in range(new_len):
            if temp_mask & 1:
                current_ones += 1
            temp_mask >>= 1
            
        remaining_slots = MS - new_len
        if current_ones + remaining_slots < MIN_MS:
            # Although returning False, we return valid shape placeholders
            return False, new_mask, new_len
        return True, new_mask, new_len
        
    else:
        # Full window check (new_len == MS or greater, but logically we enter with MS-1)
        # We check the window of size MS (which is exactly new_mask if entered with MS-1)
        
        # We assume input hist_len is at most MS-1. So new_len is at most MS.
        # If new_len == MS:
        
        # Count set bits in the window (last MS bits)
        current_ones = 0
        temp_mask = new_mask
        for _ in range(MS):
            if temp_mask & 1:
                current_ones += 1
            temp_mask >>= 1
            
        if current_ones < MIN_MS:
            return False, new_mask, MS 
            
        # Truncate to MS - 1 for state storage
        ms_minus_1 = MS - 1
        trunc_mask = new_mask & ((1 << ms_minus_1) - 1)
        return True, trunc_mask, ms_minus_1

# Type definitions for Dict
# Key: (ai_count, hist_mask, hist_len)
key_type = types.Tuple((types.int64, types.int64, types.int64))
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
            
            init_key = (int64(init_ai), int64(init_hist), int64(init_hlen))
            
            # Create list for this bucket
            val_list = List.empty_list(val_tuple_type)
            val_list.append((float64(start_cost), float64(1.0), int64(init_path)))
            current_states[init_key] = val_list
            
            # DP Loop
            for t in range(r_k + 1, tau):
                next_states = Dict.empty(key_type, val_list_type)
                
                # Iterate over current buckets
                for key, bucket in current_states.items():
                    ai_count, hist_mask, hist_len = key
                    
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
                            
                            new_key_ther = (ai_count, new_mask_ther, new_len_ther)
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
                            
                            new_key_ai = (new_ai_count, new_mask_ai, new_len_ai)
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
                ai_count, hist_mask, hist_len = key
                
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
# BRANCHING CONSTRAINTS SUPPORT
# =============================================================================
# Extended state tuple types for branching constraints
# State now includes: cost, prog, path_mask, zeta_mask (for MP branching)
# Key now includes: ai_count, hist_mask, hist_len, zeta_mask

# Key type WITH zeta: (ai_count, hist_mask, hist_len, zeta_mask)
key_with_zeta_type = types.Tuple((types.int64, types.int64, types.int64, types.int64))
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
    # Right patterns are more complex - handled via flags for now
    # (Full implementation would track mu modes)
):
    """
    Extended DP loop with branching constraint support.
    
    Supports:
    - SP Variable Fixing: forbidden_mask[j,t] = True means x[j,t] must be 0
    - MP No-Good Cuts: Track zeta deviation vector via bitmask
    - SP Left Pattern Branching: Track rho counters, prune if limit exceeded
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
                        w_pat = encoded // 1000
                        t_pat = encoded % 1000
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
            
            # Initialize state dict
            # Key: (ai_count, hist_mask, hist_len, zeta_mask)
            current_states = Dict.empty(key_with_zeta_type, val_list_with_zeta_type)
            
            init_ai = int64(0)
            init_hist = int64(1)
            init_hlen = int64(1)
            init_path = int64(1)
            
            # Include rho in key if we have left patterns (encoded in zeta_mask slot for now)
            # Actually, for proper tracking we need rho in the key or value
            # Simplification: Store rho in upper bits of zeta_mask (zeta uses lower bits, rho uses upper)
            combined_zeta_rho = init_zeta | (init_rho << 32)
            
            init_key = (init_ai, init_hist, init_hlen, combined_zeta_rho)
            val_list = List.empty_list(val_with_zeta_type)
            val_list.append((float64(start_cost), float64(1.0), init_path))
            current_states[init_key] = val_list
            
            # DP Loop
            for t in range(r_k + 1, tau):
                next_states = Dict.empty(key_with_zeta_type, val_list_with_zeta_type)
                
                for key, bucket in current_states.items():
                    ai_count, hist_mask, hist_len, combined_zeta_rho = key
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
                                
                                # Update zeta for therapist action
                                new_zeta = zeta_mask
                                if has_nogood_cuts:
                                    for cut_idx in range(num_nogood_cuts):
                                        if (new_zeta >> cut_idx) & 1 == 0:  # Not yet deviated
                                            forbidden_val = nogood_patterns[cut_idx, j, t]
                                            if forbidden_val != 1:  # We took 1, forbidden != 1
                                                new_zeta = new_zeta | (1 << cut_idx)
                                
                                # Update rho for left patterns
                                new_rho = rho_encoded
                                rho_valid = True
                                if has_left_patterns:
                                    for pat_idx in range(num_left_patterns):
                                        for elem_idx in range(left_pattern_elements.shape[1]):
                                            encoded = left_pattern_elements[pat_idx, elem_idx]
                                            if encoded < 0:
                                                break
                                            w_pat = encoded // 1000
                                            t_pat = encoded % 1000
                                            if w_pat == j and t_pat == t:
                                                current_rho = (new_rho >> (pat_idx * 8)) & 0xFF
                                                current_rho += 1
                                                if current_rho > left_pattern_limits[pat_idx]:
                                                    rho_valid = False
                                                    break
                                                new_rho = (new_rho & ~(0xFF << (pat_idx * 8))) | (current_rho << (pat_idx * 8))
                                        if not rho_valid:
                                            break
                                
                                if rho_valid:
                                    new_combined = new_zeta | (new_rho << 32)
                                    new_key_ther = (ai_count, new_mask_ther, new_len_ther, new_combined)
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
                                                is_dominated = True
                                                break
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
                                            if forbidden_val != 0 and forbidden_val > 0:  # We took 0, forbidden != 0
                                                new_zeta = new_zeta | (1 << cut_idx)
                                
                                # AI action doesn't match pattern elements (therapist only)
                                # so rho stays the same
                                new_rho = rho_encoded
                                
                                new_combined = new_zeta | (new_rho << 32)
                                new_key_ai = (new_ai_count, new_mask_ai, new_len_ai, new_combined)
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
                                            is_dominated = True
                                            break
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
                ai_count, hist_mask, hist_len, combined_zeta_rho = key
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
                            
                            # Update final rho (for completeness)
                            final_rho = rho_encoded
                            rho_valid = True
                            if has_left_patterns:
                                for pat_idx in range(num_left_patterns):
                                    for elem_idx in range(left_pattern_elements.shape[1]):
                                        encoded = left_pattern_elements[pat_idx, elem_idx]
                                        if encoded < 0:
                                            break
                                        w_pat = encoded // 1000
                                        t_pat = encoded % 1000
                                        if w_pat == j and t_pat == tau:
                                            current_rho = (final_rho >> (pat_idx * 8)) & 0xFF
                                            current_rho += 1
                                            if current_rho > left_pattern_limits[pat_idx]:
                                                rho_valid = False
                                                break
                                            final_rho = (final_rho & ~(0xFF << (pat_idx * 8))) | (current_rho << (pat_idx * 8))
                                    if not rho_valid:
                                        break
                            
                            if not rho_valid:
                                continue
                            
                            # === Terminal Feasibility Check for Zeta ===
                            # All no-good cuts must be deviated (all zeta bits = 1)
                            if has_nogood_cuts:
                                all_deviated = True
                                for cut_idx in range(num_nogood_cuts):
                                    if (final_zeta >> cut_idx) & 1 == 0:
                                        all_deviated = False
                                        break
                                if not all_deviated:
                                    continue
                            
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
                                
                                # Update final zeta for AI ending
                                final_zeta = zeta_mask
                                if has_nogood_cuts:
                                    for cut_idx in range(num_nogood_cuts):
                                        if (final_zeta >> cut_idx) & 1 == 0:
                                            forbidden_val = nogood_patterns[cut_idx, j, tau]
                                            if forbidden_val != 0 and forbidden_val > 0:
                                                final_zeta = final_zeta | (1 << cut_idx)
                                
                                # Terminal zeta check
                                if has_nogood_cuts:
                                    all_deviated = True
                                    for cut_idx in range(num_nogood_cuts):
                                        if (final_zeta >> cut_idx) & 1 == 0:
                                            all_deviated = False
                                            break
                                    if not all_deviated:
                                        continue
                                
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
