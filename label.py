"""
Core Labeling Algorithm for Column Generation (Pricing Problem Solver)

This module contains the core dynamic programming algorithm for solving 
the pricing problem in column generation, without validation, testing, 
or comparison utilities.
"""
import sys
import time
from logging_config import get_logger

# Initialize logger
logger = get_logger(__name__)


# --- Helper Functions ---

def check_strict_feasibility(history, next_val, MS, MIN_MS):
    """
    Check if adding next_val to the history satisfies rolling window constraints.
    
    Args:
        history: Tuple of recent actions (0 or 1 values)
        next_val: The next action to check (0 or 1)
        MS: Rolling window size
        MIN_MS: Minimum human services required in window
        
    Returns:
        bool: True if feasible, False otherwise
    """
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


def add_state_to_buckets(buckets, cost, prog, ai_count, hist, path, recipient_id, 
                         pruning_stats, dominance_mode='bucket', zeta=None, epsilon=1e-9):
    """
    Adds a state to buckets, applying dominance rules.
    
    Args:
        buckets: State storage structure
        cost: Current accumulated cost
        prog: Current progress towards target
        ai_count: Number of AI sessions used
        hist: History tuple for rolling window
        path: Path pattern (list of 0s and 1s)
        recipient_id: Current recipient ID (for debug output)
        pruning_stats: Statistics dictionary
        dominance_mode: 'bucket' or 'global'
        zeta: Branch constraint deviation vector (optional)
        epsilon: Tolerance for float comparisons
    """
    # Bucket key includes zeta if branch constraints are active
    if zeta is not None:
        bucket_key = (ai_count, hist, zeta)
    else:
        bucket_key = (ai_count, hist)
    
    # --- GLOBAL PRUNING CHECK (Only in Global Mode) ---
    if dominance_mode == 'global':
        is_dominated_globally = False
        dominator_global = None
        
        for (ai_other, hist_other), other_list in buckets.items():
            # Another bucket can only dominate if it is "better/equal" in AI & Hist
            if ai_other < ai_count:
                continue
                
            # Hist Check (component-wise >=)
            if len(hist_other) != len(hist):
                continue
            
            hist_better = True
            for h1, h2 in zip(hist_other, hist):
                if h1 < h2:
                    hist_better = False
                    break
            if not hist_better:
                continue
            
            # Check Cost & Prog in this bucket
            for c_old, p_old, _ in other_list:
                if c_old <= cost + epsilon and p_old >= prog - epsilon:
                    is_dominated_globally = True
                    dominator_global = (c_old, p_old, ai_other, hist_other)
                    break
            
            if is_dominated_globally:
                break
        
        if is_dominated_globally:
            pruning_stats['dominance'] += 1
            logger.info(f"    [DOMINANCE GLOBAL] Recipient {recipient_id}: State DOMINATED")
            logger.info(f"      Dominated state: Cost={cost:.4f}, Prog={prog:.4f}, AI={ai_count}, Hist={hist}")
            logger.info(f"      Dominating state: Cost={dominator_global[0]:.4f}, Prog={dominator_global[1]:.4f}, AI={dominator_global[2]}, Hist={dominator_global[3]}")
            logger.info(f"      Reason: Dominating state has Cost <= {cost:.4f} AND Prog >= {prog:.4f} with better/equal AI and History")
            if not pruning_stats['printed_dominance'].get(recipient_id, False):
                logger.print(f"    [DOMINANCE GLOBAL] Recipient {recipient_id}: First global dominance pruning occurred")
                pruning_stats['printed_dominance'][recipient_id] = True
            return

    # --- BUCKET MANAGEMENT ---
    if bucket_key not in buckets:
        buckets[bucket_key] = []
    
    bucket_list = buckets[bucket_key]

    # --- LOCAL DOMINANCE (Within the Bucket) ---
    is_dominated = False
    dominator = None
    
    for c_old, p_old, _ in bucket_list:
        if c_old <= cost + epsilon and p_old >= prog - epsilon:
            is_dominated = True
            dominator = (c_old, p_old)
            break
    
    if is_dominated:
        pruning_stats['dominance'] += 1
        logger.info(f"    [DOMINANCE BUCKET] Recipient {recipient_id}: State DOMINATED in same bucket")
        logger.info(f"      Dominated state: Cost={cost:.4f}, Prog={prog:.4f}, Bucket_key={bucket_key}")
        logger.info(f"      Dominating state: Cost={dominator[0]:.4f}, Prog={dominator[1]:.4f}")
        logger.info(f"      Reason: Dominating state in same bucket has Cost <= {cost:.4f} AND Prog >= {prog:.4f}")
        if not pruning_stats['printed_dominance'].get(recipient_id, False):
            logger.print(f"    [DOMINANCE BUCKET] Recipient {recipient_id}: First bucket dominance pruning occurred")
            pruning_stats['printed_dominance'][recipient_id] = True
        return 
    
    # --- CLEANUP ---
    # Remove existing states that are dominated by the new one
    new_bucket_list = []
    dominated_by_new = 0
    
    for c_old, p_old, path_old in bucket_list:
        if cost <= c_old + epsilon and prog >= p_old - epsilon:
            pruning_stats['dominance'] += 1
            dominated_by_new += 1
            logger.info(f"    [DOMINANCE CLEANUP] Recipient {recipient_id}: New state dominates existing state")
            logger.info(f"      New (dominating) state: Cost={cost:.4f}, Prog={prog:.4f}")
            logger.info(f"      Old (dominated) state: Cost={c_old:.4f}, Prog={p_old:.4f}")
            logger.info(f"      Reason: New state has Cost <= {c_old:.4f} AND Prog >= {p_old:.4f}")
            continue 
        new_bucket_list.append((c_old, p_old, path_old))
    
    if dominated_by_new > 0:
        logger.info(f"    [DOMINANCE CLEANUP] Recipient {recipient_id}: New state removed {dominated_by_new} dominated state(s) from bucket")
    
    new_bucket_list.append((cost, prog, path))
    buckets[bucket_key] = new_bucket_list


def generate_full_column_vector(worker_id, path_assignments, start_time, end_time, max_time, num_workers):
    """
    Generate the full column vector for a schedule.
    
    Args:
        worker_id: Worker ID (1-indexed)
        path_assignments: List of assignments (0 or 1)
        start_time: Start time of schedule
        end_time: End time of schedule
        max_time: Maximum time horizon
        num_workers: Total number of workers
        
    Returns:
        List of floats representing the full column vector
    """
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
    """
    Calculates Lower Bound for Bound Pruning.

    Assumption: Maximum productivity (only therapists with efficiency = 1.0)
    This guarantees that we don't miss any optimal solutions.
    compute_lower_bound(cost, r_k, tau, gamma_k, obj_mode)
    Returns:
        float: Minimum achievable final Reduced Cost (optimistic)
    """
    import math

    # Time Cost is fixed for the specific column length (end_time - start_time + 1)
    duration = end_time - start_time + 1
    time_cost = duration * obj_mode

    # Current cost contains the accumulated -pi values so far.
    # We assume future -pi values are 0 (optimistic, since -pi >= 0).

    lower_bound = current_cost + time_cost - gamma_k

    return lower_bound


def compute_candidate_workers(workers, r_k, tau_max, pi_dict):
    """
    Worker Dominance Pre-Elimination:
    Worker j1 dominates j2 if π_{j1,t} >= π_{j2,t} for all t in [r_k, tau_max]
    AND π_{j1,t} > π_{j2,t} for at least one t (strict dominance).
    Since π values are <= 0 (implicit costs), higher π means lower cost.
    
    Returns:
        List of non-dominated workers
    """
    candidate_workers = []

    for j1 in workers:
        is_dominated = False

        for j2 in workers:
            if j1 == j2:
                continue

            # Check if j2 dominates j1
            all_better_or_equal = True
            at_least_one_strictly_better = False

            for t in range(r_k, tau_max + 1):
                pi_j1 = pi_dict.get((j1, t), 0.0)
                pi_j2 = pi_dict.get((j2, t), 0.0)

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
            candidate_workers.append(j1)


    return candidate_workers


# --- Core Labeling Algorithm ---

def solve_pricing_for_recipient(recipient_id, r_k, s_k, gamma_k, obj_mode, pi_dict, 
                                workers, max_time, ms, min_ms, theta_lookup,
                                use_bound_pruning=True, dominance_mode='bucket', 
                                branch_constraints=None, branching_variant='mp',
                                max_columns=10):
    """
    Solve the pricing problem for a single recipient.
    
    Args:
        recipient_id: Recipient ID
        r_k: Release time
        s_k: Service target
        gamma_k: Dual value gamma
        obj_mode: Objective mode multiplier
        pi_dict: Dual values pi {(worker_id, time): value}
        workers: List of worker IDs
        max_time: Planning horizon
        ms: Rolling window size
        min_ms: Minimum human services in window
        theta_lookup: AI efficiency lookup table
        use_bound_pruning: Enable lower bound pruning
        dominance_mode: 'bucket' or 'global'
        branch_constraints: Optional branch constraints
        branching_variant: Branching strategy ('mp' or 'sp')
        max_columns: Maximum number of columns to return
        
    Returns:
        List of best columns (dictionaries)
    """
    best_columns = []
    epsilon = 1e-9
    
    pruning_stats = {
        'lb': 0,
        'dominance': 0,
        'printed_dominance': {}
    }


    time_until_end = max_time - r_k + 1

    # Worker Dominance Pre-Elimination
    candidate_workers = compute_candidate_workers(workers, r_k, max_time, pi_dict)
    if recipient_id == 22:
        print('candidate_workers', candidate_workers)
    eliminated_workers = [w for w in workers if w not in candidate_workers]

    # Print for each Recipient
    if eliminated_workers:
        logger.info(f"Recipient with entry {r_k} and req {s_k} {recipient_id:2d}: Candidate workers = {candidate_workers} (eliminated {eliminated_workers})")
    else:
        logger.info(f"Recipient with entry {r_k} and req {s_k} {recipient_id:2d}: Candidate workers = {candidate_workers} (no dominance)")
    
    # --- Parse Branch Constraints (MP Branching) ---
    forbidden_schedules = []
    use_branch_constraints = False

    
    # DEBUG: Print ALL branching constraints first
    if branch_constraints:
        logger.print(f"\n  [BRANCHING CONSTRAINTS DEBUG] Recipient {recipient_id}:")
        logger.print(f"    Type of branch_constraints: {type(branch_constraints)}")
        logger.print(f"    Length/Size: {len(branch_constraints) if hasattr(branch_constraints, '__len__') else 'N/A'}")
        
        if isinstance(branch_constraints, list):
            for idx, constraint in enumerate(branch_constraints):
                logger.print(f"    Constraint #{idx + 1}:")
                logger.print(f"      Type: {type(constraint)}")
                logger.print(f"      Has 'profile': {hasattr(constraint, 'profile')}")
                if hasattr(constraint, 'profile'):
                    logger.print(f"      Profile value: {constraint.profile}")
                logger.print(f"      Has 'direction': {hasattr(constraint, 'direction')}")
                if hasattr(constraint, 'direction'):
                    logger.print(f"      Direction value: {constraint.direction}")
                logger.print(f"      Has 'original_schedule': {hasattr(constraint, 'original_schedule')}")


    if branch_constraints:
        # Handle list of constraint objects (from Branch-and-Price)
        if isinstance(branch_constraints, list):
            for constraint in branch_constraints:
                # Check if constraint applies to this profile
                if not hasattr(constraint, 'profile') or constraint.profile != recipient_id:
                    continue
                
                # We only care about LEFT branches for MP branching (no-good cuts)
                if not hasattr(constraint, 'direction') or constraint.direction != 'left':
                    continue
                
                if branching_variant == 'mp':
                    # Check for MPVariableBranching with original_schedule
                    if hasattr(constraint, 'original_schedule') and constraint.original_schedule:
                        use_branch_constraints = True
                        forbidden_schedule = {}
                        # original_schedule keys are (p, j, t, col_id)
                        for key, val in constraint.original_schedule.items():
                            if len(key) >= 3 and val > 1e-6:
                                j, t = key[1], key[2]
                                forbidden_schedule[(j, t)] = val
                        forbidden_schedules.append(forbidden_schedule)

        # Handle dictionary (legacy format)
        elif isinstance(branch_constraints, dict):
            for constraint_key, constraint_data in branch_constraints.items():
                if constraint_data.get("profile") != recipient_id:
                    continue
                
                if constraint_data.get("direction") != "left":
                    continue
                
                if branching_variant == 'mp':
                    if "original_schedule" in constraint_data:
                        use_branch_constraints = True
                        forbidden_schedule = {}
                        for key, val in constraint_data["original_schedule"].items():
                            # key format might vary in dict, assume compatible
                            if isinstance(key, tuple) and len(key) >= 3:
                                j, t = key[1], key[2]
                                forbidden_schedule[(j, t)] = val
                        forbidden_schedules.append(forbidden_schedule)

        if use_branch_constraints:
            logger.print(f"\n  [MP BRANCHING] Recipient {recipient_id}: {len(forbidden_schedules)} no-good cut(s) active")
            for cut_idx, cut in enumerate(forbidden_schedules):
                logger.print(f"    No-Good Cut #{cut_idx + 1}:")
                # Show pattern of forbidden schedule
                schedule_pattern = sorted(cut.items(), key=lambda x: (x[0][0], x[0][1]))  # Sort by (worker, time)
                logger.print(f"      Pattern: {len(schedule_pattern)} assignments")
                # Show first few assignments as preview
                preview = schedule_pattern[:5]
                for (worker, time), val in preview:
                    logger.print(f"        Worker {worker}, Time {time}: {val}")
                if len(schedule_pattern) > 5:
                    logger.print(f"        ... and {len(schedule_pattern) - 5} more assignments")
        else:
            logger.info(f"  [MP BRANCHING] No active constraints for recipient {recipient_id}")

    for j in candidate_workers:
        effective_min_duration = min(int(s_k), time_until_end)
        start_tau = r_k + effective_min_duration - 1

        for tau in range(start_tau, max_time + 1):
            is_timeout_scenario = (tau == max_time)

            start_cost = -pi_dict.get((j, r_k), 0)
            num_cuts = len(forbidden_schedules)

            initial_zeta = tuple([0] * num_cuts) if use_branch_constraints else None

            # Show zeta initialization for recipients with active branching constraints
            # Display once per recipient (when processing first candidate worker at start_tau)
            if use_branch_constraints and tau == start_tau:
                # Check if this is the FIRST candidate worker (not necessarily worker ID 1!)
                first_candidate = candidate_workers[0] if candidate_workers else None
                if j == first_candidate:
                    logger.print(f"\n  [ZETA VECTOR] Recipient {recipient_id} (BRANCHING PROFILE): Initialized with {num_cuts} elements")
                    logger.print(f"    Initial ζ = {initial_zeta} (all zeros = no deviations yet)")
                    logger.print(f"    Terminal condition: All ζ elements must be 1 (deviated from all cuts)")
                    logger.print(f"    This recipient has active MP branching constraints!")
                    logger.print(f"    First candidate worker: {first_candidate}, All candidates: {candidate_workers}")
            
            current_states = {}
            # Initialize with start state
            initial_history = (1,)  # First action is always 1 (Therapist)
            add_state_to_buckets(current_states, start_cost, 1.0, 0, initial_history, [1], 
                               recipient_id, pruning_stats, dominance_mode, initial_zeta, epsilon)

            # DP Loop until just before Tau
            pruned_count_total = 0

            for t in range(r_k + 1, tau):
                next_states = {}
                pruned_count_this_period = 0

                # Iterate over all buckets
                for bucket_key, bucket_list in current_states.items():
                    # Extract components from bucket key
                    if use_branch_constraints:
                        ai_count, hist, zeta = bucket_key
                    else:
                        ai_count, hist = bucket_key
                        zeta = None
                    
                    # Iterate over all states in the bucket
                    for cost, prog, path in bucket_list:

                        # BOUND PRUNING: Check if state is promising
                        if use_bound_pruning:
                            lb = compute_lower_bound(cost, r_k, tau, gamma_k, obj_mode)
                            if lb >= 0:
                                pruned_count_this_period += 1
                                pruned_count_total += 1
                                pruning_stats['lb'] += 1
                                continue  # State is pruned!

                        # Feasibility Check
                        remaining_steps = tau - t + 1
                        if not is_timeout_scenario:
                            if obj_mode > 0.5:
                                if prog + remaining_steps * 1.0 < s_k - epsilon:
                                    continue

                        # A: Therapist
                        if check_strict_feasibility(hist, 1, ms, min_ms):
                            cost_ther = cost - pi_dict.get((j, t), 0)
                            prog_ther = prog + 1.0
                            new_hist_ther = (hist + (1,))
                            if len(new_hist_ther) > ms - 1: 
                                new_hist_ther = new_hist_ther[-(ms - 1):]
                            
                            # Update deviation vector ζ_t if branch constraints are active
                            new_zeta_ther = zeta
                            if use_branch_constraints:
                                new_zeta_ther = list(zeta)
                                for cut_idx, cut in enumerate(forbidden_schedules):
                                    if new_zeta_ther[cut_idx] == 0:  # Not yet deviated
                                        forbidden_val = cut.get((j, t), None)
                                        if forbidden_val is not None and forbidden_val != 1:
                                            # DEVIATION DETECTED: 0 → 1 transition
                                            new_zeta_ther[cut_idx] = 1
                                            logger.print(f"    [ZETA TRANSITION] Recipient {recipient_id}, Worker {j}, Time {t}:")
                                            logger.print(f"      Cut #{cut_idx + 1}: ζ[{cut_idx}]: 0 → 1 (Therapist action deviates from forbidden value {forbidden_val})")
                                            logger.print(f"      New ζ = {tuple(new_zeta_ther)}")
                                new_zeta_ther = tuple(new_zeta_ther)

                            add_state_to_buckets(next_states, cost_ther, prog_ther, ai_count, new_hist_ther, 
                                               path + [1], recipient_id, pruning_stats, dominance_mode, 
                                               new_zeta_ther, epsilon)

                        # B: AI
                        if check_strict_feasibility(hist, 0, ms, min_ms):
                            cost_ai = cost
                            efficiency = theta_lookup[ai_count] if ai_count < len(theta_lookup) else 1.0
                            prog_ai = prog + efficiency
                            ai_count_new = ai_count + 1
                            new_hist_ai = (hist + (0,))
                            if len(new_hist_ai) > ms - 1: 
                                new_hist_ai = new_hist_ai[-(ms - 1):]
                            
                            # Update deviation vector ζ_t if branch constraints are active
                            new_zeta_ai = zeta
                            if use_branch_constraints:
                                new_zeta_ai = list(zeta)
                                for cut_idx, cut in enumerate(forbidden_schedules):
                                    if new_zeta_ai[cut_idx] == 0:  # Not yet deviated
                                        forbidden_val = cut.get((j, t), None)
                                        if forbidden_val is not None and forbidden_val != 0:
                                            # DEVIATION DETECTED: 0 → 1 transition
                                            new_zeta_ai[cut_idx] = 1
                                            logger.print(f"    [ZETA TRANSITION] Recipient {recipient_id}, Worker {j}, Time {t}:")
                                            logger.print(f"      Cut #{cut_idx + 1}: ζ[{cut_idx}]: 0 → 1 (AI action deviates from forbidden value {forbidden_val})")
                                            logger.print(f"      New ζ = {tuple(new_zeta_ai)}")
                                new_zeta_ai = tuple(new_zeta_ai)

                            add_state_to_buckets(next_states, cost_ai, prog_ai, ai_count_new, new_hist_ai, 
                                               path + [0], recipient_id, pruning_stats, dominance_mode, 
                                               new_zeta_ai, epsilon)

                current_states = next_states
                if not current_states: 
                    break

            # Final Step (Transition to Tau)
            for bucket_key, bucket_list in current_states.items():
                # Extract components from bucket key
                if use_branch_constraints:
                    ai_count, hist, zeta = bucket_key
                else:
                    ai_count, hist = bucket_key
                    zeta = None
                    
                for cost, prog, path in bucket_list:
                    
                    # Collect possible end steps for this state
                    possible_moves = []

                    # Option 1: End with Therapist (1) - Standard
                    if check_strict_feasibility(hist, 1, ms, min_ms):
                        possible_moves.append(1)

                    # Option 2: End with App (0) - ONLY if Timeout
                    if is_timeout_scenario:
                        if check_strict_feasibility(hist, 0, ms, min_ms):
                            possible_moves.append(0)

                    for move in possible_moves:
                        # Calculate values based on Move type
                        if move == 1:
                            final_cost_accum = cost - pi_dict.get((j, tau), 0)
                            final_prog = prog + 1.0
                            final_ai_count = ai_count
                        else:  # move == 0
                            final_cost_accum = cost
                            efficiency = theta_lookup[ai_count] if ai_count < len(theta_lookup) else 1.0
                            final_prog = prog + efficiency
                            final_ai_count = ai_count + 1

                        final_path = path + [move]
                        condition_met = (final_prog >= s_k - epsilon)
                        
                        # Update final zeta for this move
                        final_zeta = zeta
                        if use_branch_constraints:
                            final_zeta = list(zeta)
                            for cut_idx, cut in enumerate(forbidden_schedules):
                                if final_zeta[cut_idx] == 0:  # Not yet deviated
                                    forbidden_val = cut.get((j, tau), None)
                                    if forbidden_val is not None and forbidden_val != move:
                                        final_zeta[cut_idx] = 1  # Deviated!
                            final_zeta = tuple(final_zeta)
                        
                        # TERMINAL FEASIBILITY CHECK: All deviation vector entries must equal 1
                        if use_branch_constraints:
                            if not all(z == 1 for z in final_zeta):
                                # This path hasn't deviated from all forbidden schedules -> REJECT
                                missing_deviations = [i for i, z in enumerate(final_zeta) if z == 0]
                                logger.print(f"    [NO-GOOD CUT ACTIVE] Recipient {recipient_id}, Worker {j}:")
                                logger.print(f"      Column REJECTED: ζ = {final_zeta}")
                                logger.print(f"      Missing deviations from cuts: {[f'#{i+1}' for i in missing_deviations]}")
                                logger.print(f"      → Column matches forbidden schedule(s), pruned by no-good cut")
                                continue

                        is_focus_patient = (obj_mode > 0.5)

                        if is_focus_patient:
                            # Focus-Patient (E_k=1):
                            is_valid_end = condition_met
                        else:
                            # Post-Patient (E_k=0):
                            is_valid_end = condition_met or is_timeout_scenario

                        if is_valid_end:
                            duration = tau - r_k + 1
                            reduced_cost = (obj_mode * duration) + final_cost_accum - gamma_k

                            col_candidate = {
                                'k': recipient_id,
                                'worker': j,
                                'start': r_k,
                                'end': tau,
                                'duration': duration,
                                'reduced_cost': reduced_cost,
                                'final_progress': final_prog,
                                'x_vector': generate_full_column_vector(j, final_path, r_k, tau, max_time, len(workers)),
                                'path_pattern': final_path
                            }

                            if reduced_cost < -epsilon:
                                best_columns.append(col_candidate)
            
            # Debug Output: Bound Pruning Statistics
            if pruned_count_total > 0:
                logger.print(f"    Worker {j}, tau={tau}: Pruned {pruned_count_total} states by Lower Bound")

    # Sort columns by reduced cost (ascending, most negative first)
    best_columns.sort(key=lambda x: x['reduced_cost'])
    
    # Keep only unique columns (based on path pattern and worker)
    unique_columns = []
    seen_patterns = set()
    
    for col in best_columns:
        pattern_key = (col['worker'], tuple(col['path_pattern']))
        if pattern_key not in seen_patterns:
            unique_columns.append(col)
            seen_patterns.add(pattern_key)
            
        if len(unique_columns) >= max_columns:
            break
            
    best_columns = unique_columns

    # Print negative reduced costs if found
    #for col in best_columns:
        #print(f"  [Labeling] Recipient {recipient_id} with gamma {gamma_k}: Negative red. cost: {col['reduced_cost']:.2f} with {col['duration']} and {col['x_vector']}")

    # Debug output for forbidden schedules and generated columns
    if use_branch_constraints and forbidden_schedules:
        logger.print(f"\n{'='*100}")
        logger.print(f"  [FORBIDDEN vs GENERATED] Recipient {recipient_id}: {len(forbidden_schedules)} No-Good Cut(s) Active")
        logger.print(f"{'='*100}")
        
        # First, show all forbidden schedules in detail
        for cut_idx, cut in enumerate(forbidden_schedules):
            logger.print(f"\n  Forbidden Schedule #{cut_idx+1}:")
            # Group by worker
            workers_in_cut = {}
            for (worker, time), val in sorted(cut.items()):
                if worker not in workers_in_cut:
                    workers_in_cut[worker] = []
                workers_in_cut[worker].append((time, val))
            
            for worker in sorted(workers_in_cut.keys()):
                times_vals = workers_in_cut[worker]
                pattern_str = "".join([str(int(v)) for _, v in sorted(times_vals)])
                time_range = f"[{min(t for t, _ in times_vals)}→{max(t for t, _ in times_vals)}]"
                logger.print(f"    Worker {worker:2d} {time_range:8s}: {pattern_str}")
        
        # Now, show all generated columns
        if best_columns:
            logger.print(f"\n  Generated Columns ({len(best_columns)} found):")
            for col_idx, col in enumerate(best_columns):
                logger.print(f"\n    Column #{col_idx+1}:")
                logger.print(f"      Worker: {col['worker']:2d}, Period: [{col['start']}→{col['end']}], Reduced Cost: {col['reduced_cost']:.4f}")
                pattern_str = "".join([str(int(v)) for v in col['path_pattern']])
                logger.print(f"      Pattern: {pattern_str}")
                
                # Compare with each forbidden schedule
                for cut_idx, cut in enumerate(forbidden_schedules):
                    comparison = []
                    deviations = []
                    all_times = list(range(col['start'], col['end'] + 1))
                    
                    for t_step_idx, t in enumerate(all_times):
                        forbidden_val = cut.get((col['worker'], t), None)
                        generated_val = col['path_pattern'][t_step_idx]
                        
                        if forbidden_val is not None:
                            if forbidden_val == generated_val:
                                comparison.append(f"t{t}:✓")  # Match
                            else:
                                comparison.append(f"t{t}:✗({int(forbidden_val)}→{int(generated_val)})")  # Deviation
                                deviations.append(f"t{t}")
                    
                    if comparison:  # Only show if there's overlap
                        match_status = "IDENTICAL - REJECTED!" if not deviations else f"DEVIATES (at {', '.join(deviations)})"
                        logger.print(f"      vs Cut #{cut_idx+1}: {match_status}")
                        if len(comparison) <= 10:
                            logger.print(f"        {' '.join(comparison)}")
                        else:
                            logger.print(f"        {' '.join(comparison[:5])} ... {' '.join(comparison[-5:])}")
        else:
            logger.print(f"\n  ⚠️  No columns generated for this recipient!")
        
        logger.print(f"\n{'='*100}\n")


    return best_columns


# --- 3b. Wrapper for Branch-and-Price Integration ---

def solve_pricing_for_profile_bnp(
    profile,
    duals_pi,
    duals_gamma,
    duals_delta,
    r_k,
    s_k,
    obj_multiplier,
    workers,
    max_time,
    theta_lookup,
    MS,
    MIN_MS,
    col_id,
    branching_constraints=None,
    max_columns=10
):
    """
    Wrapper function for Branch-and-Price integration.
    
    Solves the pricing problem for a single profile using the labeling algorithm
    and returns results in the format expected by branch_and_price.py.
    
    Args:
        profile: Profile index (k)
        duals_pi: Dict of (worker, time) -> dual value
        duals_gamma: Float, dual value for this profile's convexity constraint
        duals_delta: Float, sum of branching constraint duals
        r_k: Release time for this profile
        s_k: Service requirement
        obj_multiplier: Objective mode (0 or 1)
        workers: List of available workers
        max_time: Time horizon
        theta_lookup: Lookup table for learning curve
        MS: Milestone window size
        MIN_MS: Minimum therapist sessions in window
        col_id: Next column ID to use
        branching_constraints: Optional list of branching constraints (not yet implemented)
        max_columns: Maximum number of columns to return
    
    Returns:
        list: List of best columns in subproblem format, or empty list if no improving column found
        [
            {
                'reduced_cost': float,
                'schedules_x': dict {(p, j, t, col_id): value},
                'schedules_los': dict {(p, col_id): los_value},
                'x_list': list of x values,
                'los_list': list with single los value,
                'path_pattern': list [0, 1, 1, ...],
                'worker': int,
                'start': int,
                'end': int
            }
        ]
    """
    # Set global variables (temporary solution until we refactor to pass all parameters)
    global MAX_TIME, WORKERS, pi, gamma
    
    MAX_TIME = max_time
    WORKERS = workers
    
    # Convert duals_pi to global pi format expected by labeling algorithm
    pi = duals_pi
    # gamma for this profile
    gamma_k = duals_gamma + duals_delta
    
    # Call the labeling algorithm with correct parameters
    best_columns = solve_pricing_for_recipient(
        recipient_id=profile,
        r_k=r_k,
        s_k=s_k,
        gamma_k=gamma_k,
        obj_mode=obj_multiplier,
        pi_dict=pi,
        workers=workers,
        max_time=max_time,
        ms=MS,
        min_ms=MIN_MS,
        theta_lookup=theta_lookup,
        use_bound_pruning=False,  # Disable for now
        dominance_mode='bucket',
        branch_constraints=branching_constraints,
        branching_variant='mp',
        max_columns=max_columns
    )
    
    if not best_columns:
        return []
    
    formatted_columns = []
    current_col_id = col_id
    
    for col in best_columns:
        # Convert to subproblem format expected by branch_and_price.py
        worker = col['worker']
        start = col['start']
        end = col['end']
        path_pattern = col['path_pattern']
    
        # Build schedules_x: {(profile, worker, time, col_id): value}
        # IMPORTANT: Initialize ALL possible (profile, worker, time, col_id) combinations with 0
        # to match the format of Gurobi-generated subproblems
        schedules_x = {}
        
        # Initialize all combinations with 0 for ALL workers and times
        for w in workers:
            for t in range(1, max_time + 1):
                schedules_x[(profile, w, t, current_col_id)] = 0.0
        
        # Now set the actual values from the path_pattern
        for t_idx, val in enumerate(path_pattern):
            current_time = start + t_idx
            schedules_x[(profile, worker, current_time, current_col_id)] = float(val)
        
        # Build schedules_los: {(profile, col_id): los_value}
        duration = col['duration']
        schedules_los = {(profile, current_col_id): duration}
        
        # Build x_list (list of all x values for this column)
        x_list = list(schedules_x.values())
        
        # Build los_list (single element list with duration)
        los_list = [duration]
        
        formatted_columns.append({
            'reduced_cost': col['reduced_cost'],
            'schedules_x': schedules_x,
            'schedules_los': schedules_los,
            'x_list': x_list,
            'los_list': los_list,
            'path_pattern': path_pattern,
            'worker': worker,
            'start': start,
            'end': end,
            'final_progress': col['final_progress'],
            'x_vector': col['x_vector']
        })
        
        # Increment col_id for next column? 
        # Note: The caller (branch_and_price) manages col_ids. 
        # Here we just use the passed col_id as a placeholder or base.
        # Ideally, branch_and_price should re-assign IDs when adding to master.
        # But for now, let's keep using the passed col_id to avoid breaking structure,
        # assuming branch_and_price handles the actual ID assignment or we just return data.
        # Actually, schedules_x keys use col_id. If we return multiple, they need unique IDs locally?
        # Let's increment it locally to be safe, though BnP might overwrite it.
        current_col_id += 1
        
    return formatted_columns


def run_labeling_algorithm(recipients_r, recipients_s, gamma_dict, obj_mode_dict, 
                           pi_dict, workers, max_time, ms, min_ms, theta_lookup,
                           print_worker_selection=True, use_bound_pruning=True, 
                           dominance_mode='bucket', branch_constraints=None, 
                           branching_variant='mp', n_workers=None):
    """
    Global Labeling Algorithm Function.
    
    Labeling Algorithm for Column Generation (Pricing Problem Solver)
    
    Args:
        recipients_r: Release times {recipient_id: r_k}
        recipients_s: Service targets {recipient_id: s_k}
        gamma_dict: Dual values gamma {recipient_id: gamma_k}
        obj_mode_dict: Objective multipliers {recipient_id: multiplier}
        pi_dict: Dual values pi {(worker_id, time): pi_jt}
        workers: List of worker IDs
        max_time: Planning horizon
        ms: Rolling window size
        min_ms: Minimum human services in window
        theta_lookup: AI efficiency lookup table
        print_worker_selection: Print worker dominance info per recipient
        use_bound_pruning: Enable/Disable lower bound pruning
        dominance_mode: 'bucket' (default) or 'global' dominance strategy
        branch_constraints: Optional branch constraints dictionary
        branching_variant: Branching strategy ('mp' or 'sp')
        n_workers: Number of parallel workers (None = sequential)
        
    Returns:
        List of best columns (can be multiple per recipient if alternatives exist)
    """
    t0 = time.time()
    results = []
    
    # Pruning Statistics
    pruning_stats = {
        'lb': 0,
        'dominance': 0,
        'printed_dominance': {}
    }
    
    # === PARALLEL OR SEQUENTIAL PROCESSING ===
    
    if n_workers is not None and n_workers > 1:
        # --- PARALLEL PROCESSING ---
        from multiprocessing import Pool
        
        logger.print(f"\n[PARALLEL MODE] Using {n_workers} workers for {len(recipients_r)} recipients")
        
        # Prepare arguments for each recipient
        recipient_args = []
        for k in recipients_r:
            gamma_val = gamma_dict.get(k, 0.0)
            multiplier = obj_mode_dict.get(k, 1)
            recipient_args.append((
                k, recipients_r[k], recipients_s[k], 
                gamma_val, multiplier, pi_dict, workers, 
                max_time, ms, min_ms, theta_lookup,
                use_bound_pruning, dominance_mode, 
                branch_constraints, branching_variant
            ))
        
        # Execute in parallel
        with Pool(processes=n_workers) as pool:
            all_cols = pool.starmap(solve_pricing_for_recipient, recipient_args)
        
        # Merge results
        recipient_keys = list(recipients_r.keys())
        for k, cols in zip(recipient_keys, all_cols):
            if cols:
                results.extend(cols)
    
    else:
        # --- SEQUENTIAL PROCESSING ---
        for k in recipients_r:
            gamma_val = gamma_dict.get(k, 0.0)
            multiplier = obj_mode_dict.get(k, 1)
            
            cols = solve_pricing_for_recipient(k, recipients_r[k], recipients_s[k], 
                                              gamma_val, multiplier, pi_dict, workers, 
                                              max_time, ms, min_ms, theta_lookup,
                                              use_bound_pruning=use_bound_pruning, 
                                              dominance_mode=dominance_mode, 
                                              branch_constraints=branch_constraints, 
                                              branching_variant=branching_variant)
            
            if cols:
                results.extend(cols)
    
    runtime = time.time() - t0
    
    logger.print(f"\nRuntime: {runtime:.4f}s")
    logger.print(f"Pruning Stats: Lower Bound = {pruning_stats['lb']}, State Dominance = {pruning_stats['dominance']}")
    logger.print(f"\n--- Final Results ({len(results)} optimal schedules) ---")
    
    return results
