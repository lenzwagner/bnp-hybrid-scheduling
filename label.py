import collections
import sys
import time

# --- 1. Input Data ---

pi = {(1, 1): 0.0, (1, 2): 0.0, (1, 3): 0.0, (1, 4): 0.0, (1, 5): 0.0, (1, 6): 0.0, (1, 7): -12.0, (1, 8): 0.0,
      (1, 9): 0.0, (1, 10): 0.0, (1, 11): 0.0, (1, 12): 0.0, (1, 13): 0.0, (1, 14): 0.0, (1, 15): 0.0, (1, 16): 0.0,
      (1, 17): 0.0, (1, 18): 0.0, (1, 19): 0.0, (1, 20): 0.0, (1, 21): -43119570900367.81, (1, 22): 0.0, (1, 23): 0.0,
      (1, 24): 0.0, (1, 25): 0.0, (1, 26): 0.0, (1, 27): 0.0, (1, 28): 0.0, (1, 29): 0.0, (1, 30): 0.0, (1, 31): 0.0,
      (1, 32): 0.0, (1, 33): 0.0, (1, 34): 0.0, (1, 35): -9.0, (1, 36): 0.0, (1, 37): 0.0, (1, 38): 0.0, (1, 39): 0.0,
      (1, 40): 0.0, (1, 41): 0.0, (1, 42): 0.0, (2, 1): 0.0, (2, 2): 0.0, (2, 3): 0.0, (2, 4): 0.0, (2, 5): 0.0,
      (2, 6): 0.0, (2, 7): 0.0, (2, 8): -12.0, (2, 9): 0.0, (2, 10): 0.0, (2, 11): 0.0, (2, 12): 0.0, (2, 13): 0.0,
      (2, 14): 0.0, (2, 15): 0.0, (2, 16): 0.0, (2, 17): 0.0, (2, 18): 0.0, (2, 19): 0.0, (2, 20): 0.0, (2, 21): 0.0,
      (2, 22): -12.0, (2, 23): 0.0, (2, 24): 0.0, (2, 25): 0.0, (2, 26): 0.0, (2, 27): 0.0, (2, 28): 0.0,
      (2, 29): -21.0, (2, 30): 0.0, (2, 31): 0.0, (2, 32): 0.0, (2, 33): 0.0, (2, 34): 0.0, (2, 35): 0.0, (2, 36): 0.0,
      (2, 37): 0.0, (2, 38): 0.0, (2, 39): 0.0, (2, 40): 0.0, (2, 41): 0.0, (2, 42): 0.0, (3, 1): 0.0, (3, 2): 0.0,
      (3, 3): 0.0, (3, 4): 0.0, (3, 5): 0.0, (3, 6): 0.0, (3, 7): 0.0, (3, 8): 0.0, (3, 9): -12.0, (3, 10): 0.0,
      (3, 11): 0.0, (3, 12): 0.0, (3, 13): 0.0, (3, 14): 0.0, (3, 15): 0.0, (3, 16): -12.0, (3, 17): 0.0, (3, 18): 0.0,
      (3, 19): 0.0, (3, 20): 0.0, (3, 21): 0.0, (3, 22): 0.0, (3, 23): -12.0, (3, 24): 0.0, (3, 25): 0.0, (3, 26): 0.0,
      (3, 27): 0.0, (3, 28): 0.0, (3, 29): 0.0, (3, 30): 0.0, (3, 31): 0.0, (3, 32): 0.0, (3, 33): 0.0, (3, 34): 0.0,
      (3, 35): 0.0, (3, 36): 0.0, (3, 37): -9.0, (3, 38): 0.0, (3, 39): 0.0, (3, 40): 0.0, (3, 41): 0.0, (3, 42): 0.0}
gamma = {2: 21.0, 3: 0.0, 4: 0.0, 7: 0.0, 8: 0.0, 10: 0.0, 12: 0.0, 15: 9.0, 18: 0.0, 19: 0.0, 21: 12.0, 22: 12.0,
         24: 0.0, 29: 0.0, 31: 0.0, 33: 0.0, 34: 0.0, 35: 0.0, 37: 0.0, 38: 0.0, 41: 0.0, 42: 3.0, 46: 0.0, 47: 4.0,
         49: 0.0, 53: 0.0, 55: 2.0, 57: 3.0, 58: 0.0, 59: 2.0, 61: 2.0, 62: 0.0, 64: 3.0, 66: 0.0, 67: 0.0, 70: 0.0,
         71: 0.0, 72: 0.0, 75: 0.0, 76: 0.0, 80: 0.0, 81: 4.0}

r_i = {2: 19, 3: 15, 4: 38, 7: 22, 8: 16, 10: 17, 12: 25, 15: 34, 18: 26, 19: 13, 21: 7, 22: 16, 24: 15, 29: 40, 31: 41,
       33: 11, 34: 32, 35: 18, 37: 10, 38: 13, 41: 32, 42: 4, 46: 26, 47: 3, 49: 28, 53: 22, 55: 5, 57: 2, 58: 17,
       59: 5, 61: 2, 62: 10, 64: 1, 66: 31, 67: 37, 70: 19, 71: 9, 72: 38, 75: 9, 76: 37, 80: 13, 81: 1}
s_i = {2: 18, 3: 16, 4: 7, 7: 8, 8: 6, 10: 7, 12: 5, 15: 10, 18: 7, 19: 3, 21: 10, 22: 11, 24: 6, 29: 6, 31: 4, 33: 2,
       34: 5, 35: 4, 37: 4, 38: 10, 41: 4, 42: 5, 46: 4, 47: 6, 49: 11, 53: 6, 55: 3, 57: 5, 58: 5, 59: 2, 61: 3, 62: 8,
       64: 5, 66: 6, 67: 8, 70: 6, 71: 3, 72: 4, 75: 7, 76: 2, 80: 4, 81: 6}
obj_mode = {2: 0, 3: 0, 4: 0, 7: 0, 8: 0, 10: 0, 12: 0, 15: 0, 18: 0, 19: 0, 21: 0, 22: 0, 24: 0, 29: 0, 31: 0, 33: 0,
            34: 0, 35: 0, 37: 0, 38: 0, 41: 0, 42: 1, 46: 0, 47: 1, 49: 0, 53: 0, 55: 1, 57: 1, 58: 0, 59: 1, 61: 1,
            62: 0, 64: 1, 66: 0, 67: 0, 70: 0, 71: 0, 72: 0, 75: 0, 76: 0, 80: 0, 81: 1}

# Parameter
MS = 5
MIN_MS = 2
MAX_TIME = 42  # Hier wird es dynamisch verwendet
WORKERS = [1, 2, 3]

# Lookup Table
theta_lookup = [0.2 + 0.01 * k for k in range(50)]
theta_lookup = [min(x, 1.0) for x in theta_lookup]


# --- 2. Helper Functions ---

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


def validate_final_column(col_data, s_req, MS, MIN_MS, theta_table):
    """
    Validiert eine fertige Spalte strikt auf alle Constraints.
    Gibt eine Liste von Fehlern zurück (leer wenn alles OK).
    """
    errors = []
    path = col_data['path_pattern']

    # 1. Check Start & End Constraint
    if path[0] != 1:
        errors.append(f"Start constraint violation: First day must be Machine (1), found {path[0]}")

    # MODIFIZIERT: Check End Constraint
    # Wenn es KEIN Timeout ist, MUSS es eine 1 sein.
    # Wenn es ein Timeout ist, darf es auch eine 0 sein.
    is_timeout = (col_data['end'] == MAX_TIME)

    if path[-1] != 1:
        if not is_timeout:
            errors.append(f"End constraint violation: Last day must be Machine (1), found {path[-1]}")
        # Im Timeout-Fall (is_timeout == True) erlauben wir hier implizit die 0.

    # 2. Check Service Target (s_i)
    progress = 0.0
    ai_usage = 0
    for x in path:
        if x == 1:
            progress += 1.0
        else:
            eff = theta_table[ai_usage] if ai_usage < len(theta_table) else 1.0
            progress += eff
            ai_usage += 1

    # Toleranz 1e-9
    if progress < s_req - 1e-9:
        # Sonderfall: Timeout (Ende des Horizonts) darf Ziel verfehlen, wenn Modell das erlaubt.
        # Wir markieren es hier als Info/Fehler zur Kontrolle (wie gewünscht).
        if is_timeout:
            errors.append(f"Target NOT met (TIMEOUT case): {progress:.2f} < {s_req}")
        else:
            errors.append(f"Target NOT met: {progress:.2f} < {s_req}")

    # 3. Check Rolling Window (Strict Window-by-Window)
    if len(path) >= MS:
        for i in range(len(path) - MS + 1):
            window = path[i: i + MS]
            if sum(window) < MIN_MS:
                errors.append(
                    f"Window violation at index {i} (Days {col_data['start'] + i}-{col_data['start'] + i + MS - 1}): {window} sum={sum(window)}")

    else:
        current_sum = sum(path)
        remaining = MS - len(path)
        if current_sum + remaining < MIN_MS:
            errors.append(f"Short path violation: {path} (sum {current_sum}) cannot satisfy MIN_MS={MIN_MS}")

    return errors


# --- 3. Labeling Algorithm ---

def solve_pricing_for_recipient(k, r_k, s_k, gamma_k, obj_multiplier):
    best_reduced_cost = float('inf')
    best_columns = []
    epsilon = 1e-9

    time_until_end = MAX_TIME - r_k + 1
    candidate_workers = WORKERS  # Simplified dominance

    for j in candidate_workers:
        effective_min_duration = min(int(s_k), time_until_end)
        start_tau = r_k + effective_min_duration - 1

        for tau in range(start_tau, MAX_TIME + 1):
            is_timeout_scenario = (tau == MAX_TIME)

            start_cost = -pi.get((j, r_k), 0)
            current_states = {
                (1.0, 0, (1,)): (start_cost, [1])
            }

            # DP Loop bis kurz vor Tau
            for t in range(r_k + 1, tau):
                next_states = {}
                for state, (cost, path) in current_states.items():
                    prog, ai_count, hist = state

                    remaining_steps = tau - t + 1
                    if not is_timeout_scenario:
                        if prog + remaining_steps * 1.0 < s_k - epsilon:
                            continue

                    # A: Therapist
                    if check_strict_feasibility(hist, 1, MS, MIN_MS):
                        cost_ther = cost - pi.get((j, t), 0)
                        prog_ther = prog + 1.0
                        new_hist_ther = (hist + (1,))
                        if len(new_hist_ther) > MS - 1: new_hist_ther = new_hist_ther[-(MS - 1):]

                        state_ther = (prog_ther, ai_count, new_hist_ther)
                        if state_ther not in next_states or cost_ther < next_states[state_ther][0]:
                            next_states[state_ther] = (cost_ther, path + [1])

                    # B: AI
                    if check_strict_feasibility(hist, 0, MS, MIN_MS):
                        cost_ai = cost
                        efficiency = theta_lookup[ai_count] if ai_count < len(theta_lookup) else 1.0
                        prog_ai = prog + efficiency
                        ai_count_new = ai_count + 1
                        new_hist_ai = (hist + (0,))
                        if len(new_hist_ai) > MS - 1: new_hist_ai = new_hist_ai[-(MS - 1):]

                        state_ai = (prog_ai, ai_count_new, new_hist_ai)
                        if state_ai not in next_states or cost_ai < next_states[state_ai][0]:
                            next_states[state_ai] = (cost_ai, path + [0])

                current_states = next_states
                if not current_states: break

            # Final Step (Transition to Tau)
            # Hier ist die Änderung für den Timeout-Fall
            for state, (cost, path) in current_states.items():
                prog, ai_count, hist = state

                # Wir sammeln mögliche End-Schritte für diesen State
                possible_moves = []

                # Option 1: Enden mit Therapeut (1) - Standard
                if check_strict_feasibility(hist, 1, MS, MIN_MS):
                    possible_moves.append(1)

                # Option 2: Enden mit App (0) - NUR wenn Timeout
                if is_timeout_scenario:
                    if check_strict_feasibility(hist, 0, MS, MIN_MS):
                        possible_moves.append(0)

                for move in possible_moves:
                    # Berechne Werte basierend auf Move-Typ
                    if move == 1:
                        final_cost_accum = cost - pi.get((j, tau), 0)
                        final_prog = prog + 1.0
                        # Hier nutzen wir den alten count, da er sich nicht erhöht hat
                        final_ai_count = ai_count
                    else:  # move == 0
                        final_cost_accum = cost
                        efficiency = theta_lookup[ai_count] if ai_count < len(theta_lookup) else 1.0
                        final_prog = prog + efficiency
                        final_ai_count = ai_count + 1

                    final_path = path + [move]
                    condition_met = (final_prog >= s_k - epsilon)

                    if condition_met or is_timeout_scenario:
                        duration = tau - r_k + 1
                        reduced_cost = (obj_multiplier * duration) + final_cost_accum - gamma_k

                        col_candidate = {
                            'k': k,
                            'worker': j,
                            'start': r_k,
                            'end': tau,
                            'duration': duration,
                            'reduced_cost': reduced_cost,
                            'final_progress': final_prog,
                            'x_vector': generate_full_column_vector(j, final_path, r_k, tau, MAX_TIME, len(WORKERS)),
                            'path_pattern': final_path
                        }

                        if reduced_cost < best_reduced_cost - epsilon:
                            best_reduced_cost = reduced_cost
                            best_columns = [col_candidate]
                        elif abs(reduced_cost - best_reduced_cost) < epsilon:
                            best_columns.append(col_candidate)

    return best_columns


# --- 4. Main Execution ---

t0 = time.time()
results = []

for k in r_i:
    if k in gamma:
        gamma_val = gamma[k]
    else:
        gamma_val = 0.0
    multiplier = obj_mode.get(k, 1)

    cols = solve_pricing_for_recipient(k, r_i[k], s_i[k], gamma_val, multiplier)

    if cols:
        results.append(cols[0])

print(f"\nRuntime: {time.time() - t0:.4f}s")

print("\n--- Final Results (First found optimal per Recipient) ---")
for res in results:
    print(f"\nRecipient {res['k']}:")
    print(f"  Reduced Cost: {res['reduced_cost']:.6f}")
    print(f"  Worker: {res['worker']}, Interval: {res['start']}-{res['end']}")

    vec = res['x_vector']
    time_indices = [(i % MAX_TIME) + 1 for i, x in enumerate(vec) if x > 0.5]
    print(f"  Active Time Steps (Day 1-{MAX_TIME}): {time_indices}")

    # Kurzer Check, ob es eine App am Ende war
    last_day_val = res['path_pattern'][-1]
    last_day_type = "Therapist" if last_day_val == 1 else "App"
    print(f"  Last Session Type: {last_day_type} (Val: {last_day_val})")

    # --- NEU: VALIDIERUNGS-CHECK ---
    validation_errors = validate_final_column(res, s_i[res['k']], MS, MIN_MS, theta_lookup)

    if validation_errors:
        print("  [!] NOT CHECKED CONSTRAINTS / VIOLATIONS:")
        for err in validation_errors:
            print(f"      - {err}")
    else:
        print("  [OK] All constraints satisfied.")
