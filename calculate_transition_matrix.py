"""
Session Transition Matrix Calculator

Computes P(s' | s, theta) = Probability of transitioning from session type s to s' 
given effectiveness range theta ∈ [theta_min, theta_max]

Session types:
- 'Human' (x=1): Therapist session
- 'AI' (y=1): AI/App session  
- 'Gap': No treatment (eligible but idle)
"""

import numpy as np
import pandas as pd
import math
from collections import defaultdict
from Utils.derived_vars import compute_derived_variables


def compute_theta_value(Y, learn_type, theta_base, k_learn=None, lin_increase=None, infl_point=None):
    """
    Compute theta value for a given Y (cumulative AI sessions).
    
    Args:
        Y: Cumulative AI sessions count
        learn_type: Learning type ('exp', 'sigmoid', 'lin', or numeric constant)
        theta_base: Base effectiveness
        k_learn: Learning rate for exp/sigmoid
        lin_increase: Linear increase rate for 'lin'
        infl_point: Inflection point for sigmoid
    
    Returns:
        float: Theta effectiveness value
    """
    if learn_type == 'exp':
        return theta_base + (1 - theta_base) * (1 - math.exp(-k_learn * Y))
    elif learn_type == 'sigmoid':
        return theta_base + (1 - theta_base) / (1 + math.exp(-k_learn * (Y - infl_point)))
    elif learn_type == 'lin':
        return min(1.0, theta_base + lin_increase * Y)
    else:
        # Constant learning factor (numeric)
        return theta_base


def extract_pwl_breakpoints(app_data, max_Y=20):
    """
    Extract PWL breakpoints from app_data to use as theta bins.
    
    Creates one bin per discrete Y value (cumulative AI sessions),
    mapping directly to the PWL lookup table structure.
    
    Args:
        app_data: Dictionary with learning parameters
        max_Y: Maximum cumulative AI sessions to consider
    
    Returns:
        list: List of (theta_min, theta_max) tuples, one per Y transition
    """
    learn_type = app_data['learn_type'][0]
    theta_base = app_data['theta_base'][0]
    
    # Generate Y values (PWL breakpoints from R)
    Y_values = list(range(0, max_Y + 1))
    
    # Compute corresponding theta values
    theta_values = []
    for Y in Y_values:
        if learn_type == 'exp':
            k_learn = app_data['k_learn'][0]
            theta = compute_theta_value(Y, 'exp', theta_base, k_learn=k_learn)
        elif learn_type == 'sigmoid':
            k_learn = app_data['k_learn'][0]
            infl_point = app_data['infl_point'][0]
            theta = compute_theta_value(Y, 'sigmoid', theta_base, k_learn=k_learn, infl_point=infl_point)
        elif learn_type == 'lin':
            lin_increase = app_data['lin_increase'][0]
            theta = compute_theta_value(Y, 'lin', theta_base, lin_increase=lin_increase)
        else:
            # Constant effectiveness
            theta = theta_base
        theta_values.append(theta)
    
    # Create bins from consecutive theta values
    # Each bin corresponds to one Y value transition Y→Y+1
    theta_bins = []
    for i in range(len(theta_values) - 1):
        theta_min = theta_values[i]
        theta_max = theta_values[i + 1]
        # Only create bin if there's a meaningful change OR if it's the first few bins
        if abs(theta_max - theta_min) > 1e-6 or i < 5:
            # Store as tuple: (theta_min, theta_max, Y_value)
            theta_bins.append((theta_min, theta_max, i))
    
    # Ensure we cover the final range up to 1.0
    if theta_values and theta_values[-1] < 1.0:
        theta_bins.append((theta_values[-1], 1.01, len(theta_values) - 1))
    
    return theta_bins if theta_bins else [(theta_base, 1.01, 0)]


def compute_session_transition_matrix(cg_solver, inc_sol, app_data, 
                                       theta_bins=None, patients_list=None):
    """
    Computes session transition matrix P(s' | s, theta).
    
    Args:
        cg_solver: ColumnGeneration instance with problem data
        inc_sol: Incumbent solution dictionary
        app_data: Dictionary with learning parameters
        theta_bins: List of (theta_min, theta_max) tuples for effectiveness ranges
                   Default: [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
        patients_list: List of patients to analyze (default: cg_solver.P_F)
    
    Returns:
        dict: Nested dictionary with structure:
              {theta_range_str: {from_session: {to_session: probability}}}
        pd.DataFrame: Summary dataframe with all transitions
    """
    
    # Default theta bins: extract from PWL breakpoints if not provided
    if theta_bins is None:
        theta_bins = extract_pwl_breakpoints(app_data, max_Y=20)
        if not theta_bins:
            # Fallback to default bins if extraction fails
            theta_bins = [
                (0.0, 0.3),   # Low effectiveness
                (0.3, 0.5),   # Medium-low
                (0.5, 0.7),   # Medium-high
                (0.7, 0.9),   # High
                (0.9, 1.01)   # Very high
            ]
    
    # Get derived variables
    all_e, all_Y, all_theta, all_omega, all_g_comp, all_z, all_g_gap = \
        compute_derived_variables(cg_solver, inc_sol, app_data, patients_list)
    
    # Extract x and y from solution
    x_dict = inc_sol.get('x', {})
    y_dict = inc_sol.get('y', {})
    
    # Aggregate x and y per (patient, day)
    x_agg = defaultdict(float)
    for k, v in x_dict.items():
        if v > 1e-6:
            p, t, d = k[0], k[1], k[2]
            x_agg[(p, d)] += v
    
    y_agg = defaultdict(float)
    for k, v in y_dict.items():
        if v > 1e-6:
            p, d = k[0], k[1]
            y_agg[(p, d)] += v
    
    # Determine patients
    target_patients = patients_list if patients_list is not None else cg_solver.P_F
    
    # Initialize transition counters for each theta range
    # Structure: {theta_range: {from_session: {to_session: count}}}
    transitions = {}
    for bin_info in theta_bins:
        # Handle both old format (theta_min, theta_max) and new format (theta_min, theta_max, Y)
        if len(bin_info) == 3:
            theta_min, theta_max, Y_val = bin_info
            range_key = f"Y={Y_val} [θ={theta_min:.3f}-{theta_max:.3f})"
        else:
            theta_min, theta_max = bin_info
            range_key = f"[{theta_min:.1f}, {theta_max:.1f})"
        
        transitions[range_key] = {
            'Human': defaultdict(int),
            'AI': defaultdict(int),
            'Gap': defaultdict(int)
        }
    
    # Track transitions for each patient
    for p in target_patients:
        # Get entry and discharge days
        entry_day = None
        if hasattr(cg_solver, 'Entry') and p in cg_solver.Entry:
            entry_day = cg_solver.Entry[p]
        else:
            entry_day = cg_solver.Entry_agg.get(p, 1)
        
        # Find all eligible days (e=1)
        eligible_days = sorted([d for (pi, d) in all_e.keys() 
                               if pi == p and all_e.get((pi, d), 0) == 1])
        
        if not eligible_days:
            continue
        
        # Track session types for each day
        session_sequence = []
        theta_sequence = []
        
        for d in eligible_days:
            # Determine session type
            has_human = x_agg.get((p, d), 0) > 0.5
            has_ai = y_agg.get((p, d), 0) > 0.5
            theta_val = all_theta.get((p, d), 0.0)
            
            if has_human:
                session_type = 'Human'
            elif has_ai:
                session_type = 'AI'
            else:
                session_type = 'Gap'
            
            session_sequence.append(session_type)
            theta_sequence.append(theta_val)
        
        # Count transitions
        for i in range(len(session_sequence) - 1):
            from_session = session_sequence[i]
            to_session = session_sequence[i + 1]
            theta_current = theta_sequence[i]
            
            # Determine which theta bin this belongs to
            for bin_info in theta_bins:
                # Handle both old and new formats
                if len(bin_info) == 3:
                    theta_min, theta_max, Y_val = bin_info
                    range_key = f"Y={Y_val} [θ={theta_min:.3f}-{theta_max:.3f})"
                else:
                    theta_min, theta_max = bin_info
                    range_key = f"[{theta_min:.1f}, {theta_max:.1f})"
                    
                if theta_min <= theta_current < theta_max or \
                   (theta_max >= 1.0 and theta_current >= theta_min):  # Include upper bound for last bin
                    transitions[range_key][from_session][to_session] += 1
                    break
    
    # Convert counts to probabilities
    transition_probabilities = {}
    for range_key, from_dict in transitions.items():
        transition_probabilities[range_key] = {}
        for from_session, to_dict in from_dict.items():
            total_transitions = sum(to_dict.values())
            transition_probabilities[range_key][from_session] = {}
            
            if total_transitions > 0:
                for to_session, count in to_dict.items():
                    prob = count / total_transitions
                    transition_probabilities[range_key][from_session][to_session] = prob
            else:
                # No transitions from this state in this theta range
                transition_probabilities[range_key][from_session] = {
                    'Human': 0.0,
                    'AI': 0.0,
                    'Gap': 0.0
                }
    
    # Create summary dataframe
    rows = []
    for range_key in transition_probabilities:
        for from_session in ['Human', 'AI', 'Gap']:
            to_probs = transition_probabilities[range_key].get(from_session, {})
            
            # Calculate total transitions for this from_session in this theta range
            total = sum(transitions[range_key][from_session].values())
            
            row = {
                'Theta_Range': range_key,
                'From_Session': from_session,
                'To_Human': to_probs.get('Human', 0.0),
                'To_AI': to_probs.get('AI', 0.0),
                'To_Gap': to_probs.get('Gap', 0.0),
                'Total_Transitions': total
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Filter out bins with no transitions
    df = df[df['Total_Transitions'] > 0]
    
    return transition_probabilities, df, transitions


def print_transition_matrix(transition_probs, transitions_counts=None):
    """
    Pretty print the transition matrix.
    
    Args:
        transition_probs: Output from compute_session_transition_matrix
        transitions_counts: Optional raw counts for display
    """
    print("\n" + "="*80)
    print(" SESSION TRANSITION MATRIX P(s' | s, θ) ".center(80, "="))
    print("="*80)
    
    session_types = ['Human', 'AI', 'Gap']
    
    for range_key in sorted(transition_probs.keys()):
        # Skip bins with no transitions at all
        if transitions_counts:
            total_in_range = sum(
                sum(transitions_counts[range_key][from_s].values()) 
                for from_s in session_types
            )
            if total_in_range == 0:
                continue  # Skip empty bins
        
        print(f"\n📊 Effectiveness Range: θ ∈ {range_key}")
        print("-" * 80)
        
        # Print header
        header_label = "From \\ To"
        print(f"{header_label:<12}", end="")
        for to_session in session_types:
            print(f"{to_session:>12}", end="")
        if transitions_counts:
            print(f"{'Count':>12}", end="")
        print()
        print("-" * 80)
        
        # Print each row (only if it has transitions)
        for from_session in session_types:
            # Skip this row if no transitions from this session type
            if transitions_counts:
                row_total = sum(transitions_counts[range_key][from_session].values())
                if row_total == 0:
                    continue  # Skip empty rows
            
            print(f"{from_session:<12}", end="")
            
            probs = transition_probs[range_key].get(from_session, {})
            for to_session in session_types:
                prob = probs.get(to_session, 0.0)
                print(f"{prob:>11.2%}", end=" ")
            
            if transitions_counts:
                total = sum(transitions_counts[range_key][from_session].values())
                print(f"{total:>11}", end=" ")
            
            print()
        print()


def export_transition_matrix(df, filename='transition_matrix.xlsx'):
    """
    Export transition matrix to Excel.
    
    Args:
        df: DataFrame from compute_session_transition_matrix
        filename: Output filename
    """
    df.to_excel(filename, index=False)
    print(f"\n✅ Transition matrix exported to: {filename}")


if __name__ == "__main__":
    # Example usage (requires running from main.py or with loaded solution)
    print("This module computes session transition matrices.")
    print("Import and call compute_session_transition_matrix() with your solution data.")
