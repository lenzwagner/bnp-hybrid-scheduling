import pandas as pd
import numpy as np
import os

def analyze_results():
    # File paths
    input_file = 'results_computational.xlsx'
    output_file = 'aggr_stats.xlsx'

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return

    print(f"Loading data from {input_file}...")
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # Continuous metrics (Median, Min, Max, Std)
    continuous_metrics = [
        'total_time',
        'final_gap',
        'time_in_root',
        'time_to_first_incumbent',
        'total_nodes',
        # 'max_tree_depth',      # Note: check if column exists
        'max_tree_depth',
        'root_gap',
        'time_in_sp',
        'time_in_mp',
        'time_in_ip_heuristic',
        'time_in_branching',
        'total_cg_iterations',
        'root_iterations'
    ]
    
    # Binary metrics (Percentage share)
    binary_metrics = [
        'is_optimal',
        'root_integral'
    ]
    
    # Grouping columns
    # Adjust column names if they differ in the Excel file (e.g., 'T_count', 'D_focus_count')
    # Based on loop_main.py, likely names are 'T' or 'T_count', 'D_focus' or 'D_focus_count'
    
    # Let's inspect columns first
    print("Columns found:", df.columns.tolist())
    
    # Extract root_iterations from iterations_per_node if available
    if 'iterations_per_node' in df.columns:
        import ast
        def get_root_iterations(val):
            try:
                # Handle direct integer/float
                if isinstance(val, (int, float)):
                    return val
                # Handle string representation of list
                if isinstance(val, str):
                    lst = ast.literal_eval(val)
                    if isinstance(lst, list) and len(lst) > 0:
                        return lst[0]
                # Handle list directly (though unlikely from text file read, possible if engine handles it)
                if isinstance(val, list) and len(val) > 0:
                    return val[0]
            except:
                pass
            return np.nan
            
        print("Extracting root_iterations from iterations_per_node...")
        df['root_iterations'] = df['iterations_per_node'].apply(get_root_iterations)
        
    
    # Heuristic to find correct grouping columns
    group_col_T = 'T' if 'T' in df.columns else 'T_count'
    group_col_D = 'D_focus' if 'D_focus' in df.columns else 'D_focus_count'
    
    if group_col_T not in df.columns or group_col_D not in df.columns:
        print(f"Error: Could not find grouping columns for T and D (checked {group_col_T}, {group_col_D})")
        return

    print(f"Grouping by: {group_col_T} and {group_col_D}")

    # Validate metrics existence
    valid_continuous = [m for m in continuous_metrics if m in df.columns]
    valid_binary = [m for m in binary_metrics if m in df.columns]
    
    if not valid_continuous and not valid_binary:
        print("Error: No valid metrics found to analyze.")
        return
        
    if valid_continuous:
        missing_cont = [m for m in continuous_metrics if m not in df.columns]
        if missing_cont:
             print(f"Warning: Missing continuous metrics: {missing_cont}")

    if valid_binary:
        missing_bin = [m for m in binary_metrics if m not in df.columns]
        if missing_bin:
             print(f"Warning: Missing binary metrics: {missing_bin}")

    # Convert continuous metrics to numeric, coercing errors to NaN
    for m in valid_continuous:
        df[m] = pd.to_numeric(df[m], errors='coerce')
        
    # Convert binary metrics to boolean/numeric (1/0) for mean calculation
    for m in valid_binary:
        # Map True/False or "True"/"False" strings to 1/0
        df[m] = df[m].apply(lambda x: 1 if str(x).lower() in ['true', '1', '1.0', 'yes'] else 0)

    # Aggregation functions
    agg_funcs_continuous = ['median', 'min', 'max', 'std']

    # Perform aggregation
    print("Calculating statistics...")
    
    # 1. Continuous stats
    grouped_cont = pd.DataFrame()
    if valid_continuous:
        grouped_cont = df.groupby([group_col_T, group_col_D])[valid_continuous].agg(agg_funcs_continuous)
        # Flatten MultiIndex columns
        grouped_cont.columns = ['_'.join(col).strip() for col in grouped_cont.columns.values]
    
    # 2. Binary stats (Mean = Percentage)
    grouped_bin = pd.DataFrame()
    if valid_binary:
        # Calculate mean (percentage)
        grouped_bin = df.groupby([group_col_T, group_col_D])[valid_binary].agg('mean')
        # Rename columns to indicate percentage
        grouped_bin.columns = [f"{col}_pct" for col in grouped_bin.columns]
        
    # Merge results
    if not grouped_cont.empty and not grouped_bin.empty:
        grouped = pd.concat([grouped_cont, grouped_bin], axis=1)
    elif not grouped_cont.empty:
        grouped = grouped_cont
    else:
        grouped = grouped_bin
        
    grouped = grouped.reset_index()

    # Round before printing and saving
    grouped = grouped.round(2)

    # Print to console
    print("\n" + "="*80)
    print(" AGGREGATED STATISTICS (Grouped by T x D) ".center(80, "="))
    print("="*80)
    print(grouped.to_string(float_format="%.2f"))
    print("="*80 + "\n")

    # Copy-friendly output (for plotting)
    print("\n" + "="*80)
    print(" COPY-FRIENDLY OUTPUT (Tab-separated) ".center(80, "="))
    print("="*80)
    print(grouped.to_csv(sep='\t', index=False, float_format="%.2f"))
    print("="*80 + "\n")

    # Save to Excel
    try:
        grouped.to_excel(output_file, index=False)
        print(f"Successfully saved aggregated statistics to: {output_file}")
    except Exception as e:
        print(f"Error saving output Excel: {e}")

if __name__ == "__main__":
    analyze_results()
