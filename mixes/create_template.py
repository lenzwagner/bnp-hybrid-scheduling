#!/usr/bin/env python3
"""
Create a template Excel file for manual severity mix configuration

This creates an Excel file with example scenarios that you can edit manually.
"""

import pandas as pd
import os

def create_template():
    """Create template Excel file with severity mix examples"""
    
    # Define example scenarios
    scenarios = [
        # Baseline comparison (same seed across all mixes)
        {
            'instance_id': 'baseline_seed42',
            'scenario_nr': 1,
            'seed': 42,
            'pttr': 'medium',
            'T_count': 10,
            'D_focus_count': 30,
            'severity_mix': None,
            'severity_mix_name': None,
        },
        {
            'instance_id': 'neuro_seed42',
            'scenario_nr': 2,
            'seed': 42,
            'pttr': 'medium',
            'T_count': 10,
            'D_focus_count': 30,
            'severity_mix': '(0.7, 0.2, 0.1)',
            'severity_mix_name': 'neuro',
        },
        {
            'instance_id': 'ortho_seed42',
            'scenario_nr': 3,
            'seed': 42,
            'pttr': 'medium',
            'T_count': 10,
            'D_focus_count': 30,
            'severity_mix': '(0.1, 0.2, 0.7)',
            'severity_mix_name': 'ortho',
        },
        
        # Additional replications
        {
            'instance_id': 'baseline_seed43',
            'scenario_nr': 4,
            'seed': 43,
            'pttr': 'medium',
            'T_count': 10,
            'D_focus_count': 30,
            'severity_mix': None,
            'severity_mix_name': None,
        },
        {
            'instance_id': 'neuro_seed43',
            'scenario_nr': 5,
            'seed': 43,
            'pttr': 'medium',
            'T_count': 10,
            'D_focus_count': 30,
            'severity_mix': '(0.7, 0.2, 0.1)',
            'severity_mix_name': 'neuro',
        },
        {
            'instance_id': 'ortho_seed43',
            'scenario_nr': 6,
            'seed': 43,
            'pttr': 'medium',
            'T_count': 10,
            'D_focus_count': 30,
            'severity_mix': '(0.1, 0.2, 0.7)',
            'severity_mix_name': 'ortho',
        },
    ]
    
    # Create DataFrame
    df = pd.DataFrame(scenarios)
    
    # Add learning type columns (will be filled by loop_main_learning.py)
    df['learn_type'] = 'sigmoid'
    
    # Output path
    output_file = 'mixes/instances/scenarios_template.xlsx'
    
    # Save to Excel
    df.to_excel(output_file, index=False)
    
    print(f"\n✓ Template created: {output_file}")
    print(f"\nThis file contains {len(scenarios)} example scenarios:")
    print(f"  - 2 replications (seeds 42-43)")
    print(f"  - 3 severity mixes per replication (baseline, neuro, ortho)")
    print(f"\nYou can:")
    print(f"  1. Edit this file to add more scenarios")
    print(f"  2. Change seeds, T, D_focus, pttr parameters")
    print(f"  3. Add custom severity_mix configurations")
    print(f"\nTo run: python3 mixes/loop_main_learning.py --scenarios {output_file}")

if __name__ == "__main__":
    create_template()
