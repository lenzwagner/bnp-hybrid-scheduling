"""
Create Sample Instance File for Reverse Stress Test

Generates a small Excel file with test instances for the reverse stress test.
"""

import pandas as pd
import os
from datetime import datetime

# Create sample instances
instances = []

# Vary seeds
for seed in [12, 42, 99]:
    # Vary PTTR scenarios
    for pttr in ['light', 'medium', 'heavy']:
        instances.append({
            'instance_id': f'test_{pttr}_seed{seed}',
            'scenario_nr': len(instances) + 1,
            'seed': seed,
            'D_focus_count': 10,
            'T_count': 10,
            'pttr': pttr
        })

# Create DataFrame
df = pd.DataFrame(instances)

# Save to Excel
output_dir = 'results/instances'
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
excel_filename = f"{output_dir}/stress_test_instances_{timestamp}.xlsx"

df.to_excel(excel_filename, index=False)

print(f"Created {len(instances)} test instances")
print(f"Saved to: {excel_filename}")
print(f"\nInstances:")
print(df.to_string(index=False))
