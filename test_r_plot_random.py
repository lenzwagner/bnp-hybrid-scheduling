"""
Test script to generate random data and test the R violin plot.
"""

import numpy as np
import pandas as pd
from plot_r_violin import los_split_violin_r_plot

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples per category
n_samples = 24

# Create random dataset
data_list = []

for _ in range(n_samples):
    for pttr_category in ['light', 'medium', 'heavy']:
        for only_human in [0, 1]:  # 0 = Hybrid, 1 = Human Only
            # Generate realistic LOS values based on category and model
            base_los = {
                'light': 15,
                'medium': 25,
                'heavy': 35
            }[pttr_category]
            
            # Human Only typically has higher LOS than Hybrid
            if only_human == 1:
                mean_los = base_los + np.random.uniform(2, 5)
            else:
                mean_los = base_los
            
            # Add some variation
            sum_focus_los = np.random.normal(mean_los, mean_los * 0.15)
            sum_focus_los = max(5, sum_focus_los)  # Ensure positive values
            
            # D_focus is typically between 5-15 for focus patients
            d_focus = np.random.randint(5, 16)
            
            data_list.append({
                'pttr': pttr_category,
                'OnlyHuman': only_human,
                'sum_focus_los': sum_focus_los,
                'D_focus': d_focus
            })

# Create DataFrame
df = pd.DataFrame(data_list)

# Show statistics
print("=" * 60)
print("Random Test Dataset Statistics")
print("=" * 60)
print(f"\nTotal samples: {len(df)}")
print(f"\nSamples per category:")
print(df.groupby(['pttr', 'OnlyHuman']).size())
print(f"\nLOS statistics by category and model:")
print(df.groupby(['pttr', 'OnlyHuman'])['sum_focus_los'].describe()[['mean', 'std', 'min', 'max']])

# Convert to dictionary format expected by the plot function
data_dict = df.to_dict('list')

print("\n" + "=" * 60)
print("Creating R violin plot (NORMALIZED)...")
print("=" * 60)

# Test with normalized data
#los_split_violin_r_plot(data_dict, normalize_by_focus=True)

print("\n" + "=" * 60)
print("Creating R violin plot (NON-NORMALIZED)...")
print("=" * 60)

# Test without normalization
los_split_violin_r_plot(data_dict, normalize_by_focus=False)

print("\n✓ Test completed successfully!")
print("✓ Check 'r_split_violin_plot.png' for the final plot")
