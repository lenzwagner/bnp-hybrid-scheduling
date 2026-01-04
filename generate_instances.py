"""
Instance Generation and Export to Excel
=========================================

This script generates instances for multiple seeds and configuration combinations,
then exports all instance data to Excel files.

The user can specify:
- Seeds (default: 1-25)
- pttr variants (e.g., ['low', 'medium', 'high'])
- T variants (number of therapists, e.g., [2, 4, 6])
- D_focus variants (number of focus days, e.g., [4, 8, 12])

All combinations are generated and saved to Excel.
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from itertools import product
from CG import ColumnGeneration
from logging_config import setup_multi_level_logging, get_logger

# Setup logging
setup_multi_level_logging(base_log_dir='logs/instance_generation', enable_console=False, print_all_logs=False)
logger = get_logger(__name__)


class InstanceGenerator:
    """
    Class to generate and export instances to Excel.
    """

    def __init__(self, seeds, pttr_variants, T_variants, D_focus_variants):
        """
        Initialize the instance generator.

        Args:
            seeds: List of seeds to test
            pttr_variants: List of pttr scenarios (e.g., ['low', 'medium', 'high'])
            T_variants: List of therapist counts (e.g., [2, 4, 6])
            D_focus_variants: List of focus day counts (e.g., [4, 8, 12])
        """
        self.seeds = seeds
        self.pttr_variants = pttr_variants
        self.T_variants = T_variants
        self.D_focus_variants = D_focus_variants

        # Default learning parameters (same as main.py)
        self.app_data = {
            'learn_type': ['sigmoid'],
            'theta_base': [0.02],
            'lin_increase': [0.01],
            'k_learn': [0.01],
            'infl_point': [2],
            'MS': [5],
            'MS_min': [2],
            'W_on': [6],
            'W_off': [1],
            'daily': [4]
        }

        self.dual_improvement_iter = 20
        self.dual_stagnation_threshold = 1e-5
        self.max_itr = 100
        self.threshold = 1e-5
        self.show_plots = False
        self.pricing_filtering = True
        self.therapist_agg = False
        self.learn_method = 'pwl'
        self.deterministic = False

        # Storage for all instances
        self.instances = []
        self.instance_details = {}

    def generate_instance(self, seed, pttr, T, D_focus):
        """
        Generate a single instance with given parameters.

        Args:
            seed: Random seed
            pttr: Patient-to-therapist ratio scenario
            T: Number of therapists
            D_focus: Number of focus days

        Returns:
            dict: Instance data
        """
        try:
            # Create CG solver to generate instance
            cg_solver = ColumnGeneration(
                seed=seed,
                app_data=self.app_data,
                T=T,
                D_focus=D_focus,
                max_itr=self.max_itr,
                threshold=self.threshold,
                pttr=pttr,
                show_plots=self.show_plots,
                pricing_filtering=self.pricing_filtering,
                therapist_agg=self.therapist_agg,
                max_stagnation_itr=self.dual_improvement_iter,
                stagnation_threshold=self.dual_stagnation_threshold,
                learn_method=self.learn_method,
                save_lps=False,
                verbose=False,
                deterministic=self.deterministic
            )

            # Setup instance (this generates all data)
            cg_solver.setup()

            # Check if instance is valid (has patients in at least one category)
            if len(cg_solver.P_F) == 0 and len(cg_solver.P_Post) == 0:
                logger.warning(f"Instance has no Focus or Post patients (seed={seed}, pttr={pttr}, T={T}, D={D_focus})")
                print(f"✗ NO PATIENTS (empty instance)")
                return None

            # Check if pre_processing returned valid results
            if not isinstance(cg_solver.pre_x, dict):
                logger.warning(f"Pre-processing failed (seed={seed}, pttr={pttr}, T={T}, D={D_focus})")
                print(f"✗ PRE-PROCESSING FAILED")
                return None

            # Create instance identifier
            instance_id = f"seed{seed}_pttr{pttr}_T{T}_D{D_focus}"

            # Extract all relevant instance data
            instance_data = {
                # ===== Configuration =====
                'instance_id': instance_id,
                'seed': seed,
                'pttr': pttr,
                'T_count': T,
                'D_focus_count': D_focus,

                # ===== Learning Parameters =====
                'learn_type': self.app_data['learn_type'][0],
                'theta_base': self.app_data['theta_base'][0],
                'lin_increase': self.app_data['lin_increase'][0],
                'k_learn': self.app_data['k_learn'][0],
                'infl_point': self.app_data['infl_point'][0],
                'MS': self.app_data['MS'][0],
                'MS_min': self.app_data['MS_min'][0],
                'W_on': self.app_data['W_on'][0],
                'W_off': self.app_data['W_off'][0],
                'daily': self.app_data['daily'][0],

                # ===== Patient Counts =====
                'num_patients_total': len(cg_solver.P),
                'num_patients_full': len(cg_solver.Nr),
                'num_patients_pre': len(cg_solver.P_Pre),
                'num_patients_focus': len(cg_solver.P_F),
                'num_patients_post': len(cg_solver.P_Post),
                'num_patients_join': len(cg_solver.P_Join),

                # ===== Therapist Info =====
                'num_therapists': len(cg_solver.T),
                'num_therapist_groups': len(cg_solver.G_C),

                # ===== Horizon Info =====
                'D_length': len(cg_solver.D),
                'D_Ext_length': len(cg_solver.D_Ext),
                'D_Full_length': len(cg_solver.D_Full),

                # ===== Other =====
                'W_coeff': cg_solver.W_coeff,
                'M_p': cg_solver.M_p,
            }

            # Store detailed data separately (for Excel sheets)
            detail_key = instance_id
            self.instance_details[detail_key] = {
                # Lists
                'P': cg_solver.P,
                'Nr': cg_solver.Nr,
                'P_Pre': cg_solver.P_Pre,
                'P_F': cg_solver.P_F,
                'P_Post': cg_solver.P_Post,
                'P_Join': cg_solver.P_Join,
                'T': cg_solver.T,
                'G_C': cg_solver.G_C,
                'D': cg_solver.D,
                'D_Ext': cg_solver.D_Ext,
                'D_Full': cg_solver.D_Full,

                # Dictionaries (need to be converted for Excel)
                'Req': cg_solver.Req,
                'Req_agg': cg_solver.Req_agg,
                'Entry': cg_solver.Entry,
                'Entry_agg': cg_solver.Entry_agg,
                'Nr_agg': cg_solver.Nr_agg,
                'E_dict': cg_solver.E_dict,
                'Max_t': cg_solver.Max_t,
                'S_Bound': cg_solver.S_Bound,
                'therapist_to_type': cg_solver.therapist_to_type,
                'pre_x': cg_solver.pre_x,
                'pre_los': cg_solver.pre_los,
                'agg_to_patient': cg_solver.agg_to_patient,
            }

            return instance_data

        except ValueError as e:
            if "too many values to unpack" in str(e) or "not enough values to unpack" in str(e):
                logger.error(f"Unpacking error for seed={seed}, pttr={pttr}, T={T}, D={D_focus}: {str(e)}")
                print(f"✗ UNPACK ERROR (likely infeasible)")
                return None
            else:
                logger.error(f"ValueError generating instance (seed={seed}, pttr={pttr}, T={T}, D={D_focus}): {str(e)}")
                print(f"✗ ERROR: {str(e)}")
                return None
        except Exception as e:
            logger.error(f"Error generating instance (seed={seed}, pttr={pttr}, T={T}, D={D_focus}): {str(e)}")
            print(f"✗ ERROR: {str(e)}")
            import traceback
            print(f"   Details: {traceback.format_exc()}")
            return None

    def generate_all_instances(self):
        """
        Generate all instances for all combinations of parameters.
        """
        print("\n" + "=" * 100)
        print(" INSTANCE GENERATION ".center(100, "="))
        print("=" * 100)

        total_combinations = len(self.seeds) * len(self.pttr_variants) * len(self.T_variants) * len(self.D_focus_variants)

        print(f"\nConfiguration:")
        print(f"  - Seeds: {min(self.seeds)} to {max(self.seeds)} ({len(self.seeds)} seeds)")
        print(f"  - PTTR variants: {self.pttr_variants}")
        print(f"  - T variants: {self.T_variants}")
        print(f"  - D_focus variants: {self.D_focus_variants}")
        print(f"  - Total combinations: {total_combinations}")
        print("\n" + "=" * 100 + "\n")

        counter = 0
        successful = 0
        failed = 0

        # Generate all combinations
        for seed, pttr, T, D_focus in product(self.seeds, self.pttr_variants, self.T_variants, self.D_focus_variants):
            counter += 1
            print(f"[{counter}/{total_combinations}] Generating: seed={seed}, pttr={pttr}, T={T}, D_focus={D_focus}...", end=" ")

            instance_data = self.generate_instance(seed, pttr, T, D_focus)

            if instance_data is not None:
                self.instances.append(instance_data)
                successful += 1
                print("✓")
            else:
                failed += 1
                print("✗")

        print("\n" + "=" * 100)
        print(" GENERATION COMPLETE ".center(100, "="))
        print("=" * 100)
        print(f"\nSuccessfully generated: {successful}/{total_combinations}")
        print(f"Failed: {failed}/{total_combinations}")
        print("=" * 100 + "\n")

    def export_to_excel(self, filename=None):
        """
        Export all instances to Excel file (single sheet with all data).

        Args:
            filename: Output filename (default: auto-generated with timestamp)
        """
        if not self.instances:
            print("No instances to export!")
            return

        # Create output directory
        output_dir = 'results/instances'
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'{output_dir}/instances_{timestamp}.xlsx'
        else:
            filename = f'{output_dir}/{filename}'

        print(f"\nExporting to Excel: {filename}")
        print("  - Preparing data...")

        # Create comprehensive data for each instance
        import json

        all_data = []
        for inst_data in self.instances:
            inst_id = inst_data['instance_id']
            details = self.instance_details[inst_id]

            # Combine basic config with detailed data
            row = {
                # ===== Configuration =====
                'instance_id': inst_id,
                'seed': inst_data['seed'],
                'pttr': inst_data['pttr'],
                'T_count': inst_data['T_count'],
                'D_focus_count': inst_data['D_focus_count'],

                # ===== Learning Parameters =====
                'learn_type': inst_data['learn_type'],
                'theta_base': inst_data['theta_base'],
                'lin_increase': inst_data['lin_increase'],
                'k_learn': inst_data['k_learn'],
                'infl_point': inst_data['infl_point'],
                'MS': inst_data['MS'],
                'MS_min': inst_data['MS_min'],
                'W_on': inst_data['W_on'],
                'W_off': inst_data['W_off'],
                'daily': inst_data['daily'],

                # ===== Patient Counts =====
                'num_patients_total': inst_data['num_patients_total'],
                'num_patients_full': inst_data['num_patients_full'],
                'num_patients_pre': inst_data['num_patients_pre'],
                'num_patients_focus': inst_data['num_patients_focus'],
                'num_patients_post': inst_data['num_patients_post'],
                'num_patients_join': inst_data['num_patients_join'],

                # ===== Therapist Info =====
                'num_therapists': inst_data['num_therapists'],
                'num_therapist_groups': inst_data['num_therapist_groups'],

                # ===== Horizon Info =====
                'D_length': inst_data['D_length'],
                'D_Ext_length': inst_data['D_Ext_length'],
                'D_Full_length': inst_data['D_Full_length'],

                # ===== Other =====
                'W_coeff': inst_data['W_coeff'],
                'M_p': inst_data['M_p'],

                # ===== Lists (as JSON strings) =====
                'P': json.dumps(details['P']),
                'Nr': json.dumps(details['Nr']),
                'P_Pre': json.dumps(details['P_Pre']),
                'P_F': json.dumps(details['P_F']),
                'P_Post': json.dumps(details['P_Post']),
                'P_Join': json.dumps(details['P_Join']),
                'T': json.dumps(details['T']),
                'G_C': json.dumps(details['G_C']),
                'D': json.dumps(details['D']),
                'D_Ext': json.dumps(details['D_Ext']),
                'D_Full': json.dumps(details['D_Full']),

                # ===== Dictionaries (as JSON strings) =====
                'Req': json.dumps(details['Req']),
                'Req_agg': json.dumps(details['Req_agg']),
                'Entry': json.dumps(details['Entry']),
                'Entry_agg': json.dumps(details['Entry_agg']),
                'Nr_agg': json.dumps(details['Nr_agg']),
                'E_dict': json.dumps(details['E_dict']),
                'S_Bound': json.dumps(details['S_Bound']),
                'therapist_to_type': json.dumps(details['therapist_to_type']),
                'pre_los': json.dumps(details['pre_los']),
                'agg_to_patient': json.dumps(details['agg_to_patient']),

                # ===== Complex Dicts (convert tuples to strings for JSON) =====
                'Max_t': json.dumps({str(k): v for k, v in details['Max_t'].items()}),
                'pre_x': json.dumps({str(k): v for k, v in details['pre_x'].items()}),
            }

            all_data.append(row)

        # Create DataFrame
        df = pd.DataFrame(all_data)

        # Export to Excel (single sheet)
        print("  - Writing to Excel...")
        df.to_excel(filename, sheet_name='Instances', index=False, engine='openpyxl')

        print(f"\n✓ Excel file saved: {filename}")
        print(f"  - Total instances: {len(df)}")
        print(f"  - Total columns: {len(df.columns)}")

        # Also save as pickle for full data preservation
        pickle_file = filename.replace('.xlsx', '.pkl')
        print(f"\nSaving full instance data to pickle: {pickle_file}")
        with open(pickle_file, 'wb') as f:
            pickle.dump({
                'instances': self.instances,
                'instance_details': self.instance_details
            }, f)
        print(f"✓ Pickle file saved: {pickle_file}")

        return filename


def main():
    """
    Main function with interactive input.
    """
    print("\n" + "=" * 100)
    print(" INSTANCE GENERATION TOOL ".center(100, "="))
    print("=" * 100)
    print("\nThis tool generates instances for multiple seeds and configuration variants.")
    print("All instance data will be saved to Excel files.")
    print("\n" + "=" * 100)

    # ===== Get Seeds =====
    print("\n--- SEEDS ---")
    use_default_seeds = input("Use default seeds 1-25? (y/n): ").strip().lower()
    if use_default_seeds == 'y':
        seeds = list(range(1, 26))
    else:
        seed_input = input("Enter seeds (comma-separated, e.g., '1,2,3' or range '1-10'): ").strip()
        if '-' in seed_input and ',' not in seed_input:
            # Range input
            start, end = seed_input.split('-')
            seeds = list(range(int(start), int(end) + 1))
        else:
            # Comma-separated
            seeds = [int(s.strip()) for s in seed_input.split(',')]

    print(f"Seeds selected: {seeds}")

    # ===== Get PTTR Variants =====
    print("\n--- PTTR VARIANTS ---")
    print("Available: low, medium, high")
    pttr_input = input("Enter pttr variants (comma-separated, default='medium'): ").strip()
    if not pttr_input:
        pttr_variants = ['medium']
    else:
        pttr_variants = [p.strip() for p in pttr_input.split(',')]

    print(f"PTTR variants selected: {pttr_variants}")

    # ===== Get T Variants =====
    print("\n--- NUMBER OF THERAPISTS (T) ---")
    T_input = input("Enter T variants (comma-separated, e.g., '2,4,6', default='6'): ").strip()
    if not T_input:
        T_variants = [6]
    else:
        T_variants = [int(t.strip()) for t in T_input.split(',')]

    print(f"T variants selected: {T_variants}")

    # ===== Get D_focus Variants =====
    print("\n--- FOCUS DAYS (D_focus) ---")
    D_input = input("Enter D_focus variants (comma-separated, e.g., '4,8,12', default='30'): ").strip()
    if not D_input:
        D_focus_variants = [30]
    else:
        D_focus_variants = [int(d.strip()) for d in D_input.split(',')]

    print(f"D_focus variants selected: {D_focus_variants}")

    # ===== Confirm =====
    total = len(seeds) * len(pttr_variants) * len(T_variants) * len(D_focus_variants)
    print("\n" + "=" * 100)
    print(" CONFIGURATION SUMMARY ".center(100, "="))
    print("=" * 100)
    print(f"\nSeeds: {len(seeds)} ({min(seeds)} to {max(seeds)})")
    print(f"PTTR variants: {pttr_variants}")
    print(f"T variants: {T_variants}")
    print(f"D_focus variants: {D_focus_variants}")
    print(f"\nTotal instances to generate: {total}")
    print("=" * 100)

    confirm = input("\nProceed with generation? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return

    # ===== Generate Instances =====
    generator = InstanceGenerator(seeds, pttr_variants, T_variants, D_focus_variants)
    generator.generate_all_instances()

    # ===== Export to Excel =====
    generator.export_to_excel()

    print("\n" + "=" * 100)
    print(" DONE ".center(100, "="))
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
