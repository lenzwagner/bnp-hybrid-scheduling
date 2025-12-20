"""
Numba vs Non-Numba Stability Comparison Script
===============================================

This script compares the results of the Numba-accelerated labeling algorithm
with the pure Python implementation across multiple random seeds.

Configuration: 25 seeds × 2 strategies (sp, mp) × 2 numba modes = 100 runs
"""

import os
import sys
import time
import pandas as pd
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CG import ColumnGeneration
from branch_and_price import BranchAndPrice
from logging_config import setup_multi_level_logging, get_logger

# Setup logging
setup_multi_level_logging(base_log_dir='logs', enable_console=True, print_all_logs=False)
logger = get_logger(__name__)


class NumbaStabilityComparison:
    """
    Compare Numba and Non-Numba labeling across multiple seeds and strategies.

    Tests: 25 seeds × 2 strategies (sp, mp) × 2 numba modes (True, False) = 100 runs
    """

    def __init__(self, num_seeds=25, branching_strategies=None):
        self.num_seeds = num_seeds
        self.branching_strategies = branching_strategies or ['sp', 'mp']
        self.results = []

        # Configuration (from main.py)
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

        self.T = 3
        self.D_focus = 5
        self.max_itr = 100
        self.threshold = 1e-5
        self.pttr = 'medium'
        self.learn_method = 'pwl'

        # Labeling spec template
        self.labeling_spec_base = {
            'use_labeling': True,
            'max_columns_per_iter': 50,
            'use_parallel_pricing': False,
            'n_pricing_workers': 1,
            'debug_mode': False,
            'use_apriori_pruning': False,
            'use_pure_dp_optimization': True,
            'use_persistent_pool': True,
            'use_heuristic_pricing': False,
            'heuristic_max_labels': 20,
            'use_relaxed_history': False
        }

    def _create_cg_solver(self, seed):
        """Create a ColumnGeneration solver instance."""
        cg_solver = ColumnGeneration(
            seed=seed,
            app_data=self.app_data,
            T=self.T,
            D_focus=self.D_focus,
            max_itr=self.max_itr,
            threshold=self.threshold,
            pttr=self.pttr,
            show_plots=False,
            pricing_filtering=True,
            therapist_agg=False,
            max_stagnation_itr=20,
            stagnation_threshold=1e-5,
            learn_method=self.learn_method,
            save_lps=False,
            verbose=False,
            deterministic=True
        )
        cg_solver.setup()
        return cg_solver

    def solve_root_node(self, seed, branching_strategy, use_numba):
        """
        Solve the root node of Branch-and-Price.

        Args:
            seed: Random seed
            branching_strategy: 'sp' or 'mp'
            use_numba: Whether to use Numba labeling

        Returns:
            dict: Results including objective, timing, etc.
        """
        try:
            cg_solver = self._create_cg_solver(seed)

            labeling_spec = self.labeling_spec_base.copy()
            labeling_spec['use_numba_labeling'] = use_numba

            bnp_solver = BranchAndPrice(
                cg_solver,
                branching_strategy=branching_strategy,
                search_strategy='bfs',
                verbose=False,
                ip_heuristic_frequency=0,
                early_incumbent_iteration=100,
                save_lps=False,
                label_dict=labeling_spec
            )

            start_time = time.time()
            results = bnp_solver.solve(time_limit=600, max_nodes=300)
            solve_time = time.time() - start_time

            return {
                'success': True,
                'lp_bound': results.get('lp_bound'),
                'incumbent': results.get('incumbent'),
                'cg_iterations': results.get('cg_iterations', 0),
                'nodes_explored': results.get('nodes_explored', 0),
                'solve_time': solve_time
            }

        except Exception as e:
            logger.error(f"Error: seed={seed}, strategy={branching_strategy}, numba={use_numba}: {e}")
            return {
                'success': False,
                'error': str(e),
                'lp_bound': None,
                'incumbent': None,
                'cg_iterations': 0,
                'nodes_explored': 0,
                'solve_time': 0
            }

    def run_comparison(self, start_seed=1):
        """
        Run comparison: 25 seeds × 2 strategies × 2 numba modes = 100 runs.
        """
        print("\n" + "=" * 100)
        print(" NUMBA vs NON-NUMBA STABILITY COMPARISON ".center(100, "="))
        print("=" * 100)
        print(f"\nConfiguration:")
        print(f"  - Seeds: {self.num_seeds} (starting at {start_seed})")
        print(f"  - Strategies: {self.branching_strategies}")
        print(f"  - Numba modes: [False, True]")
        print(f"  - Total runs: {self.num_seeds * len(self.branching_strategies) * 2}")
        print("\n" + "=" * 100 + "\n")

        total_start_time = time.time()
        run_count = 0
        total_runs = self.num_seeds * len(self.branching_strategies) * 2

        for seed_idx in range(self.num_seeds):
            seed = start_seed + seed_idx

            for strategy in self.branching_strategies:
                for use_numba in [False, True]:
                    run_count += 1
                    numba_str = "Numba" if use_numba else "Python"
                    print(f"  [{run_count}/{total_runs}] Seed {seed}, {strategy.upper()}, {numba_str}...", end=" ",
                          flush=True)

                    result = self.solve_root_node(seed, strategy, use_numba)

                    self.results.append({
                        'seed': seed,
                        'branching_strategy': strategy,
                        'use_numba': use_numba,
                        'success': result['success'],
                        'lp_bound': result.get('lp_bound'),
                        'incumbent': result.get('incumbent'),
                        'cg_iterations': result.get('cg_iterations'),
                        'nodes_explored': result.get('nodes_explored'),
                        'solve_time': result.get('solve_time'),
                        'error': result.get('error', '')
                    })

                    if result['success']:
                        print(f"✓ LP={result['lp_bound']:.4f} Nodes={result['nodes_explored']} ({result['solve_time']:.1f}s)")
                    else:
                        print(f"✗ ERROR: {result.get('error', 'Unknown')[:50]}")

        total_time = time.time() - total_start_time

        # Summary
        self._print_summary(total_time)

        # Save CSV
        self._save_results()

        return pd.DataFrame(self.results)

    def _print_summary(self, total_time):
        """Print comparison summary with match statistics."""
        print("\n" + "=" * 100)
        print(" COMPARISON SUMMARY ".center(100, "="))
        print("=" * 100)

        df = pd.DataFrame(self.results)

        # Group by seed and strategy, compare Python vs Numba
        matches = 0
        mismatches = 0
        errors = 0

        for seed in df['seed'].unique():
            for strategy in df['branching_strategy'].unique():
                subset = df[(df['seed'] == seed) & (df['branching_strategy'] == strategy)]

                python_row = subset[subset['use_numba'] == False]
                numba_row = subset[subset['use_numba'] == True]

                if len(python_row) == 0 or len(numba_row) == 0:
                    errors += 1
                    continue

                py_lp = python_row['lp_bound'].values[0]
                nb_lp = numba_row['lp_bound'].values[0]

                if py_lp is None or nb_lp is None:
                    errors += 1
                elif abs(py_lp - nb_lp) < 1e-4:
                    matches += 1
                else:
                    mismatches += 1
                    print(f"  MISMATCH: Seed {seed}, {strategy}: Python={py_lp:.4f}, Numba={nb_lp:.4f}")

        total_comparisons = matches + mismatches + errors
        print(f"\nTotal runs: {len(df)}")
        print(f"Comparisons (seed+strategy pairs): {total_comparisons}")
        print(f"  ✓ Matches: {matches} ({100 * matches / max(total_comparisons, 1):.1f}%)")
        print(f"  ✗ Mismatches: {mismatches}")
        print(f"  ❌ Errors: {errors}")
        print(f"\nTotal time: {total_time:.1f}s")

        # Timing comparison
        py_times = df[df['use_numba'] == False]['solve_time'].mean()
        nb_times = df[df['use_numba'] == True]['solve_time'].mean()
        speedup = py_times / nb_times if nb_times > 0 else 1.0
        print(f"\nAverage solve times:")
        print(f"  Python: {py_times:.2f}s")
        print(f"  Numba:  {nb_times:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")

        # Max Nodes Reporting
        print("\n" + "=" * 100)
        print(" MAX NODE USAGE ".center(100, "="))
        print("=" * 100)
        
        if not df.empty and 'nodes_explored' in df.columns:
            max_nodes = df['nodes_explored'].max()
            max_row = df[df['nodes_explored'] == max_nodes].iloc[0]
            
            print(f"Run with MOST nodes explored: {max_nodes}")
            print(f"  - Seed: {max_row['seed']}")
            print(f"  - Strategy: {max_row['branching_strategy']}")
            print(f"  - Numba: {max_row['use_numba']}")
            
            # Additional detail: average nodes per strategy
            print("\nAverage nodes per strategy:")
            print(df.groupby('branching_strategy')['nodes_explored'].mean().to_string())

            # Specific: Max nodes for Numba runs
            numba_df = df[df['use_numba'] == True]
            if not numba_df.empty:
                max_numba_nodes = numba_df['nodes_explored'].max()
                max_numba_row = numba_df[numba_df['nodes_explored'] == max_numba_nodes].iloc[0]
                print(f"\nRun with MOST nodes explored (Numba Only): {max_numba_nodes}")
                print(f"  - Seed: {max_numba_row['seed']}")
                print(f"  - Strategy: {max_numba_row['branching_strategy']}")

    def _save_results(self):
        """Save results to CSV."""
        if not self.results:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = 'results/numba_comparison'
        os.makedirs(results_dir, exist_ok=True)
        filename = f'{results_dir}/numba_comparison_{timestamp}.csv'

        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"\nResults saved to: {filename}")
        print("=" * 100 + "\n")


def main():
    """Main function."""
    NUM_SEEDS = 25
    START_SEED = 1
    BRANCHING_STRATEGIES = ['sp', 'mp']

    comparison = NumbaStabilityComparison(
        num_seeds=NUM_SEEDS,
        branching_strategies=BRANCHING_STRATEGIES
    )

    results_df = comparison.run_comparison(start_seed=START_SEED)
    return results_df


if __name__ == "__main__":
    results = main()