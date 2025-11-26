import sys
from abc import ABC, abstractmethod
import gurobipy as gu


class BranchingConstraint(ABC):
    """
    Base class for all branching constraints in Branch-and-Price.

    Each branching constraint must define how it affects:
    1. The master problem (restricting which columns can be used)
    2. The subproblems (restricting which new columns can be generated)
    3. Column compatibility (checking if existing columns satisfy the constraint)
    """

    @abstractmethod
    def apply_to_master(self, master, node):
        """
        Apply this constraint to the master problem.

        Args:
            master: MasterProblem_d instance
            node: BnPNode instance (for accessing column_pool)
        """
        pass

    @abstractmethod
    def apply_to_subproblem(self, subproblem):
        """
        Apply this constraint to a subproblem.

        Args:
            subproblem: Subproblem instance
        """
        pass

    @abstractmethod
    def is_column_compatible(self, column_data):
        """
        Check if a column is compatible with this constraint.

        Args:
            column_data: Dictionary containing column information

        Returns:
            bool: True if column satisfies the constraint
        """
        pass


class SPVariableBranching(BranchingConstraint):
    """
    Branching on Subproblem Variable x_{njt}

    Master Problem Impact (Equation branch:sub2):
    - Left:  sum_{a: chi^a_{njt}=1} Lambda_{na} <= floor(beta)
    - Right: sum_{a: chi^a_{njt}=1} Lambda_{na} >= ceil(beta)

    Subproblem Impact (Equation branch:sub3):
    - Left:  x_{njt} = 0 (fix variable to 0)
    - Right: x_{njt} = 1 (fix variable to 1)

    Attributes:
        profile: Profile index n
        agent: Agent index j (or therapist type if aggregated)
        period: Period index t
        level: Level in search tree (for dual variable tracking)
        dual_left: Dual variable for left constraint (delta^L_{nl})
        dual_right: Dual variable for right constraint (delta^R_{nl})
        master_constraint: Reference to restored cosntraint
    """

    def __init__(self, profile_n, agent_j, period_t, dir, level,
                 floor_val, ceil_val):
        self.profile = profile_n
        self.agent = agent_j
        self.period = period_t
        self.level = level
        self.dir = dir
        self.floor = floor_val
        self.ceil = ceil_val
        self.dual_left = 0.0
        self.dual_right = 0.0
        self.master_constraint = None

    def apply_to_master(self, master, node):
        """Add constraint to master and set coefficients for initial columns."""
        n, j, t = self.profile, self.agent, self.period

        # Find relevant columns
        relevant_columns = []
        for (p, a), col_data in node.column_pool.items():
            if p != n:
                continue

            schedules_x = col_data.get('schedules_x', {})

            # Check if any key (n, j, t, *) exists with value > 0.5
            # Keys are now (n, j, t, col_id) where col_id can be any value
            found = False
            for key, value in schedules_x.items():
                if len(key) == 4 and key[0] == n and key[1] == j and key[2] == t and value > 0.5:
                    found = True
                    break

            if found:
                relevant_columns.append(a)

        if not relevant_columns:
            return

        # Create constraint
        lhs = gu.quicksum(master.lmbda[n, a] for a in relevant_columns if (n, a) in master.lmbda)

        if self.dir == 'left':
            self.master_constraint = master.Model.addConstr(
                lhs <= self.floor,
                name=f"sp_branch_L{self.level}_{n}_{j}_{t}"
            )
        else:
            self.master_constraint = master.Model.addConstr(
                lhs >= self.ceil,
                name=f"sp_branch_R{self.level}_{n}_{j}_{t}"
            )

        master.Model.update()

        # Set coefficients for EXISTING initial column (col_id=1)
        if (n, 1) in master.lmbda:
            if 1 in relevant_columns:
                master.Model.chgCoeff(self.master_constraint, master.lmbda[n, 1], 1)
                #print(f"    [SP Branch] Set coefficient for Lambda[{n},1] in constraint")


    def apply_to_subproblem(self, subproblem):
        """
        Fix variable in subproblem.

        Paper Equation (branch:sub3):
        - Left:  x_{njt} = 0
        - Right: x_{njt} = 1
        """
        if subproblem.P != self.profile:
            return

        # Key format in subproblem: x[p, j, t, iteration]
        var_key = (self.profile, self.agent, self.period, subproblem.itr)

        if var_key in subproblem.x:
            if self.dir == 'left':
                subproblem.x[var_key].LB = 0
                subproblem.x[var_key].UB = 0
            else:
                subproblem.x[var_key].LB = 1
                subproblem.x[var_key].UB = 1
            subproblem.Model.update()

    def is_column_compatible(self, column_data):
        """
        SP variable branching: All columns are compatible!

        The master constraint regulates usage.
        Column filtering is not necessary and can lead to bugs.

        Returns:
            bool: Always True
        """

        return True

    def __repr__(self):
        return (f"SPBranch(profile={self.profile}, agent={self.agent}, "
                f"period={self.period}, level={self.level})")


class SPPatternBranching(BranchingConstraint):
    """
    Branching on Pattern P(k) ⊆ J × T_k (set of resource pairs)
    
    This implements the hierarchical pattern-based branching from the paper.
    Instead of branching on a single x_{kjt}, we branch on patterns of 
    multiple (j,t) pairs simultaneously.
    
    Master Problem Impact:
    - Left:  sum_{a ∈ A(k,P(k))} Lambda_{ka} <= floor(beta_P(k))
    - Right: sum_{a ∈ A(k,P(k))} Lambda_{ka} >= ceil(beta_P(k))
    
    where A(k,P(k)) = {a : chi^a_{kjt}=1, ∀(j,t) ∈ P(k)}
    
    Subproblem Impact (Equation branch:sp_rest):
    - Left:  sum_{(j,t) ∈ P(k)} x_{kjt} <= |P(k)| - 1
    - Right: sum_{(j,t) ∈ P(k)} x_{kjt} = |P(k)| * w^Full_{kl}
    
    Attributes:
        profile: Profile/recipient type index k
        pattern: Set (or frozenset) of (j,t) tuples representing P(k)
        level: Level in search tree (for dual variable tracking)
        direction: 'left' or 'right'
        floor_val: floor(beta_P(k))
        ceil_val: ceil(beta_P(k))
        dual_var: Dual variable delta^L_{kl} or delta^R_{kl}
        master_constraint: Reference to master constraint
        w_full_var_name: Name for binary helper variable (right branch only)
    """
    
    def __init__(self, profile_k, pattern, direction, level, floor_val, ceil_val):
        self.profile = profile_k
        # Ensure pattern is a frozenset for hashability
        self.pattern = frozenset(pattern) if not isinstance(pattern, frozenset) else pattern
        self.direction = direction
        self.level = level
        self.floor = floor_val
        self.ceil = ceil_val
        self.dual_var = 0.0
        self.master_constraint = None
        self.w_full_var_name = f"w_full_k{profile_k}_l{level}"
    
    def apply_to_master(self, master, node):
        """
        Add constraint to master problem.
        
        Find all columns that cover the ENTIRE pattern P(k), i.e.,
        A(k,P(k)) = {a : chi^a_{kjt}=1 for all (j,t) in P(k)}
        """
        k = self.profile
        
        # Find columns that fully cover this pattern
        relevant_columns = []
        
        for (p, a), col_data in node.column_pool.items():
            if p != k:
                continue
            
            schedules_x = col_data.get('schedules_x', {})
            
            # Check if ALL pattern elements are present in this column
            pattern_fully_covered = True
            for (j_pat, t_pat) in self.pattern:
                # Look for key (k, j_pat, t_pat, *) with value > 0.5
                found = False
                for key, value in schedules_x.items():
                    if (len(key) == 4 and 
                        key[0] == k and 
                        key[1] == j_pat and 
                        key[2] == t_pat and 
                        value > 0.5):
                        found = True
                        break
                
                if not found:
                    pattern_fully_covered = False
                    break
            
            if pattern_fully_covered:
                relevant_columns.append(a)
        
        if not relevant_columns:
            # No columns cover this pattern yet - constraint is trivially satisfied
            return
        
        # Create constraint: sum_{a in A(k,P(k))} Lambda_{ka} <= floor or >= ceil
        lhs = gu.quicksum(master.lmbda[k, a] for a in relevant_columns if (k, a) in master.lmbda)
        
        if self.direction == 'left':
            self.master_constraint = master.Model.addConstr(
                lhs <= self.floor,
                name=f"sp_pattern_L{self.level}_k{k}_size{len(self.pattern)}"
            )
        else:  # right
            self.master_constraint = master.Model.addConstr(
                lhs >= self.ceil,
                name=f"sp_pattern_R{self.level}_k{k}_size{len(self.pattern)}"
            )
        
        master.Model.update()
        
        # Set coefficients for existing initial columns if they match
        if (k, 1) in master.lmbda and 1 in relevant_columns:
            master.Model.chgCoeff(self.master_constraint, master.lmbda[k, 1], 1)
    
    def apply_to_subproblem(self, subproblem):
        """
        Add pattern constraint to subproblem.
        
        Left:  sum_{(j,t) in P(k)} x_{kjt} <= |P(k)| - 1
        Right: sum_{(j,t) in P(k)} x_{kjt} = |P(k)| * w^Full
        """
        if subproblem.P != self.profile:
            return
        
        k = self.profile
        itr = subproblem.itr
        
        # Build sum of x variables over the pattern
        pattern_vars = []
        for (j_pat, t_pat) in self.pattern:
            var_key = (k, j_pat, t_pat, itr)
            if var_key in subproblem.x:
                pattern_vars.append(subproblem.x[var_key])
        
        if not pattern_vars:
            # Pattern variables don't exist in this subproblem (shouldn't happen)
            return
        
        pattern_size = len(self.pattern)
        lhs = gu.quicksum(pattern_vars)
        
        if self.direction == 'left':
            # Left: Prevent full pattern coverage
            # sum x_{kjt} <= |P(k)| - 1
            subproblem.Model.addConstr(
                lhs <= pattern_size - 1,
                name=f"pattern_left_l{self.level}_size{pattern_size}"
            )
        else:  # right
            # Right: All-or-nothing coverage
            # sum x_{kjt} = |P(k)| * w^Full
            # Add binary helper variable
            w_full = subproblem.Model.addVar(
                vtype=gu.GRB.BINARY,
                name=self.w_full_var_name,
                obj=0.0  # Dual will be added separately if needed
            )
            
            subproblem.Model.addConstr(
                lhs == pattern_size * w_full,
                name=f"pattern_right_l{self.level}_size{pattern_size}"
            )
            
            # Store reference for potential dual integration
            if not hasattr(subproblem, 'pattern_w_full_vars'):
                subproblem.pattern_w_full_vars = {}
            subproblem.pattern_w_full_vars[self.level] = w_full
        
        subproblem.Model.update()
    
    def is_column_compatible(self, column_data):
        """
        All columns are compatible - master constraint regulates usage.
        
        Returns:
            bool: Always True
        """
        return True
    
    def __repr__(self):
        pattern_str = "{" + ", ".join(f"({j},{t})" for j, t in sorted(self.pattern)) + "}"
        return (f"SPPatternBranch(profile={self.profile}, "
                f"pattern={pattern_str}, "
                f"size={len(self.pattern)}, "
                f"dir={self.direction}, level={self.level})")


class MPVariableBranching(BranchingConstraint):
    """
    Branching on Master Problem Variable Lambda_{na}

    Master Problem Impact:
    - Simply set variable bounds on Lambda_{na}

    Subproblem Impact:
    - Left branch: Add no-good cut to prevent regenerating column a
    - Right branch: No modification needed (column becomes more attractive)

    Attributes:
        profile: Profile index n
        column: Column index a
        bound: floor(Lambda) or ceil(Lambda)
        direction: 'left' or 'right'
        original_schedule: Schedule of the forbidden column (for no-good cut)
    """

    def __init__(self, profile_n, column_a, bound, direction, original_schedule=None):
        self.profile = profile_n
        self.column = column_a
        self.bound = bound
        self.direction = direction
        self.original_schedule = original_schedule

    def apply_to_master(self, master, node):
        """
        Set variable bounds on Lambda_{na}.
        """
        var = master.lmbda.get((self.profile, self.column))

        if var is None:
            print(f"    ❌ ERROR: Variable Lambda[{self.profile},{self.column}] not found in master!")
            return

        # Set bounds
        if self.direction == 'left':
            master.set_branching_bound(var, 'ub', self.bound)
        else:
            master.set_branching_bound(var, 'lb', self.bound)

        master.Model.update()

    def apply_to_subproblem(self, subproblem):
        """
        Left branch: Add no-good cut (Equation no_good_cut_disagg)
        Right branch: No modification needed

        No-good cut prevents regenerating the exact same column:
        sum_{(j,t): chi^a_{njt}=1} (1-x_{njt}) + sum_{(j,t): chi^a_{njt}=0} x_{njt} >= 1
        """
        if subproblem.P != self.profile:
            return

        if self.direction == 'right':
            return

        # Left branch: Add no-good cut
        if self.original_schedule is None or len(self.original_schedule) == 0:
            print(f"      ⚠️ WARNING: No original_schedule for no-good cut!")
            print(f"         Column ({self.profile}, {self.column}) can be regenerated!")
            return

        print(f"      [No-Good Cut] Adding for profile {self.profile}, column {self.column}")
        print(f"                    Schedule has {len(self.original_schedule)} assignments")

        # Build no-good constraint
        terms = []
        schedule_pattern = []

        for (p, j, t, a_orig), chi_value in self.original_schedule.items():
            if p != self.profile:
                continue

            var_key = (p, j, t, subproblem.itr)
            if var_key not in subproblem.x:
                continue

            x_var = subproblem.x[var_key]

            if chi_value == 1:
                terms.append(1 - x_var)
                schedule_pattern.append(f"x[{j},{t}]=1")
            else:
                terms.append(x_var)
                schedule_pattern.append(f"x[{j},{t}]=0")

        if terms:
            subproblem.Model.addConstr(
                gu.quicksum(terms) >= 1,
                name=f"no_good_p{self.profile}_c{self.column}"
            )
            print(f"                    Pattern: {', '.join(schedule_pattern[:5])}...")
            print(f"                    Added constraint with {len(terms)} terms")
        else:
            print(f"      ⚠️ WARNING: No terms in no-good cut (empty schedule?)")

    def is_column_compatible(self, column_data):
        """
        Check if column is compatible with this constraint.

        For MP Variable Branching:
        - ALL columns remain in the model!
        - We only set variable bounds, we don't filter columns
        - The no-good cut in the subproblem prevents regeneration

        Args:
            column_data: Dict with column information

        Returns:
            bool: Always True for MP branching (columns are not filtered)
        """

        return True

    def __repr__(self):
        return (f"MPBranch(profile={self.profile}, column={self.column}, "
                f"{self.direction}, bound={self.bound})")