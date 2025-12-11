# Implementation Plan: SP Branching (Pattern Branching) in Labeling Algorithm

This plan outlines the necessary changes to `label.py` to support **Resource Constrained Branching** (SP Branching) on patterns, as described in the provided methodology.

## 1. Concept Overview

We need to enforce two types of constraints on a pattern $P(k)$ of worker-time pairs $\{(j, t)\}$:
1.  **Left Branch (Limit Usage)**: $\sum_{(j,t) \in P(k)} x_{kjt} \le |P(k)| - 1$
    *   *Logic*: The column cannot contain *all* elements of the pattern.
    *   *Implementation*: Track a counter. If it reaches $|P(k)|$, prune.
2.  **Right Branch (All-or-Nothing)**: $\sum_{(j,t) \in P(k)} x_{kjt} = |P(k)| \cdot w$
    *   *Logic*: The column must contain **either ALL** elements of the pattern **OR NONE**. Partial usage is forbidden.
    *   *Implementation*: Track a state (Mode: `Exclude` vs `Cover`). Enforce consistent choices.

## 2. Data Structures

### 2.1 Constraint Parsing
We need to parse `SPPatternBranching` objects from `branch_constraints` into efficient lookup structures:
*   `left_patterns`: List of Left-Branch patterns. Each entry:
    *   `id`: constraint index
    *   `elements`: Set of $(j, t)$
    *   `limit`: $|P(k)| - 1` (actually simpler: fail if count == $|P(k)|$)
*   `right_patterns`: List of Right-Branch patterns. Each entry:
    *   `id`: constraint index
    *   `elements`: Sorted list of $(j, t)$ tuples (chronological).
    *   `first_element`: $(j_{first}, t_{first})$
    *   `dual`: $\delta^R$ (Bonus for covering)

### 2.2 State Augmentation
The state tuple $\sigma_t = (\text{cost}, \text{prog}, \text{ai\_count}, \text{hist})$ must be extended:
$$ \sigma_t' = (\sigma_t, \boldsymbol{\rho}_t, \boldsymbol{\mu}_t) $$
*   $\boldsymbol{\rho}_t$ (Left Counters): Vector of integers. $\rho_{lt}$ counts how many elements of Left-Pattern $l$ have been picked so far.
*   $\boldsymbol{\mu}_t$ (Right Modes): Vector of states for Right-Patters. $\mu_{lt} \in \{0: \text{Exclude}, 1: \text{Cover}, 2: \text{Invalid}\}$.
    *   Actually, per paper: `Exclude` (0) and `Cover` (1).

## 3. Transition Logic (in `solve_pricing_for_recipient`)

When transitioning from $t$ to $t+1$ with a specific worker $j$ (or AI $j_{AI}$):

### 3.1 Left Constraints (Counters)
For each Left-Pattern $l$:
*   If current action $(j, t+1)$ is in Pattern $l$:
    *   $\rho'_{l} = \rho_{l} + 1$
*   **Pruning Condition**: If $\rho'_{l} == |P_l|$, the path assumes the forbidden full pattern $\to$ **PRUNE IMMEDIATE**.

### 3.2 Right Constraints (Strict Modes)
For each Right-Pattern $l$:
*   Let $(j_{first}, t_{first})$ be the first chronological element of Pattern $l$.
*   **Case A: $t+1 < t_{first}$**
    *   No change, state remains initialized (usually `Exclude` effectively, but waiting for start).
*   **Case B: $t+1 == t_{first}$**
    *   If action matches $(j, t+1) == (j_{first}, t_{first})$:
        *   Transition to `Cover` ($\mu_l = 1$).
    *   Else:
        *   Transition to/Stay in `Exclude` ($\mu_l = 0$).
*   **Case C: $t+1 > t_{first}$ (Pattern active)**
    *   Is current action $(j, t+1)$ part of Pattern $l$?
    *   **If YES**:
        *   If Mode is `Exclude` $\to$ **PRUNE** (Forbidden to pick element if we skipped start).
        *   If Mode is `Cover` $\to$ OK (Continue covering).
    *   **If NO**:
        *   If Mode is `Cover` AND current time $t+1$ corresponds to an element required by pattern (i.e., we missed a required step) $\to$ **PRUNE**.
        *   *Refinement*: We only prune in `Cover` mode if we *missed* a specific $(t)$ requirement. If the pattern doesn't have an element at $t+1$, we interpret "Cover" as "must pick all elements eventually". Since elements are chronological, if $t+1$ is a time where $P_l$ has an entry $(j^*, t+1)$, and we picked $j \ne j^*$, we broke the chain.

## 4. Dominance Rules

We must adapt `add_state_to_buckets` to respect the new dimensions.

$$ \sigma^1 \succ \sigma^2 \iff V^1 \le V^2 \land \dots \land \boldsymbol{\rho}^1 \le \boldsymbol{\rho}^2 \land \boldsymbol{\mu}^1 \ge \boldsymbol{\mu}^2 $$

*   **Left (Counters)**: Smaller is better. Having used *fewer* elements of a restricted set leaves more flexibility.
*   **Right (Modes)**: `Cover` (1) dominates `Exclude` (0)?
    *   Actually, `Cover` implies we collect the dual $\delta^R$ (negative cost / gain). `Exclude` implies we don't.
    *   So usually `Cover` states will have lower Reduced Cost due to the dual.
    *   However, they act on different sub-graphs. It's safer to treat different Modes as **incomparable**.
    *   *Implementation*: Include $\boldsymbol{\mu}$ in the **Bucket Key**. States with different Right-Pattern decisions should not dominate each other.

## 5. Summary of Changes in `label.py`

1.  **Helper `parse_branching_constraints`**:
    *   Sort `branch_constraints` into `mp_cuts`, `sp_left_patterns`, `sp_right_patterns`.
2.  **Extended State**:
    *   Initialize $\rho = \vec{0}, \mu = \vec{0}$.
3.  **Inner DP Loop**:
    *   Update $\rho$ and $\mu$ for each transition.
    *   Apply Pruning Logic (Left limit reach, Right consistency violation).
4.  **Bucket/Dominance**:
    *   Add $\rho$ (counters) to dominance check.
    *   Add $\mu$ (modes) to bucket key.
5.  **Reduced Cost**:
    *   Subtract duals $\delta^R$ for paths finishing in `Cover` mode.

