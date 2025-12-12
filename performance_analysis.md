# Performance Analysis: Why Non-Zero Duals Are Faster

## Executive Summary

The labeling algorithm runs **9.7x faster** with non-zero dual values compared to all-zero duals, despite generating similar numbers of states. The key difference is the **effectiveness of pruning strategies**.

---

## Performance Comparison

### Test Case 1: All-Zero Duals (SLOW)
```python
pi = {(1,1): 0.0, (1,2): 0.0, ..., (3,42): 0.0}  # ALL ZEROS
```

**Results:**
- **Runtime:** 3.16 seconds
- **States Generated:** 2,923,764
- **States Surviving:** 1,857,312 (63.52%)
- **Lower Bound Pruning:** 192 states
- **Dominance Pruning:** 1,066,452 states
- **Recipients:** 38 with negative reduced cost

---

### Test Case 2: Non-Zero Duals (FAST)
```python
pi = {..., (1,6): -2.0, (1,16): -27.0, (2,19): -16.0, ...}  # VARIED VALUES
```

**Results:**
- **Runtime:** 0.32 seconds ⚡ **9.7x FASTER**
- **States Generated:** 299,077 (90% reduction)
- **States Surviving:** 197,748 (66.12%)
- **Lower Bound Pruning:** 7,608 states (39x more effective)
- **Dominance Pruning:** 101,329 states
- **Recipients:** 6 with negative reduced cost

---

## Root Cause Analysis

### 1. Lower Bound Pruning Effectiveness

**With pi=0 (Ineffective):**
- All paths with same length have **identical costs**
- Lower bound: `LB = 0 + time_cost - gamma`
- No differentiation → minimal pruning (192 states)

**With non-zero pi (Highly Effective):**
- Different (worker, time) combinations → **different costs**
- Paths accumulating negative pi values get **lower reduced costs**
- Clear cost signals → aggressive pruning (7,608 states, **39x more**)

### 2. State Space Explosion

**With pi=0:**
- No cost incentive to prefer any path
- Algorithm explores **all feasible combinations**
- 2.9M states generated (exponential growth)

**With non-zero pi:**
- Clear cost signals guide search
- Unpromising paths pruned early
- Only 299K states generated (**90% reduction**)

### 3. Dominance Relationships

**With pi=0:**
- States only differ by **progress** (not cost)
- Many states have identical costs
- Weaker dominance → more non-dominated states survive

**With non-zero pi:**
- States differ in **both cost AND progress**
- Clear dominance: (lower cost, higher progress)
- Stronger pruning in bucket structure

### 4. Worker Dominance

**With pi=0:**
```python
# All workers equally good → No elimination
candidate_workers = [1, 2, 3]  # All 3 considered
```

**With non-zero pi:**
```python
# Workers with better pi values dominate others
# Example: If worker 1 has π_{1,t} >= π_{2,t} for all t
#          → Worker 2 eliminated
candidate_workers = [1]  # Fewer workers explored
```

---

## Key Insight: "Harder" Problems Are Easier

This is a classic paradox in optimization:

- **Unconstrained/uniform instances** (pi=0) → No structure → Hard to prune → Slow
- **Structured instances** (varied pi) → Clear signals → Aggressive pruning → Fast

The dual values provide **economic guidance** that steers the search toward promising regions, dramatically reducing the effective search space.

---

## Recommendations

1. **Always enable bound pruning** when dual values are available
2. **Monitor survival rates**: <70% suggests pruning is working well
3. **Track LB vs Dominance pruning**: Large LB counts indicate good dual structure
4. **In Column Generation**: Later iterations (with better duals) will be faster than early iterations

---

## Mathematical Explanation

### Lower Bound Formula
```python
LB(state) = current_cost + time_cost - gamma
          = -sum(pi_used) + (end-start+1)*obj_mode - gamma
```

### Why Non-Zero Pi Enables Pruning

With pi ≤ 0 (costs), accumulated cost = `-sum(pi)`:

- **pi = 0 everywhere:** All paths → `cost = 0` (no differentiation)
- **pi varied:** Different paths → different costs (clear winner emerges early)

When `LB(state) ≥ 0`, we **prune** because adding more positive costs can't create negative reduced cost.

---

## Conclusion

**The 9.7x speedup is achieved through:**
1. 90% reduction in state generation (299K vs 2.9M)
2. 39x more effective lower bound pruning (7.6K vs 192)
3. Better worker dominance pre-elimination
4. Stronger cost-based dominance in buckets

**Bottom line:** Non-zero dual values provide the algorithm with "directions" to search, eliminating unproductive paths before they explode the state space.
