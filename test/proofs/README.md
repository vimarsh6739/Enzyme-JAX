# The recurrent projection pullback

The `vim/ttir-ad-schedules` branch emits the two TTIR functions used here.
The `post` loop adds `s * x` to each feature lane at every step. The `both`
loop sums the seeds and scales each feature once after the loop.

Check the lane identity, the arithmetic counts, and the IEEE rounding witness
from this directory with Lean 4.12 and Z3:

```sh
lean RecurrentProjection.lean
z3 projection_rounding.smt2  # expected: unsat
```

The Lean theorem quantifies over every trip count, every seed sequence and
every lane in an algebra with right distributivity and a left zero. Real
arithmetic satisfies these assumptions. For two steps, seed 1, decay 0.9 and
feature 1.2, the SMT file checks a one-ULP difference in rounded binary32.

To count logical splats in the actual generated pullbacks, point this checker
at the results from `scripts/benchmark_ttir_recurrence.py` on the schedules
branch. It inlines a caller with a constant trip count and submits both the
claimed cost and a one-off negative control to Perfify.

```sh
python3 check_projection_cost.py /path/to/results --steps 32 --opt /path/to/enzymexlamlir-opt
```

At width 128 and 32 steps the two counts are 32 and 1. At one step both are
1. With zero known steps the inliner folds both results to zero. The Lean lane
arithmetic model counts the loops as emitted, before this specialization.
