"""Check the two generated pullbacks' splat counts with Perfify.

Reads pullback.mlir from each schedule of benchmark_ttir_recurrence.py.
All other operations have explicit cost zero. Unknown operations invalidate
the cost contract. This metric counts logical splats, independent of width.
"""

import argparse
from pathlib import Path
import re
import subprocess


def contract(width, steps, expected):
    ty = f"tensor<{width}xf32>"
    costs = {"tt.splat": 1, "arith.mulf": 0, "arith.addf": 0,
             "arith.constant": 0, "tt.return": 0, "scf.for": 0, "scf.yield": 0}
    definitions = "\n".join(f'''perfify.cost "{op}" {{
      %c = arith.constant {cost} : i64
      perfify.yield %c : i64
    }}''' for op, cost in costs.items())
    return f"""
  tt.func public @probe(%w: {ty}, %x: {ty}, %a: f32, %g: f32) -> {ty} {{
    %t = arith.constant {steps} : i32
    %v = tt.call @pullback(%w, %x, %a, %t, %g) : ({ty}, {ty}, f32, i32, f32) -> {ty}
    tt.return %v : {ty}
  }}
  perfify.assumptions {{
    {definitions}
    perfify.conditions @probe true pre {{
      %start = perfify.arg 0
      perfify.estimated_cost @zero {{
        %v = arith.constant 0 : i64
        perfify.yield %v : i64
      }}
      %cmp = perfify.cmp eq, %start, @zero
      %h = perfify.hypothesis %cmp pre
      perfify.assume %h
    }} post {{
      %cost = perfify.fn_cost : !perfify.cost
      perfify.estimated_cost @count {{
        %v = arith.constant {expected} : i64
        perfify.yield %v : i64
      }}
      %cmp = perfify.cmp eq, %cost, @count
      %h = perfify.hypothesis %cmp post
      perfify.assume %h
    }}
  }}
"""


def check(opt, path, width, steps, expected, holds):
    module = path.read_text().rstrip()
    if not module.endswith("}"):
        raise ValueError(f"expected a module in {path}")
    probe = module[:-1] + contract(width, steps, expected) + "}"
    checked = subprocess.run([opt, "--inline", "--simple-cycle-analysis"],
                             input=probe, capture_output=True, text=True, check=True).stdout
    results = re.findall(r"perfify\.assume[^\n]*satres = (true|false)", checked)
    if results != ["true", str(holds).lower()] or "unknown cost" in checked:
        raise RuntimeError(checked)
    print(f"{path.parent.name}, steps={steps}, splats={expected}, proved={holds}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timing_directory", type=Path)
    parser.add_argument("--opt", default="enzymexlamlir-opt")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()
    if not 0 <= args.steps < 2**31:
        parser.error("steps must be a nonnegative i32")
    # Specializing zero trips lets the inliner fold both results to zero.
    for schedule, count in (("post", args.steps), ("both", int(args.steps > 0))):
        path = args.timing_directory / f"D{args.width}-{schedule}" / "pullback.mlir"
        check(args.opt, path, args.width, args.steps, count, True)
        check(args.opt, path, args.width, args.steps, count + 1, False)


if __name__ == "__main__":
    main()
