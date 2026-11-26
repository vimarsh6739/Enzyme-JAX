# REQUIRES: triton
# RUN: python %s

from enzyme_ad.jax import enzyme_call

ttir = """
module {
  tt.func public @fill(%out: !tt.ptr<i32>) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %two = arith.constant 2 : i32
    scf.for %i = %zero to %two step %one : i32 {
      %ptr = tt.addptr %out, %i : !tt.ptr<i32>, i32
      tt.store %ptr, %i : !tt.ptr<i32>
    } {tt.loop_unroll_factor = 2 : i32}
    tt.return
  }
}
"""
_, result = enzyme_call.run_pass_pipeline(
    [], ttir, "triton-loop-unroll,canonicalize"
)
assert "scf.for" not in result, result
assert result.count("tt.store") == 2, result
