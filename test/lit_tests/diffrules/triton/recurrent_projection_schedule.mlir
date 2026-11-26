// REQUIRES: triton
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=cell outfn=pullback argTys=enzyme_active,enzyme_const,enzyme_const,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --cse --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s --check-prefix=POST
// RUN: enzymexlamlir-opt %s --transform-interpreter --enzyme-wrap="infn=cell outfn=pullback argTys=enzyme_active,enzyme_const,enzyme_const,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --cse --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s --check-prefix=BOTH

// h[t+1] = decay * h[t] + dot(w, x). Only w is active.
// Applying the same loop-invariant code motion before reverse AD accumulates
// the adjoint of the shared projection as a scalar over time.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loops = transform.structured.match ops{["scf.for"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %loops : !transform.any_op
    transform.yield
  }

  tt.func private @cell(%w: tensor<128xf32>, %x: tensor<128xf32>, %decay: f32, %n: i32) -> f32 {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant 0.0 : f32
    %state = scf.for %i = %c0 to %n step %c1 iter_args(%hidden = %zero) -> f32 : i32 {
      %wx = arith.mulf %w, %x : tensor<128xf32>
      %projection = "tt.reduce"(%wx) <{axis = 0 : i32}> ({
      ^bb0(%a: f32, %b: f32):
        %sum = arith.addf %a, %b : f32
        tt.reduce.return %sum : f32
      }) : (tensor<128xf32>) -> f32
      %carry = arith.mulf %hidden, %decay : f32
      %sum = arith.addf %carry, %projection : f32
      scf.yield %sum : f32
    }
    tt.return %state : f32
  }
}

// POST-LABEL: tt.func private @pullback(
// POST: scf.for {{.*}} -> (f32, tensor<128xf32>)
// POST: tt.splat
// POST: arith.mulf {{.*}} : tensor<128xf32>
// POST: arith.addf {{.*}} : tensor<128xf32>
// POST: scf.yield {{.*}} : f32, tensor<128xf32>
// POST: tt.return

// BOTH-LABEL: tt.func private @pullback(
// BOTH: scf.for {{.*}} -> (f32, f32)
// BOTH-NOT: tensor<
// BOTH: scf.yield {{.*}} : f32, f32
// BOTH: tt.splat
// BOTH: arith.mulf {{.*}} : tensor<128xf32>
// BOTH: tt.return
