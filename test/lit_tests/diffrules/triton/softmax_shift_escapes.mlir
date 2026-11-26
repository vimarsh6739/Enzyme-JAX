// REQUIRES: triton
// RUN: enzymexlamlir-opt %s --detach-triton-softmax-shift | FileCheck %s

// The second returned value depends on the shift. Its tangent must survive.
// CHECK-LABEL: tt.func @exposed_numerator
// CHECK-NOT: enzyme.ignore_derivatives
// CHECK: tt.return
tt.func @exposed_numerator(%x: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
  %max = "tt.reduce"(%x) <{axis = 0 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %v = arith.maxnumf %a, %b : f32
    tt.reduce.return %v : f32
  }) : (tensor<8xf32>) -> f32
  %shift = tt.splat %max : f32 -> tensor<8xf32>
  %center = arith.subf %x, %shift : tensor<8xf32>
  %e = math.exp %center : tensor<8xf32>
  %sum = "tt.reduce"(%e) <{axis = 0 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %v = arith.addf %a, %b : f32
    tt.reduce.return %v : f32
  }) : (tensor<8xf32>) -> f32
  %den = tt.splat %sum : f32 -> tensor<8xf32>
  %p = arith.divf %e, %den : tensor<8xf32>
  tt.return %p, %e : tensor<8xf32>, tensor<8xf32>
}
