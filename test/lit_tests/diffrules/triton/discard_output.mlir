// REQUIRES: triton
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=mul outfn=rev argTys=enzyme_dup,enzyme_dup,enzyme_dup mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --discard-triton-output="function=rev argument=4" --canonicalize | FileCheck %s
// RUN: not enzymexlamlir-opt %s --discard-triton-output="function=read_then_clear argument=0" 2>&1 | FileCheck %s --check-prefix=READ
// RUN: not enzymexlamlir-opt %s --discard-triton-output="function=shared_add argument=0" 2>&1 | FileCheck %s --check-prefix=ATOMIC
// RUN: not enzymexlamlir-opt %s --discard-triton-output="function=escape argument=0" 2>&1 | FileCheck %s --check-prefix=ESCAPE

tt.func @mul(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
  %a = tt.load %x : !tt.ptr<f32>
  %b = tt.load %y : !tt.ptr<f32>
  %result = arith.mulf %a, %b : f32
  tt.store %out, %result : !tt.ptr<f32>
  tt.return
}

// The public ABI still contains the primal output. The generated pullback
// has no uses of that pointer and still consumes the independent output seed.
// CHECK-LABEL: tt.func private @rev(
// CHECK-SAME: %[[OUT:arg4]]: !tt.ptr<f32>, %[[SEED:arg5]]: !tt.ptr<f32>
// CHECK-NOT: %[[OUT]]
// CHECK: tt.load %[[SEED]]
// CHECK-NOT: %[[OUT]]
// CHECK: tt.store %[[SEED]]
// CHECK-NOT: %[[OUT]]
// CHECK: tt.atomic_rmw fadd
// CHECK-NOT: %[[OUT]]
// CHECK: tt.atomic_rmw fadd
// CHECK-NOT: %[[OUT]]
// CHECK: tt.return

// Seed consumption requires its own interface. Marking this pointer as a
// write-only output is an error even though its last access is a clear.
tt.func @read_then_clear(%seed: !tt.ptr<f32>, %observed: !tt.ptr<f32>) {
  %v = tt.load %seed : !tt.ptr<f32>
  %zero = arith.constant 0.0 : f32
  tt.store %seed, %zero : !tt.ptr<f32>
  tt.store %observed, %v : !tt.ptr<f32>
  tt.return
}
// READ: 'tt.load' op
// READ-SAME: cannot discard output argument 0 of @read_then_clear

// Every lane and every program targets the same address. No fact about
// launch dimensions or uniqueness can be inferred from this pointer.
tt.func @shared_add(%dx: !tt.ptr<f32>) {
  %p = tt.splat %dx : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %ones = arith.constant dense<1.0> : tensor<8xf32>
  %mask = arith.constant dense<true> : tensor<8xi1>
  %v = tt.atomic_rmw fadd, relaxed, gpu, %p, %ones, %mask : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, tensor<8xi1>) -> tensor<8xf32>
  tt.return
}
// ATOMIC: 'tt.atomic_rmw' op
// ATOMIC-SAME: cannot discard output argument 0 of @shared_add

tt.func @escape(%scratch: !tt.ptr<f32>) -> !tt.ptr<f32> {
  tt.return %scratch : !tt.ptr<f32>
}
// ESCAPE: 'tt.return' op
// ESCAPE-SAME: cannot discard output argument 0 of @escape
