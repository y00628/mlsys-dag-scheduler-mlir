// RUN: sched-opt %s -sched-form-groups -sched-fuse-groups -sched-select-granularity | FileCheck %s
// Hand-written 2-op chain: matmul followed by an epilogue pointwise should fuse.
module attributes {sched.fast_memory = 60000 : i64, sched.native_granularity = array<i64: 128, 128>, sched.slow_memory_bandwidth = 20 : i64} {
  func.func @dag(%a: tensor<256x256xf32> {sched.tensor_id = 0 : i64}, %b: tensor<256x256xf32> {sched.tensor_id = 1 : i64}) -> tensor<256x256xf32> {
    %0 = sched.matmul %a, %b {base_cost = 2000 : i64, op_id = 0 : i64, out_id = 2 : i64} : tensor<256x256xf32>, tensor<256x256xf32> -> tensor<256x256xf32>
    %1 = sched.pointwise %0 {base_cost = 500 : i64, op_id = 1 : i64, out_id = 3 : i64} : tensor<256x256xf32> -> tensor<256x256xf32>
    return %1 : tensor<256x256xf32>
  }
}
// CHECK: sched.group
// CHECK: sched.matmul
// CHECK: sched.pointwise
// CHECK: granularity
