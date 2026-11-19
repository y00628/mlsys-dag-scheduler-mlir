module attributes {sched.fast_memory = 60000 : i64, sched.native_granularity = array<i64: 128, 128>, sched.slow_memory_bandwidth = 20 : i64} {
  func.func @dag(%arg0: tensor<512x512xf32> {sched.tensor_id = 0 : i64}, %arg1: tensor<512x512xf32> {sched.tensor_id = 1 : i64}, %arg2: tensor<512x512xf32> {sched.tensor_id = 2 : i64}, %arg3: tensor<512x512xf32> {sched.tensor_id = 3 : i64}) -> tensor<512x512xf32> {
    %0 = sched.group -> tensor<512x512xf32> {
      %1 = sched.matmul %arg0, %arg1 {base_cost = 2000 : i64, op_id = 0 : i64, out_id = 4 : i64} : tensor<512x512xf32>, tensor<512x512xf32> -> tensor<512x512xf32>
      %2 = sched.pointwise %1 {base_cost = 500 : i64, op_id = 1 : i64, out_id = 5 : i64} : tensor<512x512xf32> -> tensor<512x512xf32>
      %3 = sched.matmul %2, %arg2 {base_cost = 2000 : i64, op_id = 2 : i64, out_id = 6 : i64} : tensor<512x512xf32>, tensor<512x512xf32> -> tensor<512x512xf32>
      %4 = sched.matmul %3, %arg3 {base_cost = 2000 : i64, op_id = 3 : i64, out_id = 7 : i64} : tensor<512x512xf32>, tensor<512x512xf32> -> tensor<512x512xf32>
      %5 = sched.pointwise %4, %arg0 {base_cost = 500 : i64, op_id = 4 : i64, out_id = 8 : i64} : tensor<512x512xf32>, tensor<512x512xf32> -> tensor<512x512xf32>
      sched.yield %5 : tensor<512x512xf32>
    } {granularity = array<i64: 128, 128, 32>}
    return %0 : tensor<512x512xf32>
  }
}
