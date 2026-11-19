# MLSys 2026 Track A — DAG Scheduler, rebuilt on MLIR

MLIR-based rewrite of [y00628/mlsys-dag-scheduler](https://github.com/y00628/mlsys-dag-scheduler).
The problem (a DAG of `MatMul`/`Pointwise` ops over 2-D tensors, scheduled into fused subgraphs with a
tile granularity, minimizing roofline latency) is now expressed as IR and solved by passes.

## Design

| Piece | Where |
|---|---|
| `sched` dialect: `sched.matmul`, `sched.pointwise`, `sched.group` (fused subgraph + `granularity = [w,h,k]`), `sched.yield` | `mlir/include/Sched/SchedOps.td`, `mlir/lib/Sched.cpp` |
| Problem JSON → IR, IR → Problem, IR → solution JSON | `mlir/lib/Convert.cpp` |
| Cost oracle (best tile granularity per group, via the original tile-level evaluator) | `mlir/lib/Cost.cpp` |
| Passes `sched-form-groups`, `sched-fuse-groups`, `sched-select-granularity` | `mlir/lib/Passes.cpp` |
| Tools `mlsys-mlir` (JSON in/out) and `sched-opt` (`mlir-opt`-style driver) | `mlir/tools/` |

Tensors are `tensor<HxWxf32>` (shape = [height, width]); hardware parameters live as module attributes.
The IR is self-contained: ids are stored as attributes, so `.mlir` files round-trip through `sched-opt`.

`sched-fuse-groups` is a steepest-descent fusion: it repeatedly merges the pair of connected groups
(data edge or shared input) whose merge keeps the group set convex and gives the largest latency drop,
as measured by the evaluator at each group's best granularity. See `docs/examples/` for scheduled IR.

The legacy (non-MLIR) solver in `source/` is kept; only its problem/solution/evaluator code is used by the MLIR build.
`tensors_to_retain` is not yet modelled in the IR (always empty).

## Build

```bash
brew install llvm cmake ninja
cmake -B build-mlir -G Ninja -DCMAKE_PREFIX_PATH=$(brew --prefix llvm) -DCMAKE_BUILD_TYPE=Release
ninja -C build-mlir
```
Also needs `third_party/json.hpp` (nlohmann/json single header) and the benchmarks in `benchmarks/`.

## Run

```bash
./build-mlir/mlsys-mlir benchmarks/mlsys-2026-5.json out.json --emit-ir=out.mlir
./build-mlir/sched-opt mlir/test/smoke.mlir -sched-form-groups -sched-fuse-groups -sched-select-granularity
```

## Results (total latency, lower is better)

| Benchmark | Baseline (no fusion) | Original solver best | This MLIR pipeline |
|---|---:|---:|---:|
| mlsys-2026-1  | 471,500.80 | 400,000 | 400,000 |
| mlsys-2026-5  | 1,013,623.47 | 650,528 | 622,308 |
| mlsys-2026-9  | 167,530,987.52 | 164,506,000 | 164,949,000 |
| mlsys-2026-13 | 166,414,742.80 | 166,403,000 | 166,402,000 |
| mlsys-2026-17 | 46,441,850.88 | 46,338,000 | 46,326,400 |

All outputs validated by the evaluator (`RecomputeLatencies`).

## License

Apache-2.0
