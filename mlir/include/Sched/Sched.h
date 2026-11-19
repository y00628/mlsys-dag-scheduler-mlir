#pragma once
// Public API of the sched MLIR layer: import/export, cost oracle, passes.

#include <optional>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/Pass.h"

#include "../../../source/problem.h"
#include "../../../source/solution.h"

namespace mlsys::sched {

// Problem JSON structs <-> IR.
mlir::OwningOpRef<mlir::ModuleOp> importProblem(mlir::MLIRContext& ctx,
                                                const Problem& problem);
// Rebuilds the Problem purely from the IR (tensor/op ids live in attributes).
Problem extractProblem(mlir::ModuleOp module);
// Reads sched.group ops (in IR order) into a Solution.
Solution exportSolution(mlir::ModuleOp module);

// Cost oracle: best tile granularity + latency for a fused group of ops.
struct GroupCost {
  Granularity granularity;
  double latency;
};
std::optional<GroupCost> bestGroupCost(const Problem& problem,
                                       const std::vector<size_t>& op_ids);

// Rebuild the func body so each element of `partition` becomes one sched.group.
// Partition entries must be in a valid topological order of the group DAG.
void regroup(mlir::func::FuncOp func,
             const std::vector<std::vector<mlir::Operation*>>& partition);

// Passes.
std::unique_ptr<mlir::Pass> createFormGroupsPass();
std::unique_ptr<mlir::Pass> createFuseGroupsPass();
std::unique_ptr<mlir::Pass> createSelectGranularityPass();
void registerSchedPasses();

}  // namespace mlsys::sched
