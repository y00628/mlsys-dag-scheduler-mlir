// mlsys-mlir: problem.json -> sched IR -> pass pipeline -> solution.json
#include <fstream>
#include <iostream>

#include "Sched/Sched.h"
#include "Sched/SchedOps.h"
#include "../../source/evaluator.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_os_ostream.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

int main(int argc, char** argv) {
  using namespace mlsys;
  std::string pipeline = "sched-form-groups,sched-fuse-groups,sched-select-granularity";
  std::string emitIR;
  std::vector<std::string> pos;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--pipeline=", 0) == 0) pipeline = a.substr(11);
    else if (a.rfind("--emit-ir=", 0) == 0) emitIR = a.substr(10);
    else pos.push_back(a);
  }
  if (pos.size() != 2) {
    std::cerr << "Usage: " << argv[0] << " <input.json> <output.json> [--pipeline=<mlir pipeline>] [--emit-ir=<file.mlir>]\n";
    return 1;
  }
  try {
    Problem problem = ReadProblem(pos[0]);
    mlir::DialectRegistry registry;
    registry.insert<sched::SchedDialect, mlir::func::FuncDialect>();
    mlir::MLIRContext ctx(registry);
    ctx.loadAllAvailableDialects();
    sched::registerSchedPasses();

    auto module = sched::importProblem(ctx, problem);
    if (mlir::failed(mlir::verify(*module))) return 1;
    mlir::PassManager pm(&ctx);
    if (mlir::failed(mlir::parsePassPipeline(pipeline, pm))) return 1;
    if (mlir::failed(pm.run(*module))) return 1;
    if (mlir::failed(mlir::verify(*module))) return 1;
    if (!emitIR.empty()) { std::ofstream f(emitIR); llvm::raw_os_ostream os(f); module->print(os); }

    Solution sol = sched::exportSolution(*module);
    WriteSolution(sol, pos[1]);
    TotalLatency total = -1;
    if (!RecomputeLatencies(problem, &sol, &total)) { std::cerr << "ERROR: solution invalid\n"; return 1; }
    WriteSolution(sol, pos[1]);
    std::cerr << sol.subgraphs.size() << " groups / " << problem.ops.size() << " ops, total latency: " << total << "\n";
  } catch (const std::exception& e) { std::cerr << "FATAL: " << e.what() << "\n"; return 1; }
  return 0;
}
