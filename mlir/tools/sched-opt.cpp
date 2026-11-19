#include "Sched/Sched.h"
#include "Sched/SchedOps.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlsys::sched::SchedDialect, mlir::func::FuncDialect>();
  mlsys::sched::registerSchedPasses();
  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "sched-opt\n", registry));
}
