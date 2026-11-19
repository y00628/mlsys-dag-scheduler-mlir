#include "Sched/SchedOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlsys::sched;

#include "SchedDialect.cpp.inc"

void SchedDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "SchedOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "SchedOps.cpp.inc"

static bool is2D(Type t) {
  auto rt = dyn_cast<RankedTensorType>(t);
  return rt && rt.getRank() == 2 && rt.hasStaticShape();
}

LogicalResult MatMulOp::verify() {
  if (!is2D(getLhs().getType()) || !is2D(getRhs().getType()) ||
      !is2D(getResult().getType()))
    return emitOpError("operands and result must be static 2-D tensors");
  auto l = cast<RankedTensorType>(getLhs().getType()).getShape();
  auto r = cast<RankedTensorType>(getRhs().getType()).getShape();
  auto o = cast<RankedTensorType>(getResult().getType()).getShape();
  // shape = [height, width]; lhs = [M, K], rhs = [K, N], result = [M, N].
  if (l[1] != r[0] || o[0] != l[0] || o[1] != r[1])
    return emitOpError("shape mismatch: expected [M,K] x [K,N] -> [M,N]");
  return success();
}

LogicalResult PointwiseOp::verify() {
  if (getOperands().empty() || getOperands().size() > 2)
    return emitOpError("expects 1 or 2 operands");
  if (!is2D(getResult().getType()))
    return emitOpError("result must be a static 2-D tensor");
  return success();
}

LogicalResult GroupOp::verify() {
  if (auto g = getGranularity()) {
    if (g->size() != 3 || llvm::any_of(*g, [](int64_t v) { return v <= 0; }))
      return emitOpError("granularity must be [w, h, k] with positive entries");
  }
  return success();
}
