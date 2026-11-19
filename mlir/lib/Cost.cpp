#include "Sched/Sched.h"

#include <algorithm>
#include <limits>
#include <set>

#include "../../source/evaluator.h"

namespace mlsys::sched {

namespace {
std::vector<int64_t> dimCandidates(int64_t dim, int64_t native) {
  std::set<int64_t> v;
  auto add = [&](int64_t x) { v.insert(std::max<int64_t>(1, std::min(dim, x))); };
  add(dim);
  for (int64_t x = native; x < dim; x *= 2) add(x);  // native * 2^i
  add(native);
  add(native / 2);
  add(native / 4);
  for (int64_t x = dim / 2; x >= 1 && x >= native / 4; x /= 2) add(x);
  return {v.rbegin(), v.rend()};
}
}  // namespace

std::optional<GroupCost> bestGroupCost(const Problem& p,
                                       const std::vector<size_t>& op_ids) {
  int64_t maxW = 1, maxH = 1, maxK = 0;
  for (size_t id : op_ids) {
    const auto& op = p.ops[id];
    for (size_t t : op.outputs) {
      maxW = std::max(maxW, p.tensors[t].width);
      maxH = std::max(maxH, p.tensors[t].height);
    }
    if (op.op_type == "MatMul")
      maxK = std::max(maxK, p.tensors[op.inputs[0]].width);
  }
  std::vector<int64_t> ks;
  if (maxK == 0) {
    ks = {1};
  } else {
    ks = dimCandidates(maxK, std::max<int64_t>(1, p.native_granularity.depth));
  }
  Subgraph sg;
  sg.op_ids = op_ids;
  std::optional<GroupCost> best;
  for (int64_t w : dimCandidates(maxW, p.native_granularity.width))
    for (int64_t h : dimCandidates(maxH, p.native_granularity.height))
      for (int64_t k : ks) {
        sg.granularity = {w, h, k};
        SubgraphEvaluation ev;
        if (!EvaluateSubgraph(p, sg, {}, &ev) || !ev.valid) continue;
        if (!best || ev.latency < best->latency) best = GroupCost{sg.granularity, ev.latency};
      }
  return best;
}

}  // namespace mlsys::sched
