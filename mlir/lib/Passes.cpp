#include "Sched/Sched.h"
#include "Sched/SchedOps.h"

#include <algorithm>
#include <map>
#include <queue>
#include <set>

#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace mlsys::sched {
namespace {

func::FuncOp getFunc(ModuleOp m) {
  func::FuncOp f;
  m.walk([&](func::FuncOp fn) { if (!f) f = fn; });
  return f;
}

// op_id -> Operation*, plus current grouping as lists of op ids (IR order).
struct IRView {
  std::map<int64_t, Operation*> byId;
  std::vector<std::vector<size_t>> groups;
};

int64_t opIdOf(Operation* op) {
  if (auto m = dyn_cast<MatMulOp>(op)) return m.getOpId();
  if (auto q = dyn_cast<PointwiseOp>(op)) return q.getOpId();
  return -1;
}

IRView readView(func::FuncOp func) {
  IRView v;
  for (Operation& top : func.getBody().front()) {
    if (auto g = dyn_cast<GroupOp>(&top)) {
      v.groups.emplace_back();
      g.walk([&](Operation* op) {
        if (opIdOf(op) >= 0) { v.byId[opIdOf(op)] = op; v.groups.back().push_back(opIdOf(op)); }
      });
    } else if (opIdOf(&top) >= 0) {
      v.byId[opIdOf(&top)] = &top;
      v.groups.push_back({(size_t)opIdOf(&top)});
    }
  }
  return v;
}

struct FormGroupsPass : PassWrapper<FormGroupsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FormGroupsPass)
  StringRef getArgument() const final { return "sched-form-groups"; }
  StringRef getDescription() const final { return "Wrap every sched op in its own sched.group (no fusion)"; }
  void getDependentDialects(DialectRegistry& r) const override { r.insert<SchedDialect>(); }
  void runOnOperation() override {
    func::FuncOp func = getFunc(getOperation());
    IRView v = readView(func);
    std::vector<std::vector<Operation*>> part;
    for (auto& [id, op] : v.byId) part.push_back({op});
    regroup(func, part);
  }
};

struct SelectGranularityPass
    : PassWrapper<SelectGranularityPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SelectGranularityPass)
  StringRef getArgument() const final { return "sched-select-granularity"; }
  StringRef getDescription() const final { return "Pick the best tile granularity [w,h,k] for each sched.group"; }
  void getDependentDialects(DialectRegistry& r) const override { r.insert<SchedDialect>(); }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Problem p = extractProblem(module);
    func::FuncOp func = getFunc(module);
    for (Operation& top : func.getBody().front()) {
      auto g = dyn_cast<GroupOp>(&top);
      if (!g) continue;
      std::vector<size_t> ids;
      g.walk([&](Operation* op) { if (opIdOf(op) >= 0) ids.push_back(opIdOf(op)); });
      auto cost = bestGroupCost(p, ids);
      if (!cost) { g.emitError("no legal tile granularity fits in fast memory"); return signalPassFailure(); }
      g.setGranularityAttr(Builder(g).getDenseI64ArrayAttr(
          {cost->granularity.width, cost->granularity.height, cost->granularity.depth}));
    }
  }
};

struct FuseGroupsPass : PassWrapper<FuseGroupsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseGroupsPass)
  StringRef getArgument() const final { return "sched-fuse-groups"; }
  StringRef getDescription() const final {
    return "Greedily merge connected, convex sched.groups while the evaluator-measured latency drops";
  }
  void getDependentDialects(DialectRegistry& r) const override { r.insert<SchedDialect>(); }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    func::FuncOp func = getFunc(module);
    Problem p = extractProblem(module);
    IRView view = readView(func);
    const size_t n = p.ops.size();

    // Op-level DAG.
    std::vector<int64_t> producer(p.tensors.size(), -1);
    for (size_t i = 0; i < n; ++i) producer[p.ops[i].outputs[0]] = i;
    std::vector<std::vector<size_t>> succ(n);
    std::vector<std::vector<size_t>> consumersOf(p.tensors.size());
    for (size_t i = 0; i < n; ++i)
      for (size_t t : p.ops[i].inputs) {
        consumersOf[t].push_back(i);
        if (producer[t] >= 0) succ[producer[t]].push_back(i);
      }

    std::map<std::vector<size_t>, std::optional<GroupCost>> cache;
    auto cost = [&](const std::vector<size_t>& g) -> std::optional<GroupCost>& {
      auto it = cache.find(g);
      if (it == cache.end()) it = cache.emplace(g, bestGroupCost(p, g)).first;
      return it->second;
    };
    // Convex: no path leaves the set and re-enters it.
    auto convex = [&](const std::vector<size_t>& s) {
      std::vector<char> in(n, 0), seen(n, 0);
      for (size_t o : s) in[o] = 1;
      std::vector<size_t> stack;
      for (size_t o : s) for (size_t v : succ[o]) if (!in[v] && !seen[v]) { seen[v] = 1; stack.push_back(v); }
      while (!stack.empty()) {
        size_t u = stack.back(); stack.pop_back();
        for (size_t v : succ[u]) {
          if (in[v]) return false;
          if (!seen[v]) { seen[v] = 1; stack.push_back(v); }
        }
      }
      return true;
    };

    auto groups = view.groups;
    for (auto& g : groups) std::sort(g.begin(), g.end());
    for (;;) {
      std::vector<int> owner(n, -1);
      for (size_t gi = 0; gi < groups.size(); ++gi) for (size_t o : groups[gi]) owner[o] = gi;
      std::set<std::pair<int, int>> pairs;  // connected via data edge or shared input
      for (size_t i = 0; i < n; ++i)
        for (size_t v : succ[i]) if (owner[i] != owner[v]) pairs.insert({std::min(owner[i], owner[v]), std::max(owner[i], owner[v])});
      for (auto& cs : consumersOf)
        for (size_t x = 0; x < cs.size(); ++x) for (size_t y = x + 1; y < cs.size(); ++y)
          if (owner[cs[x]] != owner[cs[y]]) pairs.insert({std::min(owner[cs[x]], owner[cs[y]]), std::max(owner[cs[x]], owner[cs[y]])});

      double bestGain = 1e-9; std::pair<int, int> best{-1, -1};
      for (auto [a, b] : pairs) {
        auto& ca = cost(groups[a]); auto& cb = cost(groups[b]);
        if (!ca || !cb) continue;
        std::vector<size_t> m = groups[a];
        m.insert(m.end(), groups[b].begin(), groups[b].end());
        std::sort(m.begin(), m.end());
        if (!convex(m)) continue;
        auto& cm = cost(m);
        if (!cm) continue;
        double gain = ca->latency + cb->latency - cm->latency;
        if (gain > bestGain) { bestGain = gain; best = {a, b}; }
      }
      if (best.first < 0) break;
      auto& ga = groups[best.first];
      ga.insert(ga.end(), groups[best.second].begin(), groups[best.second].end());
      std::sort(ga.begin(), ga.end());
      groups.erase(groups.begin() + best.second);
    }

    // Order the groups topologically (smallest op id first among ready groups).
    std::vector<int> owner(n, -1);
    for (size_t gi = 0; gi < groups.size(); ++gi) for (size_t o : groups[gi]) owner[o] = gi;
    std::vector<std::set<int>> gsucc(groups.size());
    std::vector<int> indeg(groups.size(), 0);
    for (size_t i = 0; i < n; ++i)
      for (size_t v : succ[i])
        if (owner[i] != owner[v] && gsucc[owner[i]].insert(owner[v]).second) indeg[owner[v]]++;
    auto cmp = [&](int a, int b) { return groups[a].front() > groups[b].front(); };
    std::priority_queue<int, std::vector<int>, decltype(cmp)> ready(cmp);
    for (size_t gi = 0; gi < groups.size(); ++gi) if (!indeg[gi]) ready.push(gi);
    std::vector<std::vector<Operation*>> part;
    while (!ready.empty()) {
      int g = ready.top(); ready.pop();
      part.emplace_back();
      for (size_t o : groups[g]) part.back().push_back(view.byId[o]);
      for (int s : gsucc[g]) if (--indeg[s] == 0) ready.push(s);
    }
    if (part.size() != groups.size()) { getOperation().emitError("cyclic group graph"); return signalPassFailure(); }
    regroup(func, part);
  }
};

}  // namespace

std::unique_ptr<Pass> createFormGroupsPass() { return std::make_unique<FormGroupsPass>(); }
std::unique_ptr<Pass> createFuseGroupsPass() { return std::make_unique<FuseGroupsPass>(); }
std::unique_ptr<Pass> createSelectGranularityPass() { return std::make_unique<SelectGranularityPass>(); }

void registerSchedPasses() {
  PassRegistration<FormGroupsPass>();
  PassRegistration<FuseGroupsPass>();
  PassRegistration<SelectGranularityPass>();
}

}  // namespace mlsys::sched
