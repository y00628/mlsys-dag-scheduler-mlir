#include "Sched/Sched.h"
#include "Sched/SchedOps.h"

#include <algorithm>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>

#include "mlir/IR/Builders.h"

#include "../../source/evaluator.h"

using namespace mlir;

namespace mlsys::sched {

namespace {
constexpr const char* kFast = "sched.fast_memory";
constexpr const char* kBw = "sched.slow_memory_bandwidth";
constexpr const char* kNative = "sched.native_granularity";
constexpr const char* kTensorId = "sched.tensor_id";

Value resolve(Value v) {
  if (auto g = v.getDefiningOp<GroupOp>()) {
    auto y = cast<YieldOp>(g.getBody().front().getTerminator());
    return resolve(y.getOperand(cast<OpResult>(v).getResultNumber()));
  }
  return v;
}

int64_t tensorId(Value v) {
  v = resolve(v);
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    auto func = cast<func::FuncOp>(arg.getOwner()->getParentOp());
    auto a = func.getArgAttrOfType<IntegerAttr>(arg.getArgNumber(), kTensorId);
    return a.getInt();
  }
  Operation* d = v.getDefiningOp();
  if (auto m = dyn_cast<MatMulOp>(d)) return m.getOutId();
  return cast<PointwiseOp>(d).getOutId();
}

func::FuncOp findFunc(ModuleOp m) {
  func::FuncOp f;
  m.walk([&](func::FuncOp fn) {
    if (!f) f = fn;
  });
  return f;
}
}  // namespace

OwningOpRef<ModuleOp> importProblem(MLIRContext& ctx, const Problem& p) {
  OpBuilder b(&ctx);
  Location loc = b.getUnknownLoc();
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  (*module)->setAttr(kFast, b.getI64IntegerAttr(p.fast_memory_capacity));
  (*module)->setAttr(kBw, b.getI64IntegerAttr(p.slow_memory_bandwidth));
  (*module)->setAttr(kNative,
                     b.getDenseI64ArrayAttr({p.native_granularity.width,
                                             p.native_granularity.height}));

  auto ttype = [&](size_t t) {
    return RankedTensorType::get({p.tensors[t].height, p.tensors[t].width},
                                 b.getF32Type());
  };
  SmallVector<Type> inTypes, outTypes;
  for (size_t t : p.graph_inputs) inTypes.push_back(ttype(t));
  for (size_t t : p.graph_outputs) outTypes.push_back(ttype(t));

  b.setInsertionPointToStart(module->getBody());
  auto func = func::FuncOp::create(b, loc, "dag", b.getFunctionType(inTypes, outTypes));
  Block* entry = func.addEntryBlock();
  std::vector<Value> value(p.tensors.size());
  for (size_t i = 0; i < p.graph_inputs.size(); ++i) {
    func.setArgAttr(i, kTensorId, b.getI64IntegerAttr(p.graph_inputs[i]));
    value[p.graph_inputs[i]] = entry->getArgument(i);
  }

  // Kahn topological order over ops.
  size_t n = p.ops.size();
  std::vector<int> indeg(n, 0);
  std::vector<std::vector<size_t>> succ(n);
  std::vector<int64_t> producer(p.tensors.size(), -1);
  for (size_t i = 0; i < n; ++i) {
    if (p.ops[i].outputs.size() != 1)
      throw std::runtime_error("sched dialect supports single-output ops only");
    producer[p.ops[i].outputs[0]] = i;
  }
  for (size_t i = 0; i < n; ++i)
    for (size_t t : p.ops[i].inputs)
      if (producer[t] >= 0) {
        succ[producer[t]].push_back(i);
        indeg[i]++;
      }
  std::priority_queue<size_t, std::vector<size_t>, std::greater<>> ready;
  for (size_t i = 0; i < n; ++i)
    if (!indeg[i]) ready.push(i);

  b.setInsertionPointToEnd(entry);
  while (!ready.empty()) {
    size_t i = ready.top();
    ready.pop();
    const Op& op = p.ops[i];
    SmallVector<Value> operands;
    for (size_t t : op.inputs) operands.push_back(value[t]);
    Type rty = ttype(op.outputs[0]);
    Operation* created;
    if (op.op_type == "MatMul") {
      created = MatMulOp::create(b, loc, rty, operands[0], operands[1], op.base_cost,
                                 (int64_t)i, (int64_t)op.outputs[0]);
    } else {
      created = PointwiseOp::create(b, loc, rty, operands, op.base_cost, (int64_t)i,
                                    (int64_t)op.outputs[0]);
    }
    value[op.outputs[0]] = created->getResult(0);
    for (size_t s : succ[i])
      if (--indeg[s] == 0) ready.push(s);
  }
  SmallVector<Value> rets;
  for (size_t t : p.graph_outputs) rets.push_back(value[t]);
  func::ReturnOp::create(b, loc, rets);
  return module;
}

Problem extractProblem(ModuleOp module) {
  Problem p;
  p.fast_memory_capacity = module->getAttrOfType<IntegerAttr>(kFast).getInt();
  p.slow_memory_bandwidth = module->getAttrOfType<IntegerAttr>(kBw).getInt();
  auto nat = module->getAttrOfType<DenseI64ArrayAttr>(kNative).asArrayRef();
  p.native_granularity = {nat[0], nat[1], nat[0]};

  func::FuncOp func = findFunc(module);
  std::map<int64_t, RankedTensorType> types;
  auto note = [&](Value v) {
    types[tensorId(v)] = cast<RankedTensorType>(v.getType());
  };
  for (BlockArgument a : func.getArguments()) note(a);

  struct Rec { int64_t id; Operation* op; };
  std::vector<Rec> recs;
  func.walk([&](Operation* op) {
    if (auto m = dyn_cast<MatMulOp>(op)) recs.push_back({(int64_t)m.getOpId(), op});
    else if (auto q = dyn_cast<PointwiseOp>(op)) recs.push_back({(int64_t)q.getOpId(), op});
  });
  std::sort(recs.begin(), recs.end(), [](auto& a, auto& b) { return a.id < b.id; });
  p.ops.resize(recs.size());
  for (size_t i = 0; i < recs.size(); ++i) {
    Operation* op = recs[i].op;
    if (recs[i].id != (int64_t)i)
      throw std::runtime_error("sched.op_id values must be dense 0..n-1");
    mlsys::Op& o = p.ops[i];
    o.op_type = isa<MatMulOp>(op) ? "MatMul" : "Pointwise";
    o.base_cost = isa<MatMulOp>(op) ? cast<MatMulOp>(op).getBaseCost()
                                    : cast<PointwiseOp>(op).getBaseCost();
    for (Value v : op->getOperands()) {
      note(v);
      o.inputs.push_back(tensorId(v));
    }
    note(op->getResult(0));
    o.outputs.push_back(tensorId(op->getResult(0)));
  }
  size_t nt = types.empty() ? 0 : types.rbegin()->first + 1;
  p.tensors.resize(nt);
  for (auto& [id, ty] : types) p.tensors[id] = {ty.getShape()[1], ty.getShape()[0]};

  std::set<size_t> produced, consumed;
  for (auto& o : p.ops) {
    for (auto t : o.outputs) produced.insert(t);
    for (auto t : o.inputs) consumed.insert(t);
  }
  for (size_t i = 0; i < nt; ++i) {
    if (!produced.count(i)) p.graph_inputs.push_back(i);
    if (!consumed.count(i)) p.graph_outputs.push_back(i);
  }
  return p;
}

Solution exportSolution(ModuleOp module) {
  Solution sol;
  Problem problem = extractProblem(module);
  func::FuncOp func = findFunc(module);
  for (Operation& top : func.getBody().front()) {
    auto g = dyn_cast<GroupOp>(&top);
    if (!g) continue;
    Subgraph sg;
    g.walk([&](Operation* op) {
      if (auto m = dyn_cast<MatMulOp>(op)) sg.op_ids.push_back(m.getOpId());
      else if (auto q = dyn_cast<PointwiseOp>(op)) sg.op_ids.push_back(q.getOpId());
    });
    auto gran = g.getGranularity();
    if (!gran) throw std::runtime_error("sched.group without granularity; run sched-select-granularity");
    sg.granularity = {(*gran)[0], (*gran)[1], (*gran)[2]};
    SubgraphEvaluation ev;
    if (EvaluateSubgraph(problem, sg, {}, &ev) && ev.valid) sg.subgraph_latency = ev.latency;
    sol.subgraphs.push_back(std::move(sg));
  }
  return sol;
}

void regroup(func::FuncOp func,
             const std::vector<std::vector<Operation*>>& partition) {
  Block& body = func.getBody().front();
  // 1. Unwrap existing groups.
  for (Operation& op : llvm::make_early_inc_range(body)) {
    auto g = dyn_cast<GroupOp>(&op);
    if (!g) continue;
    auto y = cast<YieldOp>(g.getBody().front().getTerminator());
    for (unsigned i = 0; i < g.getNumResults(); ++i)
      g.getResult(i).replaceAllUsesWith(y.getOperand(i));
    for (Operation& inner : llvm::make_early_inc_range(g.getBody().front()))
      if (!isa<YieldOp>(inner)) inner.moveBefore(g);
    g.erase();
  }
  // Original (topological) position of every op.
  llvm::DenseMap<Operation*, unsigned> pos;
  unsigned k = 0;
  for (Operation& op : body) pos[&op] = k++;

  Operation* terminator = body.getTerminator();
  OpBuilder b(func.getContext());
  for (const auto& part : partition) {
    std::vector<Operation*> ops(part.begin(), part.end());
    std::sort(ops.begin(), ops.end(), [&](Operation* a, Operation* c) { return pos[a] < pos[c]; });
    llvm::DenseSet<Operation*> in(ops.begin(), ops.end());
    SmallVector<Value> boundary;
    SmallVector<Type> types;
    for (Operation* op : ops)
      for (Value r : op->getResults())
        if (llvm::any_of(r.getUsers(), [&](Operation* u) { return !in.count(u); })) {
          boundary.push_back(r);
          types.push_back(r.getType());
        }
    b.setInsertionPoint(terminator);
    auto group = GroupOp::create(b, ops.front()->getLoc(), types, DenseI64ArrayAttr());
    Block* blk = b.createBlock(&group.getBody());
    for (Operation* op : ops) op->moveBefore(blk, blk->end());
    for (size_t i = 0; i < boundary.size(); ++i)
      boundary[i].replaceUsesWithIf(group.getResult(i), [&](OpOperand& u) {
        return !in.count(u.getOwner());
      });
    b.setInsertionPointToEnd(blk);
    YieldOp::create(b, group.getLoc(), boundary);
  }
}

}  // namespace mlsys::sched
