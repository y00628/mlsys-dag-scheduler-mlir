#ifndef SCHED_OPS
#define SCHED_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Sched_Dialect : Dialect {
  let name = "sched";
  let summary = "DAG scheduling IR for the MLSys 2026 Track A problem";
  let description = [{
    Ops are MatMul / Pointwise over 2-D tensors (tensor<HxWxf32>, shape = [height, width]).
    `sched.group` is a fused subgraph that carries its tile granularity [w, h, k].
  }];
  let cppNamespace = "::mlsys::sched";
}

class Sched_Op<string mnemonic, list<Trait> traits = []>
    : Op<Sched_Dialect, mnemonic, traits>;

def Sched_MatMulOp : Sched_Op<"matmul", [Pure]> {
  let arguments = (ins RankedTensorOf<[F32]>:$lhs, RankedTensorOf<[F32]>:$rhs,
                       I64Attr:$base_cost, I64Attr:$op_id, I64Attr:$out_id);
  let results = (outs RankedTensorOf<[F32]>:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasVerifier = 1;
}

def Sched_PointwiseOp : Sched_Op<"pointwise", [Pure]> {
  let arguments = (ins Variadic<RankedTensorOf<[F32]>>:$operands,
                       I64Attr:$base_cost, I64Attr:$op_id, I64Attr:$out_id);
  let results = (outs RankedTensorOf<[F32]>:$result);
  let assemblyFormat =
      "$operands attr-dict `:` type($operands) `->` type($result)";
  let hasVerifier = 1;
}

def Sched_YieldOp : Sched_Op<"yield", [Pure, Terminator, HasParent<"GroupOp">]> {
  let arguments = (ins Variadic<AnyType>:$operands);
  let builders = [OpBuilder<(ins), [{ /* no operands */ }]>];
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def Sched_GroupOp : Sched_Op<"group",
    [SingleBlockImplicitTerminator<"YieldOp">, RecursiveMemoryEffects]> {
  let summary = "A fused subgraph executed as one unit";
  let arguments = (ins OptionalAttr<DenseI64ArrayAttr>:$granularity);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "(`->` type($results)^)? $body attr-dict";
  let hasVerifier = 1;
}

#endif
