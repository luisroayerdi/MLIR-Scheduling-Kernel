//===- AttentionOps.cpp - Attention dialect ops ---------------*- C++ -*-===//
#include "Attention/AttentionOps.h"
#include "Attention/AttentionDialect.h"
#include "mlir/IR/BuiltinTypes.h"

#define GET_OP_CLASSES
#include "Attention/AttentionOps.cpp.inc"

using namespace mlir;
using namespace mlir::attention;

// Memory effects: reads Q, K, V, mask (optional); writes output.
// Newer MLIR EffectInstance requires OpOperand* (not Value), so iterate
// over all OpOperands and classify by type/identity.
void FusedOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  Value out = getOutput();
  for (OpOperand &operand : getOperation()->getOpOperands()) {
    Value v = operand.get();
    if (!isa<MemRefType>(v.getType()))
      continue; // skip scale (f32 scalar)
    if (v == out)
      effects.emplace_back(MemoryEffects::Write::get(), &operand,
                           SideEffects::DefaultResource::get());
    else
      effects.emplace_back(MemoryEffects::Read::get(), &operand,
                           SideEffects::DefaultResource::get());
  }
}

// Shape consistency checks.
LogicalResult FusedOp::verify() {
  auto Qtype = cast<MemRefType>(getQ().getType());
  auto Ktype = cast<MemRefType>(getK().getType());
  auto Vtype = cast<MemRefType>(getV().getType());
  auto Otype = cast<MemRefType>(getOutput().getType());

  if (Qtype.getRank() != 2)
    return emitOpError("Q must be rank-2 [seq_q, head_dim]");
  if (Ktype.getRank() != 2)
    return emitOpError("K must be rank-2 [seq_k, head_dim]");
  if (Vtype.getRank() != 2)
    return emitOpError("V must be rank-2 [seq_k, head_dim]");
  if (Otype.getRank() != 2)
    return emitOpError("output must be rank-2 [seq_q, head_dim]");

  int64_t headDim = Qtype.getShape()[1];
  if (Ktype.getShape()[1] != headDim)
    return emitOpError("K head_dim must equal Q head_dim");
  if (Vtype.getShape()[0] != Ktype.getShape()[0])
    return emitOpError("V seq_len must equal K seq_len");
  if (Otype.getShape()[0] != Qtype.getShape()[0])
    return emitOpError("output seq_q must equal Q seq_q");
  if (Otype.getShape()[1] != headDim)
    return emitOpError("output head_dim must equal Q head_dim");

  return success();
}
