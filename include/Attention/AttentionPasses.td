//===- AttentionPasses.td - Attention dialect passes -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ATTENTION_PASS
#define ATTENTION_PASS

include "mlir/Pass/PassBase.td"

def FusionPass : Pass<"fusion-pass", "::mlir::func::FuncOp"> {
  let summary = "Fuse linalg attention ops into attention.fused";
  let description = [{
    Matches the 5-op unfused attention sequence:
      linalg.generic (QK^T) -> linalg.generic (scale) ->
      linalg.generic (mask, optional) -> linalg.softmax -> linalg.matmul (PV)
    and replaces it with a single attention.fused op.

    Pattern is anchored on the final linalg.matmul (the PV matmul).
    The scale value is extracted from the body of the scale generic.
  }];
}

def TilingPass : Pass<"tiling-pass", "::mlir::func::FuncOp"> {
  let summary = "Tile attention.fused with online softmax accumulation";
  let description = [{
    Expands each attention.fused op into nested affine.for loops with
    explicit online softmax (Milakov & Gimelshein 2018) accumulation.

    After this pass no attention.fused ops remain; output is standard
    affine + linalg + memref IR that mlir-cpu-runner can lower.

    Loop structure:
      for i in [0, seq_q, tile_size]:        # Q outer loop
        init O_acc, m_acc, l_acc
        for j in [0, seq_k, tile_size]:      # K/V inner loop
          compute S_tile = Q_tile @ K_tile^T * scale  (+ optional mask)
          online softmax update of m_acc, l_acc, O_acc
        O_tile = O_acc / l_acc               # final rescale
        write O_tile to output
  }];
  let options = [
    Option<"tileSize", "tile-size", "int64_t", "128",
           "Tile size for both Q and K/V sequence dimensions. "
           "Default 128 targets A100 SRAM (192 KB). "
           "Use 32 or 64 for CPU correctness testing.">
  ];
}

#endif // ATTENTION_PASS
