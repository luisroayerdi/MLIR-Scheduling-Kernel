add_mlir_dialect(AttentionOps attention)
add_mlir_doc(AttentionDialect AttentionDialect Attention/ -gen-dialect-doc)
add_mlir_doc(AttentionOps AttentionOps Attention/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS AttentionPasses.td)
mlir_tablegen(AttentionPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRAttentionPassesIncGen)
