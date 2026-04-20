add_mlir_dialect_library(MLIRAttention
        AttentionTypes.cpp
        AttentionDialect.cpp
        AttentionOps.cpp
        AttentionPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Attention

        DEPENDS
        MLIRAttentionOpsIncGen
        MLIRAttentionPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
