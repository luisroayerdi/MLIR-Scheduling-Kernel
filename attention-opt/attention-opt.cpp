//===- attention-opt.cpp ----------------------------------------*- C++ -*-===//
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Attention/AttentionDialect.h"
#include "Attention/AttentionPasses.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::attention::registerPasses();

  mlir::DialectRegistry registry;
  // Register all MLIR core dialects so every pass in the pipeline has what
  // it needs without having to enumerate them individually.
  mlir::registerAllDialects(registry);
  registry.insert<mlir::attention::AttentionDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Attention optimizer driver\n", registry));
}
