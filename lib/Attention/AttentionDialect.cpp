//===- AttentionDialect.cpp - Attention dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Attention/AttentionDialect.h"
#include "Attention/AttentionOps.h"
#include "Attention/AttentionTypes.h"

using namespace mlir;
using namespace mlir::attention;

#include "Attention/AttentionOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Attention dialect.
//===----------------------------------------------------------------------===//

void AttentionDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Attention/AttentionOps.cpp.inc"
      >();
  registerTypes();
}
