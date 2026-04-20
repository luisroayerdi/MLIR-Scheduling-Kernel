//===- AttentionTypes.cpp - Attention dialect types -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Attention/AttentionTypes.h"

#include "Attention/AttentionDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::attention;

#define GET_TYPEDEF_CLASSES
#include "Attention/AttentionOpsTypes.cpp.inc"

void AttentionDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Attention/AttentionOpsTypes.cpp.inc"
      >();
}
