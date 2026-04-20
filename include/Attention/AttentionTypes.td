//===- AttentionTypes.td - Attention dialect types -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_TYPES
#define STANDALONE_TYPES

include "mlir/IR/AttrTypeBase.td"
include "Attention/AttentionDialect.td"

//===----------------------------------------------------------------------===//
// Attention type definitions
//===----------------------------------------------------------------------===//

class Attention_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Attention_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Attention_CustomType : Attention_Type<"Custom", "custom"> {
    let summary = "Attention custom type";
    let description = "Custom type in attention dialect";
    let parameters = (ins StringRefParameter<"the custom value">:$value);
    let assemblyFormat = "`<` $value `>`";
}

#endif // STANDALONE_TYPES
