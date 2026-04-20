// RUN: attention-opt %s | attention-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = attention.foo %{{.*}} : i32
        %res = attention.foo %0 : i32
        return
    }

    // CHECK-LABEL: func @attention_types(%arg0: !attention.custom<"10">)
    func.func @attention_types(%arg0: !attention.custom<"10">) {
        return
    }
}
