// RUN: mlir-opt -allow-unregistered-dialect %s -mlir-print-debuginfo -mlir-pretty-debuginfo -mlir-print-local-scope | FileCheck %s

#set0 = affine_set<(d0) : (1 == 0)>

// CHECK-LABEL: func @inline_notation
func.func @inline_notation() -> i32 {
  // CHECK: -> i32 "foo"
  %1 = "foo"() : () -> i32 loc("foo")

  // CHECK: arith.constant 4 : index "foo" at mysource.cc:10:8
  %2 = arith.constant 4 : index loc(callsite("foo" at "mysource.cc":10:8))

  // CHECK:      arith.constant 4 : index "foo"
  // CHECK-NEXT:  at mysource1.cc:10:8
  // CHECK-NEXT:  at mysource2.cc:13:8
  // CHECK-NEXT:  at mysource3.cc:100:10
  %3 = arith.constant 4 : index loc(callsite("foo" at callsite("mysource1.cc":10:8 at callsite("mysource2.cc":13:8 at "mysource3.cc":100:10))))

  // CHECK: } ["foo", mysource.cc:10:8]
  affine.for %i0 = 0 to 8 {
  } loc(fused["foo", "mysource.cc":10:8])

  // CHECK: } <"myPass">["foo", "foo2"]
  affine.if #set0(%2) {
  } loc(fused<"myPass">["foo", "foo2"])

  // CHECK: "foo.op"() : () -> () #test.custom_location<"foo.mlir" * 1234>
  "foo.op"() : () -> () loc(#test.custom_location<"foo.mlir" * 1234>)

  // CHECK: return %0 : i32 [unknown]
  return %1 : i32 loc(unknown)
}
