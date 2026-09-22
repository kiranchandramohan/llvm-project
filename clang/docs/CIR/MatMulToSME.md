# Experimental ClangIR MatMul to ArmSME pipeline

This experiment raises FP32 MatMul loop nests from structured CIR to linalg
and memref, tiles/vectorizes the matmul, and lowers it through ArmSME to LLVM.
Examples are in `clang/test/CIR/Lowering/matmul-to-sme.cpp` and
`clang/test/CIR/Lowering/matmul-loop.cpp`.
Enable it in the frontend with `-fclangir -fclangir-matmul-to-sme`, or use
`cir-opt` to inspect the intermediate stages. The default `clang -fclangir`
pipeline is unchanged. This is a narrow idiom recognizer for AArch64 LP64 input,
not general loop raising.

## Build and generate code

For normal compilation, including inline definitions in headers:

```sh
bin/clang++ -O3 -fclangir -fclangir-matmul-to-sme \
  -ffp-contract=fast -march=armv9-a+sme -S input.cpp -o output.s
```

Recognition runs on structured CIR before LLVM optimization and inlining.
It matches each eligible loop nest independently of its enclosing function's
signature, argument order/count, return type, and unrelated surrounding code.
The original function ABI, visibility and ODR linkage are retained. Multiple
eligible nests in one function are supported. Each matched scope is outlined
into a private dispatcher, scalar fallback, and locally streaming SME helper.
The dispatcher may subsequently inline into its caller; the SME helper stays
separate. A loop already inlined into a larger structured CIR function can also
match if it retains the supported loop/dataflow form. Unrelated functions,
globals, initializers and calls follow normal CIR lowering.

Pass options are space-separated, for example:
`-fclangir-matmul-options='allow-contract assume-no-alias'`.
`require-kernel` diagnoses a translation unit with no eligible kernel, which is
useful when checking a particular source. Without it, unmatched or
uncontractable functions keep their original code.

Enable `clang` and `mlir` in `LLVM_ENABLE_PROJECTS`, and enable
`CLANG_ENABLE_CIR`. From the build directory:

```sh
cmake -S ../llvm -B . -DCLANG_ENABLE_CIR=ON
ninja clang cir-opt mlir-translate llc
mkdir -p clangir-sme

bin/clang --target=aarch64-linux-gnu -fclangir -emit-cir -O1 \
  -ffp-contract=fast ../clang/test/CIR/Lowering/matmul-to-sme.cpp \
  -o clangir-sme/matmul.cir

bin/cir-opt clangir-sme/matmul.cir \
  --cir-raise-matmul \
  -o clangir-sme/matmul.linalg.mlir
bin/cir-opt clangir-sme/matmul.linalg.mlir --cir-matmul-to-sme \
  -o clangir-sme/matmul.sme.mlir
bin/cir-opt clangir-sme/matmul.sme.mlir --cir-sme-to-llvm \
  -o clangir-sme/matmul.llvm.mlir
bin/mlir-translate --mlir-to-llvmir clangir-sme/matmul.llvm.mlir \
  -o clangir-sme/matmul.ll
bin/llc -mtriple=aarch64-linux-gnu -mattr=+sme -O3 \
  clangir-sme/matmul.ll -o clangir-sme/matmul.s
bin/llc -mtriple=aarch64-linux-gnu -mattr=+sme -O3 -filetype=obj \
  clangir-sme/matmul.ll -o clangir-sme/matmul.o
```

The three `cir-opt` stages can also be specified in order in one invocation.
The result uses `fmopa za0.s` and the locally streaming, new-ZA ABI. Link it
with an SME-capable AArch64 runtime and execute on SME hardware or an emulator.
There is no runtime CPU-feature dispatch. The runtime must provide SME ABI
support routines such as `__arm_get_current_vg`. Select a compiler-rt runtime
with AArch64 SME support using `--rtlib=compiler-rt`.

## Semantic contract and supported input

By default, the pass inserts runtime alias checks. Each loop dispatcher runs
outside streaming mode. It computes conservative byte intervals for A, B, and C,
accounting for the transpose flags and signed leading dimensions. It calls a private SME helper when C is disjoint from both
inputs, and otherwise calls a private copy of the original loop nest lowered
through CIR's normal LLVM pipeline. A and B may overlap each other.

The checks use i128 arithmetic to avoid address-range overflow. Intervals that
cross zero or the untagged 56-bit address limit select the fallback; tagged
pointers therefore conservatively select the fallback. Padding can also cause
a conservative fallback. For nonpositive M or N the dispatcher returns without
accessing memory; for nonpositive K no input alias check is necessary. The
dispatcher bounds total scratch size using i128 arithmetic, allocates one
workspace before calling the SME helper, and checks the result. Size overflow
or allocation failure calls the original scalar function before any matrix
access. The successful path releases the workspace after the helper returns.
The fallback retains the original access order and floating-point permissions,
even when the pass override is supplied.

SME's floating-point outer product uses fused multiply-add. The pass therefore
requires permission on the reduction's operations, or an explicit override:

* `-ffp-contract=fast` records `cir.contract` on individual `cir.fmul` and
  `cir.fadd` instructions. Both members of the reduction pair must permit
  contraction. The normal CIR-to-LLVM path also preserves these as LLVM
  `contract` flags.
* `-ffp-contract=on` represents statement-local contraction as `cir.fmuladd`,
  which the matcher also accepts.
* `--cir-raise-matmul=allow-contract` explicitly permits fused multiply-add in
  the optimized reduction and alpha/beta epilogue, including input compiled
  with `-ffp-contract=off`. Without this option or instruction permission, the
  pass leaves the function unchanged (`require-kernel` diagnoses it).

The alpha/beta epilogue uses FMA only when its own multiply/add pair permits
contraction, or when `allow-contract` is set. This permission can change
rounding; it does not authorize arbitrary reassociation. Strict floating-point
environment operations remain unsupported even with the override.

`--cir-raise-matmul=assume-no-alias` skips the input/output alias checks.
Allocation checks and the scalar fallback remain. This is an explicit promise
that C does not overlap A or B. The options can be combined as `--cir-raise-matmul='assume-no-alias allow-contract'`.

The matcher infers matrix pointers, dimensions, strides, transpose flags and
scales from the loop dataflow. Function/variable names and parameter positions
are irrelevant; unused parameters may be added or removed. Bounds and strides
may share values. Constants, SSA live-ins, and initialized nonescaping locals
that remain unchanged throughout the nest can supply the inferred inputs.
Computed local values are supported when evaluated before the matched scope.
The original live-ins are captured separately for the scalar fallback.

The supported form is an i/j/k nest of signed 32-bit, zero-based, unit-step
loops with invariant bounds, an FP32 accumulator initialized to positive zero,
and `sum += A[...] * B[...]`. Matrix accesses use `row * stride + column`;
the input layouts may be fixed or selected by invariant boolean transpose
flags. The output may be `sum`, `alpha * sum`, or either of these plus C or
`beta * C`. A beta-zero guard is optional: an unconditional read of C remains
unconditional, including when beta is zero and C contains NaN. Omitted scales
do not introduce extra floating-point operations.

The complete matched scope is checked for additional effects and unsupported
control flow. Volatile/atomic accesses, strict FP, wrapping/saturating index
arithmetic, loop-dependent inputs, escaped input locals, and induction variables
whose values escape the matched scope are skipped. Snapshot values are checked
against writes and loop boundaries; a value saved before the reduction cannot
be substituted with its final value. These checks remain necessary even though
the enclosing function's signature no longer restricts recognition.

The generated code handles dynamic transpose flags, rectangular dimensions,
signed leading dimensions, masked tile edges, and nonpositive dimensions.
For nonpositive M or N it does not access the matrices. For nonpositive K it
uses a zero reduction and still applies the epilogue. When the source includes
a beta-zero guard and beta compares equal to zero, including negative zero,
C is not read. The alpha multiplication occurs after the reduction, and the
output epilogue retains i/j order.

## Lowering strategy and limits

1. Normalize A to a contiguous K-by-M memref and B to a contiguous K-by-N
   memref. Allocate and zero a contiguous M-by-N accumulation memref.
2. Emit `linalg.matmul` with transposed-A indexing maps. This allows both
   operands of each outer product to use contiguous vector loads.
3. Tile M and N by `4 * vscale`, with K steps of one, and vectorize to
   `vector<[4]x[4]xf32>`. Use explicit masks for incomplete edge tiles.
4. Form vector outer products and carry the accumulator through the K loop.
5. Convert to ArmSME, enable locally streaming mode and new ZA state, allocate
   tiles, and lower the remaining dialects to LLVM. Run these stages only on
   the generated helper and dispatcher; preserve unrelated LLVM function
   bodies through normal CIR lowering.
6. Apply alpha/beta to the completed reduction and release all scratch buffers.

This is a correctness-oriented prototype. Full-matrix packing needs
`4 * (M*K + K*N + M*N)` bytes of scratch memory (clamping dimensions at zero),
in addition to the original matrices. Allocation failure selects the original
scalar kernel. It has no cost model, cache blocking, packing reuse, or
profitability threshold, and no performance claim is made. It is not ready to
enable by default.

## Validation

```sh
bin/llvm-lit -v ../clang/test/CIR/Lowering/matmul-to-sme.cpp \
  ../clang/test/CIR/Lowering/matmul-loop.cpp \
  ../clang/test/CIR/CodeGen/fp-contract-fast.c
```

The regression covers the source-to-linalg-to-SME-to-LLVM path, runtime checks,
nonstreaming dispatch and fallback, streaming helper attributes, reduction and
epilogue FMA permissions, source pragmas that disable one member of a pair,
absence of a duplicate scalar matmul after vectorization, renamed kernels,
shared bounds, and rejection of incompatible indexing, arithmetic, volatile
accesses, constrained FP, additional stores, and stale accumulator snapshots. It also
checks mixed modules, inline callers, global constructors and hidden visibility.
`matmul-loop.cpp` covers reordered/removed arguments, non-void returns,
surrounding effects, multiple nests, constant dimensions, fixed layouts,
omitted scales, and unconditional output reads.
`CIR/CodeGen/fp-contract-fast.c` tests per-instruction permission emission and
its normal LLVM lowering.

To test the raising semantics on a native AArch64 host without SME, omit the
vector/SME stage and lower the same linalg payload to scalar LLVM:

```sh
bin/cir-opt clangir-sme/matmul.linalg.mlir --cir-sme-to-llvm \
  -o clangir-sme/matmul.scalar.mlir
bin/mlir-translate --mlir-to-llvmir clangir-sme/matmul.scalar.mlir \
  -o clangir-sme/matmul.scalar.ll
bin/llc -filetype=obj clangir-sme/matmul.scalar.ll \
  -o clangir-sme/matmul.scalar.o
bin/clang++ -O2 -ffp-contract=off -Dmatmul=matmul_reference \
  -c ../clang/test/CIR/Lowering/matmul-to-sme.cpp \
  -o clangir-sme/reference.o
bin/clang++ -O2 ../clang/test/CIR/Inputs/matmul-check.cpp \
  clangir-sme/reference.o clangir-sme/matmul.scalar.o \
  -o clangir-sme/check-matmul
clangir-sme/check-matmul
```

The differential harness checks all transpose combinations, rectangular/tail
sizes, padded/signed/zero strides, A/B aliasing, overlapping C rows,
alpha/beta values, zero and negative K, NaN-filled C
when beta is zero, untouched padding, and empty-output null-pointer cases.
For C/input overlap testing, generate a guarded variant from
`-ffp-contract=off` input with `--cir-raise-matmul=allow-contract`, and lower it
as above to `matmul.guarded.scalar.o`. Link the same reference with the alias
harness:

```sh
bin/clang++ -O2 ../clang/test/CIR/Inputs/matmul-alias-check.cpp \
  clangir-sme/reference.o clangir-sme/matmul.guarded.scalar.o \
  -o clangir-sme/check-alias
clangir-sme/check-alias
```

This adds 3,888 overlapping/disjoint cases over all transpose combinations and
positive, negative, and zero strides. Exact C/A and C/B base aliases include
864 bitwise checks of the unmodified fallback compiled with contraction off.

Replacing the scalar object with its SME equivalent tests the actual SME
kernel on a suitable machine. Scalar execution validates raising and dispatch,
but does not validate SME execution or its fused reduction. Both SME assembly
and objects can be generated on a host without SME; executing the optimized
path still requires SME hardware or an emulator.

Allocation failure can be tested without SME using either scalar-lowered object
(the guarded and `assume-no-alias` variants both retain this check):

```sh
bin/clang++ -O2 ../clang/test/CIR/Inputs/matmul-allocation-check.cpp \
  clangir-sme/matmul.scalar.o -Wl,--wrap=malloc,--wrap=free \
  -o clangir-sme/check-allocation
clangir-sme/check-allocation
```

This interposes the workspace allocator and checks successful allocation/free,
a null allocation falling back to the correct result, empty output, and K=0.

The `CIR/Inputs/matmul-loop-check.cpp` harness compares the new loop forms with
`CIR/Lowering/matmul-loop.cpp` compiled normally with `-DREFERENCE`. Compile a
second copy through CIR with `allow-contract`, omit the vector/SME stage as
above, and link both objects with the harness and
`-Wl,--wrap=malloc,--wrap=free`. This checks reordered arguments, surrounding
side effects and return values, multiple nests, fixed layouts, NaN-filled
output with an unconditional read, overlapping matrices, and allocation failure.
