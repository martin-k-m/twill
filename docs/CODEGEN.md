# Codegen: from tree walking to emitted GPU kernels

A design, not an implementation. No codegen code exists and none is written in
this pass.

## What this document is, and what it is not

`docs/gpu-feasibility.md` measured whether a GPU backend is worth building and
answered not yet. `docs/gpu.md` designed the backend that would exist when the
answer changes: which API, how buffers become resident, how the tape holds
device tensors, which kernels get written. Both are about the *runtime*, and both
assume the shape twill has today, which is one kernel per operation, dispatched
as the interpreter reaches it.

This document is about the *compiler*, and it exists because op-at-a-time
dispatch cannot pay for itself on this codebase. The number that decides it is
from `docs/gpu-feasibility.md`: the round trip costs roughly 80 microseconds per
operation and does not shrink with problem size. twill's own headline program,
`examples/montecarlo_option.tw`, is about eight elementwise operations over
200,000 elements followed by a reduction. Dispatched one at a time that is
roughly 640 microseconds of pure boundary crossing, against a measured 3.450 ms
for the whole thing on the CPU today (`docs/BENCHMARKS.md`). A backend that
launches per operation spends a fifth of the current runtime on overhead before
computing anything, and that is before the transfers.

The way out is to launch once for the whole chain, and launching once for a
chain means knowing what the chain is before running it. twill does not know
that today. Getting to know it is the subject of this document.

Read `docs/gpu.md` for the runtime. This is the layer above it.

---

## 1. What twill actually looks like today

Everything below is a consequence of these five facts, all of which were read out
of the code rather than assumed.

**There is no intermediate representation.** `internal/interp/interp.go` walks
`ast.Expr` directly. `evalBinary` takes an `*ast.Binary`, evaluates both sides
to `value.Value`, and switches on the operator string into a `tensor` package
function. An operation is discovered, executed and forgotten in one step. There
is no object representing the program's computation, which means there is
currently nothing for a compiler to consume, nothing to fuse over, and nothing
to cache a compiled kernel against. **Introducing an IR is the whole of phase one
and most of the risk.**

**A value is `any`.** `value.Value` is an empty interface; the numeric case is
`*tensor.Tensor` and the others are `value.Num`, `*value.List`, `*value.Record`,
`*value.Closure` and the rest. So the compiler cannot assume its inputs are
tensors, and the boundary between the compiled and interpreted worlds is a type
switch rather than a static property.

**A tensor is a flat `[]float64` plus a `[]int` shape, always.** From
`internal/tensor/tensor.go`. Narrow dtypes are a tag (`dtp`) and a set of
rounding rules (`docs/dtypes.md`); the storage is f64 whatever the tag says.
This matters enormously here: `docs/gpu-feasibility.md` measured 341.7 GFLOP/s
f64 against 18,010 GFLOP/s f32 on the RTX 5070, a ratio of 52.7 to 1. **Compiling
f64 kernels targets 5% of the card.** The packed layout is NEEDS-111 and it is a
hard prerequisite, not an optimisation to follow later.

**The autodiff tape is the tensor graph.** There is no tape object. Each output
`*Tensor` holds its parents inline (`p0`, `p1`, and `pRest` for einsum and
concat) and a `backward func()` closure, wired by `track1`/`track2`/`trackN` and
only when an input has `RequiresGrad`. `Backward()` walks those pointers
depth-first for a topological order and calls the closures in reverse. **A fused
forward kernel therefore destroys the backward pass unless the fusion emits a
matching fused backward kernel**, because the per-operation closures that
currently carry the derivatives will not exist. This is the hardest part of the
design and section 5 is about it.

**Parallelism is deterministic by construction.** `parallelFor` and `runChunks`
in `internal/tensor/parallel.go` split an output range into contiguous chunks
whose bodies write only their own indices, and `parallelSum` adds in fixed
4096-element blocks combined in block order, so the answer does not depend on
the core count. `docs/DECISIONS.md` entry 6 explains why this is load-bearing
and `docs/gpu.md` section 5 carries it onto the device. **The compiler inherits
the constraint: no emitted kernel may reorder a reduction.**

---

## 2. The IR

One IR, sitting between the AST and the kernels, built by the interpreter as it
evaluates rather than by a separate pass.

### Tracing, not parsing

The obvious design is a lowering pass from `ast` to the IR. It is the wrong one
here, because twill's shapes are not all statically known. The checker is
best-effort by design (`docs/DECISIONS.md` entry 3): it types `zeros(2, 3)`
precisely and `zeros(2, len(xs))` as unknown, and `docs/CORRECTNESS.md` shows a
six-line program whose shapes it cannot determine. A compiler that needs static
shapes to emit a kernel would refuse a large fraction of real programs.

So the IR is built by tracing. The interpreter runs as it does now, but instead
of calling `tensor.Add` immediately, a traced operation appends a node to a
buffer and returns a placeholder tensor carrying its shape. Shapes are concrete
because the operands are concrete values, not inferred types. When the trace has
to be forced, it is compiled and run.

This is how JAX gets its jaxprs and it is the right shape for twill for the same
reason: the language is dynamic, the values are not.

### The node

A trace node is deliberately narrow:

```
Node {
    Op      OpCode      // Add, Mul, Exp, Relu, MatMul, Sum, SumAxis, ...
    Inputs  []NodeID    // indices into the trace, or NegN for a captured buffer
    Shape   []int       // concrete, from the operand values
    DType   DType       // the existing tensor.DType
    Attrs   [2]int      // axis, k, and other small integer parameters
}
```

`OpCode` enumerates exactly the operators in `internal/tensor`, and the list is
not invented here: `TestGradientCheckCoversEveryOperator` in
`internal/tensor/fullgradcheck_test.go` already parses the package's source and
enumerates every exported operator, so the same list can generate the opcode
enum and a missing entry becomes a build failure rather than a silent fallback.

### Forcing

A trace is forced when a value escapes it: when a tensor's data is read by
`print`, by an `if` condition, by a comparison, by `save`, by any builtin with no
opcode, or when the traced program calls a closure the tracer cannot follow.
Forcing compiles the trace, runs it, and replaces the placeholders with real
tensors.

Control flow is the boundary. twill's `for` and `while` are interpreter
constructs over `value.Value`, and the tracer does not attempt to capture them:
a loop body traces, forces at the end of each iteration, and the next iteration
traces again against the same cache key. That is exactly what makes the compiled
kernel cache pay, since a training loop runs the same trace thousands of times.

---

## 3. The fusion strategy

Fusion is the point of this design, not a refinement of it. The unit of fusion
is the largest region of the trace that can be computed by one kernel launch.

### The three classes

Every operator in `internal/tensor` falls into one of three classes, and the
class determines how it fuses.

**Elementwise.** Every operation built on `broadcastBinary` or `unary`:
add, sub, mul, div, mod, neg, square, exp, log, sqrt, sin, cos, tanh, sigmoid,
relu, pow, clip, maximum, minimum, where, and the comparisons. Each output
element depends on one element of each input (after broadcasting). **Any
connected chain of these fuses into a single kernel with one loop over the output
index.** This is where the win is: the Monte Carlo pricer's whole forward pass
up to the reduction is one such chain.

**Reductions and scans.** sum, mean, max, min, prod, median, their axis
variants, softmax, logsumexp, cumsum, cumprod, cummax, cummin. **An elementwise
chain fuses *into* the producer side of a reduction** (compute the element, then
accumulate it, never materialising the intermediate), which is what turns
`mean(relu(ST - K))` into one pass. A reduction does not fuse into another
reduction, and nothing fuses across a reduction's output back into more
elementwise work in the same kernel, because the reduction needs a barrier.
Softmax and logsumexp are internally two reductions and a broadcast and get
hand-written fused kernels rather than being decomposed.

**Contractions and structural operators.** matmul, matmulNT, einsum, conv2d,
maxpool2d, gather, concat, split, reshape, transpose, broadcast_to, flip, roll,
diff, slice, sort, topk. These get their own kernels. A contraction is a tuned
kernel and fusing arbitrary work into it destroys the tuning; what fuses is the
*epilogue*, one elementwise chain applied to the contraction's output before it
is written, which is where `relu(linear(x, W) + b)` collapses from three kernels
to one. Reshape, transpose and broadcast_to are pure index arithmetic and fuse
into a consumer as an index remapping rather than as a kernel at all, which is
worth having on its own evidence: `docs/perf-baseline.md` measured
`TransposePerm` at 14% of a forward pass, all of it spent materialising copies.

### The algorithm

Walk the trace in order and grow regions greedily. A node joins the current
region when its class permits (elementwise into elementwise, elementwise into a
reduction's producer, elementwise into a contraction's epilogue), when its shape
is broadcast-compatible with the region's output shape, and when it has no
consumer outside the region. That last condition is the one that matters: a value
consumed twice must be materialised, or the fused kernel computes it twice.
Recomputing is often cheaper than a round trip to memory, so the rule is a cost
comparison and not a prohibition, but the first implementation should
materialise, because it is the version that is obviously correct.

Greedy is the right first algorithm. It is what XLA's fusion started as, the
regions it finds on the workloads in `bench/workloads` are the ones a person
would pick by hand, and it can be replaced without touching anything else.

---

## 4. Memory layout

### The packed buffer comes first

Restating the prerequisite because the design does not work without it. Today a
tensor is `[]float64` regardless of its dtype tag. Three consequences:

- f64 on the target card is 5% of its throughput
  (`docs/gpu-feasibility.md`), so a compiler emitting f64 kernels is optimising
  the wrong 5%.
- Every element is eight bytes across the PCIe boundary, doubling the transfer
  cost the fusion exists to amortise.
- `docs/BENCHMARKS.md` shows the elementwise workloads are bandwidth-bound on the
  CPU already, so f64 is costing throughput before any device is involved.

NEEDS-111, the packed layout, is therefore not a parallel workstream. It is
phase zero.

### Layout inside a fused region

Row-major, contiguous, matching `strides()` in `internal/tensor/tensor.go` and
the existing kernels. No layout assignment pass and no tiling choices in the
first version: the fused kernel walks the output in flat index order, and each
input is read through the effective strides `effStrides` already computes, which
is where broadcasting becomes free rather than a materialised expansion.

### Buffers, not tensors

`docs/gpu.md` section 3 already makes residency a property of a buffer rather
than of a tensor, and the compiler adopts that unchanged. The addition here is
that a fused region allocates output buffers for its *region* outputs only.
Intermediates inside a region never get a buffer at all, which is the memory win
that comes free with the compute win: the Monte Carlo chain currently allocates
roughly eight 200,000-element f64 buffers, 12.8 MB of traffic, and fused it
allocates one.

---

## 5. Autodiff through a fused kernel

This is the part where a design that ignored twill's actual structure would
quietly fall apart, so it is treated first among the hard problems.

The difficulty stated precisely: today the derivative of an operation lives in a
Go closure created at the moment the operation ran, capturing the operand slices
it needs (`track2` in `internal/tensor/tensor.go`, and the `da`/`db` functions
passed to `broadcastBinary`). Fusing eight operations into one kernel means those
eight closures are never created. Something has to supply the derivative.

**The answer is to differentiate the trace, not the kernel.** The trace is a
straight-line dataflow graph with concrete shapes, which is exactly the input a
source-to-source reverse-mode transform wants. So:

1. Trace the forward computation as in section 2.
2. Transform the trace into a backward trace by walking it in reverse and
   emitting, for each node, the IR nodes for its vector-Jacobian product. These
   are the same formulas the existing `da`/`db`/`backward` closures implement, so
   the transform is a transcription of code that already exists and is already
   gradient-checked, not new mathematics.
3. Fuse and compile the backward trace with the same fusion pass as the forward
   one. A chain of elementwise VJPs is itself an elementwise chain, so the
   backward pass of the Monte Carlo pricer fuses into roughly one kernel too.

Two things fall out of this that are worth stating.

**It has to fuse or it is not worth doing.** An unfused backward pass over a
fused forward pass would launch once per node and pay the 80 microseconds per
node the fusion just eliminated.

**Saved intermediates become an explicit decision.** `relu`'s backward needs the
forward input, `exp`'s needs its own output, `div`'s needs both operands. Today
the closure captures whatever it needs and the garbage collector sorts it out.
In a fused kernel an intermediate may not exist, so the transform has to decide
per value between saving it (a buffer, memory traffic) and recomputing it in the
backward kernel (arithmetic, no traffic). The first implementation should save
everything a VJP references, because it is obviously correct and matches what the
interpreter does today; recompute is an optimisation with a measurable
before-and-after.

**`hessian` does not compile in the first version.** It runs forward-mode jets
(`internal/tensor/jet.go`) through a separate `recordJets` path with its own
per-node closures, and second-order over a fused kernel is a much larger design.
Traces containing a `hessian` call force to the interpreter.

---

## 6. What compiles, and what does not

### Compiles

- Every operator in the elementwise, reduction/scan, and
  contraction/structural classes of section 3. That is the whole of
  `internal/tensor`'s differentiable surface, which
  `TestGradientCheckCoversEveryOperator` enumerates.
- Straight-line numeric code: `let` bindings, arithmetic, builtin calls,
  function calls the tracer can inline.
- `grad`, `grads` and `value_and_grad`, through the trace transform of section 5.
- Loop bodies, one iteration at a time, with the compiled trace cached and
  reused across iterations. This is the case that matters, because it is what a
  training loop is.

### Does not compile, and forces to the interpreter

Each of these forces rather than fails, so a program that mixes them still runs.
The design's correctness does not depend on the list being short.

- **Data-dependent control flow.** `if` on a computed tensor value, `while` on a
  computed condition, and `for` over a computed range. The tracer reaches the
  condition, needs the value, and forces. Capturing control flow into the IR is
  the natural second version and is not attempted here.
- **`hessian` and the forward-mode jet path**, for the reason in section 5.
- **Non-numeric values.** Records, lists, dicts, strings, bytes, closures,
  variants. A record of weights is unpacked by the optimiser into tensors before
  any arithmetic happens, so this costs less than it sounds like, but the tracer
  follows tensors and nothing else.
- **`mode systems` entirely.** The systems dialect is `I64`, byte strings,
  arrays, dicts, structs and file IO, and by design a scalar there is a machine
  word and not a rank-0 tensor (`docs/design.md`). There are no tensors to fuse.
- **IO, `print`, `save`, `load`, `read_csv`, `read_frame`.** All force.
- **The gradient-boosted trees in `internal/gbm`.** A separate engine with its
  own data structures and no tensor operations to fuse.
- **`einsum` with more than two operands**, in the first version. Today's
  `einsum` takes any number, but one and two operands cover every use in `std/`,
  including both contractions of multi-head attention in `std/nn.tw`. The
  general case needs a contraction-order decision, which is a real optimisation
  problem and belongs in a later version; until then a three-operand `einsum`,
  as in `examples/einsum.tw`, forces to the interpreter.

---

## 7. How correctness would be verified

The interpreter is the reference semantics. `docs/design.md` says so, and the
verification strategy is the direct consequence: **the compiler is correct when
it agrees with the interpreter, and that is a testable proposition rather than a
review criterion.**

### 7.1 Differential testing over generated programs

The machinery for this already exists twice in the repository and neither piece
needs to be invented.

`internal/checker/soundness_test.go` generates random twill programs from a small
grammar and runs both the checker and the interpreter over 4,000 of them, and
`internal/tensor/fullgradcheck_test.go` holds `gradCases()`, a table of 101 cases
covering every differentiable operator with inputs already chosen to sit away
from the kinks where finite differences are meaningless.

The compiler's differential harness is those two put together:

1. Generate a random trace by composing operators from the `gradCases` table,
   with random but shape-compatible operands. The generator is the one in
   `soundness_test.go` extended past its current six forms.
2. Evaluate it with the interpreter.
3. Evaluate it with the compiler, at every fusion setting: fusion off, greedy
   fusion, and one region per operator.
4. Compare.

The comparison is where the work is. Bit-exactness is the right bar for anything
that does not reorder arithmetic, and most of the compiler does not: an
elementwise fusion computes exactly the same operations in exactly the same order
on exactly the same values, so `exp(x) * y` fused must equal `exp(x) * y` unfused
to the last bit, and anything else is a bug. Reductions are the exception, and
`docs/DECISIONS.md` entry 6 already fixes the rule: the emitted reduction must
use the same fixed 4096-element blocking as `parallelSum`, in which case it too
is bit-exact. **The bar is bit-exactness everywhere except transcendentals**,
where a device's `exp` differs from Go's `math.Exp` in the last bits and the
comparison is a tolerance, sited on the ULP difference the device actually shows
rather than on a round number.

### 7.2 Gradient checking the compiled backward pass

The forward comparison above says the compiler computes the right value. It says
nothing about the derivative, and section 5 is where a fused implementation is
most likely to be wrong.

So the gradient-check harness runs a second time against the compiler.
`runCase` in `fullgradcheck_test.go` compares reverse-mode against a
Richardson-extrapolated central difference at 1e-7 relative tolerance; the
compiled version compares the *compiled* reverse-mode against the same finite
differences, over the same 101 cases, at the same tolerance. This is the test
that would catch a wrong VJP transcription, a saved intermediate that was
recomputed incorrectly, or an accumulation into the wrong buffer, and it costs
nothing to build because the harness is written and the cases are chosen.

A third comparison is stronger still and nearly free: compiled gradient against
interpreted gradient, bit-exact under the same rule as 7.1. Finite differences
agree to 1e-11; the interpreter agrees to the last bit, so it is the sharper
oracle wherever it applies.

### 7.3 The corpus

`TestExamplesRunClean` already runs every program in `examples/` and asserts a
clean check and a clean run. The compiled version asserts the stronger property:
every example produces byte-identical output under the compiler and under the
interpreter. The self-hosting work already established that comparing canonical
float renderings byte for byte is a workable oracle at corpus scale, and the
harness under `tools/diff/` exists to do it.

### 7.4 What this does not verify

The differential tests compare the compiler against the interpreter. If the
interpreter is wrong, they agree and both are wrong. The independent check on the
interpreter is the finite-difference gradient check in 7.2 and the Black-Scholes
closed form the Monte Carlo example is measured against, and neither is affected
by the compiler existing.

---

## 8. The benchmark that would prove it worked

One primary benchmark, chosen because it is the program twill leads with, its
correct answer is known in closed form, and it is exactly the shape fusion is for.

### The benchmark

`bench/workloads/mc_option_grad.tw`: the Monte Carlo European call from the
README, differentiated for delta and vega, 200,000 paths. Measured by the
existing `bench/cmd/twillbench` harness, unchanged, median and p99 over 30 runs
after 5 warmups.

It is the right benchmark for four reasons. The forward pass is one elementwise
chain into a reduction, so it exercises the whole of section 3 with nothing else
mixed in. The backward pass is a second such chain, so it exercises section 5.
The result is checkable against Black-Scholes, so a fast wrong answer is caught.
And it is already measured, so the before number exists and was not chosen after
seeing the after.

### The baseline, measured

From `docs/BENCHMARKS.md`, on the machine described there:

| | twill today, CPU, best of a GOMAXPROCS sweep |
|---|---|
| `mc_option_fwd` | 3.450 ms median, 4.410 ms p99 |
| `mc_option_grad` | 13.646 ms median, 16.443 ms p99 |

`docs/BENCHMARKS.md` section 7 is the caveat that goes with these: the absolute
milliseconds carry about 40% of thermal drift on this machine, so the thresholds
below are stated as ratios against a baseline re-measured in the same session as
the compiled version, not against these figures read off the page months later.

### What would count as success

Three thresholds, in increasing order of ambition. They are stated before the
work rather than after it, which is the only time such a number means anything.

**The floor, below which the project failed.** `mc_option_grad` no slower than
the interpreted baseline re-measured alongside it, with bit-exact
agreement on the forward value and the gradients agreeing with the interpreter
under 7.2. A GPU backend that is slower than the CPU is the outcome
`docs/gpu-feasibility.md` measured for op-at-a-time dispatch at these sizes, and
avoiding it is the entire justification for building a compiler rather than a
backend.

**Success: 5x on the differentiated workload.** `mc_option_grad` at or under a
fifth of the baseline, which against the 13.646 ms measured here is 2.73 ms.
This is the threshold to design toward. The reasoning behind the
number, stated so it can be argued with: `docs/gpu-feasibility.md` measured about
9x for f32 matmul with transfers included and about 15x resident, and the
elementwise chain here is bandwidth-bound rather than compute-bound, so it should
do better than the matmul figure once fused; against that, the packed f32 layout
gives 2x on bytes moved at best and the reduction does not parallelise as freely
as the elementwise part. 5x is deliberately below the optimistic estimate.

**The result that would justify the dependency.** `mc_option_grad` at a tenth of
the baseline, 1.36 ms against the 13.646 ms measured here, *and*
`elementwise_10000000` at a tenth of its own, 8.0 ms against 80.441 ms. The second is there because a design that only
wins on one hand-picked program has not been shown to generalise, and the large
elementwise workload is the one whose result the fusion strategy predicts most
directly.

### The result that would sink it

Stated because a design document that cannot fail is not a design document.

If fused `mc_option_grad` lands between the floor and 2x, the compiler is not
worth its cost. The cost is real and is enumerated in
`docs/gpu-feasibility.md` and `docs/DECISIONS.md` entry 7: a GPU dependency, the
end of the single dependency-free binary, a build matrix, and a second numeric
implementation to keep bit-exact with the first. A 2x does not buy that, and the
honest response would be to record the measurement and stop, exactly as
`docs/gpu-feasibility.md` did for the backend.

There is also a cheaper experiment that must run first, because it would change
the answer. The same tracing IR and the same fusion pass, emitting **CPU**
kernels, needs no dependency at all and captures the allocation and memory-traffic
win without the 80-microsecond boundary. `docs/BENCHMARKS.md` shows the
elementwise workloads are bandwidth-bound and that intermediate buffers are a
measurable share of the time, so a fused CPU backend should show a real gain on
its own. If it does, it is the correct next step regardless of whether the GPU
work ever happens, and phases 1 through 3 below deliver it.

---

## 9. Order of work

Each phase is independently useful and independently abandonable.

0. **NEEDS-111, the packed buffer layout.** Prerequisite. Without it the target
   is f64 and the target is worth 5% of the card.
1. **The trace IR and forcing**, with no fusion and no codegen: every node
   dispatches to the existing `internal/tensor` function. Correct when it is
   bit-identical to the interpreter across `examples/`, which is a strong test of
   the tracer alone, before any kernel exists to be blamed.
2. **The trace transform for reverse mode.** Correct when the gradient-check
   harness of 7.2 passes against traced execution.
3. **Greedy fusion, emitting CPU kernels.** The first phase with a number
   attached, and the one that decides whether the GPU phase is worth starting.
4. **The device backend**, as designed in `docs/gpu.md`, with the fused regions
   of phase 3 as its unit of dispatch rather than individual operations.
5. **Measure against section 8, and publish the result whichever way it goes.**
