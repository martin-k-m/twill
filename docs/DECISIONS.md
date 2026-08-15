# Decisions

The choices that shaped twill, the alternative that was seriously considered in
each case, and why it lost. Each entry is a decision that could reasonably have
gone the other way. Where one of them costs something, the cost is stated rather
than argued away.

`docs/design.md` says what twill is. This says what it is not, and why.

---

## 1. Reverse-mode autodiff, not forward-mode

**Decision.** `grad` is reverse-mode. Every operation that can carry a gradient
records the inputs it read and a closure that pushes the output's gradient back
onto them (`track1`, `track2`, `trackN` in `internal/tensor/tensor.go`), and
`Backward` walks the graph in reverse topological order from a scalar output.

**Alternative.** Forward-mode, propagating a derivative alongside the value as a
dual number, which needs no graph, no closures and no reverse traversal. It is a
substantially smaller implementation, it allocates nothing beyond the value it
is already computing, and it composes trivially with itself for higher
derivatives.

**Why it lost.** Cost scales with the wrong quantity. Forward-mode gives you one
directional derivative per pass, so gradients of a function with `n` inputs cost
`n` passes; reverse-mode gives all `n` in one pass and costs one pass per output.
Machine learning is the case where `n` is a parameter count and the output is a
single loss, which is exactly the ratio forward mode is worst at. A model with
58,000 parameters, `examples/gpt.tw`, would need 58,000 forward passes per step.

Forward mode is not absent, though, because for second derivatives the ratio
flips: `hessian` runs forward-mode jets over the core operations
(`internal/tensor/jet.go`), since an `[n, n]` Hessian is `n` directional
derivatives either way and the forward version does not need a graph that
survives being differentiated twice. The two modes coexist for the two regimes
that each suits.

---

## 2. The tape is the tensor graph, not a separate structure

**Decision.** There is no tape object. The reverse graph is the output tensors
themselves: each `*Tensor` holds up to two parents inline (`p0`, `p1`), a slice
for the rare third and beyond (`pRest`, used only by einsum and concat), and a
`backward func()` closure. `Backward` does a depth-first walk over those parent
pointers to get a topological order, then calls the closures in reverse.
Crucially the wiring only happens when an input has `RequiresGrad`, so ordinary
evaluation does not build it and does not pay for it.

**Alternative.** An explicit tape: an append-only list of operation records that
a differentiation context owns, which is what PyTorch's autograd and most
teaching implementations use. It makes the graph a thing you can inspect, reset,
checkpoint or replay, and it separates the record of what happened from the
values it happened to.

**Why it lost.** A separate tape needs to be threaded through everything that
might record onto it: a context, a thread-local, or an extra argument on every
operation. twill's whole numeric surface is a package of free functions taking
`*Tensor`, and giving all of them a context argument to serve a feature that is
off most of the time was the wrong trade for an implementation whose stated goal
is to be readable in a sitting. Storing the two common parents inline rather
than in a slice also removes a per-operation allocation from the hot path.

**The cost, which is real.** Without a tape there is nothing to replay, so
`grad(grad(f))` cannot work and is refused rather than answered with a wrong
number. The graph is also kept alive by the output tensor, so a long-lived
result pins every intermediate that produced it; a training loop that
accumulates losses into a list without extracting their values holds the whole
history. And `jacobian` has to re-run the function once per output component
(`internal/interp/builtins.go`), because there is no recorded graph to seed
repeatedly with different cotangents. An explicit tape would make all three
cheap.

---

## 3. A static shape checker rather than runtime shape errors

**Decision.** `twill check` infers shapes across the whole program before
anything runs, and `twill run` refuses to start a program with a certain
mismatch. Shapes are part of a function's signature (`fn mm(A: [n, k], B: [k,
m]) -> [n, m]`), and the checker verifies call sites against them.

**Alternative.** Leave shapes to the runtime, as NumPy, PyTorch and JAX do. Every
operation knows its own shapes when it executes, so the error message can be
exact and can name real numbers rather than inferred ones, and the language has
one fewer phase, one fewer set of rules and no possibility of a false alarm.

**Why it lost.** The cost of a shape mistake is not the mistake, it is when you
find out. A runtime shape error in a training loop arrives after the data
loading, the initialisation and however many steps ran before the code path with
the bug was reached. That is the failure the language exists to remove, and it
cannot be removed by a better error message, only by an earlier one. The measured
price is in `docs/BENCHMARKS.md`: checking the whole 27,000-line corpus costs
about 114 ms, so the guarantee is cheap enough to run on save.

**The cost.** The checker is deliberately incomplete, and this is the compromise
entry. It reports only mismatches it is certain of and stays silent otherwise, so
it does not catch everything, and a clean check is not a proof that the program
runs. `docs/CORRECTNESS.md` gives a three-line program the checker accepts and
the runtime rejects. The alternative bias, reporting anything suspicious, was
rejected on the grounds that a checker which cries wolf gets turned off, and a
checker that is turned off catches nothing at all.

---

## 4. A tree-walking interpreter, not a compiler or a bytecode VM

**Decision.** `internal/interp` walks the AST directly. There is no intermediate
representation between `ast.Expr` and the tensor kernels: `evalBinary` on an
`ast.Binary` node dispatches straight into `tensor.Add` and friends.

**Alternative.** A bytecode VM, or lowering to native code. Both are the
conventional answer for an interpreted language that wants to be fast, and both
were on the roadmap in `docs/design.md`.

**Why it lost, so far.** Profiling said the interpreter is not the cost.
`docs/perf-baseline.md` reported `Interp.Run`, `Apply`, `callClosure`,
`evalBinary` and `evalCall` all at 0% flat time on a transformer forward pass,
and the profile in `docs/BENCHMARKS.md` reproduces the same result on the Monte
Carlo pricer. Time is in the kernels. twill is already a thin orchestration layer
over tensor operations in the way Python is over NumPy, so a faster dispatch
mechanism would speed up the part that costs nothing.

**What changing it would take, and when it would be worth it.** The regime where
this decision flips is small tensors, where per-operation overhead is a real
fraction of the work: a scalar loop, or a model narrow enough that each kernel
call is a few microseconds. The change is not the VM, it is the IR. Today an
operation is discovered, executed and forgotten in one step, so there is no
program object to optimise over, no place to fuse two elementwise operations
into one pass, and no way to allocate an output buffer once for a loop that runs
a thousand times. Getting an IR is the prerequisite for compilation, for fusion
and for the GPU work in `docs/CODEGEN.md`, and it is the same prerequisite in
all three cases.

---

## 5. Every float is an f64

**Decision.** A tensor's buffer is `[]float64` and always has been. Narrower
dtypes exist as semantics: a tensor carries a dtype tag, arithmetic promotes and
rounds to it, and `docs/dtypes.md` specifies the accumulation rules. But the
storage is f64 regardless, so a bf16 tensor is a `[]float64` whose every element
happens to be a value representable in bf16.

**Alternative.** Real packed storage, a union of typed buffers or a byte slice
plus a dtype, which is what every production framework does.

**Why it lost, for now.** Splitting the storage touches every kernel in the
package: each one becomes a dispatch over element type, and the readable
`for i := range out` loops become generic or duplicated. Separating the semantics
from the layout meant the rounding rules could be specified, tested and made
bit-identical against the self-hosted implementation first, on a representation
that keeps every kernel a single loop. Getting the numerics right on a simple
layout, then changing the layout, is a smaller step than doing both at once.

**The cost, and it is the largest one in this document.** Quantisation currently
shrinks nothing. `quantize` returns a genuinely packed `QTensor` or `QTensorI4`,
so that path is real, but a `bf16` tensor occupies exactly as much memory as an
f64 one, and `docs/perf-baseline.md` is blunt that a 7B model needs 56 GB in f64
and does not fit. Every f64 element is also eight bytes to move on a workload
that `docs/BENCHMARKS.md` shows is memory-bound, and half the SIMD lanes of an
f32. This is the single decision most responsible for the gap against PyTorch,
and it is tracked as NEEDS-111.

---

## 6. Determinism over peak throughput in the parallel kernels

**Decision.** Parallelism never changes a result. `parallelSum`
(`internal/tensor/parallel.go`) adds a slice in fixed 4096-element blocks and
combines the partials in block order, so the answer is the same on one core and
on sixteen. `runChunks` splits an output range into contiguous chunks whose
bodies write only their own indices.

**Alternative.** Let each worker accumulate its own partial and combine them in
completion order, or use an atomic accumulator. Both are faster: no fixed
blocking, no barrier on a specific ordering, less bookkeeping.

**Why it lost.** Floating-point addition is not associative, so a completion-
order reduction gives a different answer run to run, and a different answer on a
different machine. That breaks two things twill claims. The first is that a
seeded program reproduces exactly. The second is the self-hosting programme:
`src/tensor.tw` is checked against the Go kernels by comparing canonical float
renderings byte for byte, and a sum whose last bits move cannot be compared that
way at all. Commit `2a4b489` is what this looks like in practice, porting the
fixed block size to the self-hosted side because a plain running sum disagreed
with the bootstrap past 8192 elements.

**The cost.** The blocking is a constraint on how the reduction may be written,
and the ordering guarantee rules out the fastest form of every reduction in the
package.

---

## 7. No dependencies, in Go

**Decision.** The reference implementation is plain Go with no third-party
packages, including no BLAS. `mm` and `mmNT` in `internal/tensor/tensor.go` are
hand-written, with four accumulators over an unrolled inner product and cache
blocking sized to L2.

**Alternative.** Link a BLAS. OpenBLAS or Intel MKL would take the matmul from
the tens of GFLOP/s the hand-written kernel reaches to something near the
machine's peak, which is most of the gap against PyTorch on the contraction
benchmarks in `docs/BENCHMARKS.md`.

**Why it lost.** A BLAS means cgo, which means the single dependency-free binary
stops being single or dependency-free: a C toolchain to build, a platform matrix
to ship, and `go install` no longer being the whole install story. It also means
the numerics are no longer twill's, which collides directly with decision 6,
since a threaded BLAS reduction does not promise a reproducible last bit.

**The cost, stated plainly.** twill's matmul is several times slower than
PyTorch's, and `docs/BENCHMARKS.md` measures exactly how much. That gap is not a
missing optimisation in twill's kernel so much as the accumulated result of years
of hand-tuned assembly in the library twill declined to link.

---

## 8. Self-hosting before bootstrapping

**Decision.** The twill compiler written in twill (`src/`) runs on the Go
bootstrap and is checked against it, rather than being compiled to a standalone
binary. The output of the exercise is `docs/needs.md`, a numbered list of what
the language still lacks, each entry naming the file and line that reached for
it.

**Alternative.** Build the standalone compiler first, which is the version of
this project that produces a demo.

**Why it lost.** The interesting question was never whether a compiler could be
written, it was what subset of the language a compiler needs, and that question
is answered by writing the compiler and recording every wall it hits. Running on
the bootstrap keeps the reference available for differential comparison at every
step, which is what makes the exercise produce evidence rather than a second
implementation nobody can check. It found real defects: the lexer panic recorded
as NEEDS-33 and fixed in `d3176a9`, and the reduction-ordering divergence in
`2a4b489`, both documented in `docs/BUGS.md`.

**The cost.** twill is not self-hosted, and will not be until the entries in
`docs/needs.md` are implemented. The README says so.
