# Exercises: answers

Answers to [EXERCISES.md](EXERCISES.md). Sources named so every claim is
checkable.

---

## Part 1

**1. Two modes.**

Forward mode gives one directional derivative per pass, so gradients of a function
with `n` inputs cost `n` passes. Reverse mode gives all `n` in one pass and costs one
pass per output. Machine learning is the case where `n` is a parameter count and the
output is a single loss, which is exactly the ratio forward mode is worst at:
`examples/gpt.tw` has 58,000 parameters and would need 58,000 forward passes per
step.

For second derivatives the ratio flips. An `[n, n]` Hessian is `n` directional
derivatives either way, and the forward version does not need a graph that survives
being differentiated twice. So `hessian` runs forward-mode jets over the core
operations (`internal/tensor/jet.go`). The two modes coexist because there are two
regimes and each suits one.

**2. No tape.**

The reverse graph **is** the output tensors. Each `*Tensor` holds up to two parents
inline (`p0`, `p1`), a slice for the rare third and beyond (`pRest`, used only by
einsum and concat), and a `backward func()` closure. `Backward` does a depth-first
walk over the parent pointers to get a topological order, then calls the closures in
reverse.

The wiring only happens when an input has `RequiresGrad`, so ordinary evaluation does
not build the graph and does not pay for it. Storing the two common parents inline
rather than in a slice also removes a per-operation allocation from the hot path.

Three costs, all real:

- `grad(grad(f))` cannot work, because there is nothing to replay. It is refused
  rather than answered with a wrong number.
- The graph is kept alive by the output tensor, so a long-lived result pins every
  intermediate that produced it. A training loop accumulating losses into a list
  without extracting their values holds the whole history.
- `jacobian` has to re-run the function once per output component, because there is
  no recorded graph to seed repeatedly with different cotangents.

The alternative, an explicit tape, would make all three cheap. It lost because a
separate tape must be threaded through everything that might record onto it, and
twill's numeric surface is a package of free functions taking `*Tensor`. Giving all
of them a context argument to serve a feature that is off most of the time was the
wrong trade for a codebase whose stated goal is to be readable in a sitting.

**3. The profile.**

Over 1,200 iterations of `mc_option_grad`, 16.71 s duration, 28.21 s of samples
across cores:

- arithmetic, about **50%**: `broadcastBinary`, `unary`, `math.archExp`, `Relu`,
  `Mul`, `Add`, `parallelSum`. This is where time should be.
- allocating and zeroing intermediates, **18%** cumulative: `runtime.mallocgc`
  18.08% cum, `makeslice` 17.48% cum, `memclrNoHeapPointers` 15.38% flat.
- goroutine coordination, about **17%**: `lock2`, `unlock2`, `semasleep`,
  `semawakeup`, `preemptM`, `procyield`, `sysUnusedOS`.
- the interpreter, **0.035%**: `Interp.Apply` at 0.01 s flat, and `evalExpr`,
  `evalCall`, `evalBinary`, `callClosure` and `execStmt` all 0% flat.

The actionable one is the 18%. The pricer computes about eight elementwise operations
over 200,000 elements, each allocating a fresh 1.6 MB output buffer that Go zeroes
before use, and every one except the last is read once and discarded. That is the
measured argument for **fusion**, and specifically for the CPU half of the design in
`docs/CODEGEN.md`, which needs no GPU and no new dependency.

The interpreter figure is the finding that stops work rather than starting it. twill
is already a thin orchestration layer over its tensor operations in the way Python is
over NumPy, so a faster dispatch mechanism would speed up the part that costs
nothing. It independently reproduces on a different workload what
`docs/perf-baseline.md` found on a transformer forward pass.

**4. The four causes.**

1. **Every float is 8 bytes and there is no f32 path.** A tensor's buffer is
   `[]float64` regardless of its dtype tag. This is most of the f32 column: PyTorch
   f32 moves half the bytes and fits twice as many SIMD lanes. PyTorch's own
   f64-to-f32 speedup on `elementwise_1000000` is 2.5x, which is the size of the
   handicap twill carries on every f32 row. Tracked as NEEDS-111.
2. **No BLAS.** `mm` and `mmNT` are hand-written Go with four accumulators to hide
   FP-add latency and cache blocking sized to L2.
3. **No SIMD.** Go does not auto-vectorise the elementwise loops, so
   `broadcastBinary` and `unary` process one f64 per iteration.
4. **No fusion**, so every intermediate is materialised. This is the smallest of the
   four, because PyTorch does not fuse by default either, but PyTorch's caching
   allocator does not return buffers to the OS and does not re-zero them while Go's
   zeroes every slice it hands out.

The 21.5 against 250 GFLOP/s is cause **2**. It is the widest and steadiest gap in
the document, it does not close with size, and it is not a missing optimisation so
much as the accumulated result of years of hand-tuned assembly in the library twill
declined to link. Linking one means cgo, a C toolchain, a platform matrix, and the
end of the single dependency-free binary. It also collides with the determinism rule,
since a threaded BLAS reduction does not promise a reproducible last bit.

The one place the gap nearly closes is `verify_deterministic` at 2.4x, where none of
the four dominate: small tensors, mixed operations, and PyTorch paying its own
per-operation dispatch overhead.

**5. The closure test.**

`TestGradientCheckCoversEveryOperator` parses the package's own source with
`go/parser`, collects every exported function that takes or returns a `*Tensor`, and
fails if one has neither a gradient-check case nor an entry in `nonDifferentiable`
saying why it has none.

The counts: **63 differentiable operators checked, 26 declared non-differentiable,
89 exported in total.**

The property it buys is that a new operator cannot be added without someone deciding,
**in writing**, whether it carries a gradient. That is the difference between "the
full operator set" as a claim and as a fact, and it is why the 103 is the less
interesting number: the 103 is what the closure currently requires, and it grows on
its own.

The non-differentiable list is not a way of quietly excusing operators. It holds the
index-valued ones (`argmax`, `argmin`, `argsort`, `argtopk`, whose outputs are
positions, integer and locally constant), the boolean comparisons, the two quantisers
(step functions, though the gradient through the product they produce is checked),
`Cast`, and the shape and dtype helpers that are not operators at all.

**6. The cotangent.**

The harness differentiates `L(x) = sum(w * out)` for a fixed deterministic cotangent
`w`. An all-ones `w` makes an entire class of **index-shuffling** bugs invisible,
because a permutation of the output has the same sum as the output. So every case
that scattered its gradient to the wrong elements would still produce the right
`L`.

The families: **transpose, flip, roll, sort, gather and concat**. Every one of them
would pass a gradient check with an all-ones cotangent while being wrong.

`w` is deterministic so there is no seed to get out of step, and irregular in both
sign and magnitude so no accidental symmetry cancels.

Two other load-bearing details of the method, worth knowing alongside it. The
comparison is against a Richardson-extrapolated central difference,
`D* = (4*D(h/2) - D(h)) / 3`, whose truncation is `O(h^4)` and lands near 1e-12,
against roughly 1e-10 for a plain central difference. That is what makes a 1e-7
tolerance honest rather than ambiguous; measured error runs 1e-13 to 1e-11, four
orders of headroom. And the step is scaled to the point, `h = 1e-4 * max(1, |x_i|)`,
so a coordinate of size 1000 is not probed with an absolute step its own rounding
swallows.

**7. The quantised gradient.**

`QLinear` and `QLinear4` returned `&Tensor{Data: out, Shape: outShape}` and nothing
else. No `track1`, no parents, no backward closure. `Backward` walks the graph
through the `p0`/`p1`/`pRest` pointers, this tensor had none, so the traversal
stopped there and every parameter upstream kept its initial zero gradient. The value
was right and the 0.02 quantisation error was right for int8; only the gradient was
wrong, and it was wrong in the way that looks like a model that has converged.

The reasoning was sound: a quantised weight is frozen, so the doc comment says the
result is a frozen weight for inference, not a differentiable tensor. That is true
**of the weight**. The conclusion does not follow because the activation `x` is an
ordinary full-precision tensor, and what the kernel computes is exactly linear in it,
so the derivative with respect to it not only exists but is exact. Freezing the
weight had been implemented as detaching the whole operation.

The fix wires `track1` and computes `dL/dx = g @ Wq` against the dequantised weight,
exact because the codes are constants.

Why exhaustiveness rather than representativeness found it: the two quantised kernels
were the least-exercised operators in the package, so a gradient check over the
operators someone thought to check would never have reached them, and the defect they
had was the worst in the package. Not a wrong number, a plausible one.

**8. The two claims.**

**It claims:** every diagnostic it reports is a real mistake. No false positives.

**It does not claim:** that a clean check means the program runs. The package comment
says it reports a diagnostic only when a mismatch is certain, and the bias toward
precision over recall is deliberate.

The separator:

```rust
let n = len([1.0, 2.0, 3.0])
let A = zeros(2, n)
let x = zeros(2)
let y = A @ x
print(y)
```

`twill check` says no shape problems found; `twill run` says
`shape mismatch in @: [2 3] @ [2] (inner 3 != 2)`. The checker cannot fold
`len(...)` to 3, so `n` types Unknown, `zeros(2, n)` types Unknown, and an Unknown
operand makes the `@` undecidable, so it stays silent as designed.

The claim it does make is tested differentially over 4,000 generated programs:
**2,646 rejected and every one really broken, 1,354 accepted and clean, 0 accepted
then failed. Zero false positives.** Programs come from a grammar of matmuls,
matrix-vector products, broadcasting combinations, reshapes, concats, calls to
shape-annotated functions, and chains where a mismatch is several operations from its
cause. Every dimension is a literal from `{1, 2, 3, 4}`, and `1` is in the set because
it broadcasts against anything and is the case a naive equality rule gets wrong.

Because every dimension is a literal, every error in that corpus is statically
decidable in principle, which is how completeness gets measured: **2,646 of 2,646, or
100%**, with the test asserting a 95% floor rather than the exact figure so improving
the checker does not break it and regressing it does.

**That corpus cannot establish the declined claim, because the corpus is by
construction the decidable case.** The gap between the two is exactly the six-line
program above, and it is not narrowing; it is where the design put it. The corpus is
also small in another way: a generated program is at most five lines and never
exercises nested closures, records, loops that reshape, or `grad`, all of which the
checker leaves Unknown. Extending the generator would lower the 100%, which is the
point of measuring it.

**9. (design) The smallest IR.**

A good answer covers these.

*What a node is.* `value.Value` is an empty interface, so the IR cannot assume its
inputs are tensors. The smallest workable node is a tensor-only expression node:
an operation code, a shape, a dtype tag, and operand references, with a distinguished
"opaque value" node for anything that is not a `*tensor.Tensor`. The IR is built by
the interpreter as it evaluates, not by a separate compilation pass, so nothing has
to be re-implemented.

*Where the boundary sits.* At a type switch, and that is unavoidable given `any`
values. The practical form is a tracing region: while every value flowing through
elementwise operations is a tensor of compatible shape, append to the IR and defer
execution; the moment something forces a value (a print, a control-flow condition, a
non-fusable operation, a reduction) flush the pending chain as one fused pass. That
gets the eight elementwise operations of the pricer without needing a whole-program
compiler, which is the point of "smallest".

*The backward pass.* This is the hard part, and the honest answer names why. The
tape is the tensor graph, so a fused chain must still produce a tensor carrying
parents and a `backward` closure. Two workable shapes: emit one backward closure for
the whole fused region that runs the chain's adjoint as a second fused pass, which is
the version that actually pays; or keep per-operation closures and fuse only the
forward, which is simpler and gives up half the win. The first requires the IR to be
differentiable, i.e. to have an adjoint per node, which is a real amount of design and
should be acknowledged rather than waved past.

*What the kernel is cached against.* The chain's structure, not its data: the sequence
of operation codes, the broadcast pattern, the dtype tags, and the rank. Not the
shapes' concrete extents, which should be a runtime argument, or the cache misses on
every new batch size.

*The determinism rule it must not break.* Entry 6: parallelism never changes a
result. `parallelSum` adds in fixed 4096-element blocks and combines partials in block
order. A fused pass that folds a reduction into an elementwise chain must keep that
blocking, or the answer moves in the last bits, seeded programs stop reproducing, and
the self-hosted differential harness, which compares canonical float renderings byte
for byte, stops being able to compare anything at all. Fusing elementwise-only chains
is safe; fusing a reduction into one needs the block structure preserved explicitly.

**10. (design) Packed storage.**

*What a tensor becomes.* Today the buffer is `[]float64` and the dtype is a tag
(`dtp`) with rounding rules. It becomes a dtype plus a typed buffer, either a union of
typed slices or a `[]byte` plus a dtype, and the honest note is that this touches
every kernel in the package: each single loop becomes a dispatch over element type,
and the readable `for i := range out` loops become generic or duplicated. That cost is
the whole reason the decision has not been taken.

*How the accumulation rules survive.* They already exist as specification
(`docs/dtypes.md`) and as tested behaviour, which is the sequencing argument for why
the semantics were built first: separating the semantics from the layout meant the
rounding rules could be specified, tested and made bit-identical against the
self-hosted implementation on a representation where every kernel is a single loop.
The rules should not change at all. The test that says so is that the packed
implementation reproduces the f64-backed one's output exactly for every dtype, which
is only possible because the f64-backed one exists first.

*The differential harness.* It compares canonical float renderings byte for byte, so
it is the strongest check available and it survives unchanged as long as the packed
kernels produce identical values. Where it needs care is `acc_add`: the self-hosted
`block_sum` ports the Go `parallelSum` form precisely so narrow dtypes accumulate at
their own width, and packing must not change what width that is.

*Order of work.* Widest-blast-radius-last. Add a packed buffer behind the existing
API with f64 still the only packed type, so nothing changes numerically and the
dispatch machinery gets tested. Then f32, which is the case that matters, with every
gradient check and the differential harness run at each step. Then the narrow types.
Never both the layout and the rules at once, which is the same argument entry 5 makes
for why it was deferred.

*What to measure.* The elementwise rows against PyTorch f32, where twill currently
carries a 2.5x handicap it has nothing to answer with, and memory footprint, since
`docs/perf-baseline.md` is blunt that a 7B model needs 56 GB in f64 and does not fit.

*What it will not fix.* The matmul. 21.5 against 250 GFLOP/s is the absence of a
BLAS, and f32 storage does not write tuned assembly. It should improve, because f32
halves the bytes and doubles the lanes, and it will not close. Nor does it touch the
18% spent allocating and zeroing intermediates, which is fusion's problem, or the 17%
in goroutine coordination.

---

## Part 2

**Scenario A: `round(x) * x`, then `hessian`.**

`grad` returns a **number**, not an error, and the contribution through `round` is
zero, because `floor`, `ceil`, `round` and the comparisons all return an **untracked
tensor** and sever the chain. So the answer is the derivative of a different function
from the one the source describes, arrived at silently.

`hessian` used to **dereference nil** whenever the input was not connected to the
output. It now checks whether the leaf is in the graph and returns **zeros**, which
is the correct second derivative of a function that does not depend on its input.

The shared property is that these operations return a bare `&Tensor` with no parents
and no backward closure, so the graph traversal simply stops there. That is exactly
the mechanism of bug 1 (`QLinear`) and bug 3 (the cumulative scans), which is why they
are the same class of hazard: **an operation that returns a bare `&Tensor` is an
operation that silently ends the graph.**

Worth noting how bug 3 was hidden inside itself. Tracking the scans removed one route
to the `hessian` nil dereference but not the cause, because these four still sever the
chain. The fix had to handle the general case, not the route that happened to be
observed.

**Scenario B: shape from a CSV.**

The program **may or may not run**, and the checker cannot decide which. If the CSV's
width happens to match, it runs; if not, it fails at runtime with an exact message
naming real numbers.

The type is **Unknown**, and the rule is that the checker reports a diagnostic only
when a mismatch is **certain**. An Unknown operand makes the `@` undecidable, so it
stays silent.

Silence here is **not a bug**. It is the declined claim, stated in
`docs/CORRECTNESS.md` and priced in `DECISIONS.md` entry 3. The alternative bias,
reporting anything suspicious, was rejected because a checker that cries wolf gets
turned off, and a checker that is turned off catches nothing at all. Note this is not
an obscure corner: any shape derived from `read_csv`, from a length, or from a
loop-carried value has the same property, and those are ordinary things to do.

What would have to change is **the language, not the checker**: shapes would have to
be expressible in the type system in a way the programmer can assert and the checker
can rely on, i.e. a dependent or refined shape annotation at the boundary
(`fn load(path) -> [n, 3]`), so the unknown dimension gets a name and a contract even
though its value arrives at runtime. Function shape signatures already do exactly this
for the annotated case, and the checker enforces them at call sites; extending that to
IO boundaries is the direction. Making the checker smarter without that cannot help,
because the information is genuinely absent at check time.

**Scenario C: the reduction order.**

`reduce_all` in `src/tensor.tw` was a plain running sum at every size. The Go
bootstrap's `parallelSum` adds in fixed 4096-element blocks and combines the partials
in block order once past 8192 elements. Floating-point addition is not associative, so
the two summation orders give different last bits. The mean scaling was off the same
way: the bootstrap forms `s * (1.0 / n)`, not `s / n`, and the two round differently.

It is a test failure and not a curiosity because the goldens compare a **canonical
float rendering byte for byte**. That is the point of them: any future reordering of a
reduction fails immediately rather than drifting.

The decision that makes one answer wrong is `DECISIONS.md` entry 6: **parallelism
never changes a result**, and the fixed block size is what pins the answer on any
machine. Given that rule, an implementation that sums in a different order is wrong,
not merely different.

The fix (`2a4b489`) ported the Go form as `block_sum`, with `acc_add` standing in for
the bare `+` so narrow dtypes accumulate at their own width. On f64, where `acc_add`
is f64 addition, the order is bit-identical to `parallelSum`.

If entry 6 had gone the other way, letting workers accumulate partials and combine in
completion order, two things break. A seeded program would stop reproducing exactly,
run to run and machine to machine. And the self-hosting programme loses its oracle
entirely: `src/tensor.tw` is checked against the Go kernels by comparing canonical
float renderings byte for byte, and a sum whose last bits move cannot be compared that
way at all. The determinism rule is not a nicety, it is what makes the differential
harness possible, and the differential harness is what found the lexer panic
(NEEDS-33, fixed in `d3176a9`) and the fractional-index bug in `transpose_origins`.

---

## Part 3: the scan reimplementation

Notes for whoever does it.

**The failure mode to expect is silence.** Bug 3 is exactly this component: all four
scans were registered through a helper that computed the fold in plain `float64` and
returned `tensor.New`. No parents, no backward closure. `grad` through them came back
zero, and where a scan was only part of an expression the answer was worse than
absent, it was wrong. `max_drawdown` divides by a `cummax` peak, so its gradient was
neither correct nor obviously broken, and `sma`, `equity`, `total_return` and `cagr`
in `std/backtest` were all in that state.

**The documentation did not save it and could not.** The comment on that helper and
`docs/language-guide.md` both said the scans were forward-only, which documented the
behaviour without enforcing it. `grad` accepted them and returned a number anyway. If
your reimplementation is going to be forward-only, `grad` must refuse it, not answer
it.

**`cumprod`'s backward must be division-free.** The obvious backward divides the
cumulative product by the element, which is `0/0` at a zero in the series. The
prefix/suffix pair gives an exact answer there instead. The gradient check will not
necessarily catch this unless a case puts a zero in the series, so put one there.

**`cummax` and `cummin` ties go to the earlier element**, matching `max` and `argmax`.
This is a convention, not a theorem: the subgradient at a tie is a set and any element
of it is defensible. It is pinned by `TestGradientKinkConventions` alongside the other
three conventions, three of which differ from PyTorch by choice, so that changing one
is a deliberate act rather than a regression nobody notices.

**Site the finite-difference cases away from the kinks.** `cummax` and `cummin` are
piecewise. A central difference straddling a kink reports the average of the two
one-sided slopes, which is not what any autodiff system returns, and reporting that as
a defect would be reporting a property of the difference method. No ties in any
ordering case. The kinks are asserted separately and directly.

**The coverage test is the backstop.** `TestGradientCheckCoversEveryOperator` parses
the package source, so a fifth scan cannot be added without a case or a written
exemption. That closure is the thing that found the `QLinear` defect, and it is the
reason this component is a good exercise: the suite will tell you what you forgot
rather than passing quietly.
