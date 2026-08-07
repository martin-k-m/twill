# Rewrite plan

Twill is to stop being a Go program and become its own system level
implementation with no dependency on anything else. This document plans that
work: which target, in what order, how correctness is proved at each step, what
it costs, and how much speed it actually returns.

The decision to rewrite is taken. What follows plans it. Where the plan
disagrees with an assumption, it says so and gives the number it disagrees on,
because a plan that hides its costs cannot be scheduled.

## What is actually being rewritten

Measured from the tree today:

| Component | Lines | Character of the work |
|---|---|---|
| `internal/checker` | 1648 | Shape, type and unit inference. Subtle. |
| `internal/interp` | 2903 (interp, builtins, serialize) | Tree-walking eval, ~90 builtins, RSTR save format |
| `internal/tensor` | 2879 (tensor, ops, scan, jet, einsum, conv, gather, parallel) | Autodiff engine, reverse tape, forward jets, parallel kernels |
| `internal/parser` | 878 | Recursive descent with a Pratt loop |
| `internal/gbm` | 505 | Gradient boosted trees |
| `internal/format` | 392 | `twill fmt` |
| `internal/value` | 356 | Value representation, `Env` scopes |
| `internal/lexer` | 223 | Hand-written scanner |
| `internal/ast` | 313 | Node types |
| `cmd/twill` | 256 | CLI: `run`, `check`, `fmt`, `repl`, `version` |
| `std/*.tw` | 543 across 6 modules | Written in Twill, embedded in the binary |

Non-test Go is about 10,300 lines. Tests are about 4,600 lines holding **279
top level test functions** (checker 60, interp 106, tensor 102, gbm 4, value 4,
format 3), plus 10 benchmarks. There are 24 files in `examples/`, of which 19
plus a `check` and a `fmt` run are smoke tested by CI on every push.

Those 279 tests are the specification. `internal/tensor/gradcheck_test.go` alone
is 989 lines of finite-difference checks at `eps = 1e-6` against a `1e-4`
absolute bar, covering matmul, conv2d, maxpool2d, einsum, softmax, logsumexp,
sort, topk, gather, cumulative ops, split, concat, reshape, transpose, where,
clip and prod-with-zeros. Any implementation that does not pass that file has
not reproduced Twill's autodiff, whatever else it does.

The language surface that must be preserved exactly is in
`docs/language-guide.md`: values, operators, equality, bindings, functions,
control flow, indexing and slicing, `grad`/`grads`/`value_and_grad`/`jacobian`/
`hessian`, shape checking, units of measure, records, imports, and the standard
library listing. The checker's diagnostics are part of that surface, not an
implementation detail, because the guide documents them and users read them.

Two further constraints that are easy to forget and expensive to rediscover:

- `internal/interp/serialize.go` defines an on-disk format: magic `RSTR`,
  version byte 1, little-endian, exact float64 bit patterns, tags
  `T L R B S U G`. Files written by v1.2.0 must load in the rewrite.
  `examples/model.bin` and `examples/params.bin` exist in the tree as fixtures.
- `std/embed.go` compiles the standard library into the binary with `go:embed`,
  with a `TWILL_STD` directory override. Every target must reproduce both
  behaviours, because "single binary, no toolchain" depends on the embedding.

## 1. Target choice

The property being defended is stated in `docs/design.md` principle 4: no
third-party packages, one binary, readable end to end. Users install a single
static binary and need no toolchain. That property is about **what the user
installs**, not about what the maintainer compiles with. Keeping those two
apart is the whole of this section, because three of the four candidates cost
nothing on the first and a great deal on the second.

The current release pipeline is the baseline to beat. `.github/workflows/release.yml`
builds five targets (linux/amd64, linux/arm64, darwin/amd64, darwin/arm64,
windows/amd64) in one loop on one ubuntu runner with `CGO_ENABLED=0`, and
`sha256sum`s them. That is nine lines of shell. No other candidate matches it,
and pretending otherwise would make the schedule wrong.

### C++

Buys: total control of memory layout, which matters for the tape and for a
value representation; mature autovectorisers; no runtime beyond libc.

Costs: no memory safety in a codebase whose hot paths are raw index arithmetic
over flat `[]float64` (see `ops.go`, `scan.go`, `einsum.go`). The current code
also leans on the garbage collector structurally: each differentiable op
captures a backward closure, values alias freely, and `value.Value` is `any`.
Porting that to C++ means picking an ownership discipline for the tape, and
picking wrong is discovered late, in gradient tests, as corruption.

Build: needs a compiler and matching libc per platform. Cross-compiling to
macOS from Linux needs an SDK you may not legally redistribute in CI. Realistic
answer is five runners plus a container matrix. "No dependencies" becomes "a
libc per platform", which is defensible for the binary and painful for the
pipeline.

Verdict: the safety cost is not paid for by anything Rust or Zig do not also
give.

### Rust

Buys: the same performance ceiling as C++ with the memory errors turned into
compile errors, which matters more here than usual because the failure mode of
a wrong autodiff port is a silently wrong gradient, not a crash. `std` alone
covers everything the interpreter needs: `std::thread` and scoped threads
replace `runtime`/`sync` in `parallel.go`, `std::fs` replaces the file IO, and
`f64` math is in `std`. A `Cargo.toml` with an empty `[dependencies]` is a
literal, checkable, CI-enforceable restatement of principle 4, stronger than
Go's, because Go's is enforced only by convention.

Costs: ownership fights the current tape design. Do not port `*Tensor` pointer
graphs to `Rc<RefCell<Tensor>>`; that is slower than Go and uglier. Port the
tape to an arena with `u32` handles: a `Vec<Node>` and indices into it. That is
a redesign of `tensor.go`, roughly two weeks on its own, and it is also the
faster design, so the cost is real but it is not waste.

Build: `rustup` is a maintainer dependency. Cross-compiling five targets needs
either five runners or `cargo-zigbuild`/`cross`, which is another pinned tool
in CI. Static linking is solved for linux via `x86_64-unknown-linux-musl` and
`aarch64-unknown-linux-musl`; windows-msvc and the darwin targets link the
system libc, which is what the Go binaries effectively do today anyway.

Verdict: recommended. Reasoning below.

### Zig

Buys: the single best build story of the four. `zig build-exe -target ...`
cross-compiles all five targets from one host, with libc headers bundled, which
would preserve today's nine-line release job almost unchanged. That is a
genuine and underrated advantage given how much of this plan's risk is in
release engineering rather than in code.

Costs: the language is pre-1.0 and still moving. Committing 10,000 lines of a
project whose defining virtue is stability to a toolchain that renames things
between releases means a recurring maintenance tax with no upper bound you
control. Zig also gives explicit allocators and no ownership checking, so on
the correctness axis it sits with C++, not with Rust.

Verdict: the runner-up, and the tripwire is stated below.

### Self-hosting Twill in Twill

This is what the goal statement literally asks for, so it deserves a direct
answer rather than deferral.

Twill today cannot express its own compiler. It has no pointers, no mutable
aggregates beyond record field assignment, no byte or char type, no sum types
or pattern matching, no arbitrary file IO, and no way to write a lexer that is
not fighting the type system. `value.Value` is Go's `any`; there is no
equivalent. Getting to self-hosting means first designing a systems dialect of
Twill, which is a strictly larger project than the interpreter it would
replace, and it means running both dialects forever or making the numeric
language absorb features it was designed to exclude.

It also does not remove the dependency, it defers it. A self-hosted compiler
needs a bootstrap compiler written in something else, and needs a pinned
prebuilt binary of itself checked in or downloadable. The honest description is
"the dependency becomes a binary artifact you must trust", which is a different
tradeoff, not an absence of one.

Verdict: a north star for after Stage 6, reached by writing the systems dialect
as a separate front end that targets the same bytecode. Not the target of this
rewrite. Choosing it as the target now is choosing a two-year project with no
shippable intermediate state.

### Recommendation

**Rust, with `[dependencies]` kept empty, and the tape rebuilt as an index
arena.**

The argument is not that Rust is fastest. On the numbers below, Rust and C++
and Zig land within noise of each other, because the wins come from the
bytecode VM and the value representation, and those are design choices
available in all three. The argument is that this rewrite's dominant risk is a
gradient that is wrong by 1e-3 in one op under one shape, found six months
later by a user, and Rust is the only candidate that removes an entire class of
the mechanisms that produce that.

The empty dependency list is not a slogan here. It is checkable in CI with
`cargo tree` and it is a stricter version of what the project already claims.

I am genuinely uncertain about one thing, and it is the build. If, after Stage
3, cross-compiling five targets from one runner has cost more than two weeks of
fiddling, switch the target to Zig and keep everything else in this plan
unchanged. The staging below is deliberately written so the target can be
changed at that point without losing the earlier work, because Stages 0 to 2
are host-language-independent by construction. That is the main reason they are
ordered first.

## 2. Staged sequence

Every stage ends with `go test ./...` green, the 19 example smoke tests
passing, and the differential harness reporting zero divergences. No stage
leaves the language broken. There is no cutover commit that flips the whole
implementation, because a 15,000 line big-bang rewrite of a working system with
279 tests has a completion probability near zero.

The Go implementation is not deleted until Stage 6. Until then it is the
executable reference, and it is a better oracle than any written specification
of this language will ever be.

### Stage 0: freeze the specification (2 to 3 weeks)

No rewrite yet. Build the thing that makes every later stage verifiable.

1. Add `twill run --dump=canonical`, which after a program finishes prints
   every top-level binding in a stable canonical form: name, kind, shape, and
   float64 values as `%x` hex bit patterns, record fields sorted by name, list
   order preserved. Hex, not decimal, because the point is bit equality.
2. Extract the `.tw` source embedded in the 106 interp tests and 60 checker
   tests into standalone fixture files under `testdata/`, each with a recorded
   canonical output or a recorded exact error string and line number. The Go
   tests keep working, they just stop being the only place the fixtures live.
3. Record golden outputs for the 24 examples and for a driver program per
   `std/` module.
4. Add a small grammar-directed `.tw` generator for fuzzing, seeded and
   reproducible.

Ends with: a corpus of roughly 400 fixtures with golden outputs, runnable
against any binary claiming to be Twill.

### Stage 1: bytecode VM in Go (6 to 10 weeks)

This is where the speed actually is, and it is where it is regardless of
language. Today's cost is not the kernels. It is a scalar loop at about 218 ns
and roughly 7 allocations per iteration.

Design a register-based instruction set over the existing `ast`, compile to it,
and run it in a new `internal/vm`. Keep the tree-walker. Select with
`TWILL_ENGINE=tree|vm`, defaulting to tree until parity holds, then flipping.
Run the whole test suite under both in CI.

What this removes: per-iteration scope allocation (partly addressed already by
inline binding slots), AST node dispatch, `value.Value` boxing on the hot path,
and the rank-0 `Tensor` wrapper around scalars. It composes with the unboxed
scalar work currently in flight rather than competing with it.

Ends with: both engines green on all 279 tests and the Stage 0 corpus, with the
VM measurably faster and no output differing by one bit.

### Stage 2: freeze the seam (2 weeks)

Define and document the boundary the port will cross: a serialized module
format carrying bytecode, constants, and checker-resolved shape information.
Front end (lexer, parser, checker) produces it; back end (VM, tensor engine,
builtins) consumes it. Both sides stay in Go for now.

This is what makes a component-by-component port possible instead of a big
bang. Without it, the first Rust component has to be linked against the whole
Go program.

### Stage 3: tensor engine in Rust (6 to 8 weeks)

Port `tensor`, `ops`, `scan`, `jet`, `einsum`, `conv`, `gather`, `parallel` to
a `staticlib` with a C ABI of maybe 40 entry points. Tape as an arena of nodes
with `u32` handles. Threading via scoped threads, keeping `parallel.go`'s
contract that chunking never changes output.

During this stage only, drive it from the Go binary through cgo so that
`gradcheck_test.go`'s 989 lines run against the Rust engine unmodified. This
cgo bridge is a test harness and never ships; `CGO_ENABLED=0` stays true for
releases throughout.

Ends with: all 102 tensor tests and all gradcheck cases passing against Rust.

### Stage 4: front end in Rust (6 to 8 weeks)

Port lexer, parser, ast, format, checker. Emit the Stage 2 module format. The
Go back end runs it. Error strings and line numbers must match byte for byte,
which is the hard part and is why the checker is the slowest 1648 lines here.

### Stage 5: back end and CLI in Rust (8 to 10 weeks)

Port the VM, the ~90 builtins, `serialize.go` including RSTR v1 compatibility,
`gbm`, the REPL, and the `std/` embedding. Remove the cgo bridge. The Rust
binary now runs the full corpus. The Go implementation stays in the tree,
building and passing, as the differential oracle.

### Stage 6: release and retirement (3 to 4 weeks)

Five-target cross-compilation, checksums, reproducible builds, install docs.
Ship the Rust binary as v2.0.0 only after a release candidate period during
which both binaries are published and the harness runs on every commit. Then
move the Go implementation to a `reference/` directory or a tag, and say in
`CHANGELOG.md` what it was for.

### Stage 7 and after: self-hosting, optional

With a bytecode target and a stable module format in place, a systems dialect
of Twill becomes a front end rather than a rewrite. That is the cheapest path
to the original goal, and it is available only because Stage 2 exists.

## 3. Differential-testing harness

The harness is the deliverable that makes the rest safe. Concretely:

**Runner.** `tools/diff/` takes two binary paths and a corpus directory, runs
every fixture through both, and diffs. Exit non-zero on any divergence. Runs on
every PR as a required check from Stage 3 onward.

**Forward values: exact.** Compare canonical dumps byte for byte. Both
implementations do IEEE-754 double arithmetic in the same order over the same
data, so the correct tolerance for `2.0 * 3.0` is zero. Do not start with a
tolerance; a tolerance hides the ordering bugs you most want to find. The known
exceptions get declared explicitly, per op, in a list that is reviewed rather
than grown: reductions whose chunk boundaries move, and anything where the Rust
build contracts a multiply-add into an FMA. Compile with FMA contraction off
first, get exact agreement, then turn it on and see exactly which ops move.

**Gradients: tolerance, and a second independent check.** For gradients where
the accumulation order is unchanged, require relative error below 1e-12. For
ops in the declared reassociation list, 1e-9. Independently of both, run the
existing finite-difference bar of 1e-4 at `eps = 1e-6` against the Rust engine,
because that check does not care which implementation is right, only that the
analytic gradient matches the numeric one. Two implementations agreeing to
1e-12 on the same wrong derivative is a real failure mode and finite
differences is the thing that catches it.

**Errors: exact.** Compare stderr text and line number byte for byte, for every
fixture that is supposed to fail. The 60 checker tests exist because these
messages are the product.

**Serialization: cross-implementation round trip.** Save with Go, load with
Rust, re-save, compare bytes. Then the reverse. Include
`examples/model.bin` and `examples/params.bin`, and a generated case per tag
(`T L R B S U G`).

**`twill fmt`: idempotence and cross-agreement.** Format every `.tw` in the
tree with both; require identical output, and require formatting twice to be a
fixed point.

**Fuzzing.** Run the Stage 0 generator continuously, diffing both
implementations. Any divergence is minimized and added to the corpus
permanently. This finds what 279 hand-written tests structurally cannot, which
is the interaction between features nobody thought to combine.

**Determinism.** `defaultSeed = 1` and the documented reproducibility guarantee
mean `rand`, `randn`, `permutation` and `seed` must produce identical streams
across implementations. That requires porting the exact PRNG algorithm, not
using Rust's, and it is worth writing down now because discovering it in Stage
5 costs a week.

## 4. Effort

One experienced engineer, full time.

| Stage | Weeks | Confidence |
|---|---|---|
| 0. Specification freeze and harness | 2 to 3 | High |
| 1. Bytecode VM in Go | 6 to 10 | Medium; instruction set design is where it slips |
| 2. Seam and module format | 2 | High |
| 3. Tensor engine in Rust | 6 to 8 | Medium; the arena tape is the risk |
| 4. Front end in Rust | 6 to 8 | Medium-low; checker error parity is underestimated by everyone |
| 5. Back end, builtins, CLI | 8 to 10 | Medium-low; 90 builtins is a long tail |
| 6. Release engineering | 3 to 4 | Low; cross-compilation always costs more than planned |
| **Total** | **33 to 45** | Call it nine to eleven months |

Add 20 percent if the language keeps gaining features during the rewrite, which
it will, since the CHANGELOG shows roughly weekly additions. Every new builtin
added during Stages 3 to 5 must be written twice.

**Delegable to agents.** Stage 0 fixture extraction, in bulk. The canonical
dumper. The lexer and parser port, since the grammar is fixed and the tests are
exhaustive. Elementwise and reduction kernels, one op per task, each gated on
its gradcheck case. The diff runner. Documentation updates. The RSTR
serializer port. Roughly 40 percent of the total by volume, and it is the
boring 40 percent, which is the right 40 percent to delegate.

**Not delegable.** The instruction set design in Stage 1, which determines the
performance ceiling of everything after it. The arena tape ownership design.
The checker port, where a subtly over-general inference rule passes all 60
tests and breaks a user's program. Error-message parity judgement, which is
taste. The build and release pipeline. Anything where a passing test suite is
not the same thing as being right, which for a differentiation engine is more
places than usual.

## 5. How much faster, honestly

Numbers, on the measured baseline of a 3-million-iteration scalar loop at 655
ms (218 ns per iteration, about 21 million allocations remaining) and 512x512
matmul at about 9 GFLOP/s.

**Bytecode VM in Go, Stage 1 alone, 6 to 10 weeks.** Removes AST dispatch,
per-iteration scope allocation, and scalar boxing. On the scalar loop I expect
655 ms to land between 90 and 170 ms, so **4x to 7x**. Kernels are untouched
and stay at 9 GFLOP/s.

**The finished Rust rewrite, 33 to 45 weeks.** The VM in Rust over the same
instruction set buys a further 1.5x to 2.5x on interpreter-bound code: no GC
write barriers, a genuinely unboxed value representation, and better dispatch.
The loop lands between 40 and 90 ms, so **7x to 16x versus today**. Kernels go
to 18 to 27 GFLOP/s with hand-tiled AVX2 or NEON microkernels, the 2 to 3x
already established.

**End to end on a real program, which is what matters.** A training example
like `examples/mlp.tw` or `examples/cnn.tw` splits its time roughly 60 percent
in kernels and 40 percent in the interpreter. Amdahl:

- Stage 1 alone: about **1.5x** end to end.
- Full Rust rewrite: about **2.6x to 3.2x** end to end.

So: the first 10 weeks buy 1.5x. The remaining 25 to 35 weeks buy another 1.8x
to 2.1x. The last doubling costs roughly three times the effort of the first
1.5x. That is the honest shape of it, and it is the ordinary shape of this kind
of work rather than an argument against it. The staging above is built so that
the 1.5x lands early and independently of which target eventually wins, which
is the correct response to that shape.

Two things the raw multipliers understate, in the rewrite's favour. First, a
bytecode VM in Go still pays for the GC on every tensor allocation, and a
training loop allocates a lot of tensors; the Rust arena removes a cost that
does not show up in a scalar-loop microbenchmark at all. Second, once the tape
is an arena, fusing elementwise chains becomes tractable, and that is worth
more on real programs than anything in this table. Neither is counted above,
because neither is measured yet.

One thing the multipliers overstate: none of this touches GPU. If large-model
throughput is the actual goal, the ranking of this work against
`docs/gpu-feasibility.md` should be settled before Stage 3 starts, because a
2.6x on CPU and a 30x on GPU are not competing for the same slot in a roadmap.

## 6. First milestone

**Stage 0: the canonical dumper, the extracted fixture corpus, and the
differential runner. Two to three weeks.**

It is the right first milestone for four reasons:

1. It is worth doing whichever target wins, and worth doing if the rewrite is
   cancelled. Nothing in it is Rust-specific, or Zig-specific, or
   rewrite-specific.
2. It pays for itself immediately. Two agents are changing the scalar
   representation in `internal/tensor`, `internal/value` and `internal/interp`
   right now. A byte-exact canonical dump across the whole example and std
   corpus catches a scalar-unboxing regression that 279 unit tests can miss,
   this week.
3. It converts the specification from prose into something executable. Right
   now the specification is 279 Go test functions that only a Go program can
   run. After Stage 0 it is a corpus any binary can be held to, which is the
   precondition for every later stage.
4. It is small enough to finish, and finishing it produces a real decision
   point with better information than exists today.

Concretely, done means: `tools/diff/run --old ./twill-v1.2.0 --new ./twill
--corpus testdata/` exits zero, covering the 24 examples, all 6 std modules,
the extracted interp and checker fixtures, the RSTR round trip, and `fmt`
idempotence, with a fuzz mode that runs for a fixed budget and reports
divergences.

Do not start Stage 1 until that runner is a required CI check.
