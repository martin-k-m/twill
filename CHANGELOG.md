# Changelog

## 0.13.0

Faster core numerics (finance roadmap #2).

- Full reductions (`sum`/`mean`) run across cores and stay deterministic:
  fixed-size blocks are summed independently and their partials combined in a
  fixed order, so the result is the same on any number of cores. ~3.3x on a
  million-element reduction.
- The backward passes for same-shape elementwise and unary ops run across cores
  too (each goroutine writes a disjoint slice of the gradient), so gradient
  computation on large tensors uses all cores — not just the forward pass.
- Together with v0.12's parallel forward ops, both the value and the gradient of
  large-tensor work (Monte Carlo, backtesting) are now multicore. Matmul is
  already row-parallel and cache-friendly; explicit blocking is left for later.

## 0.12.0

Multicore tensor ops (finance roadmap #1).

- Large elementwise, unary, and matmul forward passes now run across CPU cores.
  Each goroutine writes a disjoint slice of the output, so it's race-free and
  **bit-identical to a serial run** — parallelism never changes a program's
  result, and randomness stays deterministic. Small tensors (typical training
  parameters) run serially, below a size threshold.
- Measured scaling on large ops (1 core -> 16): `exp` ~3.8x, 256x256 matmul
  ~4.5x, elementwise add ~2x (memory-bandwidth bound). This is the biggest
  pure-Go speed lever for the Monte-Carlo and backtesting workloads.
- CI now runs the suite under the race detector.

## 0.11.0

Deterministic randomness, and the first finance step.

- Randomness is now deterministic by default: a program gives the same result
  every run. `seed(n)` chooses the starting point. Reproducibility like this is
  what model governance and audit require. (Because runs are reproducible, the
  formatter's behavior-equivalence test now covers the stochastic examples too.)
- New example `montecarlo_option.ra`: prices a European call by Monte Carlo and
  computes its Greeks (delta, vega) by autodiff — matching Black-Scholes closed
  form, with no bump-and-revalue.
- `docs/finance.md`: an honest assessment of where Raster can beat a Python
  stack for financial ML under a pure-Go, no-native-deps constraint, and the
  roadmap to get there.

## 0.10.0

A formatter, and a tape tweak.

- `raster fmt` reprints a program in a canonical style and preserves comments
  (leading and trailing). It parenthesizes only where needed to keep the
  operator tree, is idempotent, and refuses rather than move a comment it can't
  place. Add `--write` to format in place.
- The lexer now records comments (`TokenizeWithComments`), and the parser
  exposes them (`ParseWithComments`).
- Autodiff tape: parent pointers for the common one/two-input ops are stored
  inline instead of in a per-op slice, trimming an allocation per differentiable
  op. Measured as a small net win on the training benchmark; the tree-walking
  interpreter is otherwise near its floor, so a real speed jump needs a
  vectorized/bytecode backend (tracked in the roadmap). Gradient checks still
  pass.

## 0.9.0

Container-agnostic optimizers (pytrees).

- `map_leaves(f, tree)` and `zip_leaves(f, trees)` walk the tensor leaves nested
  inside lists and records, preserving structure.
- The standard optimizers (`sgd_step`, `momentum_step`, `adam_step`, and
  `zeros_like`) are rewritten on top of them, so the same code trains a model
  whether its parameters are a positional list or a named record.
- `examples/records.ra` now trains its record-based model with the library's
  Adam instead of a hand-written update.

## 0.8.0

Einsum and earlier error detection.

- `einsum(spec, ...tensors)`: a differentiable Einstein-summation primitive.
  Covers matrix multiply (`"ij,jk->ik"`), transpose (`"ij->ji"`), reductions
  (`"ij->i"`, `"ij->"`), bilinear forms, and general contractions. The gradient
  of an einsum is itself an einsum, so it backprops exactly. Repeated labels
  within a single operand (traces/diagonals) are not supported yet.
- The checker infers an einsum's output shape from a literal spec and known
  input shapes, and reports malformed specs and rank mismatches.
- The checker now checks each function body at its definition, using the
  parameter annotations — so shape mistakes, field typos, and return mismatches
  are caught even in functions that are never called. Unannotated parameters
  stay unknown, so this adds no false positives.
- New example `einsum.ra`.

## 0.7.0

Declared record types and a faster interpreter.

- Declared record types: `type Model = { w: [3, 2], b: [3] }`. A parameter can be
  typed with it (`fn f(m: Model)`), and the checker verifies the record passed
  in has the right fields with the right shapes.
- Field typos are caught: accessing a field a record doesn't have is a checker
  error (`record has no field "wong"`).
- Performance: elementwise/unary ops skip building a backward closure when no
  input needs a gradient (parameter updates and other forward-only math). A
  100-step linear-regression training loop dropped from ~372us to ~300us per run
  (~19%) with ~800 fewer allocations; environments also allocate their map
  lazily. All gradient-check tests still pass.
- Examples are now run in-process by the test suite, so `go test` covers them
  end to end (not just the built binary).

## 0.6.0

Records and modules.

- Records with named fields: `{ w: [1.0, 2.0], b: 0.5 }`, accessed with `.`
  (`p.w`). A `{` is a record when followed by `name:`, otherwise a block.
- `grad` follows record structure: differentiating a loss over a record of
  parameters returns a record of gradients with the same fields — so a model can
  live in a record instead of a positional list.
- Namespaced imports: `import "std/nn.ra" as nn` binds the module's definitions
  as a record, called as `nn.dense(...)`. Plain `import` still shares scope.
- The checker understands records and field access, and reports records used as
  numbers or called as functions.
- New example `records.ra`: an XOR net whose parameters live in a record,
  trained via a namespaced import of the nn library.

## 0.5.0

Slicing, shape variables, and a faster tensor engine.

- Slicing: `v[1:3]`, `v[:2]`, `v[2:]`, and `m[0:2]` along the first axis, for
  both tensors and lists. Tensor slicing is differentiable.
- Shape annotations can use named dimensions (shape variables). A name used in
  more than one place must resolve to the same size, so the checker ties shapes
  together and verifies the declared return, e.g.
  `fn mm(A: [n, k], B: [k, m]) -> [n, m]`.
- The checker also reports argument rank mismatches against an annotation.
- Performance: the elementwise/broadcast path was rewritten to avoid per-element
  division — fast paths for equal shapes and scalar operands, and an
  odometer-style walk for general broadcasting (~3x on the broadcast benchmark).
  All gradient-check tests still pass.
- The email in the git history was replaced with a GitHub noreply address.

## 0.4.0

Usability and distribution.

- `read_csv(path)` loads a file of numeric rows into a `[rows, cols]` tensor.
- The REPL handles multi-line input: it keeps reading until brackets balance,
  so you can define block-body functions interactively.
- Prebuilt binaries for Linux, macOS, and Windows are attached to each release;
  `go install github.com/martin-k-m/raster/cmd/raster@latest` also works.
- A release workflow builds and publishes the binaries on a version tag.
- Added a getting-started tutorial (`docs/tutorial.md`).

## 0.3.0

Broadcasting, many more operations, real optimizers, and better tooling.

- Full NumPy-style broadcasting for elementwise ops, with correct gradients
  (matrix + row vector, column broadcasting, etc.).
- New differentiable ops: `square`, `clip`, `maximum`, `minimum`, `where`,
  `softmax`, `logsumexp`, `reshape`, `transpose` (arbitrary axes), `concat`,
  and elementwise comparisons (`greater`, `less`, `equal`, ...).
- Axis-aware reductions: `sum`, `mean`, `max`, `min`, and `argmax` take an
  optional axis argument.
- List helpers: `fold`, `append`, `enumerate` (plus the existing `map`, `zip`).
- Standard library: `std/optim.ra` adds SGD, momentum, and Adam over parameter
  lists; `std/nn.ra` gains initializers (He, Xavier), `gelu`, `softplus`, and
  softmax cross-entropy (`cross_entropy`, `onehot`).
- New example `classifier.ra`: a 3-class MLP trained with softmax cross-entropy
  and Adam.
- The shape checker understands broadcasting and the new ops.
- CLI errors now show the source line and a caret.
- A parser rule so a `(` or `[` starting a new line begins a new expression,
  matching the existing rule for `+`/`-`.
- Gradient-check tests (finite differences) for every op, plus benchmarks.

## 0.2.0

- Reimplemented in Go as a single dependency-free binary (from the earlier
  TypeScript prototype).
- Static shape checking with optional parameter/return shape annotations.
- An `nn` library written in Raster, loadable via a new `import` statement.
- `grad`/`grads` differentiate through list-structured arguments.
- `map`/`zip` builtins.

## 0.1.0

- First prototype (TypeScript): lexer, parser, tree-walking interpreter, a
  reverse-mode autodiff tensor engine, and the `grad` family.
