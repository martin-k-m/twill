# Changelog

## [1.1.0] - 2026-08-04

### Added

- **`prod` and `median` reductions**, both axis-aware and both differentiable,
  bringing the built-in set to `sum`, `mean`, `max`, `min`, `prod`, `median`.
  They exist as built-ins rather than as `std` functions for the same reason
  the others do: neither can be composed out of what already existed without a
  gradient rule written by hand. `prod` handles zero factors explicitly instead
  of dividing by them, and `median` sorts indices rather than values so the
  backward pass knows which element it picked.

- **`broadcast_to(t, ...shape)`**, differentiable, expanding a tensor to a named
  shape. Broadcasting was already implicit inside every binary op, which covers
  everything until the shape you need to expand against is not one of the
  operands — the case being a reduction result you want to subtract from what
  you reduced. The gradient sums over each broadcast axis.

- **`split(t, n | sizes[, axis])`**, differentiable, the inverse of `concat`.
  A number gives that many equal pieces, a list gives those exact lengths, and
  each piece routes its gradient back into its own slice of the parent. Sizes
  that do not account for the axis exactly, and equal splits that do not divide
  evenly, are errors rather than ragged output. Pieces are copies, not views,
  matching the rest of the package.

- **Sample statistics in `std/num`**: `var_s`, `var_s_axis`, `std_s`,
  `std_s_axis`, `cov_s` and `corr_s`, dividing by n - 1. The population versions
  divide by n, which understates the spread when the tensor is a sample rather
  than the whole population, because the mean subtracted was measured from that
  same sample. A single element is left uncorrected instead of dividing by zero.

### Fixed

- **`num.var_axis` and `num.std_axis` no longer fail on any axis but the first.**
  They subtracted the mean straight back from the input, and a reduction drops
  the axis it reduced, so `var_axis(x, 1)` on a [2, 3] tried to broadcast a [2]
  against a [2, 3] and raised a shape error. Broadcasting aligns from the right,
  so only a reduction over axis 0 lined back up, and it did so by luck. The mean
  is restored to the reduced shape with `broadcast_to` now.

## [1.0.1] - 2026-08-03

### Fixed

- **A plain import no longer hollows out a namespace imported after it.** The
  load-once set was global, but "already loaded" means "already loaded into
  this scope", and a namespaced module's scope is new. So after
  `import "std/optim"`, the nested plain import inside nn was skipped as
  already loaded and its names never reached the module scope:
  `import "std/nn" as nn` came back missing `zeros_like`, `sgd_step`,
  `momentum_step` and `adam_step`, purely because of what had been imported
  before it. The same bug made a second namespace over one module come back
  empty. Each namespaced module gets its own load-once set now, which still
  guards cycles within that module.

## [1.0.0] - 2026-08-03

Cumulative scans are differentiable, equality is structural, imports are deterministic, and the standard library ships inside the binary.

## 0.28.0

Breaking. Three semantics that would be expensive to correct after a 1.0
stability promise, fixed now.

**The standard library ships inside the binary, and `std/` is its own namespace.**

- `import "std/nn"` — no extension, no directory. A path starting with `std/`
  names a module of the standard library, which is compiled into the `raster`
  binary with `go:embed`, so the import means the same thing from any working
  directory and an installed binary can find it. Before, `std/` was read off the
  disk relative to the importing file or the process cwd, so a `raster` on your
  `PATH` could not import the library it was built with. Every other import path
  is still a file.
- `std/` is reserved: a directory named `std` next to your program does not
  shadow the library. A real local file is still reachable as
  `import "./std/local.ra"`.
- A standard-library module may only import other `std/` modules. Embedded
  sources have no directory of their own to resolve a relative path against.
- `RASTER_STD=<dir>` is the escape hatch: it replaces the embedded library
  wholesale, so `import "std/nn"` reads `$RASTER_STD/nn.ra`. Meant for working
  on the library itself without rebuilding.
- **Migration.** `import "std/nn.ra"` and `import "../std/nn.ra"` become
  `import "std/nn"`; likewise for `optim`, `data`, and `backtest`. The old
  extension spelling is rejected with an error naming the new one
  (`a standard-library import names a module, not a file: write "std/nn"`), so
  the fix is mechanical:
  `sed -i -E 's|"(\.\./)*std/([a-z_]+)\.ra"|"std/\2"|g' *.ra`. Imports of your
  own files are unaffected.

**`==` and `!=` are deep structural comparison.**

- They used to answer `false` for every list and every record, including
  `a == a`, because the comparison fell through for those types and silently
  reported "not equal" rather than failing. Lists now compare elementwise,
  records field by field matched by name (so declaration order does not change
  the answer), and both recurse.
- A tensor's shape is now part of its value: `[[1.0, 2.0], [3.0, 4.0]]` and
  `[1.0, 2.0, 3.0, 4.0]` hold the same numbers and are no longer equal. This is
  the one change here that can flip a `true` to a `false`.
- `()` equals `()`. Functions compare by identity — a function equals itself,
  two separately written `fn(x) = x` do not. Values of different types are
  never equal, which is an answer rather than an error. `!=` is the negation of
  `==` in every case.
- Ordering (`<`, `<=`, `>`, `>=`) is unchanged: still scalars only, still an
  error on anything else.

**A namespaced import's field order is now declaration order.**

- `import "std/nn" as nn` built its namespace record by ranging over a Go map,
  so `columns(nn)` and `print(nn)` came out in a different order on every run.
  Reproducibility is the point of this language; a record whose field order is
  random is not that. A module scope now records the order its names were first
  defined in, including names it picked up from its own plain imports, and the
  namespace record follows it.

## 0.27.0

Differentiable cumulative scans.

- `cumsum`, `cumprod`, `cummax`, and `cummin` are now differentiable. Before,
  they returned an untracked tensor, so `grad` through a scan silently came back
  zero, and where a scan was only part of an expression (`max_drawdown`, which
  divides by a `cummax` peak) the gradient was not zero but wrong. `cumprod`'s
  backward pass avoids dividing by an input, so a zero in the series is exact.
  `cummax`/`cummin` route each output's gradient to the element the running
  extreme came from, ties going to the earlier one, matching `max`/`argmax`.
- More of `std/backtest.ra` is therefore differentiable end to end: `sma`
  (prefix sums), `equity` and `total_return` (cumulative product), `cagr`, and
  `max_drawdown` (running peak) join Sharpe and Sortino.
- Backed by new tracked `CumSum`/`CumProd`/`CumMax`/`CumMin` tensor ops, each
  with a forward-mode jet, so `hessian` flows through a scan too (it used to
  crash on one, because the scan detached the input from the graph entirely).

Also fixed:

- `hessian` no longer panics when the input is not connected to the output at
  all. Making the scans differentiable removed one way to reach that, but not
  the cause: any forward-only builtin (`floor`, `ceil`, `round`, the
  comparisons) returns an untracked tensor, so walking back from the output
  never reaches the leaf and the jet state it seeded was never allocated. It
  now returns zeros, which is both the right answer for a function whose output
  does not depend differentiably on its input and what `grad` already returned
  for the same expression.

## 0.26.0

Differentiable element indexing.

- `x[i]` (indexing a single element or row) is now differentiable — gradient
  flows to the indexed component, and `hessian` passes through it too. Before,
  element indexing silently broke the gradient graph (it returned an untracked
  tensor, so `grad` through `x[i]` was zero); slicing `x[i:i+1]` was the only
  working form. Both now work.
- Backed by a new tracked `IndexAxis0` tensor op (with a forward-mode jet);
  `x[i]` in the interpreter routes through it.

## 0.25.0

Second-order autodiff through structural ops.

- `hessian` now flows through the linear/structural ops — slicing (`x[a:b]`),
  `reshape`, `transpose`, `concat`, and `gather` — so component-wise functions
  and reshaping objectives get exact Hessians too (previously they errored).
- `examples/hessian.ra` adds a component-wise case: the Hessian of
  `(x0-x1)² + x1²` computed through slicing is `[[2,-2],[-2,4]]`.
- Extended the finite-difference cross-checks to cover slice+concat (with cross
  terms), reshape+transpose, and gather (with a repeated index).

## 0.24.1

Internal QA for the second-order engine — no API or behavior change.

- Forward-mode (jet) closures are now wired only while a Hessian is being
  computed, and the per-node jet state is boxed behind one pointer, so ordinary
  training and `grad` are back to their pre-v0.24 speed and memory (the v0.24.0
  release regressed the training hot path ~18% time / ~23% memory).
- Added finite-difference cross-checks for the Hessian across broadcasting,
  division, the general broadcast path, transcendentals, and matmul.

## 0.24.0

Second-order autodiff — exact Hessians.

- New `hessian(f)(x)` returns the matrix of second partial derivatives of a
  scalar function, exact (not finite differences). Together with `grad` it
  enables Newton's method and curvature analysis.
- Implemented as forward-mode 2-jets: each supported op now propagates a first
  and second directional derivative alongside its value, and a full Hessian
  follows by seeding basis directions and polarization. The reverse-mode engine
  is untouched — every existing gradient check still passes.
- Supported ops: `+ - * / %`, the unary math (`exp`, `log`, `sin`, `cos`,
  `tanh`, `sigmoid`, `sqrt`, `square`, `relu`, `abs`, `pow`, `neg`),
  `matmul`/`@`, `sum`, `mean`, and comparisons. A function using an op outside
  this set raises a clear error rather than returning a wrong Hessian.
- New example `hessian.ra`: the Hessian of a quadratic form recovers `A + Aᵀ`,
  and Newton's method minimizes a function with quadratic convergence.

## 0.23.0

Full Jacobians — differentiation beyond scalar outputs.

- New `jacobian(f)(x)` returns the whole `[m, n]` matrix of partial derivatives
  of a vector output `f(x)` (length `m`) with respect to a vector input `x`
  (length `n`) — every output's sensitivity to every input, exact, by one
  reverse-mode pass per output. This is the reverse-mode Jacobian (`jacrev`).
- New example `jacobian.ra`: the Jacobian of a linear map recovers its matrix,
  and a nonlinear map matches its analytic derivatives. Uses include
  risk/sensitivity matrices, input attribution, and Jacobian regularization.
- (Second-order autodiff — `grad(grad(f))`, Hessians — remains future work: it
  needs a re-differentiable reverse pass, a dedicated engine change.)

## 0.22.0

Sequences and attention — embeddings and a transformer, not just tables and
images.

- `std/nn.ra` gains `embed(table, ids)` (a differentiable embedding lookup built
  on `gather`, so embeddings are learned), `embedding_init`, and
  `self_attention(Wq, Wk, Wv, X)` — single-head self-attention, the core
  transformer operation, differentiable end to end.
- New elementwise builtins `floor`, `ceil`, `round` (forward-only) — e.g. to turn
  random draws into integer token ids.
- New example `attention.ra`: a self-attention sequence classifier (embed →
  self-attention → pool → dense) trained with Adam; `grad` differentiates the
  attention softmax and the embeddings together.
- Raster now spans tabular (boosted trees), vision (CNNs), and sequences
  (attention) — one autodiff engine, one static checker, one binary.

## 0.21.0

Data pipeline and real minibatch training.

- New differentiable builtin `gather(x, indices)` selects rows of `x` by an index
  list or 1-D tensor (gradient scatter-adds back, so repeated indices — e.g.
  embedding lookups — accumulate correctly). Gradient-checked.
- New `permutation(n)` returns a seeded random ordering of `0..n-1`, and `int(x)`
  truncates a scalar — together enabling reproducible shuffling and sizing.
- New `std/data.ra`: `standardize` (per-column z-score, returns the transform),
  `apply_standardize`, `train_test_split`, and `shuffle` (features/labels kept
  aligned).
- New example `minibatch.ra`: a genuine training loop — standardize, hold out a
  test set, then train a classifier with Adam over reshuffled minibatches each
  epoch (96%+ held-out accuracy). This is the mechanics real models train with,
  not full-batch toys.

## 0.20.0

Model persistence — train once, save, and deploy.

- New builtins `save(value, path)` and `load(path)` write and read any value —
  tensors, records, lists (a model's whole pytree), scalars, strings, bools, and
  fitted gradient-boosted models — in a compact, exact binary format (float64
  bit patterns round-trip bit-for-bit).
- `gbm.Model` now implements `encoding.BinaryMarshaler`/`BinaryUnmarshaler`, so a
  trained forest can be persisted and reloaded with identical predictions.
- New example `save_load.ra`: trains a classifier, saves it, loads it back, and
  confirms the reloaded model predicts identically; a neural net's parameter
  record round-trips too.
- Paths resolve relative to the running script (like `read_frame`), via a shared
  `resolvePath` helper.

## 0.19.0

Convolutional neural networks — general deep learning, not just MLPs.

- New differentiable builtins `conv2d(input, weight)` and `maxpool2d(input, k)`.
  `conv2d` is a 2-D cross-correlation (`input` `[Cin, H, W]`, `weight`
  `[Cout, Cin, KH, KW]` → `[Cout, H-KH+1, W-KW+1]`); `maxpool2d` does
  non-overlapping `k×k` max pooling per channel. Both have gradient-checked
  backward passes (input and weight), so `grad` trains a conv net end-to-end.
- New `std/nn.ra` helpers: `conv` (a conv layer with per-channel bias) and
  `conv_init` (He-initialized kernel + zero bias).
- New example `cnn.ra`: a real conv net (conv → relu → max-pool → dense →
  sigmoid) that learns to tell vertical from horizontal bars in noisy images,
  trained with Adam over the whole model (the nested conv kernel included).
- The checker infers conv/pool output shapes where the inputs are known.
- Positioning: Raster now spans the ML stack — neural nets (incl. CNNs),
  gradient-boosted trees, and backtesting — in one dependency-free binary.

## 0.18.0

Gradient-optimized trading signals.

- Because the backtest Sharpe is differentiable in the return series and a smooth
  signal's returns are differentiable in its weights, `grad` gives the gradient
  of Sharpe with respect to the weights — so a signal can be tuned by gradient
  ascent, straight through a backtest. This is the kind of end-to-end autodiff a
  plain Python backtest can't do without JAX.
- New example `signal_opt.ra`: learns a linear signal's weights on a synthetic
  market by climbing its annualized Sharpe, recovering the true signal direction
  and turning a negative-Sharpe asset into a positive-Sharpe strategy.
- New `sortino(r, periods)` in `std/backtest.ra`: the annualized Sortino ratio
  (downside-deviation-adjusted, differentiable).
- Internal: removed dead code and added a CI lint job (`deadcode` + `staticcheck`)
  so it can't return.

## 0.17.0

Backtesting toolkit (finance roadmap #6).

- New cumulative-scan builtins: `cumsum`, `cumprod`, `cummax`, `cummin` — the
  vectorized primitives for signals, equity curves, and running peaks.
- New `std/backtest.ra` library: `returns`/`log_returns`, `sma` (moving average
  via prefix sums), `equity` (cumulative-product equity curve), `max_drawdown`,
  `sharpe`, `ann_vol`, `total_return`, and `cagr`. The Sharpe ratio is
  differentiable in the return series, so a smooth signal can be tuned by
  gradient ascent.
- New example `backtest.ra`: a long-only k-day momentum strategy on a synthetic
  price series, reported against buy-and-hold (total return, CAGR, vol, Sharpe,
  max drawdown), with the position lagged a day to avoid look-ahead.
- This completes the finance roadmap in `docs/finance.md` (#1–#6).

## 0.16.0

Native gradient-boosted trees (finance roadmap #5).

- A pure-Go gradient boosting engine (`internal/gbm`) using the second-order
  (Newton) formulation, so squared-error regression and logistic binary
  classification share one tree builder. No XGBoost, no Python, no native deps —
  it stays a single static binary.
- `gbm_fit(X, y)` or `gbm_fit(X, y, opts)` trains on a `[n, d]` feature matrix
  and an `[n]` target/label vector. `opts` is a record of hyperparameters:
  `rounds`, `learning_rate`, `max_depth`, `min_leaf`, `lambda`, `gamma`, and
  `objective` (`"squared"` or `"logistic"`). It returns an opaque model.
- `gbm_predict(model, X)` scores a `[n, d]` matrix, returning `[n]` raw scores
  for regression or probabilities for a logistic model.
- Deterministic: exact-greedy splits with pre-sorted features, and the
  per-feature split search parallelizes across cores while reducing in fixed
  order — so fits are bit-identical run to run regardless of scheduling.
- New example `gbm.ra`: a train/test split on a synthetic loan book, fitting a
  logistic default classifier and a regression model on the same features.

## 0.15.0

Units of measure (finance roadmap #4).

- Declare base units with `unit USD` (like `type`, top-level). Annotate scalars
  with a unit or a compound unit expression: `px: USD/share`, `rate: USD/year`,
  `t: year^-1`, `-> USD`.
- The checker tracks units through arithmetic: `*` multiplies them, `/` divides,
  `+`/`-`/comparisons require a match, `^` with a constant exponent raises them,
  and `sqrt` halves them. `exp`/`log`/`sin`/`cos`/`tanh`/`sigmoid` require a
  dimensionless argument. `matmul`/`dot` multiply the operand units.
- Introduce a unit on a value with a `let` annotation — `let px: USD/share =
  150.0` — bare numeric literals are dimensionless and adopted into the
  declared unit. Undeclared unit names in an annotation are reported.
- Units are checked statically and fully erased at runtime: annotated code
  computes the same plain numbers with zero overhead, and unannotated code is
  unaffected (everything is dimensionless).
- New example `units.ra`: notional value and accrued interest, with the checker
  proving price × quantity is money and rejecting dollars + shares.

## 0.14.0

Data frames (finance roadmap #3).

- A frame is a record whose fields are named column tensors — so field access,
  slicing, and `grad` all work on it, and a numeric time column makes it a time
  series. No new type, and it composes with everything.
- `read_frame(path)` loads a CSV with a header row into such a record;
  `write_frame(frame, path)` writes one back.
- `columns(rec)` lists the field names, `field(rec, name)` looks one up by
  string, and `with_field(rec, name, value)` returns a copy with a field set.
- New example `frames.ra`: loads prices, computes daily log returns and realized
  (annualized) volatility, and adds a column with `with_field`.
- Kept pure Go / zero dependencies — Parquet/Arrow would need a third-party
  module, so it's deferred.

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
