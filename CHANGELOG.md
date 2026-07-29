# Changelog

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
