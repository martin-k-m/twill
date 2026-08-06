# Raster language guide

This is the reference for Raster v0.28. The language is small, so this is short.

## Running programs

```bash
raster path/to/program.ra    # shape-check, then run
raster run path/to/program.ra
raster check path/to/program.ra   # shape-check only
raster fmt path/to/program.ra     # canonically format (add --write to edit in place)
raster                             # REPL
```

`raster fmt` reprints a program in a canonical style, preserving comments. It
refuses rather than move a comment it can't place.

Pass `--no-check` to run without the static shape check. In the REPL, each line's
value is printed; `:help` and `:quit` do the obvious things.

## Lexical structure

- Comments run from `#` to end of line.
- Whitespace is insignificant, with one exception: a token that could either
  continue the previous line or start a new one begins a new statement when it
  appears at the start of a line. This applies to a leading `+`/`-` (which would
  otherwise read as subtraction) and to a leading `(`/`[` (which would otherwise
  read as a call or index). To continue an expression across lines, end the line
  with the operator, or keep the call/index on the same line as its target.
- Identifiers match `[A-Za-z_][A-Za-z0-9_]*`.
- Numbers are floating point: `3`, `3.14`, `1e-3`, `.5`.
- Strings use double quotes with `\n`, `\t`, `\"`, `\\` escapes.

## Values

| Type | Example | Notes |
| --- | --- | --- |
| Tensor | `3.0`, `[1.0, 2.0]`, `[[1.0],[2.0]]` | The core type. Scalars are rank-0 tensors. |
| Bool | `true`, `false` | From comparisons and logic. |
| String | `"hello"` | For `print` and messages. |
| List | `range(5)`, `[grad(f), 2]` | Heterogeneous; from `[...]` of non-numbers, `list(...)`, or `range`. |
| Record | `{ w: [1.0], b: 0.0 }` | Named fields; access with `.`. |
| Function | `fn(x) = x + 1` | Closures capture their scope. |
| Unit | `()` | The result of `print`, loops, etc. |

A bracketed literal whose elements are all numbers (or nested numeric brackets)
is a tensor. If any element isn't numeric, it's a list. Build a tensor from
computed values with `tensor([...])`.

```rust
[1.0, 2.0, 3.0]           # tensor, shape [3]
[[1.0, 2.0], [3.0, 4.0]]  # tensor, shape [2, 2]
[grad(f), "x", true]      # list
```

## Operators

Lowest to highest precedence:

| Operators | Meaning |
| --- | --- |
| `or` / `\|\|`, `and` / `&&` | short-circuiting logic |
| `==` `!=` `<` `<=` `>` `>=` | comparison (returns Bool); see [Equality](#equality) |
| `+` `-` | add / subtract (elementwise) |
| `*` `/` `%` `@` | multiply / divide / modulo (elementwise), matmul (`@`) |
| `^` | power (right-associative, scalar exponent) |
| unary `-`, `not` / `!` | negation, logical not |

Elementwise operators broadcast NumPy-style: a scalar against a tensor, a row
vector across a matrix, a column vector down its rows, and so on. Two shapes
combine when, aligned from the right, each pair of dimensions is equal or one of
them is 1. `@` covers vector·vector (dot), matrix·vector, vector·matrix, and
matrix·matrix.

## Equality

`==` and `!=` are **deep structural comparison**. Two values are equal when they
have the same type and the same contents, all the way down:

```rust
[1.0, 2.0] == [1.0, 2.0]                       # true (tensors)
[1.0, "x"] == [1.0, "x"]                       # true (lists, element by element)
{ w: [1.0], b: 0.5 } == { w: [1.0], b: 0.5 }   # true (records, field by field)
```

The details:

- A tensor's **shape is part of its value**: `[[1.0, 2.0], [3.0, 4.0]]` and
  `[1.0, 2.0, 3.0, 4.0]` hold the same numbers but are not equal. Numbers compare
  by IEEE rules, so a tensor holding a `NaN` is not equal to itself.
- Lists compare elementwise, and must be the same length.
- Records compare field by field, **matched by name**, so declaration order
  doesn't change the answer: `{ a: 1.0, b: 2.0 } == { b: 2.0, a: 1.0 }`. A record
  with an extra field is not equal.
- Values of different types are never equal — `[1.0] == 1.0` is false, not an
  error.
- Functions have no structure worth walking, so they compare by **identity**: a
  function equals itself, and two separately written `fn(x) = x` do not.
- `!=` is exactly the negation of `==`.

The ordering operators (`<`, `<=`, `>`, `>=`) are only defined on scalars;
applying one to a list, record, string, or non-scalar tensor is an error.

For elementwise comparison of two tensors into a tensor of 1s and 0s, use the
`equal` builtin — `==` on tensors gives one Bool for the whole value.

## Bindings and assignment

```rust
let x = 10     # new binding in the current scope
x = x + 1      # reassign an existing binding (error if not yet bound)
```

`let` always introduces a new variable. Plain assignment updates the nearest
existing binding, which is what makes training loops work.

## Functions

```rust
fn square(x) = x * x       # expression body
fn norm(v) {               # block body; last expression is returned
  let s = sum(v * v)
  sqrt(s)
}
let inc = fn(x) = x + 1    # anonymous function
```

Functions are values and close over their environment:

```rust
fn adder(n) = fn(x) = x + n
let add5 = adder(5)
add5(10)                   # 15
```

`return` exits early; a bare `return` yields `()`.

Parameters may carry shape annotations, and a function may declare its return
shape. These are checked statically (see below); at runtime they're ignored.

```rust
fn matvec(A: [3, 2], x: [2]) -> [3] {
  A @ x
}
```

A dimension can be a concrete size, `_` for an unknown, or a name (a shape
variable). A name used in more than one place must stand for the same size, so
the checker can tie shapes together and verify the return:

```rust
fn matmul2(A: [n, k], B: [k, m]) -> [n, m] {
  A @ B
}
```

Here `k` must match between `A` and `B`, and the result is checked against
`[n, m]`.

## Control flow

`if` is an expression:

```rust
let sign = if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }
```

`while` and `for` are statements:

```rust
while i < n { i = i + 1 }

for k in range(5) { print(k) }      # over a list
for xi in [1.0, 2.0, 3.0] { ... }   # over a 1-D tensor
```

## Indexing and slicing

```rust
let v = [10.0, 20.0, 30.0]
v[0]                  # 10 (scalar)

let m = [[1.0, 2.0], [3.0, 4.0]]
m[1]                  # tensor([3, 4], shape=[2]) — a row
m[1][0]               # 3
```

Indexing a tensor along the first axis returns a scalar (rank-1) or a slice
(higher rank). Lists index directly.

Slicing takes a half-open range along the first axis; either bound may be
omitted. Both indexing (`x[i]`) and slicing (`x[a:b]`) a tensor are
differentiable — gradient flows to the selected element or rows.

```rust
v[1:3]                # tensor([20, 30], shape=[2])
v[:2]                 # first two elements
v[1:]                 # from index 1 to the end
m[0:1]                # the first row, kept as a [1, 2] tensor
range(10)[2:5]        # works on lists too
```

## Differentiation

```rust
grad(f)            # -> function returning df/d(arg0)
grads(f)           # -> function returning [df/d(arg0), df/d(arg1), ...]
value_and_grad(f)  # -> function returning [f(args), df/d(arg0)]
jacobian(f)        # -> function returning the [m, n] Jacobian of a vector output
hessian(f)         # -> function returning the [n, n] Hessian of a scalar output
```

`grad`, `grads`, and `value_and_grad` require the differentiated function to
return a scalar; a gradient has the same shape as the argument it corresponds to,
including nested lists. `jacobian(f)(x)` instead takes a function with a *vector*
output and returns the full matrix of partials — row `i` is the gradient of
output `i` — computed by one reverse-mode pass per output. See
`examples/jacobian.ra`.

```rust
grad(fn(x) = x * x)(4.0)                 # 8
grad(fn(w) = sum(w * w))([1.0, 2.0])     # [2, 4]

let g = grads(fn(a, b) = sum(a * b))([1.0, 2.0], [3.0, 4.0])
g[0]   # [3, 4]   d/da
g[1]   # [1, 2]   d/db
```

Differentiable primitives: `+ - * / % @ ^`, `relu`, `sigmoid`, `tanh`, `exp`,
`log`, `sin`, `cos`, `sqrt`, `sum`, `mean`, `abs`, `pow`.

`hessian(f)(x)` gives the exact matrix of second partial derivatives of a scalar
function — second-order autodiff via forward-mode jets (see `examples/hessian.ra`
for Newton's method). It supports functions built from arithmetic, the unary
math functions, `matmul`, `sum`, `mean`, and the structural ops indexing
(`x[i]`), slicing (`x[a:b]`), `reshape`, `transpose`, `concat`, and `gather`; a
function using an op outside this set raises a clear error. The reverse-mode `grad` remains
first-order, so the general nested form `grad(grad(f))` is not supported — use
`hessian` for second derivatives.

## Shape checking

`raster check` (and the check that runs before `raster run`) infers tensor shapes
and reports mismatches it can prove. It stays quiet when a shape can't be
determined, so dynamic code doesn't produce false alarms.

```
$ raster check bad.ra
bad.ra:3: shape error: shape mismatch in @: [2, 3] @ [2] (inner 3 != 2)
```

Annotations (`[3, 2]`, `[2]`, `[]`, `_` for unknown, or named shape variables)
let you state a contract that the checker enforces at call sites and against the
function body. A shape variable used more than once must resolve to the same
size. Annotated function bodies are also checked at their definition, so a
mistake is caught even if the function is never called.

## Units of measure

Declare a base unit at the top level with `unit`, then annotate scalar
quantities with it. The checker tracks units through arithmetic and reports a
mismatch, the same way it does for shapes — but units are erased at runtime, so
annotated code runs as plain numbers with zero overhead.

```rust
unit USD
unit share

fn notional(px: USD/share, qty: share) -> USD { px * qty }
```

An annotation is a single unit (`USD`) or a compound expression: a product
(`USD*share`), a quotient (`USD/year`), or a power (`year^-1`, `USD^2`). The
checker applies the natural rules:

- `*` multiplies units, `/` divides them, and `^` with a constant integer
  exponent raises them (`sqrt` halves them).
- `+`, `-`, `%`, and comparisons require both sides to share a unit — adding
  `USD` to `share` is an error.
- `matmul`/`dot` multiply the operand units; indexing and slicing preserve them.
- `exp`, `log`, `sin`, `cos`, `tanh`, and `sigmoid` require a dimensionless
  argument (their result is dimensionless).

A bare numeric literal is dimensionless. To give a value a unit, annotate the
`let` that binds it — the literal is adopted into the declared unit:

```rust
let price: USD/share = 150.0
let qty: share = 200.0
let value = notional(price, qty)   # inferred: USD
```

Naming a unit that was never declared (a typo like `USD/yr`) is a checker error.
Code with no unit annotations is entirely dimensionless and unaffected.

## Records

A record groups named fields. Fields are accessed with `.`.

```rust
let p = { w: [1.0, 2.0], b: 0.5 }
p.w                   # tensor([1, 2], shape=[2])
p.b                   # 0.5
{ inner: { x: 3.0 } }.inner.x   # 3
```

`grad` follows record structure: if a loss takes a record of parameters, the
gradient is a record with the same fields.

```rust
fn loss(m) = sum(m.w) + m.b
grad(loss)({ w: [1.0, 2.0], b: 0.5 })   # { w: [1, 1], b: 1 }
```

This makes a record a natural container for a model's parameters. A `{` starts a
record only when it is followed by `name:`; otherwise it is a block.

A record type can be declared and used to annotate a parameter. The checker then
verifies that the record passed in has the declared fields with the declared
shapes, and that field accesses name real fields:

```rust
type Model = { w: [3, 2], b: [3] }

fn predict(m: Model, x: [2]) -> [3] {
  m.w @ x + m.b
}
```

Accessing a field a record doesn't have (`m.wieght`) is a checker error, whether
the record is a literal or a declared type.

## Imports

There are two kinds of import path, and the spelling tells you which is which.

```rust
import "std/nn"             # a standard-library module (ships inside the binary)
import "helpers.ra"         # a file, relative to the importing file
```

A path beginning with `std/` names a **module** of the standard library, not a
file: no extension, no directory, and it means the same thing from anywhere,
because the library is compiled into the `raster` binary. `std/` is reserved — a
directory called `std` next to your program does not shadow it. Every other path
is a **file**, resolved relative to the importing file first, then the working
directory; `import "./std/local.ra"` reaches a real directory named `std`.

Either kind can be namespaced:

```rust
import "std/nn"             # drops the module's definitions into this scope
import "std/nn" as nn       # binds them as a namespace record instead
```

A plain import evaluates the module and adds its top-level definitions to the
importing scope; each module loads once, so re-imports and cycles are fine. An
`as name` import instead evaluates it into its own scope and binds a record of
its definitions under `name`, so you call `nn.dense(...)`. That record's fields
are in the module's declaration order, so printing or iterating a namespace gives
the same result on every run.

A standard-library module may only import other `std/` modules — it has no
directory of its own to resolve a relative path against.

To work on the library itself without rebuilding, set `RASTER_STD` to a directory
of `.ra` files; it replaces the embedded library wholesale, so `import "std/nn"`
reads `$RASTER_STD/nn.ra`. Unset it and you are back to the copy in the binary.

## Standard library

Elementwise math (differentiable): `relu`, `sigmoid`, `tanh`, `exp`, `log`,
`sin`, `cos`, `sqrt`, `square`, `abs`, `pow(x, p)`, `clip(x, lo, hi)`.

Elementwise combine: `maximum(a, b)`, `minimum(a, b)`, `where(cond, a, b)`, and
the comparisons `greater`, `less`, `greater_equal`, `less_equal`, `equal`
(each returns a tensor of 1s and 0s).

Reductions: `sum`, `mean`, `max`, `min`, `prod` and `median` reduce the whole
tensor to a scalar, or one axis with a second argument (`sum(t, 0)`).
`argmax(t[, axis])` gives the index of the maximum.

All of them are differentiable, including the two order-based ones, though what
that means is worth being clear about. `median` routes the whole gradient to
whichever element was selected, the way `max` does, and splits it in half
between the middle two when the run has even length. `prod` gives each factor
the product of the others, which is the total divided by that factor — except
where a factor is zero and the division is not available. There, a single zero
takes the product of the rest and everything else gets nothing, and two or more
zeros flatten the gradient entirely, because every product of the others still
contains a zero. `softmax(t[, axis])` and `logsumexp(t[, axis])` default to
the last axis.

`split(t, n | sizes[, axis])` is the inverse of `concat`, returning a list of
pieces. A number means that many equal pieces (`split(x, 2, 1)` halves the
columns) and a list means those exact lengths (`split(x, list(1, 3), 1)`). The
axis defaults to 0. The sizes must account for the axis exactly and an equal
split must divide evenly; both are errors rather than ragged output, because a
split that quietly loses a row shows up later as a wrong loss rather than as a
crash. Each piece keeps its own gradient path, so
`concat(split(t, 2, 1), 1)` is `t` in both directions.

`broadcast_to(t, ...shape)` expands a tensor to a given shape under the usual
right-aligned rules, where every axis must already match or be 1. It is what
you need after a reduction: reducing axis 1 of a `[2, 3]` gives a `[2]`, and
`[2]` will not broadcast back against `[2, 3]`, because alignment is from the
right. `broadcast_to(reshape(mu, list(2, 1)), list(2, 3))` puts it back. Other
array libraries spell this as `keepdims=True` on the reduction itself; here it
is an operation, and `num.keep` wraps the two steps.

Sorting: `sort(t[, axis[, descending]])` and `argsort` give the values and the
positions; `topk(t, k[, axis[, smallest]])` and `argtopk` keep the k largest,
largest first, shrinking that axis to k. All four default to the last axis,
because sorting a matrix almost always means sorting each row. The flags are
numbers, since a comparison in Raster already yields 1 and 0.

`sort` and `topk` are differentiable and exactly so. Sorting is a permutation
and the derivative of a permutation is its inverse: whatever gradient arrives at
the element now in a position belongs to whichever element started there. A
value outside the top k does not move the output at all, so its gradient is
zero, which is correct rather than a simplification. The sort is stable, so ties
keep their original order and therefore their own gradients.

`argmin(t[, axis])` is `argmax`'s counterpart, and `flip(t[, axis])` reverses
along an axis. `flip` is differentiable and exactly so, since a reversal is a
permutation and is its own inverse, which makes the backward pass the same
reversal. All three default to the last axis. Ties in `argmax` and `argmin` go to
the first occurrence, the same rule the cumulative extremes and the sort use.

`roll(t, shift[, axis])` shifts along an axis and wraps what falls off the end
back to the start; `diff(t[, axis])` is the difference between neighbours,
shortening that axis by one. Both are differentiable. A positive shift moves
elements towards the end, so `roll(x, 1)` is the previous value and
`x - roll(x, 1)` compares a series with its own past. `diff` shortens rather than
pads, because there is no honest first difference: a zero there claims "no
change" about data that does not exist, and it is exactly the claim whatever
consumes the series next will believe.

`argsort` and `argtopk` are not differentiable, and not by omission: an index
does not move when an input moves slightly, then jumps when two values cross.
The derivative is zero almost everywhere and undefined on the boundaries.

Cumulative scans (preserving length): `cumsum`, `cumprod`, `cummax`, `cummin`.
These build signals, equity curves, and running peaks, and they are
differentiable: `cumsum` and `cumprod` have exact gradients (`cumprod` handles
zeros in the series), and `cummax`/`cummin` send each output's gradient to the
element the running extreme came from, ties going to the earlier one.

Each takes an optional axis: `cumsum(t)` scans the tensor's elements in order
and `cumsum(t, 1)` scans along axis 1, one run per row, keeping the shape. The
split follows the reductions, where `sum(t)` covers everything and `sum(t, 0)`
works per axis. On a 1-D tensor, which is what a sequence is, the two forms are
the same thing, so the axis is a widening rather than a second meaning.
Elementwise rounding `floor`, `ceil`, and `round` are forward-only (their
derivative is zero wherever it exists), handy for turning random draws into
integer ids.

Linear algebra / shape: `matmul(a, b)` / `dot(a, b)` (same as `@`),
`transpose(t[, ...axes])`, `reshape(t, ...shape)`, `concat(list, axis)`,
`einsum(spec, ...tensors)`.

Indexing / batching: `gather(x, indices)` selects rows of `x` (its first axis)
by an index list or 1-D tensor, and is differentiable — the gradient scatters
back to the selected rows, so repeated indices (embedding lookups) accumulate.
`permutation(n)` returns a seeded random ordering of `0..n-1` (for shuffling),
and `int(x)` truncates a scalar toward zero.

Convolutions (differentiable): `conv2d(input, weight)` is a 2-D cross-correlation
with `input` shaped `[Cin, H, W]` and `weight` shaped `[Cout, Cin, KH, KW]`,
producing `[Cout, H-KH+1, W-KW+1]` (valid padding, unit stride).
`maxpool2d(input, k)` does non-overlapping `k×k` max pooling over each channel of
a `[C, H, W]` tensor. `grad` flows through both, so a convolutional net trains
like any other model — see `examples/cnn.ra`.

`einsum` is a general Einstein-summation contraction and is differentiable:

```rust
einsum("ij,jk->ik", A, B)   # matrix multiply
einsum("ij->ji", A)         # transpose
einsum("ij->i", A)          # sum over the second axis
einsum("i,ij,j->", x, W, y) # bilinear form x' W y
```

Each label names an axis; repeated labels across operands are summed, and only
the labels in the output remain. Omitting `->` keeps the labels that appear once.
(A label repeated within one operand — a trace or diagonal — isn't supported
yet.)

Construction: `tensor(list)`, `scalar(x)`, `zeros(...shape)`, `ones(...shape)`,
`fill(value, ...shape)`, `eye(n)`, `randn(...shape)` (standard normal),
`rand(...shape)` (uniform), `seed(n)`. Shapes may be separate args or a list.
Randomness is **deterministic by default** — a program gives the same result
every run — and `seed(n)` chooses the starting point. That reproducibility
matters for model governance and audit.

Lists / higher-order: `range(...)`, `list(...)`, `map(f, xs)`, `zip(...)`,
`fold(f, init, xs)`, `append(xs, x)`, `enumerate(xs)`, `len(x)`.

Trees (tensors nested in lists/records): `map_leaves(f, tree)` applies `f` to
every tensor leaf; `zip_leaves(f, trees)` walks a list of same-shaped trees in
parallel, calling `f` with the list of leaves at each position. Optimizers use
these, so they work on any model structure.

Inspection: `shape(t)`, `item(t)`, `str(x)`, `print(...)`.

Data: `read_csv(path)` loads a file of numeric rows (comma- or
whitespace-separated, `#` lines skipped) into a `[rows, cols]` tensor.

Persistence: `save(value, path)` writes any value — a tensor, a record or list
of tensors (a model's whole parameter tree), or a fitted `gbm` model — to a file
in an exact binary format, and `load(path)` reads it back. Paths are relative to
the running script. This is the deploy path: train once, `save` the model, and
ship it with the single binary for inference (`examples/save_load.ra`).

Frames: a frame is a record whose fields are named column tensors, so field
access, slicing, and `grad` all work on it. `read_frame(path)` loads a CSV whose
first row is a header into such a record; `write_frame(frame, path)` writes one
back. `columns(rec)` lists the field names, `field(rec, name)` looks one up by
string, and `with_field(rec, name, value)` returns a copy with a field set. See
`examples/frames.ra`.

Gradient-boosted trees: `gbm_fit(X, y)` (or `gbm_fit(X, y, opts)`) trains a
native gradient-boosting model on a `[n, d]` feature matrix and an `[n]`
target/label vector, and `gbm_predict(model, X)` scores a `[n, d]` matrix into an
`[n]` vector. `opts` is a record of hyperparameters — `rounds`, `learning_rate`,
`max_depth`, `min_leaf`, `lambda`, `gamma`, and `objective` (`"squared"` for
regression, `"logistic"` for binary classification, where predictions are
probabilities). The engine is pure Go and deterministic. See `examples/gbm.ra`.

Libraries written in Raster ship inside the binary and are imported as
`std/<module>`: `std/nn` (layers including `dense`, `conv`, `embed`, and
`self_attention`; activations, initializers, losses), `std/optim` (SGD,
momentum, Adam), `std/data` (`standardize`, `train_test_split`, `shuffle` for
real training loops — see `examples/minibatch.ra`), and `std/backtest`
(returns, moving averages, equity curves, drawdown, Sharpe, Sortino, volatility,
CAGR). Their sources are the `.ra` files in `std/` in the repository. The
optimizers are container-agnostic — the same `sgd_step`/`adam_step`
update a model held in a positional list or a named record. The backtest Sharpe
and Sortino are differentiable in the return series, so a smooth signal can be
tuned by gradient ascent through the backtest (`examples/signal_opt.ra`).

## Example

```rust
# Fit y = X w + b by gradient descent.
let X = [[1.0, 1.0], [2.0, 1.0], [1.0, 3.0]]
let y = [-0.5, 1.5, -6.5]

fn loss(w, b) {
  let err = X @ w + b - y
  mean(err * err)
}

let w = [0.0, 0.0]
let b = 0.0
for step in range(400) {
  let g = grads(loss)(w, b)
  w = w - g[0] * 0.05
  b = b - g[1] * 0.05
}
print("w =", w, "b =", b)
```
