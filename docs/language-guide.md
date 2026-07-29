# Raster language guide

This is the reference for Raster v0.9. The language is small, so this is short.

## Running programs

```bash
raster path/to/program.ra    # shape-check, then run
raster run path/to/program.ra
raster check path/to/program.ra   # shape-check only
raster                             # REPL
```

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
| `==` `!=` `<` `<=` `>` `>=` | comparison (returns Bool) |
| `+` `-` | add / subtract (elementwise) |
| `*` `/` `%` `@` | multiply / divide / modulo (elementwise), matmul (`@`) |
| `^` | power (right-associative, scalar exponent) |
| unary `-`, `not` / `!` | negation, logical not |

Elementwise operators broadcast NumPy-style: a scalar against a tensor, a row
vector across a matrix, a column vector down its rows, and so on. Two shapes
combine when, aligned from the right, each pair of dimensions is equal or one of
them is 1. `@` covers vector·vector (dot), matrix·vector, vector·matrix, and
matrix·matrix.

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
omitted. Slicing a tensor is differentiable.

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
```

The differentiated function must return a scalar. A gradient has the same shape
as the argument it corresponds to, including nested lists.

```rust
grad(fn(x) = x * x)(4.0)                 # 8
grad(fn(w) = sum(w * w))([1.0, 2.0])     # [2, 4]

let g = grads(fn(a, b) = sum(a * b))([1.0, 2.0], [3.0, 4.0])
g[0]   # [3, 4]   d/da
g[1]   # [1, 2]   d/db
```

Differentiable primitives: `+ - * / % @ ^`, `relu`, `sigmoid`, `tanh`, `exp`,
`log`, `sin`, `cos`, `sqrt`, `sum`, `mean`, `abs`, `pow`.

Note: autodiff is reverse-mode and first-order, so `grad(grad(f))` is not
supported.

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

```rust
import "std/nn.ra"          # drops the file's definitions into this scope
import "std/nn.ra" as nn    # binds them as a namespace record instead
```

A plain import evaluates another file and adds its top-level definitions to the
importing scope; each file loads once, so re-imports and cycles are fine. An
`as name` import instead evaluates the file into its own scope and binds a record
of its definitions under `name`, so you call `nn.dense(...)`. Paths are resolved
relative to the importing file first, then the working directory.

## Standard library

Elementwise math (differentiable): `relu`, `sigmoid`, `tanh`, `exp`, `log`,
`sin`, `cos`, `sqrt`, `square`, `abs`, `pow(x, p)`, `clip(x, lo, hi)`.

Elementwise combine: `maximum(a, b)`, `minimum(a, b)`, `where(cond, a, b)`, and
the comparisons `greater`, `less`, `greater_equal`, `less_equal`, `equal`
(each returns a tensor of 1s and 0s).

Reductions: `sum`, `mean`, `max`, `min` reduce the whole tensor to a scalar, or
one axis with a second argument (`sum(t, 0)`). `argmax(t[, axis])` gives the
index of the maximum. `softmax(t[, axis])` and `logsumexp(t[, axis])` default to
the last axis.

Linear algebra / shape: `matmul(a, b)` / `dot(a, b)` (same as `@`),
`transpose(t[, ...axes])`, `reshape(t, ...shape)`, `concat(list, axis)`,
`einsum(spec, ...tensors)`.

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
`rand(...shape)` (uniform). Shapes may be separate args or a list.

Lists / higher-order: `range(...)`, `list(...)`, `map(f, xs)`, `zip(...)`,
`fold(f, init, xs)`, `append(xs, x)`, `enumerate(xs)`, `len(x)`.

Trees (tensors nested in lists/records): `map_leaves(f, tree)` applies `f` to
every tensor leaf; `zip_leaves(f, trees)` walks a list of same-shaped trees in
parallel, calling `f` with the list of leaves at each position. Optimizers use
these, so they work on any model structure.

Inspection: `shape(t)`, `item(t)`, `str(x)`, `print(...)`.

Data: `read_csv(path)` loads a file of numeric rows (comma- or
whitespace-separated, `#` lines skipped) into a `[rows, cols]` tensor.

Libraries written in Raster live in `std/`: `nn.ra` (layers, activations,
initializers, losses) and `optim.ra` (SGD, momentum, Adam). The optimizers are
container-agnostic — the same `sgd_step`/`adam_step` update a model held in a
positional list or a named record.

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
