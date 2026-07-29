# Aster language guide

This is the reference for Aster v0.2. The language is small, so this is short.

## Running programs

```bash
aster path/to/program.ast    # shape-check, then run
aster run path/to/program.ast
aster check path/to/program.ast   # shape-check only
aster                             # REPL
```

Pass `--no-check` to run without the static shape check. In the REPL, each line's
value is printed; `:help` and `:quit` do the obvious things.

## Lexical structure

- Comments run from `#` to end of line.
- Whitespace is insignificant, with one exception: a `+` or `-` at the start of
  a line begins a new statement instead of continuing the previous one. To
  break a long expression across lines, end the line with the operator.
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

Elementwise operators broadcast a scalar against a tensor. `@` covers
vector·vector (dot), matrix·vector, vector·matrix, and matrix·matrix.

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

## Indexing

```rust
let v = [10.0, 20.0, 30.0]
v[0]                  # 10 (scalar)

let m = [[1.0, 2.0], [3.0, 4.0]]
m[1]                  # tensor([3, 4], shape=[2]) — a row
m[1][0]               # 3
```

Indexing a tensor along the first axis returns a scalar (rank-1) or a slice
(higher rank). Lists index directly.

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

Note: v0.2 is reverse-mode and first-order, so `grad(grad(f))` is not supported.

## Shape checking

`aster check` (and the check that runs before `aster run`) infers tensor shapes
and reports mismatches it can prove. It stays quiet when a shape can't be
determined, so dynamic code doesn't produce false alarms.

```
$ aster check bad.ast
bad.ast:3: shape error: shape mismatch in @: [2, 3] @ [2] (inner 3 != 2)
```

Annotations (`[3, 2]`, `[2]`, `[]`, `_` for unknown) let you state a contract
that the checker enforces at call sites and against the function body.

## Imports

```rust
import "std/nn.ast"     # relative to the importing file, then the working dir
```

An import evaluates another file and drops its top-level definitions into the
global scope. Each file is loaded once, so re-imports and cycles are fine. Paths
are resolved relative to the importing file first, then the working directory.

## Standard library

Math (differentiable): `relu`, `sigmoid`, `tanh`, `exp`, `log`, `sin`, `cos`,
`sqrt`, `abs`, `pow(x, p)`, `sum`, `mean`.

Linear algebra: `matmul(a, b)` / `dot(a, b)` (same as `@`), `transpose(m)`.

Construction: `tensor(list)`, `scalar(x)`, `zeros(...shape)`, `ones(...shape)`,
`fill(value, ...shape)`, `eye(n)`, `randn(...shape)` (standard normal),
`rand(...shape)` (uniform). Shapes may be separate args or a list.

Lists / higher-order: `range(...)`, `list(...)`, `map(f, xs)`, `zip(...)`,
`len(x)`.

Inspection: `shape(t)`, `item(t)`, `str(x)`, `print(...)`.

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
