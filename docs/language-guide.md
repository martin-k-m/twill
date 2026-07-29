# The Aster Language Guide

This is the reference for Aster v0.1. It covers the whole language — which is
small by design.

## Running programs

```bash
node src/cli.ts path/to/program.ast   # run a file
node src/cli.ts                        # start the REPL
```

In the REPL, the value of each line is printed. `:help` and `:quit` control it.

## Lexical structure

- **Comments** run from `#` to the end of the line.
- **Whitespace and newlines are insignificant.** A `;` may separate statements
  but is never required.
- **Identifiers** match `[A-Za-z_][A-Za-z0-9_]*`.
- **Numbers** are floating point: `3`, `3.14`, `1e-3`, `.5`.
- **Strings** use double quotes with `\n`, `\t`, `\"`, `\\` escapes.

## Values

Every value is one of:

| Type | Example | Notes |
| --- | --- | --- |
| Tensor | `3.0`, `[1.0, 2.0]`, `[[1.0],[2.0]]` | The core type. Scalars are rank-0 tensors. |
| Bool | `true`, `false` | From comparisons and logic. |
| String | `"hello"` | For `print` and messages. |
| List | `[grad(f), 2]`, `range(5)` | Heterogeneous; created by `[...]` of non-numbers, `list(...)`, or `range`. |
| Function | `fn(x) = x + 1` | Closures capture their defining scope. |
| Unit | `()` | The result of statements like `print` or a `while` loop. |

### Tensors vs. lists

A bracketed literal whose elements are **all numbers (or nested numeric
brackets)** is a **tensor**:

```rust
[1.0, 2.0, 3.0]          # tensor, shape [3]
[[1.0, 2.0], [3.0, 4.0]] # tensor, shape [2, 2]
```

If any element is not numeric, it is a **list**:

```rust
[grad(f), "x", true]     # list
```

Build a tensor from computed values with `tensor([...])`.

## Operators

From lowest to highest precedence:

| Operators | Meaning |
| --- | --- |
| `or` / `\|\|`, `and` / `&&` | Short-circuiting logic |
| `==` `!=` `<` `<=` `>` `>=` | Comparison (return Bool) |
| `+` `-` | Add / subtract (elementwise) |
| `*` `/` `%` `@` | Multiply / divide / modulo (elementwise), matmul (`@`) |
| `^` | Power (right-associative, scalar exponent) |
| unary `-`, `not` / `!` | Negation, logical not |

Elementwise ops broadcast a scalar against a tensor. `@` performs vector·vector
(dot), matrix·vector, vector·matrix, and matrix·matrix products.

## Bindings and assignment

```rust
let x = 10        # introduce a new binding
x = x + 1         # reassign an existing binding (error if not yet bound)
```

`let` always creates a new variable in the current scope. Plain assignment
updates the nearest existing binding, which is what makes training loops work.

## Functions

```rust
fn square(x) = x * x              # expression body
fn norm(v) {                      # block body — last expression is returned
  let s = sum(v * v)
  sqrt(s)
}
let inc = fn(x) = x + 1           # anonymous function (lambda)
```

Functions are first-class values and close over their environment:

```rust
fn adder(n) = fn(x) = x + n
let add5 = adder(5)
add5(10)          # 15
```

`return` exits a function early; a bare `return` yields `()`.

## Control flow

`if` is an expression and yields a value:

```rust
let sign = if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }
```

`while` and `for` are statements (they yield `()`):

```rust
while i < n { i = i + 1 }

for k in range(5) { print(k) }     # iterate a list
for xi in [1.0, 2.0, 3.0] { ... }  # iterate a 1-D tensor
```

## Indexing

```rust
let v = [10.0, 20.0, 30.0]
v[0]                 # 10  (scalar)

let m = [[1.0, 2.0], [3.0, 4.0]]
m[1]                 # tensor([3, 4], shape=[2])  — a row
m[1][0]              # 3
```

Indexing a tensor along the first axis returns a scalar (for 1-D) or a slice
(for higher rank). Lists index directly.

## Automatic differentiation

The defining feature. `grad`, `grads`, and `value_and_grad` turn a function
into its derivative.

```rust
grad(f)            # -> fn returning df/d(arg0)
grads(f)           # -> fn returning [df/d(arg0), df/d(arg1), ...]
value_and_grad(f)  # -> fn returning [f(args), df/d(arg0)]
```

The differentiated function **must return a scalar** (rank-0 tensor) — the usual
shape of a loss. Gradients have the same shape as the argument they correspond
to.

```rust
# scalar -> scalar
grad(fn(x) = x * x)(4.0)                 # 8

# tensor -> scalar
grad(fn(w) = sum(w * w))([1.0, 2.0])     # [2, 4]

# multiple parameters
let g = grads(fn(a, b) = sum(a * b))([1.0, 2.0], [3.0, 4.0])
g[0]   # [3, 4]   (d/da)
g[1]   # [1, 2]   (d/db)
```

Differentiable primitives: `+ - * / @ ^ %`, `relu`, `sigmoid`, `tanh`, `exp`,
`log`, `sin`, `cos`, `sqrt`, `sum`, `mean`, `abs`, `pow`.

> **Note:** v0.1 uses reverse-mode autodiff and does not yet support
> higher-order derivatives (`grad(grad(f))`). See the roadmap in
> [`design.md`](design.md).

## Standard library

**Math (differentiable):** `relu`, `sigmoid`, `tanh`, `exp`, `log`, `sin`,
`cos`, `sqrt`, `abs`, `pow(x, p)`, `sum`, `mean`.

**Linear algebra:** `matmul(a, b)` / `dot(a, b)` (same as `@`), `transpose(m)`.

**Construction:** `tensor(list)`, `scalar(x)`, `zeros(...shape)`,
`ones(...shape)`, `fill(value, ...shape)`, `eye(n)`, `randn(...shape)`
(standard normal), `rand(...shape)` (uniform [0,1)). Shapes may be passed as
separate args (`zeros(2, 3)`) or a list (`zeros([2, 3])`).

**Inspection:** `shape(t)` (list of dims), `len(x)`, `item(t)` (single-element
tensor → scalar).

**Utility:** `range(n)` / `range(start, end)` / `range(start, end, step)`,
`list(...)`, `str(x)`, `print(...)`.

## A complete example

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
