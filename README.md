# Raster

[![CI](https://github.com/martin-k-m/raster/actions/workflows/ci.yml/badge.svg)](https://github.com/martin-k-m/raster/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/martin-k-m/raster?sort=semver)](https://github.com/martin-k-m/raster/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Go](https://img.shields.io/badge/go-1.23%2B-00ADD8.svg)](go.mod)

Raster is a small programming language for numeric and machine-learning code.
Tensors are the built-in data type, differentiation is part of the language
(`grad`, not a library call), and a static checker catches shape mistakes
before a program runs.

It's an early prototype (v0.6). The reference implementation is a single Go
binary with no dependencies, so it's easy to build and easy to read.

```rust
fn f(x) = x * x * x     # f(x) = x^3
let df = grad(f)        # df(x) = 3x^2
print(df(2.0))          # 12
```

## Why

Most machine-learning code today is Python plus a numeric framework. That works,
but the framework is bolted onto a language that predates it: autodiff is a
runtime library, tensor shapes are only known once you run the code, and a lot
of glue sits between the math and the program.

Raster is an experiment in the other direction — a language built around
differentiable tensor programs from the start. Three things fall out of that:

- Tensors are the primitive. Every number is a rank-0 tensor, vectors and
  matrices are literals, and `@` is matrix multiply.
- `grad` is a keyword-like builtin backed by a real reverse-mode autodiff
  engine. No `requires_grad`, no `.backward()`, no optimizer objects.
- Shapes are checked statically. `[2,3] @ [4]` is an error you see before the
  program runs, not a stack trace halfway through training.

The language is deliberately small. The whole implementation is a few thousand
lines of Go you can read in a sitting.

## Install

Download a prebuilt binary for your platform from the
[releases page](https://github.com/martin-k-m/raster/releases) and put it on your
`PATH`. With a Go toolchain (1.23+) you can also:

```bash
go install github.com/martin-k-m/raster/cmd/raster@latest
```

Or build from source:

```bash
git clone https://github.com/martin-k-m/raster.git
cd raster
go build -o raster ./cmd/raster
```

## Run

```bash
raster examples/autodiff.ra      # run a program
raster check examples/shapes.ra  # shape-check without running
raster                           # start the REPL (multi-line aware)
```

The REPL keeps reading until brackets balance, so you can define block-body
functions interactively. Without installing, `go run ./cmd/raster <file.ra>`
works too, and `go test ./...` runs the suite.

## The language in a few lines

```rust
# Comments start with '#'.

let a = 3.0
let v = [1.0, 2.0, 3.0]           # a vector, shape [3]
let m = [[1.0, 2.0], [3.0, 4.0]]  # a matrix, shape [2, 2]

let d  = v @ v                    # dot product -> 14
let mv = m @ [1.0, 1.0]           # matrix-vector -> [3, 7]

# Functions are one expression or a block; the last expression is returned.
fn rms(t) {
  let n = len(t)
  sqrt(sum(t * t) / n)
}

# Loops for training code.
let total = 0.0
for i in range(10) { total = total + i }

# Differentiation.
fn energy(w) = sum(relu(w) * relu(w)) / 2.0
let g = grad(energy)([-1.0, 2.0, -3.0, 4.0])   # [0, 2, 0, 4]
```

The [language guide](docs/language-guide.md) covers everything; the
[design notes](docs/design.md) explain how it works and what's next.

## Differentiation

`grad`, `grads`, and `value_and_grad` turn a function into its derivative.

| Builtin | Returns |
| --- | --- |
| `grad(f)` | a function computing `df/d(arg0)`, for scalar or tensor args |
| `grads(f)` | a function returning the gradient of every argument, as a list |
| `value_and_grad(f)` | a function returning `[f(x), df/d(arg0)]` |

The function being differentiated has to return a scalar, as a loss does. The
autodiff graph is only built while a value is being differentiated, so ordinary
evaluation doesn't pay for it. Gradients also follow the structure of their
argument, so a model held in a list gets a matching list of gradients back —
see `examples/nn_xor.ra`.

## Shape checking

Before running, Raster infers tensor shapes and reports the ones that can't line
up. It only flags a mismatch when it's certain, so dynamic code (shapes that
depend on runtime values) is left alone rather than guessed at.

```
$ raster check bad.ra
bad.ra:3: shape error: shape mismatch in @: [2, 3] @ [2] (inner 3 != 2)
```

Function parameters can carry optional shape annotations that document and
enforce a contract:

```rust
fn matvec(A: [3, 2], x: [2]) -> [3] {
  A @ x
}
```

Use `[2]` for a vector, `[3, 2]` for a matrix, `[]` for a scalar, and `_` for a
dimension you don't want to pin down. A dimension can also be a name (a shape
variable): a name used more than once must be the same size, which lets the
checker tie shapes together and verify the return —
`fn mm(A: [n, k], B: [k, m]) -> [n, m]`.

## Tensors and operations

Elementwise ops broadcast NumPy-style (a row vector across a matrix, a column
against rows, a scalar against anything), and the gradients broadcast back
correctly. Beyond arithmetic and `@`, the built-in ops include `relu`,
`sigmoid`, `tanh`, `exp`, `log`, `sqrt`, `square`, `abs`, `clip`; `softmax` and
`logsumexp`; `maximum`, `minimum`, `where`, and elementwise comparisons; the
reductions `sum`, `mean`, `max`, `min`, and `argmax` (with an optional axis);
and shape ops `reshape`, `transpose`, and `concat`. Tensors and lists also
support differentiable first-axis slicing (`v[1:3]`, `m[:2]`). See the
[language guide](docs/language-guide.md) for the full list.

## A small standard library

The `std/` libraries are written in Raster itself. `std/nn.ra` has dense layers,
activations (`gelu`, `softplus`, ...), initializers (He, Xavier), and losses
including softmax cross-entropy. `std/optim.ra` has SGD, momentum, and Adam that
operate on a model held as a list of tensors. Import with `import "std/nn.ra"`
(it pulls in the optimizers too).

`examples/classifier.ra` trains a 3-class MLP with softmax cross-entropy and
Adam; `examples/nn_xor.ra` is a smaller net using `grad` over a whole
parameter list.

Load your own data with `read_csv("data.csv")`, which returns a `[rows, cols]`
tensor (comma- or whitespace-separated, `#` lines skipped).

Parameters can also live in a record with named fields instead of a positional
list. `grad` follows the record structure, so differentiating a loss over a
record returns a record of gradients — see `examples/records.ra`. Libraries can
be imported as a namespace with `import "std/nn.ra" as nn` and called as
`nn.dense(...)`.

## Layout

```
cmd/raster/          the `raster` command (run / check / repl)
internal/lexer/      source text -> tokens
internal/parser/     tokens -> AST
internal/ast/        AST node types
internal/tensor/     the differentiable tensor engine
internal/value/      runtime values and environments
internal/interp/     the tree-walking interpreter + builtins
internal/checker/    static shape analysis
std/                 libraries written in Raster (nn.ra, optim.ra)
examples/            runnable .ra programs
editors/vscode/      syntax highlighting for .ra files
docs/                language guide and design notes
```

## What's not done yet

This is a prototype, and some of it is deliberately left for later:

- It's interpreted. Tensor ops loop in Go; there's no vectorized or GPU
  backend yet. The interpreter is the reference for the semantics.
- Autodiff is reverse-mode and first-order; `grad(grad(f))` isn't supported.
- The shape checker is best-effort, not a full type system — it catches
  mismatches when shapes are statically knowable and stays quiet otherwise.
- There are no records or a module namespace yet; imports drop definitions into
  the global scope.

The [design notes](docs/design.md) go into the roadmap.

## License

[MIT](LICENSE).
