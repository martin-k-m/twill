# Aster

Aster is a small programming language for numeric and machine-learning code.
Tensors are the built-in data type, differentiation is part of the language
(`grad`, not a library call), and a static checker catches shape mistakes
before a program runs.

It's an early prototype (v0.2). The reference implementation is a single Go
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

Aster is an experiment in the other direction — a language built around
differentiable tensor programs from the start. Three things fall out of that:

- Tensors are the primitive. Every number is a rank-0 tensor, vectors and
  matrices are literals, and `@` is matrix multiply.
- `grad` is a keyword-like builtin backed by a real reverse-mode autodiff
  engine. No `requires_grad`, no `.backward()`, no optimizer objects.
- Shapes are checked statically. `[2,3] @ [4]` is an error you see before the
  program runs, not a stack trace halfway through training.

The language is deliberately small. The whole implementation is a few thousand
lines of Go you can read in a sitting.

## Build and run

You need Go 1.23 or newer.

```bash
git clone https://github.com/martin-k-m/aster.git
cd aster
go build -o aster ./cmd/aster
```

That produces a single `aster` binary.

```bash
./aster examples/autodiff.ast      # run a program
./aster check examples/shapes.ast  # shape-check without running
./aster                            # start the REPL
go test ./...                      # run the test suite
```

Without building first, `go run ./cmd/aster <file.ast>` works too.

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
see `examples/nn_xor.ast`.

## Shape checking

Before running, Aster infers tensor shapes and reports the ones that can't line
up. It only flags a mismatch when it's certain, so dynamic code (shapes that
depend on runtime values) is left alone rather than guessed at.

```
$ aster check bad.ast
bad.ast:3: shape error: shape mismatch in @: [2, 3] @ [2] (inner 3 != 2)
```

Function parameters can carry optional shape annotations that document and
enforce a contract:

```rust
fn matvec(A: [3, 2], x: [2]) -> [3] {
  A @ x
}
```

Use `[2]` for a vector, `[3, 2]` for a matrix, `[]` for a scalar, and `_` for a
dimension you don't want to pin down.

## A small standard library

`std/nn.ast` is a neural-network toolkit written in Aster itself — dense layers,
a few activations and losses, and an SGD step built from `map`/`zip`. Import it
with `import "std/nn.ast"` (adjust the path to your file). `examples/nn_xor.ast`
trains a network with it.

## Layout

```
cmd/aster/           the `aster` command (run / check / repl)
internal/lexer/      source text -> tokens
internal/parser/     tokens -> AST
internal/ast/        AST node types
internal/tensor/     the differentiable tensor engine
internal/value/      runtime values and environments
internal/interp/     the tree-walking interpreter + builtins
internal/checker/    static shape analysis
std/nn.ast           neural-net library written in Aster
examples/            runnable programs
editors/vscode/      syntax highlighting for .ast files
docs/                language guide and design notes
```

## What's not done yet

This is a prototype, and some of it is deliberately left for later:

- It's interpreted. Tensor ops loop in Go; there's no vectorized or GPU
  backend yet. The interpreter is the reference for the semantics.
- Broadcasting is scalar-to-tensor only, not full NumPy-style broadcasting.
- Autodiff is reverse-mode and first-order; `grad(grad(f))` isn't supported.
- There are no records or a module namespace yet; imports drop definitions into
  the global scope.

The [design notes](docs/design.md) go into the roadmap.

## License

[MIT](LICENSE).
