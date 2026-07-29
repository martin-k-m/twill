# Aster

**A lightweight, tensor-first programming language with automatic differentiation built into the language itself.**

Aster treats tensors as the primitive data type and `grad` as a language keyword — not a library import. If you write a function, you can differentiate it. The result is ML code that reads like the math it implements, with none of the framework ceremony.

```rust
# The gradient of any function is just grad(f).
fn f(x) = x * x * x        # f(x) = x³
let df = grad(f)           # df(x) = 3x²

print(df(2.0))             # 12
```

```rust
# Train a model in a dozen lines — no tape, no optimizer object, no boilerplate.
fn loss(w, b) {
  let err = X @ w + b - y
  mean(err * err)
}

for step in range(400) {
  let g = grads(loss)(w, b)   # gradients w.r.t. every parameter
  w = w - g[0] * lr
  b = b - g[1] * lr
}
```

> **Status:** early prototype (v0.1). The language runs today via a complete, zero-dependency reference interpreter. The design is stable enough to build on and simple enough to change. This is a foundation to iterate on, not a finished product.

---

## Why Aster?

Modern ML is written in a general-purpose language (Python) bolted to a numerical framework (PyTorch/JAX/TensorFlow). That stack is powerful but heavy: autodiff is a runtime library, shapes are discovered at runtime, and a mountain of glue code sits between your idea and the math.

Aster starts from the other end — **what if a language were designed around differentiable tensor programs from the first line?**

- **Tensors are the primitive.** Every number is a rank-0 tensor. Vectors and matrices are literals. `@` is matrix multiply.
- **Differentiation is a keyword.** `grad(f)` and `grads(f)` are language builtins backed by a real reverse-mode autodiff engine. No `requires_grad`, no `.backward()` wiring, no optimizer objects.
- **Small and readable.** The whole language is a few hundred lines. You can read the interpreter in an afternoon and understand exactly what your program does.
- **Zero dependencies.** The reference implementation runs on plain Node.js. Nothing to install, nothing to compile, easy to embed.

## Quick start

Requires **Node.js ≥ 22.6** (uses native TypeScript execution — no build step).

```bash
git clone https://github.com/martin-k-m/aster.git
cd aster

# Run a program
node src/cli.ts examples/autodiff.ast

# Or start the REPL
node src/cli.ts
```

```
aster> let df = grad(fn(x) = sin(x))
aster> df(0.0)
1
```

Run the examples and tests:

```bash
node src/cli.ts examples/linreg.ast   # linear regression by gradient descent
node src/cli.ts examples/mlp.ast      # a 2-layer net learning XOR
node --test                           # the test suite
```

## A tour of the language

```rust
# Comments start with '#'.

# Scalars, vectors, matrices — all tensors.
let a = 3.0
let v = [1.0, 2.0, 3.0]
let m = [[1.0, 2.0], [3.0, 4.0]]

# Operators are elementwise; '@' is matrix/vector multiply.
let dot = v @ v            # 14
let mv  = m @ [1.0, 1.0]   # [3, 7]

# Functions: one-liners or blocks. The last expression is the return value.
fn relu6(x) = if x > 6.0 { 6.0 } else { relu(x) }
fn rms(t) {
  let n = len(t)
  sqrt(sum(t * t) / n)
}

# Control flow for training loops.
let total = 0.0
for i in range(10) { total = total + i }

# Differentiation — the whole point.
fn energy(w) = sum(relu(w) * relu(w)) / 2.0
let g = grad(energy)([-1.0, 2.0, -3.0, 4.0])   # [0, 2, 0, 4]
```

See [`docs/language-guide.md`](docs/language-guide.md) for the full reference and [`docs/design.md`](docs/design.md) for the design rationale and roadmap.

## Built-in autodiff

| Builtin | Meaning |
| --- | --- |
| `grad(f)` | Returns a function computing `df/d(arg₀)` (works for scalar and tensor args). |
| `grads(f)` | Returns a function giving the gradient w.r.t. **every** argument, as a list. |
| `value_and_grad(f)` | Returns a function giving `[f(x), df/d(arg₀)]` in one pass. |

The target function must return a scalar (as is standard for a loss). Under the hood, Aster builds a reverse-mode autodiff graph only when a value is being differentiated, so ordinary evaluation stays fast.

Differentiable operations include `+ - * / @ ^`, and `relu`, `sigmoid`, `tanh`, `exp`, `log`, `sin`, `cos`, `sqrt`, `sum`, `mean`, `abs`, `pow`.

## Project layout

```
src/
  lexer.ts        # source text  -> tokens
  parser.ts       # tokens       -> AST (Pratt parser)
  ast.ts          # AST node types
  tensor.ts       # the differentiable tensor engine (autodiff core)
  values.ts       # runtime value model + environments
  interpreter.ts  # tree-walking evaluator
  builtins.ts     # standard library (grad, math, tensor construction)
  cli.ts          # `aster` command + REPL
examples/         # runnable .ast programs
test/             # node:test suite
docs/             # language guide + design notes
```

## Roadmap

Aster is intentionally minimal today. Natural next steps:

- **Static shape checking** — catch shape mismatches before running, using shapes as part of the type system.
- **A compiled/vectorized backend** — the current interpreter is for clarity, not speed; a bytecode VM or codegen to a numeric backend comes next.
- **Broadcasting** beyond scalar↔tensor, and named axes.
- **Higher-order autodiff** (grad of grad) and forward-mode.
- **Records/structs** for model parameters, and a module system.

Contributions and design discussion are welcome — see [`docs/design.md`](docs/design.md).

## License

[MIT](LICENSE) © 2026 Martin (martin-k-m)
