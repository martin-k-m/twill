# Getting started with Raster

This walks through Raster from nothing to training a small model. It assumes you
have the `raster` binary on your `PATH` (see the README for install options).

## Hello

Put this in `hello.ra`:

```rust
print("hello from raster")
let x = 3.0
print("x squared is", x * x)
```

Run it:

```bash
raster hello.ra
```

Or start the REPL and type expressions:

```bash
raster
raster> 2 + 2 * 3
8
```

The REPL keeps reading while brackets are open, so you can paste or type a
multi-line function.

## Tensors

Numbers are rank-0 tensors. Bracketed literals of numbers are vectors and
matrices.

```rust
let v = [1.0, 2.0, 3.0]            # shape [3]
let m = [[1.0, 2.0], [3.0, 4.0]]  # shape [2, 2]

print(v + 10.0)       # scalar broadcasts: [11, 12, 13]
print(m @ [1.0, 1.0]) # matrix-vector: [3, 7]
print(sum(m, 0))      # column sums: [4, 6]
```

Elementwise operators broadcast: a row vector spreads across the rows of a
matrix, a column vector down them.

```rust
let a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
print(a + [10.0, 20.0, 30.0])   # adds the row to each row
```

## Functions

```rust
fn square(x) = x * x

fn normalize(v) {
  let s = sum(v)
  v / s
}

print(square(5.0))            # 25
print(normalize([1.0, 3.0]))  # [0.25, 0.75]
```

## Differentiation

`grad(f)` returns the derivative of `f` with respect to its first argument. It
works for scalars and tensors, and the function must return a scalar.

```rust
fn f(x) = x * x * x     # x^3
print(grad(f)(2.0))     # 12

fn energy(w) = sum(w * w)
print(grad(energy)([1.0, 2.0, 3.0]))   # [2, 4, 6]
```

`grads(f)` gives the gradient for every argument; `value_and_grad(f)` returns the
value and the first gradient together.

## Training a model

Here is linear regression by gradient descent. `loss` returns the mean squared
error, and `grads` gives the gradient for each parameter.

```rust
let X = [[1.0, 1.0], [2.0, 1.0], [1.0, 3.0], [3.0, 2.0]]
let y = [1.5, 3.5, 3.5, 6.5]

fn loss(w, b) {
  let err = X @ w + b - y
  mean(square(err))
}

let w = [0.0, 0.0]
let b = 0.0
for step in range(300) {
  let g = grads(loss)(w, b)
  w = w - g[0] * 0.05
  b = b - g[1] * 0.05
}
print("w =", w, " b =", b)
```

## Using the library

`std/nn` and `std/optim` are written in Raster and ship inside the `raster`
binary, so the import works from any directory. Importing `nn` also pulls in the
optimizers. A model is just a list of tensors, so `grad` differentiates the whole
thing at once and an optimizer updates it.

```rust
import "std/nn"

let p = [he_init(8, 2), zeros(8), he_init(3, 8), zeros(3)]

fn logits(p, x) {
  let h = tanh(dense(p[0], p[1], x))
  dense(p[2], p[3], h)
}
```

See `examples/classifier.ra` for a full training loop with softmax
cross-entropy and Adam, and `examples/nn_xor.ra` for a smaller one.

## Loading data

```rust
let data = read_csv("data.csv")   # a [rows, cols] tensor
print("rows:", shape(data)[0])
```

## Catching shape mistakes

Run `raster check file.ra` to shape-check without running. Aster infers tensor
shapes and reports mismatches it can prove:

```
$ raster check bad.ra
bad.ra:3: shape error: shape mismatch in @: [2, 3] @ [2] (inner 3 != 2)
  3 | let y = A @ x
```

You can annotate parameters to make the contract explicit:

```rust
fn matvec(A: [3, 2], x: [2]) -> [3] {
  A @ x
}
```

That's the whole language. The [language guide](language-guide.md) is the
complete reference.
