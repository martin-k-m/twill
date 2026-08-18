# Getting started with Twill

This walks through Twill from nothing to training a small model. It assumes you
have the `twill` binary on your `PATH` (see the README for install options).

## Hello

Put this in `hello.tw`:

```rust
print("hello from twill")
let x = 3.0
print("x squared is", x * x)
```

Run it:

```bash
twill hello.tw
```

Or start the REPL and type expressions:

```bash
twill
twill> 2 + 2 * 3
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

`std/nn` and `std/optim` are written in Twill and ship inside the `twill`
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

See `examples/classifier.tw` for a full training loop with softmax
cross-entropy and Adam, and `examples/nn_xor.tw` for a smaller one.

## Loading data

```rust
let data = read_csv("data.csv")   # a [rows, cols] tensor
print("rows:", shape(data)[0])
```

## Catching shape mistakes

Run `twill check file.tw` to shape-check without running. Twill infers tensor
shapes and reports mismatches it can prove:

```
$ twill check bad.tw
bad.tw:3: shape error: shape mismatch in @: [2, 3] @ [2] (inner 3 != 2)
  3 | let y = A @ x
```

You can annotate parameters to make the contract explicit:

```rust
fn matvec(A: [3, 2], x: [2]) -> [3] {
  A @ x
}
```

In the REPL, `:shape` asks the same question about one expression, and answers
it without running anything -- so it costs nothing even for a result that would
not fit in memory:

```
twill> :shape zeros(4, 8) @ zeros(8, 2)
[4, 2]
twill> :shape zeros(2, 3) + zeros(4)
shape mismatch: [2, 3] vs [4] cannot broadcast
```

## Units, which are checked and then disappear

A number can carry a unit, and the checker does the algebra:

```rust
unit USD
unit share

fn notional(px: USD/share, qty: share) -> USD { px * qty }

let price: USD/share = 150.0
let quantity: share = 200.0
let value = notional(price, quantity)   # inferred: USD
```

Adding `USD` to `share` is an error before the program runs. Multiplying them
is not, and the result's unit is what the multiplication says it is. Units are
erased at run time, so none of this costs anything: the program that runs is
plain arithmetic.

## Checking a gradient

`grad` gives an answer quickly and a wrong one looks exactly like a right one.
A model with a bad gradient does not crash; it trains to a worse loss than it
should have. `std/gradcheck` compares the gradient against a difference
quotient, which is the same derivative reached a different way:

```rust
import "std/gradcheck" as gc

let report = gc.check_tree(fn(m) = loss(m, x, y), model)
if not report.ok {
  print("the gradient disagrees by", report.error)
}
```

It costs two evaluations per parameter, so it is a thing to run once while
writing a model and not something a training loop calls.

## Packages, and the rest of the ecosystem

Twill's standard library ships inside the binary and is imported by name:

```rust
import "std/nn" as nn        # a module of the library
import "helpers.tw"          # a file, next to this one
```

Everything larger lives in its own package, written in twill:
[spool](https://github.com/twill-lang/spool) resolves and vendors dependencies,
[loom](https://github.com/twill-lang/loom) is the training loop,
[warp](https://github.com/twill-lang/warp) is data pipelines,
[weft](https://github.com/twill-lang/weft) plots,
[skein](https://github.com/twill-lang/skein) tokenises,
[heddle](https://github.com/twill-lang/heddle) does Bayesian inference,
[selvedge](https://github.com/twill-lang/selvedge) serialises models,
[shuttle](https://github.com/twill-lang/shuttle) serves them, and
[bobbin](https://github.com/twill-lang/bobbin) benchmarks.

That's the whole language. The [language guide](language-guide.md) is the
complete reference, and it has a second half -- `mode systems`, with integers,
strings, arrays, dictionaries, structs, enums and `?` -- which is what twill's
own compiler is written in.
