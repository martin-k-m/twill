# Design notes

Why Raster is built the way it is, and what's left to do.

## The idea

The usual machine-learning stack — Python plus a numeric framework — was
assembled over time, not designed as a whole. Autodiff, shapes, and device
placement are all added on top of a language that came first. Raster asks a
narrower question: if you designed a language around differentiable tensor
programs from the start, what would it look like?

The answer we're trying: a small language where the tensor is the primitive
type and differentiation is a language operation rather than a library call.
The rest follows from keeping those two ideas central.

## Principles

1. Tensors first. There's no separate scalar type; a scalar is a rank-0 tensor.
   That keeps autodiff, broadcasting, and printing uniform.
2. Differentiation is built in. `grad`/`grads` are part of the language. You
   never wire up a tape or call `.backward()`.
3. Small enough to read. The implementation is a few thousand lines of Go, and
   predictability matters more than feature count right now.
4. No dependencies. The reference implementation is plain Go with no third-party
   packages, so it builds to one binary and can be read end to end.

## Why Go

The implementation language went through a couple of iterations (an early
TypeScript prototype, then a look at Rust). Go won for this stage: it builds to
a single dependency-free binary, it's quick to compile and easy to read, and the
standard library covers everything the interpreter needs. Rust would give better
numeric performance and a real ML crate ecosystem (Burn, candle, ndarray), and
it's a reasonable target later — the language design doesn't depend on the host.
For a small, readable reference implementation, Go is the better fit.

## Architecture

A straightforward tree-walking interpreter:

```
source ─lexer─▶ tokens ─parser─▶ AST ─┬─ checker ─▶ shape diagnostics
                                      └─ interpreter ─▶ values
                                                 │
                                                 ▼
                                           tensor engine (autodiff)
```

- `internal/lexer` — a hand-written scanner with source positions.
- `internal/parser` — recursive descent with a Pratt loop for operators.
  Expression-oriented, so `if` and blocks produce values.
- `internal/tensor` — the autodiff engine. A tensor is a flat `[]float64` plus a
  shape; operations optionally record a reverse-mode graph.
- `internal/interp` — evaluates the AST against lexical scopes. All arithmetic
  lowers to tensor-engine ops, so any computed value can be differentiated.
- `internal/checker` — static shape inference over the AST.
- `cmd/raster` — the CLI and REPL.

## How autodiff works

Raster uses reverse-mode automatic differentiation, the same approach as PyTorch
and JAX, implemented directly in the tensor engine.

Besides computing its output, each operation can record the inputs it depended
on and a closure that pushes gradient from the output back to those inputs using
the local derivatives. That graph is only built when an input requires
gradients, so ordinary evaluation allocates nothing extra — the tape appears
only inside a `grad(...)` call.

`grad(f)` re-wraps the call's arguments as fresh gradient-tracking leaves, runs
`f` (which builds the graph as a side effect), seeds the scalar output's
gradient to 1, and walks the graph in reverse topological order. Arguments can
be nested lists, in which case the returned gradient mirrors that structure —
which is how a whole model held in a list gets differentiated at once.

So differentiation is just running the program with gradients turned on.

## The shape checker

The checker infers a shape (or "unknown") for every expression and reports a
diagnostic only when a mismatch is certain — both operand shapes fully known and
incompatible. Everything it can't determine stays unknown, so it never flags
correct dynamic code. That bias toward precision over recall is deliberate: a
checker that cries wolf gets turned off.

It knows the shapes of tensor literals, of construction builtins called with
literal sizes (`zeros(4)`, `randn(4, 2)`), and of the operations that combine
them. Optional parameter annotations give it more to work with and let it check
call sites against a declared contract. It does not follow shapes through
`grad`, loops that reshape, or values read at runtime — those are left unknown.

## Known limitations (v0.5)

Deliberate, for a prototype:

- Interpreted, not compiled. Tensor ops loop in Go (the elementwise/broadcast
  hot path avoids per-element division, but there's still no vectorized or
  native backend). The interpreter is the reference semantics.
- Reverse-mode, first-order autodiff. No `grad(grad(f))`.
- Shape checking is stronger than it was — annotations support shape variables
  that must agree — but it's still best-effort, not a full type system that
  catches every mismatch.
- No records or module namespaces. Imports share the global scope.
- No named axes; broadcasting and reductions work on positional axes.

## Roadmap

Roughly in order of value:

1. Push static shapes further: infer shape variables without annotations and
   catch more mismatches, moving from best-effort toward a real type system.
2. A faster backend: a bytecode VM, or lowering tensor ops to a vectorized or
   native library. Keep the interpreter as the reference.
3. More autodiff: higher-order derivatives, forward mode, batching.
4. Named axes and general einsum-style contraction.
5. Records and a module system for organizing parameters and code.

## Non-goals for now

- Being a general-purpose language. Raster is aimed at differentiable numeric
  programs.
- Matching a mature framework's operator coverage on day one. The point is the
  core model; operators are easy to add once that's right.
