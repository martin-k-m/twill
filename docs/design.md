# Aster Design Notes

This document records *why* Aster is the way it is, and where it's headed. It's
meant to make the project easy to reason about and contribute to.

## Thesis

The dominant ML stack — Python plus a numerical framework — was assembled, not
designed. Autodiff, shapes, and device placement are all bolted onto a language
that predates deep learning. Aster asks a narrower question:

> If you designed a language *around differentiable tensor programs* from the
> first line, what would it look like?

The answer we're exploring: a small language where **the tensor is the
primitive type** and **differentiation is a language operation**, not a library
call. Everything else follows from keeping those two ideas front and center.

## Principles

1. **Tensors first.** There is no separate "scalar" type; a scalar is a rank-0
   tensor. This makes autodiff and broadcasting uniform, with no special cases.
2. **Differentiation is a primitive.** `grad`/`grads` are part of the language,
   not an SDK. A user should never wire up a tape or call `.backward()`.
3. **Small enough to hold in your head.** The whole implementation is a few
   hundred readable lines. Predictability beats features at this stage.
4. **No magic, no dependencies.** The reference interpreter is plain, dependency-
   free code you can audit end to end.

## Architecture

The reference implementation is a classic tree-walking interpreter:

```
source ──lexer──▶ tokens ──parser──▶ AST ──interpreter──▶ values
                                              │
                                              ▼
                                        tensor engine
                                     (autodiff happens here)
```

- **`lexer.ts`** — a hand-written scanner. No dependencies, precise error spans.
- **`parser.ts`** — recursive descent with a Pratt loop for binary operators.
  Expression-oriented: `if`/blocks are expressions with values.
- **`tensor.ts`** — the interesting part. A `Tensor` is a flat `Float64Array`
  plus a shape. Operations optionally record a reverse-mode autodiff graph.
- **`interpreter.ts`** — evaluates the AST against lexically-scoped
  environments. All arithmetic lowers to tensor-engine ops, so *any* computed
  value is differentiable for free.
- **`builtins.ts`** — the standard library, including the `grad` family.

### How autodiff works

Aster uses **reverse-mode automatic differentiation** (the same algorithm as
PyTorch/JAX), implemented directly in `tensor.ts`.

Each tensor operation, in addition to computing its output, can record:
- `_prev`: the input tensors it depends on, and
- `_backward`: a closure that pushes gradient from the output back to the
  inputs using the local partial derivatives.

Crucially, this graph is **only built when an input has `requiresGrad` set**.
Ordinary evaluation allocates no graph and pays no autodiff cost — the tape
materializes only inside a `grad(...)` call.

`grad(f)` works by:
1. Re-wrapping the call's tensor arguments as fresh gradient-tracking leaves.
2. Running `f`, which builds the graph as a side effect of normal evaluation.
3. Seeding the scalar output's gradient to 1 and walking the graph in reverse
   topological order.
4. Returning the accumulated gradient of the requested argument(s) as ordinary
   (non-tracked) tensors.

This keeps the mental model simple: **differentiation is just running your
program with gradients turned on.**

### Why a rank-0 tensor for scalars?

Treating `3.0` as a tensor of shape `[]` means the autodiff engine, broadcasting
rules, and printing all have exactly one code path. The alternative — a distinct
`Number` type — would force conversions and special cases at every op.

## Deliberate limitations (v0.1)

These are known and intentional for a first prototype:

- **Interpreted, not compiled.** Speed is not a goal yet; clarity is. A vector
  op loops in JS. A real backend (bytecode VM or codegen to BLAS/GPU) is future
  work.
- **Scalar↔tensor broadcasting only.** No general NumPy broadcasting yet.
- **Reverse-mode, first-order only.** `grad(grad(f))` is not supported. Adding
  higher-order and forward-mode autodiff is a natural extension.
- **Dynamic shapes.** Shapes are checked at runtime, not statically. Static
  shape typing is the single most valuable next step (see below).
- **No records/modules.** Parameters are passed positionally. Structs and a
  module system are planned.

## Roadmap

Ordered roughly by value-to-effort:

1. **Static shape checking.** Make tensor shapes part of the type system so
   `[2,3] @ [4]` is a *compile-time* error. This is the feature that would most
   distinguish Aster from Python-based stacks.
2. **A performant backend.** Compile to a bytecode VM, or lower tensor ops to a
   vectorized/native numeric library. Keep the interpreter as the reference
   semantics.
3. **Richer autodiff.** Higher-order derivatives, forward-mode (JVPs), and
   `vmap`-style batching.
4. **General broadcasting and named axes.**
5. **Records/structs and modules** for organizing model parameters and code.
6. **A `nn` standard library** — layers, initializers, and optimizers written
   in Aster itself.

## Non-goals (for now)

- Being a general-purpose systems language. Aster is a domain-specific language
  for differentiable numeric programs.
- Matching a mature framework's op coverage on day one. The point is the *core
  model*; ops are easy to add once the model is right.
