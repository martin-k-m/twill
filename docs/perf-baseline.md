# Performance baseline: what it takes to run large models

This is the measured starting point for making twill run large neural networks —
eventually billion-parameter models on a laptop. It exists so the optimisation
work is aimed by data rather than by intuition. Re-measure with `bench/forward.tw`
and `bench/matmul.tw`; re-profile with the recipe at the bottom.

Measured on an Intel Core Ultra 9 285H, f64 build, August 2026.

## The two walls: memory and compute

Every tensor element in twill is a `float64` (8 bytes). That fixes the memory
cost of a model before any kernel runs:

| Model      | f64 (today) | f32   | bf16  | int8  | int4    |
|------------|-------------|-------|-------|-------|---------|
| 1B params  | 8 GB        | 4 GB  | 2 GB  | 1 GB  | 0.5 GB  |
| 7B params  | 56 GB       | 28 GB | 14 GB | 7 GB  | 3.5 GB  |

A laptop with 16 GB of RAM cannot hold a 7B model in f64 (56 GB) or even f32
(28 GB). The int4 column is the target, and it is reachable only once tensors
store their elements in real byte width — the packed-buffer layout of NEEDS-111.
The dtype *semantics* already exist (`docs/dtypes.md`); the *layout* does not, so
today quantisation shrinks nothing.

## The end-to-end baseline works

`examples/gpt.tw` trains a real decoder-only GPT (58k params) end to end with
`grad`, loss 3.0 -> 0.17 over 40 steps, and generates coherent text. The
pipeline is real; this is an optimisation problem, not a bring-up problem.

Forward-pass (inference) scaling, `bench/forward.tw`:

| dim | params | fwd_ms |
|-----|--------|--------|
| 48  | 60k    | 1.5    |
| 96  | 231k   | 2.8    |
| 192 | 904k   | 9.0    |
| 384 | 3.58M  | 27.9   |

Raw square matmul ceiling, `bench/matmul.tw`: ~24–30 GFLOP/s (f64).

At dim 384 the forward pass does ~`2·params·tokens` = ~0.115 GFLOP in 27.9 ms —
about **4 GFLOP/s effective**, roughly one-sixth of the raw-matmul ceiling. Most
inference time is therefore *not* in the large matmuls.

## The profile: where inference time actually goes

CPU profile of a repeated forward pass (dim 256, depth 4, T 32). Flat time —
actual CPU spent *in* each function:

| function                    | flat % | what it is                     |
|-----------------------------|--------|--------------------------------|
| `tensor.mm.func1`           | 38%    | the matmul kernel inner loop   |
| `tensor.einsumRaw`          | 20%    | attention (batched over heads) |
| `tensor.TransposePerm`(+fn) | 14%    | materialising transposed copies|
| `runtime.memclr…`           | 6%     | zeroing freshly allocated bufs |
| `broadcastBinary`           | 6%     | elementwise add/mul/…          |
| gelu/softmax/rand/rest      | ~5%    | small elementwise + sampling   |

The decisive result: **the tree-walking interpreter costs essentially nothing.**
`Interp.Run`, `Apply`, `callClosure`, `evalBinary`, and `evalCall` all show 0%
flat — they are pure `cum`, dispatching into the kernels. twill is already a thin
orchestration layer over tensor ops, the way Python is over NumPy. There is no
need to rewrite the interpreter or compile to native code to go fast.

## What this means, in priority order

1. **The matmul kernel (38%) is the main lever — partly done.** `mmNT` (the
   fused-linear kernel, now the dominant op after the transpose fusions) runs
   four independent accumulators over its inner product, hiding FP-add latency:
   ~1.7× on square inputs, and at prefill scale `linear()` sustains **~48–50
   GFLOP/s vs ~25–30 for the general `@`**. Remaining kernel wins: the general
   `mm` (streaming AXPY form, memory-bound, harder to reorder); cache blocking for
   the large case (a 128×4096×4096 still drops to ~19 GFLOP/s, i.e. bandwidth-
   bound); then f32 (halves bytes moved, doubles SIMD width) and quantised weights
   (shrink the bytes read). The Rust port (`raster-tensor`) is the SIMD vehicle.
   Small-batch decode (m≈1) is bandwidth-bound and will not show the accumulator
   win — that regime needs f32/quantisation, not ILP.

2. **Transpose (14%) — DONE.** `nn.dense_apply` was `x @ transpose(p.W)`, which
   allocated a transposed weight copy every forward pass. Fused into `linear(x, W)`
   (`tensor.MatMulNT` / `mmNT`), which reads `W` in its stored `[nout,nin]` layout
   and computes `sum_p x[i,p]·W[j,p]` in the same accumulation order — bit-for-bit
   identical, parity-safe. Measured win: dim-384 forward **27.9ms → 11.1ms (2.5×)**,
   larger than the transpose share alone because the fused dot-product is
   cache-friendlier and skips the allocation and `memclr`. The next transpose to
   fuse the same way is the single-head `scores = (X@Wq) @ transpose(X@Wk)` in
   `std/nn.tw`, and attention's einsum path is the larger remaining kernel.

3. **Attention einsum (20%) is the second kernel** to give the f32/SIMD treatment
   after matmul, since the two share the same inner product.

4. **Allocation churn (~12%, memclr + intermediate buffers)** falls to buffer
   reuse once the hot kernels are fused.

Only after these does the memory work (NEEDS-111 packed layout) turn into speed
as well as size. The order is deliberate: prove each kernel win against the
benchmarks here, keeping Go and self-hosted `src/tensor.tw` in bit-exact parity
at every step.

## Reproduce

```
tw run bench/forward.tw
tw run bench/matmul.tw
```

CPU profile of a forward pass (throwaway Go bench, from repo root):

```
# a _test.go that runs std/transformer's gpt_logits in a loop under -cpuprofile,
# then:  go tool pprof -top -nodecount=25 cpu.prof
```
