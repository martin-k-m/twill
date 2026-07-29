# Raster for finance ML

This is an honest assessment of where Raster can beat a Python + NumPy/PyTorch
stack for financial machine learning, and a roadmap to get there. The design
constraint is deliberate: **pure Go, no native dependencies** — one static,
auditable binary, no CUDA, no BLAS via cgo, no Python.

## What that constraint means

Being clear up front avoids a costly surprise later:

- **We will not beat GPU/BLAS on large deep-learning matmul.** That is a
  hardware-and-vendor-library gap, not an effort gap. Under a pure-Go, no-deps
  constraint, dense deep learning at scale is not the place to compete.
- **We can win where the work is parallel rather than matmul-bound.** Monte
  Carlo, backtesting, and tree building are embarrassingly parallel; pure-Go
  goroutines beat single-threaded NumPy and rival multicore, with no dependency
  stack.
- **We win on the non-speed axes regardless.** Static shape checking catches
  bugs before a model runs; results are deterministic and reproducible by
  default; the whole thing deploys as one binary with nothing to install. In
  regulated finance those are not nice-to-haves — they are the job.

## The edge Raster already has

- **Greeks by autodiff.** Differentiate a pricer and get delta/vega/etc.
  directly — no bump-and-revalue, no finite-difference noise. See
  `examples/montecarlo_option.ra`: a Monte-Carlo European call whose price and
  Greeks match Black-Scholes closed form, computed with `grad`.
- **Deterministic by default.** Randomness is seeded, so a run reproduces
  exactly. `seed(n)` picks the starting point. This is what model-risk
  governance and audit trails require.
- **Static shape (and soon unit) checking.** A shape mistake is caught before a
  single path is simulated. The natural next step — dimensioned quantities
  (currency, rates, time) checked at compile time — would catch a whole class of
  finance bugs Python cannot.
- **One auditable binary.** No environment drift between research and
  production; the artifact you validate is the artifact you run.

## Beachhead: derivatives pricing and risk

This is the workload where pure-Go Raster can be *provably better than Python
soon*, because it wins on speed (parallel), correctness, reproducibility, and
deployment at the same time — and its infrastructure (fast RNG, parallelism,
autodiff at scale) is shared with backtesting and MC risk.

Delivered so far: deterministic seeded RNG and the Monte-Carlo pricer with
autodiff Greeks.

## Roadmap

Ordered so each step is usable on its own and compounds toward the beachhead,
then widens to the other workloads.

1. **Parallelism.** *(delivered v0.12)* Elementwise, unary, and matmul forward
   passes run across cores for large tensors — deterministic and race-free
   (each goroutine writes disjoint outputs, so results are bit-identical to a
   serial run). Measured ~2–4.5× on large ops; small tensors stay serial. This
   is the single biggest pure-Go speed lever for Monte-Carlo and backtesting.
2. **Faster core numerics.** *(delivered v0.13)* Full reductions (`sum`/`mean`)
   are parallel and deterministic (fixed-block partials, ~3.3× on large data),
   and the backward passes for elementwise/unary ops run across cores too — so
   both the forward and the gradient use all cores on large tensors. (Matmul is
   already row-parallel and cache-friendly; explicit blocking is a later
   micro-optimization.)
3. **Data I/O.** CSV is here (`read_csv`); add Parquet/Arrow and a time-indexed
   frame type. Finance is data-first.
4. **Dimensioned types.** Optional units on quantities (USD, bps, years) checked
   statically — correctness that Python structurally cannot offer.
5. **Gradient-boosted trees.** A native GBM engine for tabular credit/fraud/
   default work — the model class that dominates finance tabular ML today and
   that Raster has none of yet. This is a large, separate build.
6. **Backtesting toolkit.** Time model, event loop, and vectorized signal ops in
   the standard library.

## Non-goals

- Competing with CUDA/PyTorch on large dense deep learning under a pure-Go, no
  native deps constraint.
- Matching the breadth of the Python ecosystem. Raster aims to be *better for a
  specific, high-stakes finance workflow*, not broader than Python.
