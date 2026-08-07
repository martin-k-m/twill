# Twill for finance ML

This is an honest assessment of where Twill can beat a Python + NumPy/PyTorch
stack for financial machine learning, and a roadmap to get there. The design
constraint is deliberate: **pure Go, no native dependencies**: one static,
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
  regulated finance those are not nice-to-haves. They are the job.

## The edge Twill already has

- **Greeks by autodiff.** Differentiate a pricer and get delta/vega/etc.
  directly, with no bump-and-revalue, no finite-difference noise. See
  `examples/montecarlo_option.tw`: a Monte-Carlo European call whose price and
  Greeks match Black-Scholes closed form, computed with `grad`.
- **Deterministic by default.** Randomness is seeded, so a run reproduces
  exactly. `seed(n)` picks the starting point. This is what model-risk
  governance and audit trails require.
- **Static shape and unit checking.** A shape mistake is caught before a single
  path is simulated, and dimensioned quantities (currency, rates, time) are
  checked at compile time too, so adding dollars to shares, or passing a rate
  where money is expected, is a compile error, not a silent production bug. This
  catches a whole class of finance bugs Python cannot.
- **One auditable binary.** No environment drift between research and
  production; the artifact you validate is the artifact you run.

## Beachhead: derivatives pricing and risk

This is the workload where pure-Go Twill can be *provably better than Python
soon*, because it wins on speed (parallel), correctness, reproducibility, and
deployment at the same time, and its infrastructure (fast RNG, parallelism,
autodiff at scale) is shared with backtesting and MC risk.

Delivered so far: deterministic seeded RNG and the Monte-Carlo pricer with
autodiff Greeks.

## Roadmap

Ordered so each step is usable on its own and compounds toward the beachhead,
then widens to the other workloads.

1. **Parallelism.** *(delivered v0.12)* Elementwise, unary, and matmul forward
   passes run across cores for large tensors, deterministic and race-free
   (each goroutine writes disjoint outputs, so results are bit-identical to a
   serial run). Measured ~2–4.5× on large ops; small tensors stay serial. This
   is the single biggest pure-Go speed lever for Monte-Carlo and backtesting.
2. **Faster core numerics.** *(delivered v0.13)* Full reductions (`sum`/`mean`)
   are parallel and deterministic (fixed-block partials, ~3.3× on large data),
   and the backward passes for elementwise/unary ops run across cores too, so
   both the forward and the gradient use all cores on large tensors. (Matmul is
   already row-parallel and cache-friendly; explicit blocking is a later
   micro-optimization.)
3. **Data I/O.** *(delivered v0.14)* A frame is a record of named column tensors;
   `read_frame`/`write_frame` do header CSV, and `columns`/`field`/`with_field`
   manipulate columns by name. Field access, slicing, and `grad` all work on a
   frame, and a numeric time column makes it a time series. See
   `examples/frames.tw`. Parquet/Arrow would need a third-party module, so it's
   deferred to keep the zero-dependency single binary.
4. **Dimensioned types.** *(delivered v0.15)* Declare base units (`unit USD`) and
   annotate scalars with units or unit expressions (`px: USD/share`, `-> USD`).
   The checker tracks units through arithmetic, `matmul`/`dot`, and powers,
   requires dimensionless arguments to transcendentals, and rejects nonsense
   like dollars + shares, all statically, then fully erased at runtime for zero
   overhead. See `examples/units.tw`. This is correctness Python structurally
   cannot offer.
5. **Gradient-boosted trees.** *(delivered v0.16)* A native, pure-Go GBM engine
   (`internal/gbm`) using the second-order (Newton) formulation, so regression
   (squared error) and binary classification (logistic) share one exact-greedy
   tree builder. `gbm_fit(X, y, opts)` trains, `gbm_predict(model, X)` scores;
   the split search runs across cores but reduces in fixed order, so fits are
   deterministic. This is the model class that dominates finance tabular ML
   (credit/fraud/default), now native and dependency-free. See
   `examples/gbm.tw`. Boosting-specific extras (early stopping, feature
   importance, categorical splits, missing-value handling) are natural
   follow-ups.
6. **Backtesting toolkit.** *(delivered v0.17)* Cumulative-scan builtins
   (`cumsum`/`cumprod`/`cummax`/`cummin`) plus a `std/backtest` library:
   returns, moving averages, equity curves, drawdown, Sharpe, volatility, and
   CAGR, all vectorized on tensors, no event loop needed. The Sharpe ratio is
   differentiable in the return series, so a smooth signal can be *tuned by
   gradient ascent*, a genuine edge over a Python backtest. See
   `examples/backtest.tw`.

With #1–#6 delivered, the original roadmap is complete: Twill now covers
parallel numerics, data frames, dimensioned types, gradient-boosted trees, and
backtesting, the pure-Go, deterministic, single-binary core of the finance
pitch.

Beyond the roadmap, the first follow-up is delivered: **gradient-optimized
signals** *(v0.18)*. Because the backtest Sharpe/Sortino are differentiable in
the return series, `grad` differentiates a whole backtest, so a signal's weights
can be tuned by gradient ascent on risk-adjusted return, see
`examples/signal_opt.tw`. Remaining ideas live in each item's notes above (GBM
early stopping / feature importance / histogram splits, Parquet I/O, second-order
autodiff for risk sensitivities, and a vectorized interpreter backend for speed).

## Non-goals

- Competing with CUDA/PyTorch on large dense deep learning under a pure-Go, no
  native deps constraint.
- Matching the breadth of the Python ecosystem. Twill aims to be *better for a
  specific, high-stakes finance workflow*, not broader than Python.
