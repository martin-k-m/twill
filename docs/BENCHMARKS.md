# Benchmarks

How fast twill is, measured against PyTorch on the same mathematics, with the
commands that regenerate every number.

The summary, stated first because it is the point of the document: **twill is
between 3x and 20x slower than PyTorch on identical f64 work, and between 3x and
29x slower than the f32 a PyTorch user would actually run.** That is the expected
result and it is not a defect. Section 6 says where the gap comes from, which is
more useful than how large it is.

---

## 1. Environment

Everything below was measured on this machine, on 2026-08-15. Nothing is
estimated, scaled or carried over from another document.

| | |
|---|---|
| CPU | Intel Core Ultra 9 285H, 16 physical cores, 16 logical, 2.9 GHz base |
| RAM | 15.43 GB |
| Storage | NVMe SSD (Timetec 35TT2280GEN4P-2TB) |
| OS | Windows 11 Pro, build 26200 |
| Go | go1.26.5 windows/amd64 |
| Python | 3.12.10 |
| PyTorch | 2.13.0+cpu |

**Every number in this document is a CPU number.** The machine has two GPUs, an
NVIDIA GeForce RTX 5070 Laptop and an Intel Arc 140T, and neither is used by
anything measured here. twill has no GPU backend, so there is nothing to run on
one; PyTorch is deliberately the CPU-only wheel, so
`torch.cuda.is_available()` is `False` and the comparison is CPU against CPU. A
GPU comparison would be a different document and would not be a comparison of
these two implementations.

This is a laptop, and section 7 is about what that costs the measurement.

## 2. Reproducing

From the repository root:

```
powershell -File bench/run_all.ps1 -Python <interpreter-with-torch> -Runs 30 -Warmup 5
python bench/interleave.py --python <interpreter-with-torch> --rounds 5
```

The first drives everything: the twill sweep, the front-end timings, the
gradient check, the PyTorch sweep and the profile. The second is the
drift-controlled comparison of section 7. Results land in `bench/results/` as
console output and JSON.

The individual pieces, if you want one of them:

```
go run ./bench/cmd/twillbench -runs 30 -warmup 5 -procs 1,2,4,8,16
go run ./bench/cmd/checkbench -runs 30 -warmup 5
python bench/torch_bench.py --threads 1,2,4,8,16 --runs 30 --warmup 5
go test ./bench/profile/ -run TestProfileMonteCarlo -count=30 -cpuprofile mc.prof -o interp.test.exe
go tool pprof -top -nodecount=22 interp.test.exe mc.prof
```

## 3. Method

**What is timed.** Each workload is a `.tw` file under `bench/workloads/` whose
top level does the setup and whose final expression is a nullary closure holding
the work. The harness runs the top level once, then calls that closure
repeatedly. Setup, parsing and shape checking therefore cost nothing in the
reported figures; what is measured is the evaluation of the work. The PyTorch
side is structured identically, so the two measure the same quantity.

**Warmup and run count.** 5 untimed warmups, then 30 timed runs.

**Median and p99, not mean.** A mean over a laptop is a report on what the
operating system decided to do during the run. The median says what a typical
call costs; the p99 says how bad the tail is. Neither can be moved by one
outlier. The p99 here is the nearest-rank p99 of 30 samples, which is the slowest
one: with 30 samples an interpolated p99 is mostly invention.

**No outliers are discarded.** Nothing is trimmed, winsorised or rejected. The
tail is reported instead, which is what the p99 column is for.

**Inner repeats.** Windows' wall clock has roughly half-millisecond granularity,
so a call taking 40 microseconds times as either 0 or 1 ms. Both harnesses
calibrate an inner repeat count until the timed span exceeds 20 ms and divide.
The consequence, stated because it matters for reading the p99: a workload with
an inner count above 1 reports the distribution of batch means rather than of
individual calls, which understates the tail. The inner count is printed for
every row in the raw output.

**Thread counts are swept, and each side is reported at its own best.** 1, 2, 4,
8 and 16, via `GOMAXPROCS` for twill and `torch.set_num_threads` for PyTorch.
Pinning both to 16 would be the obvious choice and the wrong one: the best count
is workload-dependent for both, and on the smaller workloads both are faster on
8 threads than on 16. Reporting either at a count that suits the other is how a
benchmark ends up saying what its author wanted.

**Both dtypes for PyTorch.** twill's only float type is f64. PyTorch defaults to
f32, which is half the bytes to move and twice the SIMD lanes, so comparing
twill-f64 against torch-f32 charges twill for a design decision rather than for
its kernels. The f64 column is the like-for-like comparison. The f32 column is
what a PyTorch user actually gets, and it is the more honest answer to "should I
use this instead".

### The workloads compute the same mathematics, and that is checked

Timing two programs is only a comparison if they do the same thing.
`bench/workloads/verify_deterministic.tw` and its PyTorch counterpart contain no
RNG: every input is a fixed function of its index, so both sides build
bit-identical inputs. The workload runs an elementwise chain, a reshape, a
matmul, a softmax, a logsumexp and a relu, differentiates the result, and returns
the loss, the gradient sum and the gradient sum of squares.

| implementation | result |
|---|---|
| twill (f64) | `[359.026899, 31.267037, 0.583661]` |
| PyTorch f64 | `[359.026899, 31.267037, 0.583661]` |
| PyTorch f32 | `[359.026886, 31.267042, 0.583662]` |

Identical to every printed digit against f64, gradient included, with f32
differing only where f32 rounding puts it. The other workloads draw their inputs
from each implementation's own RNG, so their printed results are not comparable
term by term and are not presented as though they were; the Monte Carlo pricer
is the exception worth noting, since twill's 10.442696 and PyTorch's 10.460652
are two independent estimates of the same Black-Scholes price of 10.4506.

---

## 4. Results

Median milliseconds, best thread count in parentheses. `x64` is twill's time
divided by PyTorch f64's, so higher means twill is slower.

### Forward and backward on representative programs

| workload | twill | p99 | torch f64 | p99 | torch f32 | x64 | x32 |
|---|---|---|---|---|---|---|---|
| `mc_option_fwd` | 3.450 (16) | 4.41 | 0.168 (8) | 0.19 | 0.119 (8) | 20.5 | 28.9 |
| `mc_option_grad` | 13.646 (16) | 16.44 | 0.952 (8) | 1.35 | 0.720 (8) | 14.3 | 19.0 |
| `mlp_train_step` | 5.521 (16) | 7.22 | 1.118 (8) | 1.85 | 0.819 (4) | 4.9 | 6.7 |
| `attention_head` | 2.688 (16) | 3.18 | 0.141 (8) | 0.18 | 0.130 (4) | 19.1 | 20.7 |
| `verify_deterministic` | 1.317 (4) | 1.71 | 0.445 (1) | 0.53 | 0.515 (1) | 3.0 | 2.6 |

`mc_option_fwd` and `mc_option_grad` are the Monte Carlo European call from the
README, 200,000 paths, forward only and then differentiated for delta and vega.
`mlp_train_step` is one forward and backward pass of a 256-512-256-10 MLP at
batch 64 with softmax cross-entropy. `attention_head` is one self-attention head
at sequence 256, model dimension 64.

The backward pass costs twill about 4x its forward pass on the Monte Carlo
program (13.646 against 3.450) and PyTorch about 5.7x (0.952 against 0.168), so
the *ratio* of backward to forward is similar and it is the forward kernels where
twill loses.

`verify_deterministic` is the closest twill gets, at 3.0x, and the reason is
instructive: it is a mixture of small operations on 64x64 and 4096-element
tensors, where PyTorch's per-operation dispatch overhead is a real fraction of
the work and twill's is not.

### Scaling with tensor size

The same elementwise chain, `tanh(exp(x/4) * x)` reduced to a scalar, at growing
sizes. Three passes over the buffer plus a reduction, so this is memory traffic
and transcendental throughput rather than any one kernel.

| n | twill | torch f64 | torch f32 | x64 | x32 |
|---|---|---|---|---|---|
| 10,000 | 0.283 | 0.053 | 0.037 | 5.3 | 7.6 |
| 100,000 | 1.554 | 0.190 | 0.109 | 8.2 | 14.3 |
| 1,000,000 | 9.652 | 1.438 | 0.574 | 6.7 | 16.8 |
| 10,000,000 | 80.441 | 21.835 | 8.865 | 3.7 | 9.1 |

And differentiated:

| n | twill | torch f64 | torch f32 | x64 | x32 |
|---|---|---|---|---|---|
| 100,000 | 3.693 | 0.441 | 0.342 | 8.4 | 10.8 |
| 1,000,000 | 27.960 | 4.461 | 1.631 | 6.3 | 17.1 |

The gap against f64 *narrows* as n grows, from 8.2x at 100,000 to 3.7x at ten
million. Both implementations become bandwidth-bound at that size and the memory
bus does not care which language issued the load. The gap against f32 does not
narrow the same way, because f32 halves the bytes and twill has no f32 storage to
offer.

### Matrix multiply

| size | twill | torch f64 | torch f32 | x64 | x32 |
|---|---|---|---|---|---|
| 128x128 | 0.569 | 0.039 | 0.025 | 14.7 | 22.5 |
| 256x256 | 2.141 | 0.215 | 0.104 | 10.0 | 20.6 |
| 512x512 | 13.826 | 1.359 | 0.644 | 10.2 | 21.5 |
| 1024x1024 | 99.772 | 8.594 | 3.964 | 11.6 | 25.2 |
| 512x512 fwd+bwd | 28.437 | 2.471 | 1.214 | 11.5 | 23.4 |

At 1024, twill's 99.772 ms for 2 x 1024^3 flops is **21.5 GFLOP/s f64**, against
PyTorch's 250 GFLOP/s f64 and 542 GFLOP/s f32. This is the widest and steadiest
gap in the document, it does not close with size, and section 6 says why.

### Parallel scaling

twill scales as its kernels should. From the sweep:

| workload | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| `matmul_512` | 105.032 | 48.770 | 25.037 | 15.038 | 13.826 |
| `elementwise_10000000` | 351.319 | 205.110 | 125.259 | 90.421 | 80.441 |
| `mc_option_fwd` | 6.045 | 5.138 | 3.714 | 3.912 | 3.450 |

The matmul gets 7.6x on 16 cores, close to linear to 8 and then flattening as it
runs out of bandwidth. The large elementwise chain gets 4.4x and flattens
earlier, which is what a memory-bound kernel does. `mc_option_fwd` gets 1.75x and
is the least parallel of the three, because at 200,000 elements each individual
kernel is only just above the 8,192-element threshold at which `parallelFor`
decides parallelism is worth its overhead.

### Compile and check time for the static shape checker

This is the one measurement with no PyTorch counterpart, because PyTorch has
nothing to compare against: shapes there are discovered by running the program,
so the equivalent cost is not a compile step but the wait until the offending
line executes.

Over the whole corpus of 53 files, 27,472 lines, 1,012,677 bytes
(`examples/*.tw`, `std/*.tw`, `src/*.tw`), summed medians:

| phase | whole corpus | per 1000 lines |
|---|---|---|
| lex | 20.96 ms | 0.763 ms |
| parse | 30.12 ms | 1.096 ms |
| check | 88.33 ms | 3.215 ms |

**Checking the entire corpus costs 88 ms.** That is the price of the guarantee,
and it is cheap enough to run on every save, which is the number that matters.

The per-file figures show the checker is not linear in file size:

| file | lines | lex | parse | check | check p99 |
|---|---|---|---|---|---|
| `src/tensor.tw` | 5617 | 5.534 | 7.463 | 10.045 | 20.370 |
| `src/eval.tw` | 4425 | 3.730 | 5.196 | 40.868 | 45.264 |
| `src/check.tw` | 3507 | 2.684 | 4.367 | 13.117 | 21.301 |
| `src/parse.tw` | 1149 | 0.970 | 1.527 | 14.664 | 18.347 |
| `std/linalg.tw` | 1172 | 0.895 | 1.154 | 0.450 | 1.233 |

`src/eval.tw` is 21% smaller than `src/tensor.tw` and takes 4x as long to check.
`src/parse.tw` is a fifth the size of `src/tensor.tw` and takes 1.5x as long. Lex
and parse track line count closely, as they should; check does not. The cause is
that the checker re-infers a function's body at call sites in order to check the
contract, so cost follows the call graph rather than the file, and a module of
many small mutually-calling functions costs more than a module of a few large
ones. Nothing here is slow enough to need fixing, but a file ten times the size
of `src/eval.tw` would not cost ten times as much.

---

## 5. Where the time actually goes

CPU profile of `bench/workloads/mc_option_grad.tw`, the differentiated Monte
Carlo pricer, over 1,200 iterations. 16.71 s duration, 28.21 s of samples across
cores.

```
go test ./bench/profile/ -run TestProfileMonteCarlo -count=30 \
    -cpuprofile mc.prof -o interp.test.exe
go tool pprof -top -nodecount=22 interp.test.exe mc.prof
```

Grouped by what the time is actually doing:

| what | share of flat time | the functions |
|---|---|---|
| arithmetic | ~50% | `broadcastBinary.func4/5/6`, `unary.func1/3.1`, `math.archExp`, `Relu`, `Mul`, `Add`, `parallelSum` |
| allocating and zeroing intermediates | 18% cumulative | `runtime.mallocgc` 18.08% cum, `makeslice` 17.48% cum, `memclrNoHeapPointers` 15.38% flat |
| goroutine coordination | ~17% | `lock2`, `unlock2`, `semasleep`, `semawakeup`, `preemptM`, `procyield`, `sysUnusedOS` |
| the interpreter | 0.035% | `Interp.Apply` 0.01 s flat; `evalExpr`, `evalCall`, `evalBinary`, `callClosure`, `execStmt` all 0% flat |

**Three conclusions, and the third is the actionable one.**

The interpreter costs nothing. `Interp.Apply` accounts for 0.01 seconds of flat
time out of 28.21, and every other interpreter entry point is 0% flat and
appears only in the cumulative column, dispatching into kernels. This
independently reproduces on a different workload what `docs/perf-baseline.md`
found on a transformer forward pass. twill is already a thin orchestration layer
over its tensor operations in the way Python is over NumPy, so the tree-walking
interpreter is not what to fix, and `docs/DECISIONS.md` entry 4 is the decision
this measurement supports.

Half the time is real arithmetic, which is where it should be, and a third of it
is not. Allocating and zeroing buffers for intermediates is 18%, and goroutine
coordination is another 17%. Together that is a third of the runtime spent on
neither the maths nor the language.

The 18% is the case for fusion. The pricer computes about eight elementwise
operations over 200,000 elements, and each one allocates a fresh 1.6 MB output
buffer that Go must zero before use, which is exactly what
`memclrNoHeapPointers` at 15.38% flat is. Every one of those buffers except the
last is read once and discarded. This is measured evidence for the fused-kernel
design in `docs/CODEGEN.md`, and it is evidence for the CPU half of that design,
which needs no GPU and no new dependency.

---

## 6. Where the gap comes from

Being slower than PyTorch is expected. PyTorch dispatches matmul to a tuned BLAS
and carries years of hand-written kernel work, and twill is a dependency-free Go
program. But "PyTorch is more optimised" is not an explanation. Four specific
things account for the difference, and each is a decision recorded in
`docs/DECISIONS.md` rather than an oversight.

**1. Every twill float is 8 bytes, and there is no f32 path.** A tensor's buffer
is `[]float64` regardless of its dtype tag; narrow dtypes are semantics without a
layout (`docs/DECISIONS.md` entry 5, tracked as NEEDS-111). This is most of the
f32 column: PyTorch f32 moves half the bytes and fits twice as many lanes in a
SIMD register, and twill has nothing to answer with. On `elementwise_1000000`
PyTorch's own f64-to-f32 speedup is 2.5x, which is the size of the handicap twill
carries on every row of the f32 comparison.

**2. No BLAS, so the matmul is hand-written Go.** `mm` and `mmNT` in
`internal/tensor/tensor.go` are decent for what they are: four independent
accumulators to hide FP-add latency, cache blocking sized to L2. They reach 21.5
GFLOP/s f64 at 1024x1024. A tuned BLAS reaches 250. That factor of twelve is the
matmul row of every table, and it is not a missing optimisation so much as the
accumulated result of years of hand-tuned assembly in the library twill declined
to link (`docs/DECISIONS.md` entry 7). Linking one means cgo, a C toolchain, a
platform matrix, and the end of the single dependency-free binary.

**3. No SIMD.** Go's compiler does not auto-vectorise the elementwise loops, so
`broadcastBinary` and `unary` process one f64 per iteration where PyTorch
processes eight f32 lanes. This is the elementwise rows.

**4. No fusion, so every intermediate is materialised.** Measured at 18% in
section 5. PyTorch does not fuse either by default, which is why this is the
smallest of the four, but PyTorch's allocator is a caching one that does not
return buffers to the OS and does not re-zero them, while Go's allocator zeroes
every slice it hands out.

The one place the gap nearly closes, `verify_deterministic` at 3.0x, is where
none of the four dominate: small tensors, mixed operations, and PyTorch paying
its own per-operation Python and dispatch overhead. It is the shape of workload
twill is least bad at, and it is worth knowing that such a shape exists.

---

## 7. What these numbers do not support

**Absolute times are good to about 40%, and the ratios are better than that.**
This is a laptop, and a laptop under sustained all-core load does not hold its
clocks. Across three full sweeps taken over about forty minutes, the same twill
workload drifted substantially with nothing changed but the machine's
temperature: `mc_option_fwd` reported 1.985 ms on a cold machine and 3.450 ms on
a hot one, a 74% spread. The tables above are one clean run with nothing else
running, which is the best single snapshot available, but a reader should treat
the absolute milliseconds as approximate.

That matters most for the comparison, because `run_all.ps1` measures every twill
workload first and every PyTorch workload afterwards, so twill is timed on a
cooler machine than PyTorch. That is a systematic bias in twill's favour, of the
same order as some of the smaller gaps being reported.

`bench/interleave.py` exists to remove it. It alternates twill and PyTorch
measurements of the same workload within the same few seconds, repeats that for
several rounds, and takes the median of the per-round ratios, so whatever the
clocks are doing they are doing it to both sides. The interleaved ratios are
reported in `bench/results/interleaved.json`.

**One early finding was wrong and is recorded here rather than deleted.** An
earlier sweep showed PyTorch's 16-thread numbers more than an order of magnitude
worse than its single-thread numbers, which looked like a thread-pool pathology
worth writing up. It was contention from other work on the same machine. On a
quiet machine PyTorch scales cleanly to 8 threads and regresses only slightly at
16. Nothing about a benchmark is trustworthy if the machine is doing something
else at the time, including a benchmark of the other implementation.

**These are single-machine numbers.** One CPU, one OS, one Go version, one
PyTorch build. Nothing here should be read as a general statement about either
implementation on other hardware.

**The workloads are small.** The largest is a 1024x1024 matmul and a ten-million
element vector. Nothing here says anything about how either implementation
behaves on a model that does not fit in RAM, and `docs/perf-baseline.md` is blunt
that twill in f64 cannot hold one.

---

## 8. Related

- `docs/CORRECTNESS.md`, the evidence that the numbers being computed are right,
  which is the prerequisite for caring how fast they are computed.
- `docs/perf-baseline.md`, the earlier optimisation-directed measurements of
  transformer forward-pass scaling and the kernel work queue.
- `docs/gpu-feasibility.md`, the measured case against a GPU backend today.
- `docs/CODEGEN.md`, the fused-kernel design that section 5's 18% is the argument
  for.
- `docs/DECISIONS.md`, entries 5, 6 and 7, which are the decisions section 6
  attributes the gap to.
