# The GPU backend

`docs/gpu-feasibility.md` measured the question and answered **not yet**. This
document does not overturn that answer. It designs the thing that would be built
when the answer changes, so that the design exists before the temptation does,
and so that the list of native primitives it needs is written down in
`docs/needs.md` rather than discovered halfway through.

Read the feasibility document first. Every performance number quoted below is
from it, measured on one machine on 2026-08-07. Nothing new was measured for
this document. Anything that is a projection is labelled as one.

Two facts from that document govern everything here.

**Fact one: f64 on the RTX 5070 runs at 341.7 GFLOP/s against 18,010 for f32, a
ratio of 52.7 to 1.** Twill's tensors are f64 throughout, so a GPU backend today
buys the slow 5% of the card. This is being addressed elsewhere: f32 and bf16
storage types are in progress in `src/tensor.tw`. This design is written to be
dtype-parametric so that it does not have to be rewritten when they land, but it
is honest to say that the design's value is contingent on that work. A GPU
backend restricted to f64 is worth much less than the same backend over f32.

**Fact two: at 64x64, matmul on the GPU measured 92.8 us round trip against
74.2 us on the CPU, a 0.8x.** Transfer and launch overhead was measured at
roughly 80 us per op and it does not shrink with problem size. No dtype fixes
this. It is a fixed cost per boundary crossing, and the only defence against it
is to cross the boundary fewer times. That is why the memory model below is the
centre of this design and not an appendix to it.

---

## 1. Which backend

Four candidates, scored against twill's constraints rather than against each
other in the abstract. The constraints that matter are: design principle 4 is
"no dependencies"; the release pipeline cross-compiles every target from one
runner with cgo off; and the machine the project is developed on has two GPUs
from two vendors.

### CUDA

Rejected. It is NVIDIA-only, which excludes the Intel Arc 140T in the
development machine itself and every contributor without an NVIDIA card. It
needs a toolkit to compile kernels ahead of time; the feasibility work found no
local `nvcc` and the 5GB devel container wedged the Docker engine on a 16GB
machine. Adopting CUDA converts twill from a project with one portable binary
into a project with a vendor-specific build. The performance argument for CUDA
over OpenCL on the same silicon is real but small, and it is not worth that.

### Vulkan compute

Plausible and rejected for now. Both vendors ship ICDs on the development
machine, and `vulkan-1.dll` is present in `System32` with no SDK installed, so
the runtime story is as good as OpenCL's. The problem is the compile story:
Vulkan consumes SPIR-V, not source, so shipping kernels means either shipping
precompiled SPIR-V blobs (which have to be built by a toolchain that is not
present, and which then become opaque binary artifacts in a repository whose
whole argument is that you can read the implementation) or writing a SPIR-V
emitter in twill. The second is genuinely interesting and is the right answer
some years from now. It is not the right answer for the first version.

Vulkan is also a much larger API. The OpenCL surface below is about fifteen
entry points. The equivalent Vulkan path needs instance and device creation,
queue families, descriptor set layouts, pipeline layouts, command buffers,
memory type selection and explicit barriers. Every one of those is a native
symbol that has to be reachable from twill.

### WebGPU / wgpu

Rejected outright, on a fact rather than a preference: **WGSL has no f64 type.**
There is no double precision in WebGPU at all. Twill's numeric semantics are
f64, `docs/finance.md` sells reproducibility, and a backend that cannot represent
the language's own number type is not a backend. It would also mean either cgo
against a native library or shelling out to a separate binary, neither of which
is compatible with the release model.

### OpenCL

**Recommended.** The reasons, in order of weight:

1. **It is the only one demonstrated end to end on this hardware.** The
   feasibility work ran an f64 vector add on both the RTX 5070 and the Arc 140T
   and verified the result against the host at `max abs err = 0` over a million
   elements. That is not a paper evaluation.

2. **No SDK, no headers, no container, no admin.** The ICD loader ships with the
   driver. A host program that resolves `OpenCL.dll` with `LoadLibrary` and
   declares the dozen entry points it needs compiled with the mingw `gcc`
   already on the box. The dependency is the driver, which the user already has
   if they have a GPU, and nothing at build time.

3. **Kernels are source, compiled at run time.** This is the property that makes
   OpenCL fit a language project rather than a binary product. The kernels in
   `src/gpu/` are text that is readable in the repository and handed to the
   driver at run time, so there is no build step, no toolchain matrix, and no
   opaque artifact. It also means a kernel can be specialised at run time on the
   shapes it is about to run, which the tiling design in section 6 uses.

4. **It keeps both GPUs in play.** The vendor-neutral story is worth something
   even though the feasibility numbers show the Arc iGPU is slower than the CPU
   at everything except N=1024. It is worth something because correctness gets
   tested on two independent implementations, which catches kernels that
   accidentally depend on one vendor's warp width.

The honest weakness: OpenCL is a legacy-ish API that vendors maintain rather
than invest in. NVIDIA's implementation is capped at OpenCL 3.0 with a 1.2
feature baseline in practice, and new hardware features arrive in CUDA first and
in OpenCL late or never. This design accepts that. Twill is not chasing tensor
cores; it is trying to get a dense f64 (later f32) matmul onto a wide machine,
and OpenCL 1.2 is entirely sufficient for that.

### The dependency question, stated plainly

Recommending OpenCL does not make the no-dependency property survive. It does
not. `docs/gpu-feasibility.md` calls the alternative framing a word game and it
is right. What OpenCL gets is a *smaller and better-shaped* breach:

- Nothing is added to the build. The default binary and the release matrix are
  unchanged, because the GPU path is absent unless the driver library loads at
  run time.
- The dependency is discovered, not declared. There is no version to pin, no
  package to vendor, no lockfile.
- Failure is graceful and total. If the loader is absent, `gpu_available()` is
  false and every tensor stays on the CPU with identical results. There is no
  degraded mode to reason about.

That is the most that can be claimed, and this document claims exactly that.

---

## 2. The dispatch rule

At 64x64 the GPU is a 0.8x. Somewhere between 64 and 128 it crosses one. So
there is a threshold, and the design question is where it comes from.

### A fixed constant is a guess

The obvious implementation is a constant, by analogy with `minParallel = 8192`
in `internal/tensor/parallel.go`. It would be wrong. `minParallel` picks between
one CPU and sixteen cores of the same CPU, and the ratio between those is a
property of the program. A GPU threshold picks between two different pieces of
silicon whose ratio varies by two orders of magnitude across machines: the
feasibility numbers show the same threshold sitting in a completely different
place for the RTX 5070 (crosses one between N=64 and N=128) and for the Arc
140T (never crosses until N=1024). A constant tuned on the 5070 would send work
to the Arc that the Arc loses on, on every machine that has only the Arc.

**So: no constant. The threshold is measured on the machine it will run on.**

### How it is measured

`src/gpu/dispatch.tw` carries a calibration routine that runs once per device,
the first time a GPU dispatch is considered, and produces a small cost model.
The model is deliberately crude, because a crude model that was measured beats a
precise model that was assumed.

Three numbers per device:

- `launch_us`: the fixed cost of a kernel launch plus synchronise, measured with
  an empty kernel over a one-element buffer, repeated enough times that the
  total is comfortably above the resolution of `now_ms` (NEEDS-39). This is the
  term that made 64x64 a loss.
- `transfer_us_per_mb`: measured by writing and reading back buffers of two
  sizes and taking the slope, so the fixed part falls out.
- `flops_per_us`: measured with the same matmul kernel that will actually run,
  at one large size, device-resident, so it is throughput and not throughput
  plus overhead.

Against those, a candidate op is offloaded when

    gpu_cost(op) + transfer_cost(op) < cpu_cost(op) * margin

where `cpu_cost` comes from a matching one-time CPU calibration of the same
kernel, and `margin` is 1.2. The margin exists because being wrong in the
direction of staying on the CPU costs a missed speedup, and being wrong in the
direction of offloading costs a regression on a workload that used to be fine.
Those are not symmetric, so the rule is not symmetric.

`transfer_cost(op)` is zero for operands already resident on the device. That is
not a detail of the formula, it is the whole point of it: the same op that loses
at 64x64 with transfers wins at 64x64 without them, 10.0 us against 74.2 us,
which is 7.4x. The dispatch rule and the memory model are one mechanism.

### Calibration cost and caching

Calibration takes a handful of milliseconds and would be absurd to repeat per
process for a script that runs for two hundred. It is cached on disk, keyed by
the device name and driver version string reported by `gpu_device_info`. A
missing or unreadable cache means recalibrate, never means guess.

Calibration is also skippable: an explicit `--gpu=off` never touches the device,
and `--gpu=force` offloads everything above rank 0 and exists only so that the
threshold's effect can be measured by turning it off.

**Projection, not measurement:** on the RTX 5070 this rule is expected to place
the matmul threshold between N=64 and N=128, because that is where the measured
round-trip table crosses. It has not been run, because the calibration routine
described here does not exist yet.

---

## 3. The memory model

This is the core of the design. The measured 80 us per boundary crossing is the
thing that decides whether any of this is worth doing, and it is entirely under
the design's control.

### Residency is a property of a buffer, not of a tensor

A tensor's data has a **location**: host, device, or both. Both is not a
contradiction, it is the common case: a tensor uploaded and not since modified
has a valid copy in each place, and reading it costs nothing.

    enum Loc { LocHost, LocDevice, LocBoth }

Twill tensors are immutable values, which makes this far simpler than it is in
a framework with in-place mutation. A `DeviceTensor` pairs a shape with a device
buffer handle, and because nothing writes through it after construction, a
host copy once taken is valid forever. There is no invalidation protocol,
no dirty bit, and no coherence problem. That is a real dividend of the value
semantics `src/tensor.tw` already commits to, and it is worth naming because it
is the part of a GPU backend that is normally hardest.

### Transfers happen at exactly two points

**Up**, when a host-only tensor is an operand of an op that has been chosen for
the device. **Down**, when a device-only tensor is read by something that needs
host bytes: printing, comparison against a scalar in twill control flow,
`to_nested`, writing a file, or an op with no device kernel.

Everything else stays where it is. In particular:

- The **output** of a device op is allocated on the device and left there. It is
  never read back speculatively. This is the single rule that turns a chain of
  ops into one upload, n launches and one download, instead of n round trips.
- A tensor that is used by two device ops is uploaded once.
- Constants and weights are uploaded once and stay resident for the life of the
  program, because nothing can write to them.

### Chains stay resident

Consider a forward pass `relu(X @ W1 + b1) @ W2 + b2`. Naive per-op offload
crosses the boundary ten times. Under this model:

- `X`, `W1`, `b1`, `W2`, `b2` upload once each, on first use. Weights upload
  once for the whole training run.
- `@`, `+`, `relu`, `@`, `+` each launch a kernel over device buffers and leave
  their output on the device.
- Nothing comes down until the loss is read.

The residency is achieved by the ordinary dataflow of the interpreter and does
not need a graph optimiser. Each op asks each operand for a device handle, the
operand either has one or uploads itself once to get one, and the op leaves its
result on the device. That is it. There is no fusion pass and no scheduling
here, deliberately: fusion is a large second project and it is not what the
measured numbers say is missing.

### Eviction

8GB on the test card, shared with the display. A device allocation that fails
is not an error. `gpu_alloc` returning `Err` triggers eviction of device copies
whose host copy is already valid (`LocBoth`), oldest first, and one retry. If
that still fails, the op falls back to the CPU and the run continues with a
slower answer and an identical one. Nothing in this design is allowed to turn a
memory pressure event into a failed program.

### Asynchrony, and why there is none yet

OpenCL commands are queued and the queue can run ahead of the host. Enqueuing a
chain of kernels without synchronising between them would hide launch latency
and is worth real time. It is deliberately not in the first version, because a
non-blocking queue makes an error report arrive at a point unrelated to the op
that caused it, and debugging a numerical difference is hard enough with the
error attached to the right line. `gpu_finish` is called after every launch in
version one. The queue depth is the first optimisation to take once the answers
are trusted, and it is the reason `gpu_finish` is a named primitive rather than
folded into `gpu_launch`.

---

## 4. Autodiff on the device

`src/tensor.tw` records a flat tape: each `TapeEntry` holds an op, the indices of
earlier entries, and the output tensor. `backward` walks it in reverse.

### The tape holds device tensors

The `out` field of a `TapeEntry` becomes a tensor whose data may be a device
handle. This changes nothing structurally, because the tape never inspects the
bytes of `out`. It stores it and hands it to a `vjp` function. Only the `vjp`
functions look inside, and they are the ops.

This matters more than it sounds. A tape entry pins its output alive, which on
the CPU costs memory and on the GPU costs *device* memory, which is scarcer. A
long training step with device-resident activations can exceed 8GB where the
host copy would have been fine. The eviction rule above covers it, because an
activation whose host copy exists is evictable, but the honest statement is that
device memory is the resource this design is most likely to run out of first.

### The backward pass must not force a transfer per node

Every gradient rule in `src/tensor.tw` is built from the same forward kernels.
`vjp_matmul` is `transpose2d` and two `mm` calls. `vjp_softmax` is a sum along a
run and two elementwise ops. `vjp_binary` is elementwise plus a `sum_to_shape`.
So the backward pass has no primitive that the forward pass does not, and if the
forward kernels are device-resident then the backward ones are too, by
construction. There is no separate backward device path to write.

Two places would force a transfer and both are avoidable:

- **`accumulate`** adds an incoming cotangent into a slot. On the CPU it is a
  loop over a buffer. On the device it must be the elementwise add kernel
  writing into the existing device buffer, not a read, add and write. This is
  the one place in the whole design where a buffer is written twice, and it is
  safe because a cotangent slot is private to one backward pass.
- **`touched`**, the flag array saying whether a cotangent slot has been written
  yet, is host-side bookkeeping over node indices and never touches tensor data.
  It stays on the host and costs nothing.

Ops with no device kernel (sort, median, topk, einsum with an unusual spec) pull
their operands down, run on the host, and leave their result on the host. That
is correct, it is slow, and the dispatch rule already accounts for it because
the transfer cost of the *next* device op will include the re-upload. A backward
pass through a median is going to be slow. That is a true statement about the
op and not a flaw in the design.

### Second order

`hessian` in `src/tensor.tw` runs forward-mode jets over the same kernels. Jets
double the arithmetic and keep the same shapes, so nothing about residency
changes. No device work is planned for jets in the first version; a jet run
stays on the host. This is stated so that it is a decision and not an omission.

---

## 5. Determinism

`README.md` line 148: "across CPU cores, deterministically: parallelism never
changes a result." `internal/tensor/parallel.go` says the result is
"bit-identical to a serial run" and `parallelSum` blocks by a fixed size
specifically so the answer does not depend on the worker count. `docs/finance.md`
sells reproducibility as a thing twill wins on.

A GPU reduction is not deterministic by default. Atomics land in arrival order,
tree reductions have a shape that depends on the work-group size the driver
chose, and a compiler is free to contract `a*b+c` into an FMA with different
rounding. This is a direct conflict with a documented language guarantee and it
does not get to be resolved by not mentioning it.

**The resolution: the guarantee is kept, by construction, and the cost is
paid in parallelism rather than in accuracy.** Three rules.

### Rule 1: parallelise across independent outputs, never within a reduction

This is the whole trick and it is why this design does not need atomics
anywhere.

For every reduction twill has, the output is not one number, it is a grid of
independent runs. `sum_axis` over a `[512, 768]` tensor along axis 1 is 512
independent sums of 768 elements. Assign **one work-item per output element**,
and have that work-item walk its run sequentially in ascending index order,
exactly as `reduce_axis` in `src/tensor.tw` does. There are 512 work-items, which
saturates nothing on a 36-CU card but is plenty of parallelism for the sizes at
which offloading is chosen at all, and each output is the sum of the same
numbers in the same order as the CPU produced. Bit-identical, with no barrier,
no local memory and no atomic.

The same shape applies to `softmax` (one work-item per run, doing max, exp, sum,
divide in the source order), `logsumexp`, `prod`, `max`, `min`, `cumsum` and
`conv2d` (one work-item per output pixel, walking the window in scan order).

Matmul is the same statement: one work-item per output element accumulating over
`k` ascending. Tiling, which section 6 describes, changes which *memory* the
work-item reads and does not change the order it adds in. That is a load
requirement on the tiled kernel and it is testable.

The cost is real and worth naming. `sum_all` of a large tensor has exactly one
output, so this rule makes it serial, which is absurd on a GPU. The escape is
in rule 2.

### Rule 2: where a tree is needed, use the CPU's tree

`internal/tensor/parallel.go` already defines a deterministic tree for the
whole-tensor sum: below `minParallel = 8192` a plain running sum, and at or
above it, fixed blocks of `sumChunk = 4096` summed independently and then
combined in block order. Because the block size is fixed rather than derived
from the core count, that tree is a *specification*, not an implementation
detail, and a GPU kernel can reproduce it exactly: one work-item per 4096-element
block, then a second pass combining partials in ascending block order. Same
numbers, same order, same answer.

**A divergence found while writing this, which is not a GPU problem.**
`src/tensor.tw` `reduce_all` is a plain sequential sum with no blocking, so for
`n >= 8192` it already disagrees with `internal/tensor` in the last bits, before
any GPU exists. The three implementations cannot all be right. The Go blocking
form is the one to adopt, because it is the one that is both parallelisable and
pinned, and `src/tensor.tw` should take it. That file is owned elsewhere at the
time of writing, so this is recorded here and in `docs/needs.md` rather than
changed. The GPU kernel in `src/gpu/reduce.tw` implements the blocked form and
says so at the call site.

### Rule 3: forbid every compiler transform that changes an answer

The kernels are compiled at run time with a fixed options string, and what is
*absent* from it matters as much as what is present:

    -cl-opt-disable is NOT used            it would cost most of the speedup
    -cl-fast-relaxed-math is NOT used      it permits reassociation
    -cl-unsafe-math-optimizations NOT used
    -cl-mad-enable is NOT used             it permits a*b+c to lose a rounding
    -cl-no-signed-zeros is NOT used        f64_max(-0, +0) depends on it

and every kernel source opens with

    #pragma OPENCL FP_CONTRACT OFF

because contraction into an FMA is otherwise implementation-defined, and an FMA
keeps an extra rounding that the CPU's separate multiply and add does not. That
one pragma is the difference between a matmul that matches and one that is off
by an ulp per k step. No `native_*` or `half_*` function appears anywhere in
`src/gpu/`, and there is a test for that, because they are exactly the functions
whose accuracy is unspecified.

### What this does not fix: transcendentals

`exp`, `log`, `sin`, `cos`, `tanh` and `sqrt` are the hole, and it is a real one.
OpenCL specifies accuracy bounds for these in ulp, and those bounds are looser
than "the same bits as Go's `math` package". Two vendors' `exp` may both be
conformant and differ from each other and from Go in the last bit.

NEEDS-68 is explicit that agreement in the last bits is a requirement and not a
nicety, because `testdata/` compares byte for byte after a canonical float
rendering, so a one-ulp `exp` turns every test touching a sigmoid into a diff.

Three options, and the design takes the first now and the second later:

1. **Interim: unary transcendental ops stay on the CPU.** `exp`, `log`, `sin`,
   `cos`, `tanh`, `sigmoid` and `sqrt` are not offloaded. They are memory-bound
   elementwise ops whose GPU speedup is bounded by bandwidth anyway, so this
   costs less than it sounds. It does cost the residency of a chain that
   contains one, which is the real price: `relu` offloads, `sigmoid` does not,
   and a network built from sigmoids gets no residency at all. `softmax` and
   `logsumexp` are affected too, since both call `exp`, so under option 1 they
   run on the host despite having device kernels written for them. Those kernels
   are written anyway, because they become correct the moment option 2 lands.

2. **Target: transcribe the same algorithm into the kernel source.** Whatever
   supplies `f64_exp` for NEEDS-68 is a specific algorithm, and the same
   algorithm written in OpenCL C over the same f64 arithmetic produces the same
   bits, because every step of it is a correctly-rounded IEEE operation. This is
   a few hundred lines of kernel source per function and it is the only route to
   a device softmax that matches. It is real work and it is not free, and it
   is the single largest piece of unbudgeted effort in this whole design.

3. **Rejected: relax and document.** Rejected because the guarantee is load
   bearing for the audience `docs/finance.md` names, and because a difference
   that only appears on some hardware in the last bit is the worst possible
   failure mode to debug.

### The opt-out

There is one, it is explicit, and it is off by default.

    twill run model.tw --gpu-fast

`--gpu-fast` permits atomic reductions, driver-chosen tree shapes, and
`-cl-fast-relaxed-math`. It is expected to be meaningfully faster for whole-
tensor reductions and for softmax over short runs, though by how much is
unmeasured. When it is on:

- The run header prints `determinism: relaxed (--gpu-fast)`, every time, on
  stderr, so it cannot be on in a log without being visible in that log.
- `twill check` refuses to accept it, because a shape check has no business
  being nondeterministic.
- The differential test harness refuses to run under it.

The reason this exists at all rather than being refused outright is that the
person who wants a fast approximate hyperparameter sweep and the person who
wants a reproducible finance model are the same person on different days, and
the second person's guarantee is protected by making the first person type nine
extra characters, not by taking the option away.

**Default is deterministic. Always. On every device.**

---

## 6. The kernels

`src/gpu/` contains the following. Each file states its parallel decomposition
at the top, and the reason for it.

| File | What it is |
| --- | --- |
| `device.tw` | The declared native surface. Every foreign symbol the backend needs, in one place, and nothing else. |
| `source.tw` | Kernel source as twill strings, and the compile options that keep results deterministic. |
| `buffer.tw` | Residency: `DeviceTensor`, upload, download, eviction. |
| `dispatch.tw` | Calibration, the cost model, and the offload decision. |
| `elementwise.tw` | Binary and unary elementwise, including broadcasting. |
| `matmul.tw` | Tiled matmul, and the host side that picks a tile size. |
| `reduce.tw` | Axis reductions and the blocked whole-tensor reduction. |
| `softmax.tw` | Softmax and logsumexp, one work-item per run. |
| `conv.tw` | conv2d and maxpool2d, one work-item per output pixel. |
| `autodiff.tw` | Device-aware `accumulate` and the residency rules for `backward`. |

The decompositions, with the reason for each:

**Elementwise, one work-item per output element.** The trivially correct
decomposition and the only one that makes sense: every output depends on one
input element per operand, there is no sharing to exploit and no ordering to
preserve. Broadcasting is handled by computing the source offsets from the
output index with the same `eff_strides` arithmetic `src/tensor.tw` uses, so a
broadcast operand is read repeatedly rather than materialised. This matters:
materialising a `[1, 768]` bias against a `[512, 768]` activation would be 512
times the memory traffic for no reason.

**Matmul, one work-item per output element, tiled over local memory.** Each
work-item owns one `c[i][j]` and accumulates over `k` ascending into a private
register. A work-group of 16 by 16 cooperatively stages a 16 by 16 tile of `a`
and of `b` into local memory, barriers, and every work-item in the group reads
its 16 values for that tile from local memory rather than global. That is the
whole optimisation: it changes each element of `a` and `b` from being read
`n` times out of global memory to being read once, and it is why a tiled matmul
is several times a naive one.

It is worth being precise that this does not touch determinism. The `k` loop
still runs 0, 1, 2 and so on, in tiles, and the additions into the accumulator
happen in that order. Tiling changed where the operands were read from. It did
not reorder a sum. The zero-skip in `src/tensor.tw` `mm` (`if aip != 0.0`) is
reproduced, because it is load-bearing for a zero times an infinity, not a
speed hack, and dropping it would turn a sparse row against an infinity into a
NaN on the GPU and not on the CPU.

**Reductions, one work-item per output run.** Rule 1 above. For `reduce_axis`,
the grid is `before` by `after` and each work-item strides its run by `after`,
which is exactly the loop `src/tensor.tw` writes. For the whole-tensor case, one
work-item per 4096-element block and a second kernel to combine partials in
block order, reproducing `parallelSum`.

**Softmax, one work-item per run.** All four passes (maximum, exponentiate,
accumulate, divide) run inside one work-item over its own run. The alternative
is four kernels over the whole tensor with the run maximum and sum computed by
group reductions, which is faster for long runs and reorders the sum, so it is
not taken. The maximum shift is kept exactly as `src/tensor.tw` writes it,
because it is what stops `exp` overflowing to infinity and turning the ratio
into a NaN, and it is not an optimisation to be dropped for a shorter kernel.

**conv2d, one work-item per output pixel.** The grid is `[cout, oh, ow]` and each
work-item walks the full `cin * kh * kw` window in scan order, accumulating into
a register. Weights are small and read by every work-item, which is the ideal
case for the constant address space. This is deliberately the direct form and
not im2col: `src/tensor.tw` says the loop nest is the definition of the operation
and the gradient rule is read against it, and the same reasoning applies with
more force here, since a GPU im2col would materialise a `cin*kh*kw` by `oh*ow`
matrix on a device where memory is the scarce resource.

**maxpool2d, one work-item per output window.** The `>` and not `>=` tie-break
is reproduced exactly, so the first maximum in scan order wins. That one
character decides which input gets the gradient, and a kernel that used `>=`
would produce a different gradient on any flat region, which is every pooled
relu output.

### What has no kernel, and why

`sort_axis`, `argsort_axis`, `topk_axis`, `median_axis`, `einsum`. Sorting on a
GPU is a real project (a bitonic sort is the standard answer and it is a
different kernel per size class), median is built on sorting, and einsum is a
general contraction whose kernel would have to be generated per spec. All five
run on the host and pull their operands down. This is a listed limitation, not
an oversight.

---

## 7. Testing

The backend is worth nothing if it is not bit-identical, so the test is not "is
it close".

- **Differential, every op, every kernel.** Run each op on the host and on the
  device over the same inputs and compare the f64 bit patterns. Not a tolerance.
  Equality. Under the determinism rules of section 5 this must hold, and where
  it does not the kernel is wrong.
- **Two devices where two exist.** The development machine has an NVIDIA and an
  Intel device, and a kernel that passes on one and fails on the other has an
  assumption in it about work-group size or warp width.
- **Grep the kernel source for `native_` and `half_` and fail if present.**
- **The threshold is tested by being turned off.** `--gpu=force` and `--gpu=off`
  must produce identical output on every test in `testdata/`. If they do not,
  the dispatch rule is changing an answer, which it is never allowed to do.

---

## 8. What this cannot be yet

Everything in `src/gpu/` is twill that calls primitives that do not exist. The
primitives are foreign function calls into a driver library, and there is no
mechanism in twill for a foreign function call, and under the current no-Go rule
there is nowhere for the layer that would implement one to live.

That is not a gap that more twill closes. `docs/needs.md` NEEDS-100 through
NEEDS-107 name the primitives precisely and NEEDS-108 states the problem
directly. The value of `src/gpu/` today is that the surface is small, named and
argued for, so that the day an FFI exists the requirement is a list of fifteen
symbols rather than an open question.

The recommendation of `docs/gpu-feasibility.md` is unchanged: **do not build
this yet.** Settle f32. Add the matmul benchmarks at N=256, 512 and 1024 that
the repository does not have. Find a real twill program that is matmul-bound at
256x256 or larger. This document is what gets built when those three things are
true, and not before.
