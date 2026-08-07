# GPU feasibility

An empirical answer to a question `docs/rewrite-plan.md` defers to this file:
should Twill get a GPU backend, and if so when. Everything below was measured
on one machine on 2026-08-07. Nothing here is estimated.

The short version: a GPU backend is technically reachable, the honest speedup is
about 9x and not the 30x the rewrite plan guesses at, it only applies to matmul
above roughly 128x128, it is *slower* than the current CPU code at the sizes
Twill actually benchmarks, and it costs the no-dependencies property that the
project is built on. Recommendation is do not do this yet.

## The machine

ASUS ROG Zephyrus G16. Intel Core Ultra 9 285H, 16 cores, 16 threads.
NVIDIA GeForce RTX 5070 Laptop GPU (Blackwell, 36 CUs, 8GB) plus an
Intel Arc 140T integrated GPU (128 CUs, shared memory). Windows 11.

This is a hybrid laptop, not a workstation. That matters for the conclusion:
any CUDA-only story silently excludes one of the two GPUs in this very machine,
and excludes every contributor without an NVIDIA card.

## What was measured

### 1. Driver and CUDA support

`nvidia-smi` on the host:

```
NVIDIA-SMI 610.88     KMD Version: 610.88     CUDA UMD Version: 13.3
0  NVIDIA GeForce RTX 5070 ...  WDDM | 00000000:01:00.0 Off |  N/A
   57C  P8  3W / 65W  |  114MiB / 8151MiB  |  0%  Default
```

A current driver supporting CUDA 13.3. No local CUDA toolkit is installed
(`nvcc` is absent), so compiling CUDA means a container.

### 2. Docker GPU passthrough

Docker 29.6.2, with `nvidia` present in the runtime list. The literal test:

```
$ MSYS_NO_PATHCONV=1 docker run --rm --gpus all \
    nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi

NVIDIA-SMI 610.57.01   KMD Version: 610.88   CUDA UMD Version: 13.3
0  NVIDIA GeForce RTX 5070 ...  On | 00000000:01:00.0 Off |  N/A
   58C  P8  3W / 65W | 114MiB / 8151MiB | 0% Default
```

Passthrough works. The card is visible inside a container with no extra setup.

It is worth recording what happened next, because it is part of the cost. Pulling
`nvidia/cuda:12.8.1-devel-ubuntu22.04`, which is roughly 5GB, wedged the Docker
Linux engine. The API began returning `500 Internal Server Error` on every route
including `_ping`, and `docker` calls then hung indefinitely. Restarting Docker
Desktop and terminating the `docker-desktop` WSL distro did not bring the engine
back within the session. The end-to-end CUDA kernel compile was therefore not
completed here. On a 16GB machine the CUDA devel image is not a casual
dependency, and a build pipeline that needs it is not a casual pipeline.

That gap is covered, and more usefully, by the next test, which runs a real f64
kernel on the same card without any of that machinery.

### 3. The vendor-neutral options

Both GPUs register ICDs on this host, read out of the driver registry:

```
NVIDIA GeForce RTX 5070 Laptop GPU
  OpenCL: ...\nvami.inf_amd64_...\nvopencl64.dll
  Vulkan: ...\nvami.inf_amd64_...\nv-vk64.json
Intel(R) Arc(TM) 140T GPU (8GB)
  OpenCL: ...\iigd_dch.inf_amd64_...\igdrcl64.dll
  Vulkan: ...\iigd_dch.inf_amd64_...\igvk64.json
```

`OpenCL.dll` and `vulkan-1.dll` are both present in `System32`. Neither needs an
SDK to *use*: the ICD loader is part of the driver install.

OpenCL was tested properly. A host program that resolves the loader with
`LoadLibrary`/`GetProcAddress`, declares the dozen API entry points it needs, and
carries no headers and no SDK, compiles with the mingw `gcc` already on this box
and runs against both GPUs:

```
OpenCL platforms: 2

[NVIDIA CUDA]
  device: NVIDIA GeForce RTX 5070 Laptop GPU
  OpenCL 3.0 CUDA | CUs=36 | 1545MHz | mem=8.5GB
  cl_khr_fp64 extension: YES   DOUBLE_FP_CONFIG=0x3f
  f64 vecadd n=1048576: host-verified max abs err=0  c[0]=1.0000 c[n-1]=786432.2500
  fma32:    0.13 ms    18009.9 GFLOP/s
  fma64:    6.90 ms      341.7 GFLOP/s
  >> f32:f64 ratio = 52.7 : 1

[Intel(R) OpenCL Graphics]
  device: Intel(R) Arc(TM) 140T GPU (8GB)
  OpenCL 3.0 NEO | CUs=128 | 2350MHz | mem=8.4GB
  cl_khr_fp64 extension: YES   DOUBLE_FP_CONFIG=0x3f
  f64 vecadd n=1048576: host-verified max abs err=0  c[0]=1.0000 c[n-1]=786432.2500
  fma32:    2.60 ms     3223.4 GFLOP/s
  fma64:   72.44 ms      115.8 GFLOP/s
  >> f32:f64 ratio = 27.8 : 1
```

So a double-precision kernel does run, on both vendors, and its result is
bit-exact against the host (`max abs err = 0` over a million elements).

Scoring the four candidates on this machine:

- **OpenCL.** Viable, and the only one demonstrated end to end here. Works on
  both GPUs, no SDK, no admin, no container. Its weakness is that it is a
  legacy-ish API that vendors maintain rather than invest in.
- **CUDA.** Passthrough proven, but NVIDIA-only, needs a container to compile,
  and the container broke the engine on this machine. It would leave the Arc GPU
  unused.
- **Vulkan compute.** Plausible: both vendors ship ICDs. Not tested, because
  compiling GLSL to SPIR-V needs `glslc`, which is absent. More portable than
  CUDA and considerably more work than OpenCL.
- **WebGPU/wgpu.** Not testable here. There is no local Rust toolchain, and
  wgpu would mean either cgo against a native library or shelling out to a
  separate binary. Also, WebGPU has no f64 type at all, which by itself rules it
  out for Twill.

### 4. The f64 throughput reality

The claim to check was that consumer NVIDIA cards run double precision at about
1/64 of single. Measured on this card: **1/52.7**. So the claim is close to
right, slightly pessimistic, and the correction does not change anything.

The RTX 5070 does 18.0 TFLOP/s in f32 and 342 GFLOP/s in f64. Twill's tensors
are `[]float64` throughout, so Twill gets the 342, which is 5% of the number
the card is sold on. The Arc iGPU is less lopsided in ratio (27.8:1) but slower
in absolute terms (116 GFLOP/s f64), so it is worse either way.

### 5. GPU against Twill's actual CPU code

`internal/tensor.mm` is a parallel triple loop over `[]float64`, split across
cores by `runChunks`. Measured, both in-repo and as a standalone copy of `mm`
at sizes the repo does not yet benchmark:

```
BenchmarkMatMul64-16       74233 ns/op
BenchmarkMatMul256-16    1925365 ns/op

standalone mm(), GOMAXPROCS=16:
    N          time      f64 GFLOP/s
   64       74.2 us              7.1
  128      289.1 us             14.5
  256     1762.8 us             19.0
  512     9939.5 us             27.0
 1024    85878.0 us             25.0
```

Against the same matmul in OpenCL f64 on the RTX 5070, timed two ways: kernel
only, which is what a device-resident tensor would cost, and round trip, which
is what a naive per-op offload would cost:

```
    N   kernel-only    round-trip   f64 GFLOP/s (rt)
   64      10.0 us        92.8 us          5.7
  128      17.6 us       187.5 us         22.4
  256     100.6 us       438.9 us         76.5
  512     707.3 us      1841.1 us        145.8
 1024    5666.6 us      9378.0 us        229.0
```

Putting those side by side is the whole argument:

| N | CPU | GPU round trip | speedup | GPU kernel only | speedup |
|---|---|---|---|---|---|
| 64 | 74.2 us | 92.8 us | **0.8x** | 10.0 us | 7.4x |
| 128 | 289 us | 187 us | 1.5x | 17.6 us | 16.4x |
| 256 | 1763 us | 439 us | 4.0x | 101 us | 17.5x |
| 512 | 9940 us | 1841 us | 5.4x | 707 us | 14.1x |
| 1024 | 85878 us | 9378 us | 9.2x | 5667 us | 15.2x |

Three things follow.

First, at 64x64, the size `BenchmarkMatMul64` uses, offloading is a **loss**.
The GPU takes 92.8 us to do what the CPU does in 74.2 us. Transfer and launch
overhead is roughly 80 us per op and it does not shrink.

Second, the ceiling is about 9x with transfers and about 15x if tensors live on
the device. `docs/rewrite-plan.md` weighs "a 2.6x on CPU and a 30x on GPU". The
30x is not available here, in f64, on this card. The real comparison is 2.6x
against 15x, and the 15x only for large matmul.

Third, the Intel iGPU never beats the CPU except at N=1024, where it manages
65.5 GFLOP/s round trip against the CPU's 25. The vendor-neutral story is real
for correctness and useless for speed on the integrated part.

## What integrating this into Twill would cost

### The backward pass

`Tensor` is `Data []float64` plus `Grad []float64`, and every backward closure
reaches straight into those slices. `MatMul`'s is representative:

```go
return track2(res, a, b, func() {
    g := res.Grad
    if a.RequiresGrad {
        bt := transpose2d(b.Data, k, n)
        dA := mm(g, m, n, bt, k)
        ga := a.ensureGrad()
        for i := range dA { ga[i] += dA[i] }
    }
    ...
})
```

A device-resident tensor means `Data` and `Grad` are handles, not slices, and
every one of those closures has to be rewritten. That is the entire operator
surface: `ops.go` (1234 lines), `einsum.go`, `conv.go`, `scan.go`, `gather.go`,
and `jet.go`, which does forward mode and would need each op's JVP on the device
too. There are 152 direct `.Data` uses in non-test code, 37 of them outside
`internal/tensor`, in `internal/value`, `internal/interp` and `cmd/twill`. So
the boundary is not contained inside the tensor package today, and containing it
is a prerequisite, not a detail.

Halfway is worse than either end. If only matmul is offloaded, every op around
it pays the 80 us round trip, and the table above says that loses below N=128.
A GPU backend that pays for itself has to keep tensors resident across a whole
training step, which means the autodiff graph itself becomes device-aware.

There is a second, quieter cost. `parallel.go` is explicit that results are
"bit-identical to a serial run" and "the same on any number of cores", and
`parallelSum` uses fixed-size blocks specifically so the answer does not depend
on the worker count. GPU reductions do not give that for free. `docs/finance.md`
sells determinism and reproducibility as a thing Twill wins on "regardless".
A GPU backend puts that in tension with itself, and for finance that is not a
small print item.

### The zero-dependency policy

`go.mod` has no requires. The release workflow builds with `CGO_ENABLED=0` across
a GOOS/GOARCH matrix. Design principle 4 is "No dependencies", and the README
leads with a binary that has none.

Every GPU option breaks that in practice. CUDA needs cgo or a driver-API
binding. OpenCL, even in the header-free form used above, still needs a runtime
`dlopen` of a vendor library and a way to call into it, which under
`CGO_ENABLED=0` means either purego-style assembly thunks or a third-party
package. Calling it "no dependency" because the loader ships with the driver
would be a word game. The dependency is the driver, and it is not in the binary,
and it is not portable across the release matrix.

### The build

This is the hard blocker and it is worth stating plainly. The current pipeline
cross-compiles to every target from one runner with cgo off. A CUDA build cannot
be produced that way at all. Even the OpenCL path cannot: `CGO_ENABLED=0` means
no C calls, and turning cgo on means a per-target C toolchain, which means the
matrix stops being a matrix. Adding a GPU backend is not a new file in
`internal/tensor`. It is a different release model, and a second binary flavour
to test and support.

## Recommendation

**Do not do this yet.** Not never, but not now, and not before the things ahead
of it.

The evidence for that, in order of weight:

1. **The workload is not there.** Twill's benchmarks are 64x64 matmul, 128x128
   broadcast add, 256x32 softmax. `minParallel` is 8192, so most ops do not even
   go multi-core today. At those sizes the GPU is measurably slower. Buying a
   GPU backend now means buying a 0.8x.
2. **f64 throws away 95% of the card.** 342 GFLOP/s out of 18 TFLOP/s. The
   biggest single GPU win available to Twill is not a GPU backend at all, it is
   f32 support, which is worth up to 52.7x on the same silicon and costs a type
   parameter rather than a rewrite. That question should be settled first,
   because a device-resident f64 tensor engine is a lot of work aimed at the
   slow path of the hardware.
3. **The ceiling is 15x, not 30x.** `docs/rewrite-plan.md` should be corrected.
   15x is still more than the Rust rewrite's 2.6x to 3.2x, so this does not
   settle the roadmap ranking on its own, but it should be ranked on the real
   number and only for large dense matmul.
4. **It contradicts stated positions.** `docs/finance.md` already lists
   "competing with CUDA/PyTorch on large dense deep learning under a pure-Go, no
   native deps constraint" as a non-goal. Nothing measured here argues that
   non-goal was wrong. It argues it was right.

What is worth doing instead, cheaply:

- Add matmul benchmarks at N=256, 512, 1024. They are the sizes where any of
  this becomes a question, and right now there is no CPU baseline at 512 or 1024
  in the repo.
- Treat f32 as the open performance question rather than GPU.
- If someone does want to prototype, use **OpenCL**, not CUDA. It ran on both
  GPUs here with no SDK, no container, and no admin, and it keeps the Arc part
  in play. Keep it out of tree and behind a build tag so the default binary and
  the release matrix stay exactly as they are.

Revisit when there is a real Twill program whose matmuls are 256x256 or larger
and whose profile is matmul-bound. Until such a program exists, this is
optimising a workload nobody has run.

## Reproducing

The probes are standalone C and Go, not part of the build. The OpenCL ones need
only `gcc` and a GPU driver:

```
gcc -O2 clprobe.c -o clprobe.exe && ./clprobe.exe   # devices, fp64, f32:f64 ratio
gcc -O2 clrt.c    -o clrt.exe    && ./clrt.exe      # matmul kernel vs round trip
```

The CPU column is a verbatim copy of `internal/tensor.mm` and `runChunks` run
under `go run`, so it is the same code path the interpreter uses.
