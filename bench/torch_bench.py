"""PyTorch counterparts of the twill workloads under bench/workloads.

Every workload here computes the same mathematics as the .tw file of the same
name, is timed the same way (setup outside the loop, warmup discarded, median
and p99 rather than mean, inner repeats calibrated against the clock), and
prints the same result value so the two sides can be checked to be doing the
same arithmetic rather than merely arithmetic of the same shape.

Two things make this a fair comparison rather than a rigged one.

Precision. twill's only float type is f64. PyTorch's default is f32, which is
half the bytes to move and twice the SIMD lanes per instruction, so comparing
twill-f64 against torch-f32 would charge twill for a design choice rather than
for its kernels. Both dtypes are run and both are reported: f64 is the
like-for-like number, f32 is what a PyTorch user would actually get.

Threads. Fixing both sides at the machine's core count turned out to be the
unfair option, not the fair one. PyTorch's intra-op pool has a pathological
regime on this hardware: on a workload whose individual kernels are a fraction
of a millisecond, the pool's barrier cost swamps the work and sixteen threads
run more than an order of magnitude slower than one. Reporting that as
"PyTorch's number" would flatter twill for a reason that has nothing to do with
twill. So both sides are swept over a range of thread counts and each is
reported at its own best, with the full sweep kept in the JSON so the pathology
is visible rather than merely averaged away.

Run:
    python bench/torch_bench.py --threads 1,2,4,8,16 --runs 30 --warmup 5
"""

import argparse
import json
import math
import platform
import statistics
import sys
import time

import torch


def quantile(sorted_samples, q):
    """Nearest-rank quantile, matching twillbench's definition exactly."""
    if not sorted_samples:
        return 0.0
    i = int(q * len(sorted_samples))
    return sorted_samples[min(i, len(sorted_samples) - 1)]


def bench(name, setup, dtype, runs, warmup, target_span=0.020):
    """Time one workload. setup(dtype) returns a nullary callable holding the work."""
    work = setup(dtype)

    last = None
    for _ in range(warmup):
        last = work()

    # Same calibration as twillbench: repeat until the timed span clears the
    # clock's resolution, then divide. perf_counter is high resolution on
    # Windows, but the inner loop is kept anyway so both sides measure the same
    # quantity and a row with inner > 1 is marked as such on both.
    inner = 1
    while True:
        start = time.perf_counter()
        for _ in range(inner):
            last = work()
        span = time.perf_counter() - start
        if span >= target_span or inner >= 1 << 16:
            break
        if span <= 0:
            inner *= 8
            continue
        nxt = int(inner * target_span / span * 1.2)
        inner = nxt if nxt > inner else inner * 2

    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        for _ in range(inner):
            last = work()
        samples.append((time.perf_counter() - start) * 1000.0 / inner)
    samples.sort()

    return {
        "workload": name,
        "impl": f"torch-{str(dtype).replace('torch.', '')}",
        "runs": runs,
        "warmup": warmup,
        "inner": inner,
        "median_ms": quantile(samples, 0.5),
        "p99_ms": quantile(samples, 0.99),
        "min_ms": samples[0],
        "max_ms": samples[-1],
        "result": fmt_result(last),
    }


def fmt_result(v):
    if isinstance(v, torch.Tensor):
        v = v.detach()
        if v.numel() == 1:
            return f"{float(v):.6f}"
        # A short vector is printed elementwise, so a verification workload can
        # be compared term by term rather than through a sum that could hide a
        # compensating pair of errors.
        if v.numel() <= 8:
            return "[" + ", ".join(f"{float(e):.6f}" for e in v.flatten()) + "]"
        return f"{float(v.sum()):.6f}"
    return f"{float(v):.6f}"


# --------------------------------------------------------------------------
# Workloads. Each mirrors the .tw file of the same name.
# --------------------------------------------------------------------------


def mc_option_fwd(dtype):
    # Same shocks every run, as the .tw file does: the RNG is not the workload.
    g = torch.Generator().manual_seed(42)
    Z = torch.randn(200000, generator=g, dtype=torch.float64).to(dtype)

    def price(S0, K, r, sigma, T):
        drift = (r - 0.5 * sigma * sigma) * T
        ST = S0 * torch.exp(drift + sigma * math.sqrt(T) * Z)
        return math.exp(-r * T) * torch.relu(ST - K).mean()

    return lambda: price(100.0, 100.0, 0.05, 0.2, 1.0)


def mc_option_grad(dtype):
    g = torch.Generator().manual_seed(42)
    Z = torch.randn(200000, generator=g, dtype=torch.float64).to(dtype)

    def price(S0, K, r, sigma, T):
        drift = (r - 0.5 * sigma * sigma) * T
        ST = S0 * torch.exp(drift + sigma * torch.sqrt(T) * Z)
        return torch.exp(-r * T) * torch.relu(ST - K).mean()

    def work():
        # Two separate reverse passes, matching the two grad() calls in the .tw
        # file, rather than one pass returning both. A single pass would be the
        # faster way to do it in both systems and would not be the same program.
        S0 = torch.tensor(100.0, dtype=dtype, requires_grad=True)
        T = torch.tensor(1.0, dtype=dtype)
        p = price(S0, 100.0, 0.05, 0.2, T)
        (delta,) = torch.autograd.grad(p, S0)

        sigma = torch.tensor(0.2, dtype=dtype, requires_grad=True)
        p2 = price(100.0, 100.0, 0.05, sigma, T)
        (vega,) = torch.autograd.grad(p2, sigma)
        return delta + vega

    return work


def matmul(n):
    def setup(dtype):
        g = torch.Generator().manual_seed(1)
        A = torch.randn(n, n, generator=g, dtype=torch.float64).to(dtype)
        B = torch.randn(n, n, generator=g, dtype=torch.float64).to(dtype)
        return lambda: (A @ B).sum()

    return setup


def matmul_512_grad(dtype):
    g = torch.Generator().manual_seed(1)
    B = torch.randn(512, 512, generator=g, dtype=torch.float64).to(dtype)
    A0 = torch.randn(512, 512, generator=g, dtype=torch.float64).to(dtype)

    def work():
        A = A0.clone().requires_grad_(True)
        (gA,) = torch.autograd.grad((A @ B).sum(), A)
        return gA.sum()

    return work


def elementwise(n):
    def setup(dtype):
        g = torch.Generator().manual_seed(1)
        x = torch.randn(n, generator=g, dtype=torch.float64).to(dtype)
        return lambda: torch.tanh(torch.exp(x / 4.0) * x).mean()

    return setup


def elementwise_grad(n):
    def setup(dtype):
        g = torch.Generator().manual_seed(1)
        x0 = torch.randn(n, generator=g, dtype=torch.float64).to(dtype)

        def work():
            x = x0.clone().requires_grad_(True)
            loss = torch.tanh(torch.exp(x / 4.0) * x).mean()
            (gx,) = torch.autograd.grad(loss, x)
            return gx.sum()

        return work

    return setup


def mlp_train_step(dtype):
    g = torch.Generator().manual_seed(1)
    X = torch.randn(64, 256, generator=g, dtype=torch.float64).to(dtype)
    W10 = (torch.randn(512, 256, generator=g, dtype=torch.float64) * 0.05).to(dtype)
    W2 = (torch.randn(256, 512, generator=g, dtype=torch.float64) * 0.05).to(dtype)
    W3 = (torch.randn(10, 256, generator=g, dtype=torch.float64) * 0.05).to(dtype)
    Y = torch.full((64, 10), 0.1, dtype=dtype)

    def work():
        W1 = W10.clone().requires_grad_(True)
        h1 = torch.relu(torch.nn.functional.linear(X, W1))
        h2 = torch.relu(torch.nn.functional.linear(h1, W2))
        logits = torch.nn.functional.linear(h2, W3)
        lse = torch.logsumexp(logits, dim=1, keepdim=True)
        loss = -(Y * (logits - lse)).sum(dim=1).mean()
        (gW1,) = torch.autograd.grad(loss, W1)
        return gW1.sum()

    return work


def attention_head(dtype):
    g = torch.Generator().manual_seed(1)
    X = torch.randn(256, 64, generator=g, dtype=torch.float64).to(dtype)
    return lambda: (torch.softmax(X @ X.T, dim=1) @ X).sum()



def verify_deterministic(dtype):
    """Mirror of bench/workloads/verify_deterministic.tw, with no RNG anywhere.

    Every input is a fixed function of its index, so both sides build bit-
    identical inputs and the two results are comparable as numbers. This is the
    workload that shows the two implementations compute the same mathematics;
    the others show how fast they do it.
    """
    n = 4096
    idx = torch.arange(n, dtype=torch.float64)
    x0 = (torch.sin(idx * 0.001) * 2.0 + 0.5).to(dtype)
    A = torch.sin(torch.arange(64 * 64, dtype=torch.float64) * 0.01).reshape(64, 64).to(dtype)
    B = torch.cos(torch.arange(64 * 64, dtype=torch.float64) * 0.02).reshape(64, 64).to(dtype)

    def loss(v):
        h = torch.tanh(torch.exp(v / 4.0) * v)
        m = h[0:4096].reshape(64, 64)
        p = torch.softmax(A @ (m + B), dim=1)
        return (
            h.mean()
            + (p * m).sum()
            + torch.logsumexp(m, dim=0).sum()
            + torch.relu(m - 0.25).mean()
        )

    def work():
        v = x0.clone().requires_grad_(True)
        (g,) = torch.autograd.grad(loss(v), v)
        with torch.no_grad():
            return torch.stack([loss(x0), g.sum(), (g * g).sum()])

    return work


WORKLOADS = {
    "verify_deterministic": verify_deterministic,
    "attention_head": attention_head,
    "elementwise_10000": elementwise(10000),
    "elementwise_100000": elementwise(100000),
    "elementwise_1000000": elementwise(1000000),
    "elementwise_10000000": elementwise(10000000),
    "elementwise_grad_100000": elementwise_grad(100000),
    "elementwise_grad_1000000": elementwise_grad(1000000),
    "matmul_128": matmul(128),
    "matmul_256": matmul(256),
    "matmul_512": matmul(512),
    "matmul_1024": matmul(1024),
    "matmul_512_grad": matmul_512_grad,
    "mc_option_fwd": mc_option_fwd,
    "mc_option_grad": mc_option_grad,
    "mlp_train_step": mlp_train_step,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--threads",
        default="1,2,4,8,16",
        help="comma-separated thread counts to sweep; each workload is reported at its best",
    )
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    thread_counts = [int(t) for t in args.threads.split(",") if t.strip()]

    print(
        f"# torch_bench: torch={torch.__version__} threads_swept={thread_counts} "
        f"cuda_available={torch.cuda.is_available()} python={platform.python_version()}"
    )
    print(
        f"{'workload':<26} {'dtype':>5} {'thr':>4} {'inner':>6} {'median_ms':>11} "
        f"{'p99_ms':>11} {'min_ms':>11} {'max_ms':>11}  result"
    )

    results = []
    for name, setup in WORKLOADS.items():
        if args.only and args.only not in name:
            continue
        for dtype in (torch.float64, torch.float32):
            sweep = []
            for t in thread_counts:
                torch.set_num_threads(t)
                r = bench(name, setup, dtype, args.runs, args.warmup)
                r["threads"] = t
                sweep.append(r)
                results.append(r)
            best = min(sweep, key=lambda r: r["median_ms"])
            best["best_of_sweep"] = True
            short = "f64" if dtype is torch.float64 else "f32"
            for r in sweep:
                mark = " <- best" if r is best else ""
                print(
                    f"{name:<26} {short:>5} {r['threads']:>4} {r['inner']:>6} "
                    f"{r['median_ms']:>11.4f} {r['p99_ms']:>11.4f} {r['min_ms']:>11.4f} "
                    f"{r['max_ms']:>11.4f}  {r['result']}{mark}"
                )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
