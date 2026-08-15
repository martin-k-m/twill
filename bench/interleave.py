"""Interleaved twill-against-PyTorch comparison, to control for thermal drift.

Why this exists. The machine these benchmarks run on is a laptop, and a laptop
under sustained all-core load does not hold its clocks. Measured across three
full sweeps taken over about forty minutes, the same twill workload drifted by
up to 75%: `mc_option_fwd` reported 1.985 ms on a cold machine and 3.450 ms on a
hot one, with nothing changed but the temperature.

That makes the sequential design of run_all.ps1 unsound for the comparison it is
used for. It measures every twill workload first and every PyTorch workload
afterwards, so twill is timed on a cool machine and PyTorch on a hot one, which
is a systematic bias in twill's favour of the same order as some of the gaps
being reported.

This script removes it. For each workload it alternates twill and PyTorch
measurements within the same few seconds, repeats that for several rounds, and
reports the median ratio across rounds. Both sides see the same thermal state on
every round, so whatever the clocks are doing, they are doing it to both.

Each side runs at the best setting its sweep found, read out of the sweep JSON,
so neither is handicapped by a thread count that happens to suit the other.

Run, from the repository root, after run_all.ps1 has produced the sweeps:

    python bench/interleave.py --python <interpreter-with-torch>
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

# The workloads worth the extra time: one per regime, plus the two the README
# and docs/CODEGEN.md quote directly.
DEFAULT_WORKLOADS = [
    "mc_option_fwd",
    "mc_option_grad",
    "matmul_512",
    "matmul_512_grad",
    "elementwise_1000000",
    "attention_head",
    "mlp_train_step",
    "verify_deterministic",
]


def best_settings(twill_json, torch_json):
    """Read each side's best thread count per workload out of the sweep results."""
    procs, threads = {}, {}
    if os.path.exists(twill_json):
        for r in json.load(open(twill_json)):
            if r.get("best_of_sweep"):
                procs[r["workload"]] = r["gomaxprocs"]
    if os.path.exists(torch_json):
        for r in json.load(open(torch_json)):
            if r.get("best_of_sweep"):
                threads[(r["workload"], r["impl"])] = r["threads"]
    return procs, threads


def build_twillbench():
    """Compile the harness once.

    `go run` recompiles on every invocation, and a compile between two timed
    measurements is exactly the contention this script exists to avoid. The
    first version of this script used it and produced a PyTorch number twelve
    times worse than the same workload in the sweep.
    """
    exe = os.path.join(tempfile.gettempdir(), "twillbench_interleave.exe")
    subprocess.run(["go", "build", "-o", exe, "./bench/cmd/twillbench"], check=True)
    return exe


def run_twill(exe, workload, procs, runs):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out = f.name
    try:
        subprocess.run(
            [exe, "-only", workload,
             "-procs", str(procs), "-runs", str(runs), "-warmup", "3", "-out", out],
            check=True, stdout=subprocess.DEVNULL,
        )
        rows = json.load(open(out))
        return rows[0]["median_ms"] if rows else None
    finally:
        os.unlink(out)


def run_torch(python, workload, threads, runs):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out = f.name
    try:
        subprocess.run(
            [python, "bench/torch_bench.py", "--only", workload,
             "--threads", str(threads), "--runs", str(runs), "--warmup", "3", "--out", out],
            check=True, stdout=subprocess.DEVNULL,
        )
        rows = json.load(open(out))
        by = {}
        for r in rows:
            by[r["impl"]] = r["median_ms"]
        return by.get("torch-float64"), by.get("torch-float32")
    finally:
        os.unlink(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--runs", type=int, default=12, help="timed runs inside each measurement")
    ap.add_argument("--results", default="bench/results")
    ap.add_argument("--workloads", default=",".join(DEFAULT_WORKLOADS))
    ap.add_argument("--out", default="bench/results/interleaved.json")
    ap.add_argument("--settle", type=float, default=1.0,
                    help="seconds of idle between measurements, so one side's "
                         "thread pool has wound down before the other starts")
    args = ap.parse_args()

    exe = build_twillbench()

    procs, threads = best_settings(
        os.path.join(args.results, "twill.json"), os.path.join(args.results, "torch.json")
    )
    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]

    samples = {w: {"twill": [], "f64": [], "f32": []} for w in workloads}

    for rnd in range(args.rounds):
        for w in workloads:
            p = procs.get(w, 16)
            t64 = threads.get((w, "torch-float64"), 8)
            t32 = threads.get((w, "torch-float32"), 8)
            # twill and PyTorch back to back, so the round is one thermal state.
            time.sleep(args.settle)
            tw = run_twill(exe, w, p, args.runs)
            time.sleep(args.settle)
            f64, f32 = run_torch(args.python, w, t64, args.runs)
            if tw:
                samples[w]["twill"].append(tw)
            if f64:
                samples[w]["f64"].append(f64)
            if f32:
                samples[w]["f32"].append(f32)
            print(f"round {rnd + 1}/{args.rounds}  {w:<26} "
                  f"twill {tw:8.3f} (p{p})   f64 {f64:8.3f} (t{t64})   f32 {f32:8.3f} (t{t32})",
                  flush=True)

    print()
    print(f"{'workload':<26}{'twill_ms':>10}{'t64_ms':>10}{'t32_ms':>10}"
          f"{'x_f64':>8}{'x_f32':>8}{'drift':>8}")
    out = []
    for w in workloads:
        s = samples[w]
        if not s["twill"] or not s["f64"]:
            continue
        tw = statistics.median(s["twill"])
        f64 = statistics.median(s["f64"])
        f32 = statistics.median(s["f32"]) if s["f32"] else float("nan")
        # Per-round ratios, then the median of those. Taking the ratio inside the
        # round is the whole point: it cancels whatever the clocks did.
        r64 = statistics.median([a / b for a, b in zip(s["twill"], s["f64"])])
        r32 = (statistics.median([a / b for a, b in zip(s["twill"], s["f32"])])
               if s["f32"] else float("nan"))
        # How much twill's own measurement moved across rounds, as a check that
        # the drift being corrected for is real and how large it was.
        drift = (max(s["twill"]) - min(s["twill"])) / min(s["twill"]) * 100
        print(f"{w:<26}{tw:>10.3f}{f64:>10.3f}{f32:>10.3f}{r64:>8.1f}{r32:>8.1f}{drift:>7.0f}%")
        out.append({
            "workload": w, "rounds": args.rounds,
            "twill_median_ms": tw, "torch_f64_median_ms": f64, "torch_f32_median_ms": f32,
            "ratio_vs_f64": r64, "ratio_vs_f32": r32, "twill_drift_pct": drift,
            "twill_samples": s["twill"], "torch_f64_samples": s["f64"],
            "torch_f32_samples": s["f32"],
        })

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
