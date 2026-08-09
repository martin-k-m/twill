//! Threading, ported from `internal/tensor/parallel.go`.
//!
//! The contract carried over verbatim: chunking never changes a program's
//! output. Every body here writes only its own disjoint output range, and the
//! one reduction that could not honour that (a sum) blocks by a fixed size
//! instead of by core count, so the answer does not depend on the machine.

use std::thread::available_parallelism;

/// Work size below which an op runs serially. Thread setup is not worth it for
/// tensors the size of a typical training parameter.
pub const MIN_PARALLEL: usize = 8192;

/// Fixed block size for [`parallel_sum`]. Fixed rather than core-derived so the
/// result is bit-identical on any machine, which costs a little parallel
/// efficiency on very wide hosts and buys the reproducibility guarantee.
pub const SUM_CHUNK: usize = 4096;

pub fn num_workers() -> usize {
    available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// Worker count for a task of the given total work.
pub fn workers_for(work: usize) -> usize {
    if work < MIN_PARALLEL {
        1
    } else {
        num_workers()
    }
}

/// Splits `n` logical units across `workers` scoped threads. `out` holds
/// `stride` elements per unit and is split so each thread owns its slice
/// outright; that is what makes the race-freedom a compile-time fact here
/// rather than a comment as it is in the Go original.
pub fn run_chunks<F>(n: usize, workers: usize, stride: usize, out: &mut [f64], body: F)
where
    F: Fn(usize, &mut [f64]) + Sync,
{
    if workers < 2 || n < 2 {
        body(0, out);
        return;
    }
    let w = workers.min(n);
    let chunk = n.div_ceil(w);
    std::thread::scope(|s| {
        let mut lo = 0usize;
        let mut rest = out;
        while lo < n {
            let hi = (lo + chunk).min(n);
            let (head, tail) = rest.split_at_mut((hi - lo) * stride);
            rest = tail;
            let b = &body;
            s.spawn(move || b(lo, head));
            lo = hi;
        }
    });
}

/// Runs `body` over `[0, n)` across cores when `n` is large enough to pay for
/// it. `out` must have exactly `n` elements; `body` receives the chunk's start
/// index and the chunk itself.
pub fn parallel_for<F>(n: usize, out: &mut [f64], body: F)
where
    F: Fn(usize, &mut [f64]) + Sync,
{
    let workers = if n >= MIN_PARALLEL { num_workers() } else { 1 };
    run_chunks(n, workers, 1, out, body);
}

/// Adds a slice deterministically: fixed-size blocks summed independently, then
/// their partials combined in index order.
pub fn parallel_sum(data: &[f64]) -> f64 {
    let n = data.len();
    if n < MIN_PARALLEL {
        let mut s = 0.0;
        for &x in data {
            s += x;
        }
        return s;
    }
    let nblocks = n.div_ceil(SUM_CHUNK);
    let mut partials = vec![0.0; nblocks];
    run_chunks(nblocks, workers_for(n), 1, &mut partials, |lo, slots| {
        for (bi, slot) in slots.iter_mut().enumerate() {
            let start = (lo + bi) * SUM_CHUNK;
            let end = (start + SUM_CHUNK).min(n);
            let mut s = 0.0;
            for &x in &data[start..end] {
                s += x;
            }
            *slot = s;
        }
    });
    let mut total = 0.0;
    for &p in &partials {
        total += p;
    }
    total
}
